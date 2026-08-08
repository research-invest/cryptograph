"""
Инкрементальный прогон: добрать свежие бары, разметить их уже обученной
моделью состояний и выпустить кандидатов по последним переходам.

Модель состояний здесь не переобучается — иначе group_id поехали бы, и
накопленный в btc-graph граф перестал бы соответствовать реальности.
Переобучение — это отдельный полный прогон.
"""
from __future__ import annotations

import logging
import time

import pandas as pd

from btcproc import config
from btcproc.candidates import builder as cand
from btcproc.candidates.outcomes import compute_outcomes
from btcproc.db import repo, runs
from btcproc.features import builder as feat
from btcproc.features import events as ev
from btcproc.ingest import binance
from btcproc.pipeline.train import emit_pending
from btcproc.states import assign, graph

logger = logging.getLogger(__name__)


def run_live(
    *,
    model_run_id: int | None = None,
    lookback_minutes: int = 240,
    do_emit: bool = True,
    symbol: str | None = None,
) -> dict:
    """
    lookback_minutes — насколько назад от последнего бара считать кандидатов
    «свежими». Всё старше уже выпускалось прошлыми запусками.
    """
    symbol = symbol or config.data.symbol
    started = time.time()

    source = (
        runs.get_run(model_run_id) if model_run_id else runs.latest_completed_run("train")
    )
    if not source:
        raise RuntimeError(
            "Нет обученной модели состояний — сначала выполни полный прогон (train)."
        )
    model_run_id = int(source["run_id"])
    model = repo.load_state_model(model_run_id)
    if model is None:
        raise RuntimeError(f"У прогона {model_run_id} нет сохранённой модели состояний.")

    run_id = runs.start_run(
        "live", {"model_run_id": model_run_id, "lookback_minutes": lookback_minutes}
    )
    stats: dict = {"run_id": run_id, "model_run_id": model_run_id}

    try:
        runs.log(run_id, "Добор свежих баров", stage="ingest", progress=0.1)
        new_bars = binance.sync_recent(symbol, config.data.base_tf)
        binance.rebuild_context_timeframes(symbol)
        stats["new_bars"] = new_bars

        base = binance.load_ohlcv(symbol, config.data.base_tf)
        context = {tf: binance.load_ohlcv(symbol, tf) for tf in config.data.context_tfs}

        runs.log(run_id, "Признаки и события", stage="features", progress=0.3)
        features = feat.build_features(base, context)
        missing = set(model.feature_names) - set(features.columns)
        if missing:
            raise RuntimeError(
                f"Набор признаков изменился с момента обучения модели "
                f"(прогон {model_run_id}): не хватает {sorted(missing)}. "
                "Нужен новый полный прогон train."
            )
        # Порядок колонок должен совпадать с тем, на котором обучалась модель.
        features = features[model.feature_names]
        matrix = feat.apply_scale(features, model.scale)
        labels = model.predict(matrix)
        states = assign.assign_states(features.index, labels)

        events = ev.build_event_blocks(base).reindex(features.index).dropna(
            subset=["event_block_id"]
        )
        blocks = ev.block_statistics(events)
        outcomes = compute_outcomes(base).reindex(features.index)

        runs.log(run_id, "Разметка и граф", stage="states", progress=0.5)
        repo.save_bar_states(run_id, states, symbol)
        transitions = graph.transition_stats(states, outcomes)
        rarity_map = dict(zip(transitions["transition_id"], transitions["rarity"]))
        block_map = blocks.set_index("event_block_id").to_dict("index")

        runs.log(run_id, "Сборка кандидатов", stage="candidates", progress=0.7)
        snapshots = cand.build_snapshots(states, events, outcomes)
        cutoff = features.index[-1] - pd.Timedelta(minutes=lookback_minutes)

        fresh = []
        for candidate in cand.generate(snapshots, rarity_map, block_map, symbol):
            ts = pd.Timestamp(candidate["_meta"]["ts"])
            if ts >= cutoff:
                fresh.append(candidate)
        if fresh:
            repo.save_candidates(run_id, fresh)
        stats["candidates"] = len(fresh)
        runs.log(run_id, f"Свежих кандидатов: {len(fresh)}", progress=0.85)

        if do_emit and fresh and config.sink.mode != "none":
            stats["emit"] = emit_pending(run_id)
        else:
            stats["emit"] = {"sent": 0, "skipped": True}

        stats["seconds"] = round(time.time() - started, 1)
        runs.finish_run(run_id, stats)
        return stats

    except Exception as exc:  # noqa: BLE001
        logger.exception("Live-прогон %s упал", run_id)
        runs.fail_run(run_id, f"{type(exc).__name__}: {exc}")
        raise
