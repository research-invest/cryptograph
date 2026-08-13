"""
Проигрывание истории уже обученной моделью состояний.

Штатных прогонов два, и оба не годятся, когда нужно пересобрать кандидатов
существующей моделью: `train` переобучает состояния (перенумеровывает
`group_id` и сносит граф монеты в Neo4j — инвариант 13), а `live` выпускает
кандидатов только от точки продолжения.

Между ними и живёт этот модуль: разметить ВСЮ историю моделью конкретного
прогона, ничего не переобучая и ничего не записывая. На нём стоят замер
`scripts/measure_block_a.py` и выгрузка `scripts/dump_candidates.py` —
раньше каждый нёс свою копию этих сорока строк, и копии успели разъехаться
в том, откуда берутся карты редкости.

**Карты редкости считаются по всей истории — как в train и live.** Это
незакрытый look-ahead (A7 внешнего аудита), и здесь он воспроизводится
намеренно: модуль обязан давать ровно то, что даёт боевой расчёт. Валидация
на отложенной части, которой нужны карты по префиксу, пользуется
`analysis.holdout.prefix_maps` и этот модуль не зовёт.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from btcproc import config, symbols
from btcproc.candidates import builder as cand
from btcproc.candidates.outcomes import compute_outcomes
from btcproc.db import repo, runs
from btcproc.features import builder as feat
from btcproc.features import events as ev
from btcproc.ingest import bars
from btcproc.states import assign, graph

logger = logging.getLogger(__name__)


class ReplayError(RuntimeError):
    """Историю нечем или не по чему размечать — вызывающий печатает диагноз."""


@dataclass
class Replay:
    """Размеченная история и всё, из чего собираются кандидаты."""

    symbol: str
    model_run: int
    bars: pd.DataFrame
    states: pd.DataFrame
    events: pd.DataFrame
    outcomes: pd.DataFrame
    snapshots: pd.DataFrame
    rarity: dict[str, str]
    blocks: dict[str, dict]

    def candidates(self, cfg: config.CandidateConfig | None = None):
        """Кандидаты по всей истории, в хронологическом порядке."""
        return cand.generate(
            self.snapshots, self.rarity, self.blocks, self.symbol, cfg=cfg
        )


def label_history(
    symbol: str,
    end: str | None = None,
    model_run: int | None = None,
    start: str | None = None,
    log=logger.info,
) -> Replay:
    """
    Размечает историю монеты моделью её последнего `train` (или указанного).

    Ничего не пишет в БД и ничего не отправляет: вызывающий сам решает, что
    делать с результатом.
    """
    spec = symbols.get(symbol)
    start = start or spec.start_date()

    if model_run is None:
        run = runs.latest_completed_run("train", symbol)
        if run is None:
            raise ReplayError(
                f"{symbol}: нет успешного train — модели состояний не существует."
            )
        model_run = int(run["run_id"])
    else:
        run = runs.get_run(model_run)
        if run is None:
            raise ReplayError(f"Прогона #{model_run} нет.")
        # Центроиды обучены в пространстве признаков конкретной монеты:
        # чужой моделью история размечается молча и полностью бессмысленно.
        if run.get("symbol") and run["symbol"] != symbol:
            raise ReplayError(
                f"Прогон #{model_run} обучен на {run['symbol']}, а размечать "
                f"просят {symbol}. Модели состояний между монетами несопоставимы."
            )

    model = repo.load_state_model(model_run)
    if model is None:
        raise ReplayError(f"У прогона #{model_run} нет сохранённой модели состояний.")

    base = bars.load_ohlcv(symbol, config.data.base_tf, start, end)
    if base.empty:
        raise ReplayError(f"{symbol}: в БД нет баров за {start}…{end or 'конец'}.")
    context = {
        tf: bars.load_ohlcv(symbol, tf, start, end) for tf in config.data.context_tfs
    }

    log(f"[{symbol}] модель прогона #{model_run}, баров {len(base)}; признаки…")
    features = feat.build_features(base, context)
    missing = set(model.feature_names) - set(features.columns)
    if missing:
        raise ReplayError(
            f"{symbol}: набор признаков разошёлся с моделью #{model_run} — "
            f"не хватает {sorted(missing)}. Нужен новый train."
        )
    features = features[model.feature_names]

    labels = model.predict(feat.apply_scale(features, model.scale))
    states = assign.assign_states(features.index, labels)
    events = ev.build_event_blocks(base).reindex(features.index).dropna(
        subset=["event_block_id"]
    )
    outcomes = compute_outcomes(base).reindex(features.index)

    transitions = graph.transition_stats(states, outcomes)
    rarity = dict(zip(transitions["transition_id"], transitions["rarity"]))
    blocks = ev.block_statistics(events).set_index("event_block_id").to_dict("index")

    snapshots = cand.build_snapshots(states, events, outcomes)
    log(f"[{symbol}] снимков конфигураций: {len(snapshots)}")

    return Replay(
        symbol=symbol, model_run=model_run, bars=base, states=states,
        events=events, outcomes=outcomes, snapshots=snapshots,
        rarity=rarity, blocks=blocks,
    )
