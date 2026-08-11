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

from btcproc import config, notify, symbols
from btcproc.candidates import builder as cand
from btcproc.candidates.outcomes import compute_outcomes
from btcproc.db import repo, runs
from btcproc.features import builder as feat
from btcproc.features import events as ev
from btcproc.ingest import binance
from btcproc.pipeline.train import emit_pending
from btcproc.states import assign, graph

logger = logging.getLogger(__name__)


DEFAULT_LOOKBACK_MINUTES = 240
MAX_LOOKBACK_DAYS = 30


def bar_states_window(states: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Какой кусок разметки live-прогон записывает в `bar_states`.

    Размечается по-прежнему вся история — это нужно аккумуляторам кандидатов,
    сглаживанию и энтропии траектории. А вот ПИСАТЬ всю историю под каждый
    свежий run_id нельзя: PK таблицы — `(symbol, ts, run_id)`, то есть каждый
    live делает не upsert поверх старого, а чистые вставки на весь объём.
    При ~300 тыс. баров и прогоне раз в полчаса на три монеты это порядка
    30 млн строк в сутки, которые не удаляет никто: retention у таблицы нет,
    а `prune_runs` сознательно бережёт live-прогоны действующей модели.

    Технологический запас перед cutoff нужен, чтобы записанный кусок был
    самодостаточен для чтения: сглаживание переписывает начало серии задним
    числом (`smoothing_bars`), энтропия считается по окну траектории
    (`trajectory_window`), а снимки кандидатов смотрят назад на максимальный
    офсет (`SNAPSHOT_OFFSETS_MIN`).
    """
    from btcproc.candidates.builder import SNAPSHOT_OFFSETS_MIN

    base_minutes = config.data.base_minutes
    margin_bars = (
        config.states.smoothing_bars
        + config.states.trajectory_window
        + -(-max(SNAPSHOT_OFFSETS_MIN) // base_minutes)  # ceil
    )
    start = cutoff - pd.Timedelta(minutes=margin_bars * base_minutes)
    return states[states.index >= start]


def resolve_cutoff(
    last_bar: pd.Timestamp,
    last_candidate: pd.Timestamp | None,
    lookback_minutes: int | None,
    max_lookback_days: int = MAX_LOOKBACK_DAYS,
) -> tuple[pd.Timestamp, str]:
    """
    С какого момента выпускать кандидатов.

    По умолчанию (lookback_minutes=None) точка берётся не из календаря, а из
    данных: продолжаем с самого свежего уже выпущенного кандидата. Иначе
    запуск в понедельник после пятничного прогона потерял бы все выходные —
    бары бы догрузились, а кандидаты по ним нет.

    Верхняя граница (max_lookback_days) страхует от первого запуска на давно
    заполненной базе: без неё пауза в полгода вылилась бы в разовый выпуск
    десятков тысяч кандидатов.

    Возвращает границу и причину — она пишется в лог прогона.
    """
    if lookback_minutes is not None:
        cutoff = last_bar - pd.Timedelta(minutes=lookback_minutes)
        # Явное окно короче фактического пропуска = дыра, которую не закроет
        # уже никто: кандидаты за интервал не выпустятся, но last_candidate_ts
        # уедет в свежее время, и следующий запуск продолжит с него.
        if last_candidate is not None and last_candidate < cutoff:
            return cutoff, (
                f"окно задано явно: {lookback_minutes} мин. "
                f"ВНИМАНИЕ: последний кандидат от {last_candidate:%Y-%m-%d %H:%M} "
                f"старше начала окна ({cutoff:%Y-%m-%d %H:%M}) — интервал между "
                f"ними останется без кандидатов навсегда. Для «догнать пропуск» "
                f"запусти live без явного окна."
            )
        return cutoff, f"окно задано явно: {lookback_minutes} мин"

    limit = last_bar - pd.Timedelta(days=max_lookback_days)
    if last_candidate is None:
        return (
            last_bar - pd.Timedelta(minutes=DEFAULT_LOOKBACK_MINUTES),
            f"кандидатов ещё не было, окно по умолчанию {DEFAULT_LOOKBACK_MINUTES} мин",
        )
    if last_candidate < limit:
        return limit, (
            f"последний кандидат от {last_candidate:%Y-%m-%d %H:%M} старше "
            f"{max_lookback_days} дней — окно обрезано"
        )
    gap_hours = (last_bar - last_candidate).total_seconds() / 3600
    return last_candidate, (
        f"продолжаем с {last_candidate:%Y-%m-%d %H:%M} "
        f"(пропуск {gap_hours:.1f} ч)"
    )


def run_live(
    *,
    model_run_id: int | None = None,
    lookback_minutes: int | None = None,
    max_lookback_days: int = MAX_LOOKBACK_DAYS,
    do_emit: bool = True,
    symbol: str | None = None,
) -> dict:
    """
    lookback_minutes=None — продолжить с последнего выпущенного кандидата
    (см. resolve_cutoff). Число — жёсткое окно в минутах от последнего бара.
    """
    spec = symbols.resolve(symbol)
    symbol = spec.ticker
    started = time.time()

    # Модель ищется среди прогонов ЭТОЙ монеты. Без фильтра live по ETH взял бы
    # модель последнего train'а вообще — то есть разметил бы бары ETH
    # состояниями BTC, и обнаружилось бы это только по бессмысленным кандидатам.
    source = (
        runs.get_run(model_run_id) if model_run_id
        else runs.latest_completed_run("train", symbol)
    )
    if not source:
        raise RuntimeError(
            f"Нет обученной модели состояний для {symbol} — выполни "
            f"train --symbol {symbol}."
        )

    # Явно переданный model_run обязан принадлежать той же монете: центроиды
    # обучены в пространстве признаков конкретного инструмента.
    run_symbol = source.get("symbol")
    if run_symbol and run_symbol != symbol:
        raise RuntimeError(
            f"Прогон #{source['run_id']} обучен на {run_symbol}, а live запущен "
            f"для {symbol}. Модели состояний между монетами несопоставимы."
        )

    model_run_id = int(source["run_id"])
    model = repo.load_state_model(model_run_id)
    if model is None:
        raise RuntimeError(f"У прогона {model_run_id} нет сохранённой модели состояний.")

    run_id = runs.start_run(
        "live",
        {
            "symbol": symbol,
            "model_run_id": model_run_id,
            "lookback_minutes": lookback_minutes or "auto",
            "max_lookback_days": max_lookback_days,
        },
        symbol=symbol,
    )
    stats: dict = {"run_id": run_id, "symbol": symbol, "model_run_id": model_run_id}

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

        # Cutoff нужен раньше записи разметки: под свежий run_id пишется только
        # окно от него (см. bar_states_window), а не вся размеченная история.
        cutoff, reason = resolve_cutoff(
            features.index[-1], repo.last_candidate_ts(symbol),
            lookback_minutes, max_lookback_days,
        )
        stats["cutoff"] = cutoff.isoformat()

        runs.log(run_id, "Разметка и граф", stage="states", progress=0.5)
        written = bar_states_window(states, cutoff)
        repo.save_bar_states(run_id, written, symbol)
        stats["bar_states"] = len(written)
        transitions = graph.transition_stats(states, outcomes)
        rarity_map = dict(zip(transitions["transition_id"], transitions["rarity"]))
        block_map = blocks.set_index("event_block_id").to_dict("index")

        runs.log(run_id, "Сборка кандидатов", stage="candidates", progress=0.7)
        snapshots = cand.build_snapshots(states, events, outcomes)
        runs.log(run_id, f"Кандидаты начиная с {cutoff:%Y-%m-%d %H:%M} — {reason}")

        # Граница включающая: перекрытие с прошлым запуском безопаснее дыры.
        # candidate_id детерминирован (symbol|ts|переход|блок|смещение), запись
        # идёт upsert'ом, а emitted_at при этом не сбрасывается — повторно
        # отправлен уже отправленный кандидат не будет.
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

        # Второй заход рассылки — для кандидатов, до оценки не доехавших:
        # отправка выключена (`--no-emit`, `SINK_MODE=none`) либо btc-graph
        # отсеял их своим фильтром. Повтора не будет: захват в журнале
        # доставок уникален по паре «правило + кандидат», и всё, что уже
        # ушло из emit_pending, здесь молча отсеется. Правила с фильтром по
        # рейтингу тут не сработают — рейтинга у неоценённого кандидата нет,
        # см. Rule.matches.
        if fresh:
            stats["notify"] = notify.dispatch(
                [c["candidate_id"] for c in fresh], reason=f"live run={run_id}"
            )
            # `duplicates` здесь — норма, а не тревога: это ровно те, кого уже
            # отправил emit_pending выше. Поэтому в лог пишем, только когда
            # что-то действительно ушло.
            if stats["notify"].get("queued"):
                runs.log(run_id, f"Уведомления: {runs.dumps(stats['notify'])}")
        notify.flush()

        stats["seconds"] = round(time.time() - started, 1)
        runs.finish_run(run_id, stats)
        return stats

    except Exception as exc:  # noqa: BLE001
        logger.exception("Live-прогон %s упал", run_id)
        runs.fail_run(run_id, f"{type(exc).__name__}: {exc}")
        raise
