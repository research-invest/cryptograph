"""
Выбор точки, с которой live-прогон продолжает выпускать кандидатов.

Смысл проверок один: пауза между запусками не должна создавать дыру.
Бары догружаются по max(ts) в БД, кандидаты — по max(ts) уже выпущенных;
ни то, ни другое не зависит от того, когда именно запустили процесс.
"""
from __future__ import annotations

import pandas as pd

from btcproc.pipeline.live import DEFAULT_LOOKBACK_MINUTES, resolve_cutoff


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def test_gap_over_weekend_is_covered():
    """Прогон в пятницу вечером, следующий — в понедельник утром."""
    friday = ts("2026-08-07 20:00")
    monday = ts("2026-08-10 09:00")

    cutoff, reason = resolve_cutoff(monday, friday, lookback_minutes=None)

    assert cutoff == friday, "выходные не должны выпасть из выпуска кандидатов"
    assert "пропуск 61.0 ч" in reason


def test_regular_run_continues_from_last_candidate():
    last_bar = ts("2026-08-10 09:00")
    last_candidate = ts("2026-08-10 08:15")

    cutoff, _ = resolve_cutoff(last_bar, last_candidate, lookback_minutes=None)

    assert cutoff == last_candidate


def test_first_run_uses_default_window():
    """Кандидатов ещё нет — берём разумное окно, а не всю историю."""
    last_bar = ts("2026-08-10 09:00")

    cutoff, reason = resolve_cutoff(last_bar, None, lookback_minutes=None)

    assert cutoff == last_bar - pd.Timedelta(minutes=DEFAULT_LOOKBACK_MINUTES)
    assert "по умолчанию" in reason


def test_long_pause_is_capped():
    """Полгода простоя не должны вылиться в разовый выпуск за полгода."""
    last_bar = ts("2026-08-10 09:00")
    ancient = ts("2026-01-10 09:00")

    cutoff, reason = resolve_cutoff(
        last_bar, ancient, lookback_minutes=None, max_lookback_days=30
    )

    assert cutoff == last_bar - pd.Timedelta(days=30)
    assert "обрезано" in reason


def test_explicit_window_wins():
    last_bar = ts("2026-08-10 09:00")
    last_candidate = ts("2026-08-10 08:00")

    cutoff, reason = resolve_cutoff(last_bar, last_candidate, lookback_minutes=120)

    assert cutoff == last_bar - pd.Timedelta(minutes=120)
    assert "явно" in reason
    assert "ВНИМАНИЕ" not in reason, "окно шире пропуска — дыры нет"


def test_explicit_window_shorter_than_gap_warns():
    """
    Сценарий дыры: пауза больше явного окна. Кандидаты за пропуск не выйдут,
    но last_candidate_ts уедет вперёд, и следующий запуск продолжит уже с него —
    интервал не закроет никто.
    """
    last_bar = ts("2026-08-10 09:00")
    last_candidate = ts("2026-08-01 00:00")

    cutoff, reason = resolve_cutoff(last_bar, last_candidate, lookback_minutes=240)

    assert cutoff == last_bar - pd.Timedelta(minutes=240)
    assert "ВНИМАНИЕ" in reason
    assert "без кандидатов" in reason


# ── Что live записывает в bar_states (B2) ───────────────────────────────────


def _states(n: int, end: str = "2026-08-10 09:00") -> pd.DataFrame:
    """Разметка на n баров базового ТФ, заканчивающаяся в end."""
    from btcproc import config

    index = pd.date_range(
        end=ts(end), periods=n, freq=f"{config.data.base_minutes}min", tz="UTC"
    )
    return pd.DataFrame({"group_id": 1.0}, index=index)


def test_live_writes_only_the_tail():
    """
    Разметка считается на всей истории, но пишется только окно от cutoff:
    PK (symbol, ts, run_id) делает каждый live чистой вставкой на весь объём,
    и полная история под каждым run_id — это десятки млн строк в сутки.
    """
    from btcproc.pipeline.live import bar_states_window

    states = _states(20_000)
    cutoff = ts("2026-08-10 05:00")

    written = bar_states_window(states, cutoff)

    assert len(written) < len(states), "вся история писаться не должна"
    assert written.index[-1] == states.index[-1], "хвост обязан попасть целиком"
    assert written.index[0] < cutoff, "перед cutoff нужен технологический запас"


def test_written_window_covers_technological_margin():
    """
    Запас перед cutoff — сглаживание + окно энтропии + максимальный офсет
    снимка: записанный кусок должен быть самодостаточен для чтения.
    """
    from btcproc import config
    from btcproc.candidates.builder import SNAPSHOT_OFFSETS_MIN
    from btcproc.pipeline.live import bar_states_window

    states = _states(5_000)
    cutoff = ts("2026-08-10 05:00")

    written = bar_states_window(states, cutoff)

    need = (
        config.states.smoothing_bars + config.states.trajectory_window
    ) * config.data.base_minutes + max(SNAPSHOT_OFFSETS_MIN)
    assert cutoff - written.index[0] >= pd.Timedelta(minutes=need)


# ── Хвост признаков: панель графика не должна обрываться на дате train ──────
def _features(n: int, end: str = "2026-08-10 12:00") -> pd.DataFrame:
    from btcproc import config

    index = pd.date_range(
        end=ts(end), periods=n, freq=f"{config.data.base_minutes}min", tz="UTC"
    )
    return pd.DataFrame({"rsi": 0.5, "pos_1d": 0.4}, index=index)


def test_features_tail_continues_from_the_last_stored_bar():
    """
    `live` считает признаки на всей истории, но дописывает только то, чего в
    таблице ещё нет: PK здесь `(symbol, ts, version)`, повтор был бы upsert'ом
    тех же чисел на триста тысяч строк двенадцать раз в сутки.
    """
    from btcproc.pipeline.live import features_tail

    features = _features(2_000)
    last_stored = features.index[-5]

    tail = features_tail(features, last_stored, cutoff=ts("2026-08-01 00:00"))

    assert len(tail) == 4, "дописываются только бары строго после сохранённого"
    assert tail.index[0] > last_stored
    assert tail.index[-1] == features.index[-1]


def test_features_tail_falls_back_to_the_cutoff_window():
    """
    Набора нет в таблице вовсе (чистили базу) — пишем окно cutoff, а не всю
    историю: восстановить панель на видимом куске достаточно.
    """
    from btcproc.pipeline.live import features_tail

    features = _features(2_000)
    cutoff = features.index[-50]

    tail = features_tail(features, None, cutoff)

    assert len(tail) == 50
    assert tail.index[0] == cutoff


def test_features_tail_accepts_naive_timestamp_from_the_db():
    """
    `max(ts)` приезжает из БД, и на драйвере без tz строка была бы наивной:
    сравнение с tz-aware индексом упало бы прямо в боевом прогоне.
    """
    from btcproc.pipeline.live import features_tail

    features = _features(100)
    naive = features.index[-3].tz_localize(None)

    tail = features_tail(features, naive, cutoff=ts("2026-01-01 00:00"))

    assert len(tail) == 2


def test_features_tail_is_empty_when_nothing_is_new():
    """Прогон без новых баров ничего не пишет — и это не ошибка."""
    from btcproc.pipeline.live import features_tail

    features = _features(100)

    assert features_tail(features, features.index[-1], ts("2026-01-01")).empty
