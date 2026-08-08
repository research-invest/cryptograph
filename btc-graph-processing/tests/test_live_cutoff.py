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
    last_candidate = ts("2026-08-01 00:00")

    cutoff, reason = resolve_cutoff(last_bar, last_candidate, lookback_minutes=120)

    assert cutoff == last_bar - pd.Timedelta(minutes=120)
    assert "явно" in reason
