"""
Сетевой тест на карту конвенций `create_time` (docs/tz_deriv_ingest_14-08-26.md,
§0.8, A3). Ходит в data.binance.vision, поэтому помечен `network` и не входит
в общий прогон (`pytest.ini`: `addopts = -m "not network"`).

Обязателен к запуску перед КАЖДЫМ изменением
`metrics.CREATE_TIME_CONVENTION_PERIODS`:

    pytest -m network tests/test_metrics_convention.py

Один день из каждого периода сверяется `detect_convention` (сверка
`sum_taker_long_short_vol_ratio` с 5m-klines фьючерса) с тем, что говорит
карта. 2022-01-01 в карту сознательно не входит отдельной проверкой — колонка
в этот день пуста целиком (§0.9), сверка на нём невозможна по построению.
"""
from __future__ import annotations

import httpx
import pytest

from btcproc.ingest import metrics

SYMBOL = "BTCUSDT"

# По одному дню из середины каждого периода CREATE_TIME_CONVENTION_PERIODS —
# не на границе, чтобы не зависеть от точности самой границы.
PROBE_DAYS = [
    "2021-06-01",
    "2023-01-01",
    "2024-07-15",
    "2025-07-01",
    "2026-06-20",
]


@pytest.mark.network
@pytest.mark.parametrize("day", PROBE_DAYS)
def test_convention_map_matches_detected_fact(day):
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        detected = metrics.detect_convention(SYMBOL, day, client)
    assert detected is not None, f"{day}: sum_taker_long_short_vol_ratio пуст — выбери другой день"
    assert detected == metrics.create_time_convention(day), (
        f"{day}: карта говорит {metrics.create_time_convention(day)!r}, "
        f"сверка с klines даёт {detected!r} — CREATE_TIME_CONVENTION_PERIODS устарела"
    )


@pytest.mark.network
def test_positive_control_detects_a_known_shift():
    """
    Позитивный контроль измерителя (раздел 6 ТЗ): `detect_convention` обязан
    СМОЧЬ различить смещение на 5 минут, а не тихо вернуть один и тот же
    ответ независимо от входа. Берём два дня из РАЗНЫХ периодов (§0.8:
    2021-06-01 — "конец", 2025-07-01 — "начало") — они обязаны дать разные
    ответы; совпадение означало бы, что измеритель не чувствителен к сдвигу.
    """
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        end_period = metrics.detect_convention(SYMBOL, "2021-06-01", client)
        start_period = metrics.detect_convention(SYMBOL, "2025-07-01", client)
    assert end_period == "end"
    assert start_period == "start"
    assert end_period != start_period
