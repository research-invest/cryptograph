"""
Сборка строк для вставки: itertuples обязан давать ровно то же, что iterrows.

O5 (аудит 2026-08-15). Три горячих `save_*` (бары-состояния, события, исходы)
идут по DataFrame на трёхстах тысячах строк за train, и `iterrows` собирал на
каждую строку отдельный Series. Замена механическая, но у `itertuples` есть
своя ловушка: колонку, чьё имя совпадает с методом namedtuple (`count`,
`index`), он молча переименовывает в `_N`. Поэтому здесь сверяется РЕЗУЛЬТАТ,
а не факт замены.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcproc.db import repo


@pytest.fixture
def captured(monkeypatch):
    box: dict = {}

    def fake_bulk_upsert(table, columns, rows, conflict_columns, **kw):
        box["table"], box["columns"], box["rows"] = table, columns, list(rows)
        return len(box["rows"])

    monkeypatch.setattr(repo, "bulk_upsert", fake_bulk_upsert)
    return box


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")


def test_bar_states_rows(captured):
    index = _index(3)
    states = pd.DataFrame({
        "group_id": [1.0, 2.0, 2.0],
        "prev_group_id": [np.nan, 1.0, 1.0],
        "state_seq": [0, 1, 1],
        "age_minutes": [0, 0, 15],
        "age_bucket": ["age_lt_30"] * 3,
        "entropy": ["low"] * 3,
        "is_transition": [False, True, False],
        "transition_id": [None, "1->2", None],
    }, index=index)

    repo.save_bar_states(7, states, "BTCUSDT")

    assert len(captured["rows"]) == 3
    first, second = captured["rows"][0], captured["rows"][1]
    assert first[0] == "BTCUSDT"
    assert first[1] == index[0].to_pydatetime()
    assert first[3] == 1.0
    assert first[4] is None, "NaN prev_group_id обязан стать NULL"
    assert second[4] == 1.0
    assert second[9] is True and second[10] == "1->2"


def test_outcomes_rows(captured):
    index = _index(2)
    outcomes = pd.DataFrame({
        "ret_pct": [1.5, np.nan],
        "mfe_pct": [2.0, np.nan],
        "mae_pct": [-0.5, np.nan],
        "is_up": [True, None],
        "valid": [True, False],
    }, index=index)

    repo.save_outcomes(outcomes, "ETHUSDT", "24h")

    good, empty = captured["rows"]
    assert good[:3] == ("ETHUSDT", index[0].to_pydatetime(), "24h")
    assert good[3] == 1.5 and good[6] is True and good[7] is True
    # NaN исход — NULL по всем метрикам и is_up, а не 0 и не False.
    assert empty[3] is None and empty[6] is None and empty[7] is False


def test_events_rows(captured):
    index = _index(2)
    events = pd.DataFrame({
        "event_block_id": ["abc", "def"],
        "atoms": [["breakout_up"], []],
        "families": [["breakout"], []],
        "atom_count": [1, 0],
        "family_count": [1, 0],
        "intensity": ["low", "none"],
        "primary_family": ["breakout", None],
        "context_atoms": [["trend_up_align"], []],
    }, index=index)

    repo.save_events(events, "SOLUSDT", version="v1")

    row = captured["rows"][0]
    assert row[0] == "SOLUSDT"
    assert row[1] == index[0].to_pydatetime()
    assert row[2] == "abc"
    assert row[3] == ["breakout_up"] and row[9] == ["trend_up_align"]
    assert row[10] == "v1"
