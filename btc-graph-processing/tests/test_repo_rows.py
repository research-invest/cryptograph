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


def _by_name(captured) -> list[dict]:
    """Строки как словари: позиционные индексы ломались при каждой новой колонке."""
    return [dict(zip(captured["columns"], row)) for row in captured["rows"]]


def test_outcomes_rows(captured):
    index = _index(2)
    outcomes = pd.DataFrame({
        "ret_pct": [1.5, np.nan],
        "mfe_pct": [2.0, np.nan],
        "mae_pct": [-0.5, np.nan],
        "is_up": [True, None],
        "range_pct": [3.0, np.nan],
        "rv_fwd": [0.004, np.nan],
        "range_ratio": [1.25, np.nan],
        "valid": [True, False],
    }, index=index)

    repo.save_outcomes(outcomes, "ETHUSDT", "24h")

    good, empty = _by_name(captured)
    assert (good["symbol"], good["ts"], good["horizon"]) == (
        "ETHUSDT", index[0].to_pydatetime(), "24h")
    assert good["ret_pct"] == 1.5 and good["is_up"] is True and good["valid"] is True
    assert (good["range_pct"], good["rv_fwd"], good["range_ratio"]) == (3.0, 0.004, 1.25)
    # NaN исход — NULL по всем метрикам и is_up, а не 0 и не False.
    assert empty["ret_pct"] is None and empty["is_up"] is None
    assert empty["valid"] is False
    assert empty["range_pct"] is None and empty["range_ratio"] is None


def test_outcomes_rows_without_range_columns(captured):
    """
    Кадр без величин размаха обязан сохраняться, отдавая по ним NULL.

    Так выглядит любой вызывающий код, написанный до 2026-08-19, и так же —
    чужой кадр в тестах. Падение здесь означало бы, что добавление колонки
    сломало совместимость на ровном месте; NULL честно означает «не считалось».
    """
    index = _index(1)
    outcomes = pd.DataFrame({
        "ret_pct": [1.5], "mfe_pct": [2.0], "mae_pct": [-0.5],
        "is_up": [True], "valid": [True],
    }, index=index)

    repo.save_outcomes(outcomes, "ETHUSDT", "24h")

    row, = _by_name(captured)
    assert row["ret_pct"] == 1.5
    assert row["range_pct"] is None and row["rv_fwd"] is None
    assert row["range_ratio"] is None


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
