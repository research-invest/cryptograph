"""
Семантика `run_id` у сохранённого кандидата.

Окна live намеренно перекрываются — это дешевле дыры. Значит одного и того же
кандидата видит несколько прогонов подряд, и upsert срабатывает многократно.
Пока `run_id` входил в update_columns, каждый такой прогон «перевозил»
кандидата к себе: выборка «кандидаты прогона N» худела со временем сама,
счётчики prune_runs плыли, и ни то, ни другое не выглядело ошибкой.
"""
from __future__ import annotations

import pytest

from btcproc.db import repo


@pytest.fixture
def upserts(monkeypatch):
    calls = []
    monkeypatch.setattr(
        repo, "bulk_upsert",
        lambda table, columns, rows, **kw: calls.append(
            {"table": table, "columns": columns, "rows": list(rows), **kw}
        ) or len(rows),
    )
    return calls


def _candidate(cid: str = "c1") -> dict:
    return {
        "candidate_id": cid,
        "symbol": "BTCUSDT",
        "transition_id": "t1",
        "event_block_id": "b1",
        "candidate_family_key": "f1",
        "research_side": "long",
        "research_score": 0.7,
        "sample_size": 1200,
        "_meta": {"ts": "2026-08-09T00:00:00Z"},
    }


def test_run_id_is_not_overwritten_on_repeat(upserts):
    repo.save_candidates(7, [_candidate()])

    assert "run_id" not in upserts[0]["update_columns"], (
        "перекрытие окон live не должно перевозить кандидата в новый прогон"
    )


def test_payload_and_metrics_still_refresh(upserts):
    """Пересчитанные величины обновляться обязаны — выборка кандидата растёт."""
    repo.save_candidates(7, [_candidate()])

    assert set(upserts[0]["update_columns"]) == {
        "payload", "research_score", "sample_size"
    }


def test_conflict_key_is_candidate_id(upserts):
    repo.save_candidates(7, [_candidate()])

    assert upserts[0]["conflict_columns"] == ["candidate_id"]
