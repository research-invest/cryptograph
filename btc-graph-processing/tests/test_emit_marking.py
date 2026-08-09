"""
Маркировка кандидатов при отправке в btc-graph.

Два разных события легко путаются, и путаница молча портит данные:

* btc-graph НЕ принял кандидата — он у него не сохранён, локальной оценки быть
  не может, причина «отсеян фильтром btc-graph» верна;
* btc-graph принял и сохранил, но кандидат не прошёл ЛОКАЛЬНЫЙ
  emit_min_quality — оценка у btc-graph есть, и она обязана быть сохранена
  у нас тоже, иначе базы расходятся, а причина отсева другая.
"""
from __future__ import annotations

import pytest

from btcproc.pipeline import train as train_module


@pytest.fixture
def harness(monkeypatch):
    """Отправка и БД заменены на записывающие заглушки."""
    state = {"saved": [], "marked": [], "sent": []}

    pending = [
        {"candidate_id": "strong"},
        {"candidate_id": "weak"},
        {"candidate_id": "dropped"},
    ]

    monkeypatch.setattr(
        train_module, "fetch_all", lambda *a, **kw: [], raising=False
    )
    monkeypatch.setattr(
        "btcproc.db.session.fetch_all",
        lambda *a, **kw: [{"payload": c} for c in pending],
    )
    # Добор сорвавшихся отправок ходит в runs — к предмету этих тестов он
    # отношения не имеет, но SQL строит до выборки.
    monkeypatch.setattr("btcproc.db.runs.model_root", lambda run_id: run_id)
    monkeypatch.setattr(
        "btcproc.db.runs.model_run_scope",
        lambda run_id, alias=None: ("run_id = ANY(%s)", [[run_id]]),
    )

    def fake_emit(batch, *a, **kw):
        state["sent"].append([c["candidate_id"] for c in batch])
        # "dropped" не вернулся вовсе — btc-graph его не принял.
        return [
            {"candidate_id": "strong", "quality_score": 0.80},
            {"candidate_id": "weak", "quality_score": 0.40},
        ]

    monkeypatch.setattr(train_module.graph_sink, "emit_batch", fake_emit)
    monkeypatch.setattr(
        train_module.repo, "save_evaluations",
        lambda results: state["saved"].extend(r["candidate_id"] for r in results),
    )
    monkeypatch.setattr(
        train_module.repo, "mark_emit_error",
        lambda ids, reason, mark_emitted=False: state["marked"].append((list(ids), reason)),
    )
    monkeypatch.setattr(train_module.runs, "log", lambda *a, **kw: None)
    return state


def test_evaluation_is_saved_even_below_local_threshold(harness):
    """Оценка, которая уже есть у btc-graph, не должна теряться у нас."""
    train_module.emit_pending(run_id=1, min_quality=0.6)

    assert "weak" in harness["saved"], (
        "btc-graph этого кандидата принял и оценил — его оценка обязана "
        "сохраниться, иначе базы расходятся"
    )
    assert "strong" in harness["saved"]


def test_local_threshold_reason_is_distinct(harness):
    train_module.emit_pending(run_id=1, min_quality=0.6)

    reasons = {cid: reason for ids, reason in harness["marked"] for cid in ids}

    assert reasons["dropped"] == "отсеян фильтром btc-graph"
    assert "emit_min_quality" in reasons["weak"]
    assert reasons["weak"] != reasons["dropped"], "разные события — разные причины"
    assert "strong" not in reasons


def test_statistics_count_only_accepted(harness):
    stats = train_module.emit_pending(run_id=1, min_quality=0.6)

    assert stats["sent"] == 3
    assert stats["evaluated"] == 1, "успешным считается прошедший оба фильтра"


def test_without_local_threshold_nothing_changes(harness):
    """Без min_quality поведение прежнее: локального отсева нет."""
    stats = train_module.emit_pending(run_id=1)

    reasons = {cid: reason for ids, reason in harness["marked"] for cid in ids}
    assert set(reasons) == {"dropped"}
    assert stats["evaluated"] == 2
    assert sorted(harness["saved"]) == ["strong", "weak"]


# ── Добор сорвавшихся отправок ──────────────────────────────────────────────
#
# Пока `run_id` кандидата переписывался при каждом перекрытии окон live,
# сорвавшаяся отправка чинилась сама: кандидат «переезжал» в свежий прогон и
# тот видел его как своего. После M6 (run_id закреплён за первым прогоном)
# этого пути нет, и залипший кандидат не отправился бы уже никогда.


def _captured_sql(monkeypatch) -> list:
    seen = []
    monkeypatch.setattr(
        "btcproc.db.session.fetch_all",
        lambda sql, params=None: seen.append((sql, list(params or []))) or [],
    )
    monkeypatch.setattr("btcproc.db.runs.model_root", lambda run_id: 5)
    monkeypatch.setattr(
        "btcproc.db.runs.model_run_scope",
        lambda run_id, alias=None: ("run_id = ANY(%s)", [[5, 6]]),
    )
    return seen


def test_failed_emits_of_same_model_are_retried(monkeypatch):
    seen = _captured_sql(monkeypatch)

    train_module.emit_pending(run_id=7)

    sql, _ = seen[0]
    assert "emit_error IS NOT NULL" in sql, (
        "сорвавшиеся отправки той же модели обязаны добираться заново"
    )
    assert "emitted_at IS NULL" in sql


def test_untouched_backlog_is_not_flushed(monkeypatch):
    """
    Добираются именно СОРВАВШИЕСЯ, а не все неотправленные: `train --no-emit`
    копит десятки тысяч кандидатов без ошибки, и первый же live после него
    не должен молча взяться их разгребать.
    """
    seen = _captured_sql(monkeypatch)

    train_module.emit_pending(run_id=7)

    sql, _ = seen[0]
    condition = sql.split("WHERE", 1)[1]
    assert "emit_error IS NOT NULL" in condition
    # Добор ограничен ошибочными: скоуп модели без этого условия не встречается.
    assert condition.count("emit_error IS NOT NULL") == condition.count("run_id = ANY")


def test_retry_can_be_switched_off(monkeypatch):
    seen = _captured_sql(monkeypatch)

    train_module.emit_pending(run_id=7, retry_failed=False)

    sql, params = seen[0]
    assert "emit_error" not in sql
    assert params == [7]
