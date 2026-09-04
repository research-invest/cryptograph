"""
Отметка свежести графика: по ней страница решает, приехали ли новые данные.

Проверяется ровно то, на чём держится автообновление: токен меняется, когда
меняются бары или завершается прогон, и не меняется, пока в базе всё то же;
идущий прогон виден отдельным флагом, потому что обновляться на его середине
нельзя — бары он пишет раньше кандидатов.

БД не нужна: fetch_one и active_run подменяются.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from btcproc.admin import queries

SYMBOL = "ETHUSDT"


def _state(monkeypatch, *, last_bar, run, busy):
    def fake_fetch_one(sql, params=None):
        if "FROM ohlcv" in sql:
            assert params[0] == SYMBOL, "отметка обязана быть помонетной"
            return {"ts": last_bar}
        if "FROM runs" in sql:
            return run
        raise AssertionError(f"неожиданный запрос: {sql}")

    monkeypatch.setattr(queries, "fetch_one", fake_fetch_one)
    monkeypatch.setattr(queries.runs_repo, "active_run",
                        lambda symbol=None, kind=None: {"run_id": 7} if busy else None)
    return queries.chart_freshness(SYMBOL)


def _run(run_id: int, finished_day: int, status: str = "done") -> dict:
    return {
        "run_id": run_id, "kind": "live", "status": status,
        "finished_at": datetime(2026, 9, finished_day, 12, 3, tzinfo=timezone.utc),
    }


def test_token_stable_without_changes(monkeypatch):
    """Те же данные — тот же токен: иначе страница перезагружалась бы вхолостую."""
    bar = datetime(2026, 9, 4, 11, 45, tzinfo=timezone.utc)
    first = _state(monkeypatch, last_bar=bar, run=_run(10, 4), busy=False)
    second = _state(monkeypatch, last_bar=bar, run=_run(10, 4), busy=False)
    assert first["token"] == second["token"]


@pytest.mark.parametrize("later_bar, later_run", [
    (datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc), _run(10, 4)),
    (datetime(2026, 9, 4, 11, 45, tzinfo=timezone.utc), _run(11, 4)),
])
def test_token_changes_on_new_data(monkeypatch, later_bar, later_run):
    """Новый бар ИЛИ новый завершённый прогон — повод перерисовать график."""
    before = _state(
        monkeypatch,
        last_bar=datetime(2026, 9, 4, 11, 45, tzinfo=timezone.utc),
        run=_run(10, 4), busy=False,
    )
    after = _state(monkeypatch, last_bar=later_bar, run=later_run, busy=False)
    assert before["token"] != after["token"]


def test_failed_run_also_moves_token(monkeypatch):
    """
    Упавший прогон завершает ожидание так же, как успешный: страница ждёт
    приезда данных, а после падения ждать больше нечего.
    """
    bar = datetime(2026, 9, 4, 11, 45, tzinfo=timezone.utc)
    done = _state(monkeypatch, last_bar=bar, run=_run(10, 4), busy=False)
    failed = _state(monkeypatch, last_bar=bar, run=_run(11, 4, "failed"), busy=False)
    assert done["token"] != failed["token"]


def test_busy_reported(monkeypatch):
    """Идущий прогон виден отдельно от токена — на его середине не обновляемся."""
    bar = datetime(2026, 9, 4, 11, 45, tzinfo=timezone.utc)
    assert _state(monkeypatch, last_bar=bar, run=_run(10, 4), busy=True)["busy"] is True
    assert _state(monkeypatch, last_bar=bar, run=_run(10, 4), busy=False)["busy"] is False


def test_no_history_does_not_break(monkeypatch):
    """Монета без баров и прогонов отдаёт токен, а не падает."""
    state = _state(monkeypatch, last_bar=None, run=None, busy=False)
    assert isinstance(state["token"], str) and state["last_run"] is None
