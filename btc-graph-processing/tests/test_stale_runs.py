"""
Протухшие прогоны: `running` без heartbeat.

Статус `running` снимает только сам процесс. Если его убили (OOM killer на
расчёте признаков, ребут машины, kill -9), строка остаётся `running` навсегда.
Дальше отказ становится тихим: крон стоит с --skip-if-busy и молча пропускает
каждый следующий live этой монеты, а админка отдаёт 409 «идёт прогон».
Единственный признак живости — heartbeat `updated_at`, который `update_run`
трогает на каждой стадии.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from btcproc.db import runs as runs_repo


def _run(minutes_ago: float, status: str = "running", **extra) -> dict:
    return {
        "run_id": 42,
        "kind": "live",
        "symbol": "BTCUSDT",
        "status": status,
        "updated_at": datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
        **extra,
    }


def test_fresh_heartbeat_is_alive():
    assert runs_repo.is_stale(_run(5), stale_after_minutes=120) is False


def test_silent_run_is_stale():
    assert runs_repo.is_stale(_run(500), stale_after_minutes=120) is True


def test_finished_run_is_never_stale():
    """У завершённого прогона heartbeat старый по определению."""
    assert runs_repo.is_stale(_run(10_000, status="done"), stale_after_minutes=120) is False


def test_old_rows_fall_back_to_started_at():
    """Строки, заведённые до появления колонки, сравниваются по старту."""
    run = _run(5)
    run["updated_at"] = None
    run["started_at"] = datetime.now(timezone.utc) - timedelta(minutes=500)

    assert runs_repo.is_stale(run, stale_after_minutes=120) is True


def test_alive_run_is_not_reaped(monkeypatch):
    failed = []
    monkeypatch.setattr(runs_repo, "fail_run", lambda rid, err: failed.append(rid))

    assert runs_repo.reap_if_stale(_run(5), stale_after_minutes=120) is False
    assert failed == []


def test_stale_run_is_marked_failed(monkeypatch):
    failed = []
    monkeypatch.setattr(runs_repo, "fail_run", lambda rid, err: failed.append((rid, err)))

    assert runs_repo.reap_if_stale(_run(500), stale_after_minutes=120) is True
    assert failed[0][0] == 42
    assert "heartbeat" in failed[0][1]


def test_reap_ignores_missing_run(monkeypatch):
    monkeypatch.setattr(runs_repo, "fail_run", lambda *a: pytest.fail("не должно вызываться"))
    assert runs_repo.reap_if_stale(None) is False


def test_guard_lets_through_after_reaping_dead_run(monkeypatch):
    """
    Ради этого всё и делалось: мёртвый прогон не должен вечно блокировать
    следующий запуск монеты.
    """
    import typer

    from btcproc import cli

    dead = _run(500)
    monkeypatch.setattr(cli.runs_repo, "active_run", lambda **kw: dead)
    monkeypatch.setattr(cli.runs_repo, "fail_run", lambda *a: None)

    cli._guard_active_run(force=False, symbol="BTCUSDT", skip_if_busy=True)


def test_guard_still_blocks_on_live_run(monkeypatch):
    from btcproc import cli

    alive = _run(5, progress=0.4, stage="features")
    monkeypatch.setattr(cli.runs_repo, "active_run", lambda **kw: alive)
    monkeypatch.setattr(
        cli.runs_repo, "fail_run", lambda *a: pytest.fail("живой прогон не трогаем")
    )

    with pytest.raises(cli.SkipBusy):
        cli._guard_active_run(force=False, symbol="BTCUSDT", skip_if_busy=True)
