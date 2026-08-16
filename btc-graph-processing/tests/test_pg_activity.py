"""
Живой срез запросов PostgreSQL на странице «Сервер».

Блок появился после случая из журнала 43: запрос на открытие узла графа шёл
десять с лишним минут и клал базу, а увидеть виновника можно было только через
psql на сервере. Здесь проверяется разметка снимка (что считается долгим, что
висящей транзакцией), выбор способа снятия бэкенда и главное свойство блока —
недоступная база не должна ронять страницу, ведь смотрят на неё как раз тогда,
когда стеку плохо.
"""
from __future__ import annotations

import contextlib

import pytest

from btcproc import config
from btcproc.admin import auth
from btcproc.db import activity


def _row(**over):
    row = {
        "pid": 100, "state": "active", "query_seconds": 1.0, "xact_seconds": 1.0,
        "state_seconds": 1.0, "wait_event_type": None, "wait_event": None,
        "application_name": "btcproc", "backend_type": "client backend",
        "usename": "btc_user", "client": "172.18.0.1", "query": "SELECT 1",
    }
    row.update(over)
    return row


@pytest.fixture
def fake_db(monkeypatch):
    """
    Подменяет соединение. Снимок берёт ВСЕ бэкенды одним запросом, поэтому
    фикстура отдаёт именно их — включая простаивающие, которые в таблицу не
    попадут, но в сводке обязаны быть.
    """
    def _install(backends, max_connections=100):
        state = {"backends": backends, "max": max_connections}

        class Cursor:
            def execute(self, sql, params=None):
                state["last"] = sql

            def fetchall(self):
                return list(state["backends"])

            def fetchone(self):
                return {"n": state["max"]}

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        class Conn:
            def cursor(self, **kwargs):
                return Cursor()

        @contextlib.contextmanager
        def fake_connect(**kwargs):
            state["timeout"] = kwargs.get("timeout_ms")
            yield Conn()

        monkeypatch.setattr(activity, "connect", fake_connect)
        return state

    return _install


def test_long_query_is_marked_slow_and_short_one_is_not(fake_db):
    fake_db([
        _row(pid=1, query_seconds=45.0, query="INSERT INTO state_context"),
        _row(pid=2, query_seconds=0.4, query="SELECT count(*) FROM runs"),
    ])

    rows = activity.snapshot(slow_seconds=30)["rows"]

    assert {r["pid"]: r["slow"] for r in rows} == {1: True, 2: False}


def test_rows_are_ordered_by_duration(fake_db):
    """Смотрят на этот блок ради самого долгого запроса — он обязан быть первым."""
    fake_db([_row(pid=1, query_seconds=2.0), _row(pid=2, query_seconds=600.0),
             _row(pid=3, query_seconds=None)])

    rows = activity.snapshot(slow_seconds=30)["rows"]

    assert [r["pid"] for r in rows] == [2, 1, 3]


def test_idle_in_transaction_is_flagged(fake_db):
    """
    Висящая транзакция — не «простаивающее соединение», а держатель блокировок,
    и снимается она иначе (terminate вместо cancel). Флаг решает и то, и другое.
    """
    fake_db([_row(pid=1, state="idle in transaction", query_seconds=None),
             _row(pid=2, state="active")])

    rows = {r["pid"]: r["stuck_in_transaction"] for r in activity.snapshot(30)["rows"]}

    assert rows == {1: True, 2: False}


def test_multiline_query_is_collapsed(fake_db):
    """SQL проекта многострочный; в ячейке таблицы отступы съедают пол-экрана."""
    fake_db([_row(query="SELECT a,\n       b\nFROM   t\nWHERE  x = 1")])

    assert activity.snapshot(30)["rows"][0]["query"] == "SELECT a, b FROM t WHERE x = 1"


def test_snapshot_caps_its_own_query_time(fake_db):
    """Диагностика не имеет права стать причиной проблемы, которую показывает."""
    state = fake_db([_row()])

    activity.snapshot(30)

    assert state["timeout"] == activity.TIMEOUT_MS


def test_duration_keeps_subsecond_and_minutes():
    """
    Общий `human_seconds` админки округляет до минут: быстрый запрос стал бы
    «0 с», а десятиминутный потерял бы секунды — ровно то, за чем тут следят.
    """
    assert activity.format_seconds(0.34) == "0.34 с"
    assert activity.format_seconds(12.7) == "12.7 с"
    assert activity.format_seconds(641.0) == "10 м 41 с"
    assert activity.format_seconds(None) == "—"


def test_stop_uses_cancel_for_a_running_query(monkeypatch):
    calls = []

    def fake_fetch_one(sql, params=None, timeout_ms=None):
        calls.append(sql)
        if "pg_stat_activity" in sql and "pg_cancel" not in sql:
            return {"pid": 7, "state": "active", "backend_type": "client backend",
                    "query": "SELECT pg_sleep(600)"}
        return {"ok": True}

    monkeypatch.setattr(activity, "fetch_one", fake_fetch_one)

    result = activity.stop_query(7)

    assert result == {"action": "cancel", "ok": True, "note": "Запрос 7 отменён."}
    assert any("pg_cancel_backend" in sql for sql in calls)
    assert not any("pg_terminate_backend" in sql for sql in calls)


def test_stop_terminates_a_hanging_transaction(monkeypatch):
    """
    `pg_cancel_backend` на 'idle in transaction' не действует — отменять там
    нечего. Кнопка, которая ничего не делает, хуже отсутствующей.
    """
    calls = []

    def fake_fetch_one(sql, params=None, timeout_ms=None):
        calls.append(sql)
        if "pg_stat_activity" in sql and "pg_terminate" not in sql:
            return {"pid": 7, "state": "idle in transaction",
                    "backend_type": "client backend", "query": "BEGIN"}
        return {"ok": True}

    monkeypatch.setattr(activity, "fetch_one", fake_fetch_one)

    assert activity.stop_query(7)["action"] == "terminate"
    assert any("pg_terminate_backend" in sql for sql in calls)


def test_stop_reports_a_backend_that_already_finished(monkeypatch):
    """Между отрисовкой и нажатием проходят секунды — это не ошибка."""
    monkeypatch.setattr(activity, "fetch_one",
                        lambda sql, params=None, timeout_ms=None: None)

    result = activity.stop_query(7)

    assert result["action"] == "none" and result["ok"] is False


# ─── Страница ───────────────────────────────────────────────────────────────
fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        config, "admin",
        config.AdminConfig(user="operator", password="very-long-password-42",
                           secret_key="k" * 64, ip_allowlist=[]),
    )
    from btcproc.admin import app as admin_app

    monkeypatch.setattr(auth, "current_user", lambda request: "operator")
    monkeypatch.setattr(admin_app, "init_schema", lambda: None, raising=False)
    with fastapi_testclient.TestClient(admin_app.app) as test_client:
        yield test_client


def test_page_survives_an_unreachable_database(client, monkeypatch):
    """
    Главное свойство страницы «Сервер»: смотрят на неё именно тогда, когда
    docker-стеку плохо. Недоступный Postgres — это факт для показа, а не 500.
    """
    def boom(*args, **kwargs):
        raise RuntimeError("could not connect to server")

    monkeypatch.setattr(activity, "snapshot", boom)

    resp = client.get("/server")

    assert resp.status_code == 200
    assert "could not connect to server" in resp.text


def test_summary_counts_idle_connections_the_table_hides(fake_db):
    """
    Простаивающие соединения пула из таблицы убраны — их полтора десятка и
    смотреть в них нечего. Но лимит соединений они расходуют, поэтому в сводке
    остаются: «60 из 100, все простаивают» — диагноз утечки соединений.
    """
    fake_db([
        _row(pid=1, state="active", query_seconds=5.0),
        _row(pid=2, state="idle", query_seconds=None),
        _row(pid=3, state="idle", query_seconds=None),
        _row(pid=4, state="idle in transaction", query_seconds=None),
    ])

    snap = activity.snapshot(slow_seconds=30)

    assert snap["summary"] == {
        "total": 4, "active": 1, "idle": 2, "in_transaction": 1,
        "waiting": 0, "max_connections": 100,
    }
    assert [r["pid"] for r in snap["rows"]] == [1, 4], "idle в таблице не нужны"


def test_summary_and_table_come_from_one_snapshot(fake_db):
    """
    Сводка считается из тех же строк, что и таблица, а не отдельным запросом.
    Отдельным она врала: между двумя запросами проходят миллисекунды, и
    «активных 2» появлялось над таблицей с одной строкой.
    """
    fake_db([_row(pid=1, state="active", query_seconds=1.0),
             _row(pid=2, state="active", query_seconds=2.0)])

    snap = activity.snapshot(slow_seconds=30)

    assert snap["summary"]["active"] == len(snap["rows"]) == 2
