"""
Поведение пула соединений на исчерпании.

Пул psycopg2 сам по себе не ждёт свободного соединения, а кидает PoolError.
Для админки это означало бы 500-е при всплеске параллельных запросов:
thread-pool uvicorn — 40 потоков, соединений — 16. Семафор в `connect`
превращает исчерпание в ожидание; здесь проверяется, что ожидание работает,
что таймаут даёт внятную ошибку и что слот не утекает ни при ошибке в теле
запроса, ни при падении возврата соединения.

БД тесты не трогают: пул подменяется фейком той же формы.
"""
from __future__ import annotations

import threading
import time

import pytest

from btcproc.db import session


class FakeCursor:
    def execute(self, sql):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeConn:
    def __init__(self):
        self.autocommit = False

    def cursor(self):
        return FakeCursor()

    def commit(self):
        pass

    def rollback(self):
        pass


class FakePool:
    """Та же дисциплина, что у ThreadedConnectionPool: лимит и PoolError."""

    def __init__(self, minconn, maxconn, dsn):
        self.maxconn = maxconn
        self.given = 0
        self._lock = threading.Lock()

    def getconn(self):
        with self._lock:
            if self.given >= self.maxconn:
                raise session.psycopg2.pool.PoolError("connection pool exhausted")
            self.given += 1
        return FakeConn()

    def putconn(self, conn, close=False):
        with self._lock:
            self.given -= 1

    def closeall(self):
        pass


@pytest.fixture
def fake_pool(monkeypatch):
    session.close_pool()
    monkeypatch.setattr(session.psycopg2.pool, "ThreadedConnectionPool", FakePool)
    # Лимит 1, чтобы исчерпание наступало от одного держателя.
    monkeypatch.setattr(session, "_pool_slots", threading.BoundedSemaphore(1))
    yield
    session.close_pool()


def _hold_connection(entered: threading.Event, release: threading.Event) -> threading.Thread:
    def holder():
        with session.connect():
            entered.set()
            release.wait(timeout=5)

    t = threading.Thread(target=holder)
    t.start()
    assert entered.wait(timeout=5)
    return t


def test_exhausted_pool_waits_instead_of_raising(fake_pool):
    entered, release = threading.Event(), threading.Event()
    holder = _hold_connection(entered, release)

    got = {}

    def waiter():
        with session.connect():
            got["ok"] = True

    w = threading.Thread(target=waiter)
    w.start()
    time.sleep(0.1)
    assert "ok" not in got, "второй заём должен ЖДАТЬ, а не падать PoolError"

    release.set()
    w.join(timeout=5)
    holder.join(timeout=5)
    assert got.get("ok"), "после освобождения слота ожидавший должен пройти"


def test_timeout_names_the_limit(fake_pool, monkeypatch):
    monkeypatch.setattr(session, "POOL_WAIT_SECONDS", 0.05)
    entered, release = threading.Event(), threading.Event()
    holder = _hold_connection(entered, release)

    try:
        with pytest.raises(TimeoutError, match="заняты"):
            with session.connect():
                pass
    finally:
        release.set()
        holder.join(timeout=5)


def test_slot_survives_an_error_in_the_body(fake_pool):
    with pytest.raises(RuntimeError):
        with session.connect():
            raise RuntimeError("ошибка запроса")

    # Слот вернулся: следующий заём не ждёт и не падает.
    with session.connect():
        pass


def test_slot_survives_a_failing_putconn(fake_pool):
    with session.connect():
        pass  # инициализировать пул

    def boom(conn, close=False):
        raise RuntimeError("пул закрыт параллельно")

    session._pool.putconn = boom
    with pytest.raises(RuntimeError):
        with session.connect():
            pass

    # Пул пересоздаётся (счётчик фейка не узнал о возврате), но семафор обязан
    # быть сбалансирован — иначе этот connect завис бы до таймаута.
    session.close_pool()
    with session.connect():
        pass
