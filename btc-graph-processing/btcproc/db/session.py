"""
Подключение к PostgreSQL и применение схемы.

Работаем на psycopg2 напрямую: весь тяжёлый путь — это bulk-вставка сотен
тысяч строк, и ORM здесь только мешает.
"""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import psycopg2
import psycopg2.extras
import psycopg2.pool

from btcproc import config

logger = logging.getLogger(__name__)

SCHEMA_SQL = Path(__file__).with_name("schema.sql")


# Пул соединений вместо connect/close на каждый запрос.
#
# Раньше каждый fetch_one / fetch_all / bulk_upsert открывал новое соединение
# и закрывал его, плюс отдельным round-trip'ом выставлял search_path. На
# тяжёлом пути (bulk-вставка сотен тысяч строк) это тонет в самой вставке, но
# на путях с множеством мелких запросов — `cli.status`, страницы админки,
# `runs.log` на каждой стадии — накладные расходы доминируют.
#
# Threaded, потому что админка многопоточна: uvicorn отдаёт синхронные
# обработчики в пул потоков.
_pool = None
_pool_lock = threading.Lock()

# maxconn держим заметно ниже дефолтных 100 у postgres: тут же живут btc-graph,
# celery и neo4j, а параллельных потребителей у нас единицы.
POOL_MIN, POOL_MAX = 1, 16

# На исчерпании пул psycopg2 не ждёт, а кидает PoolError. Для админки это
# значило бы 500-е при всплеске: thread-pool uvicorn — 40 потоков, соединений
# 16. Семафор превращает исчерпание в ожидание свободного слота. Таймаут —
# страховка от вечного зависания, если кто-то не вернул соединение: заметно
# дольше любого страничного запроса, но не бесконечность.
POOL_WAIT_SECONDS = 30
_pool_slots = threading.BoundedSemaphore(POOL_MAX)


def _get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = psycopg2.pool.ThreadedConnectionPool(
                    POOL_MIN, POOL_MAX, config.db.url
                )
    return _pool


def close_pool() -> None:
    """Закрывает все соединения пула. Нужен тестам и корректному выходу."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.closeall()
            _pool = None


@contextmanager
def connect(autocommit: bool = False,
            timeout_ms: int | None = None) -> Iterator[psycopg2.extensions.connection]:
    """
    Соединение с выставленным search_path на нашу схему.

    `timeout_ms` — потолок на КАЖДЫЙ запрос этого соединения. Ставится не
    глобально в конфиге и не на пул, потому что пул общий: в том же процессе
    админки фоновые потоки гоняют train с bulk-вставками на десятки минут, и
    единый потолок убивал бы их. Значение задают вызывающие — чтения админки
    им ограничены, расчётные пути нет.

    Ставится обычным SET, рядом с search_path, и обязательно снимается перед
    возвратом соединения в пул: пул общий, и унаследованный чужой потолок
    оборвал бы bulk-вставку прогона. SET LOCAL здесь не годится — он
    откатился бы ближайшим commit, то есть ещё до первого запроса.
    """
    if not _pool_slots.acquire(timeout=POOL_WAIT_SECONDS):
        raise TimeoutError(
            f"Свободное соединение с БД не дождалось за {POOL_WAIT_SECONDS} с: "
            f"все {POOL_MAX} заняты. Либо всплеск параллельных запросов, либо "
            f"кто-то держит соединение и не возвращает."
        )
    try:
        pool = _get_pool()
        conn = pool.getconn()
    except Exception:
        _pool_slots.release()
        raise
    broken = False
    try:
        conn.autocommit = autocommit
        with conn.cursor() as cur:
            cur.execute(f"SET search_path TO {config.db.schema}, public")
            if timeout_ms is not None:
                cur.execute("SET statement_timeout = %s", (int(timeout_ms),))
        if not autocommit:
            conn.commit()
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        # Соединение могло остаться в незавершённой транзакции или вовсе
        # оборваться. Откатываем, а на ошибке отката — выбрасываем из пула:
        # вернуть в оборот битое соединение хуже, чем открыть новое.
        try:
            if not autocommit:
                conn.rollback()
        except psycopg2.Error:
            broken = True
        raise
    finally:
        # autocommit — свойство соединения, а не запроса: следующий, кто
        # возьмёт его из пула, не должен унаследовать чужой режим. Ровно то
        # же и с потолком запроса: чужой statement_timeout, унаследованный
        # прогоном, оборвал бы ему bulk-вставку — снимаем явно.
        try:
            if timeout_ms is not None:
                with conn.cursor() as cur:
                    cur.execute("SET statement_timeout = 0")
                if not conn.autocommit:
                    conn.commit()
            conn.autocommit = False
        except psycopg2.Error:
            broken = True
        # Слот освобождается даже если putconn упал (например, пул закрыли
        # параллельно): потерянный слот — это навсегда уменьшившийся лимит.
        try:
            pool.putconn(conn, close=broken)
        finally:
            _pool_slots.release()


def init_schema() -> None:
    """Создаёт схему и таблицы. Идемпотентно."""
    sql = SCHEMA_SQL.read_text(encoding="utf-8").replace("{schema}", config.db.schema)
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql)
    logger.info("Схема %s готова", config.db.schema)


def fetch_all(sql: str, params: Sequence[Any] | None = None,
              timeout_ms: int | None = None) -> list[dict]:
    with connect(timeout_ms=timeout_ms) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]


def fetch_one(sql: str, params: Sequence[Any] | None = None,
              timeout_ms: int | None = None) -> dict | None:
    rows = fetch_all(sql, params, timeout_ms=timeout_ms)
    return rows[0] if rows else None


def execute(sql: str, params: Sequence[Any] | None = None,
            timeout_ms: int | None = None) -> None:
    with connect(timeout_ms=timeout_ms) as conn, conn.cursor() as cur:
        cur.execute(sql, params)


def bulk_upsert(
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
    conflict_columns: Sequence[str],
    update_columns: Sequence[str] | None = None,
    page_size: int = 5000,
) -> int:
    """
    INSERT ... ON CONFLICT DO UPDATE пачками.

    Возвращает число переданных строк (не число фактически изменённых —
    execute_values его не отдаёт).
    """
    rows = list(rows)
    if not rows:
        return 0

    cols = ", ".join(columns)
    conflict = ", ".join(conflict_columns)
    if update_columns is None:
        update_columns = [c for c in columns if c not in conflict_columns]

    if update_columns:
        setters = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_columns)
        action = f"DO UPDATE SET {setters}"
    else:
        action = "DO NOTHING"

    sql = (
        f"INSERT INTO {table} ({cols}) VALUES %s "
        f"ON CONFLICT ({conflict}) {action}"
    )
    with connect() as conn, conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, page_size=page_size)
    return len(rows)
