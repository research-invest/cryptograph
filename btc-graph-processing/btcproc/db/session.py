"""
Подключение к PostgreSQL и применение схемы.

Работаем на psycopg2 напрямую: весь тяжёлый путь — это bulk-вставка сотен
тысяч строк, и ORM здесь только мешает.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import psycopg2
import psycopg2.extras

from btcproc import config

logger = logging.getLogger(__name__)

SCHEMA_SQL = Path(__file__).with_name("schema.sql")


@contextmanager
def connect(autocommit: bool = False) -> Iterator[psycopg2.extensions.connection]:
    """Соединение с выставленным search_path на нашу схему."""
    conn = psycopg2.connect(config.db.url)
    conn.autocommit = autocommit
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET search_path TO {config.db.schema}, public")
        if not autocommit:
            conn.commit()
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        if not autocommit:
            conn.rollback()
        raise
    finally:
        conn.close()


def init_schema() -> None:
    """Создаёт схему и таблицы. Идемпотентно."""
    sql = SCHEMA_SQL.read_text(encoding="utf-8").replace("{schema}", config.db.schema)
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql)
    logger.info("Схема %s готова", config.db.schema)


def fetch_all(sql: str, params: Sequence[Any] | None = None) -> list[dict]:
    with connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]


def fetch_one(sql: str, params: Sequence[Any] | None = None) -> dict | None:
    rows = fetch_all(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: Sequence[Any] | None = None) -> None:
    with connect() as conn, conn.cursor() as cur:
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
