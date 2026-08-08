"""
Учёт прогонов: один run_id связывает модель состояний, разметку баров,
статистику переходов и выпущенных кандидатов.

Админка читает отсюда прогресс и лог, поэтому запись идёт отдельным
соединением с autocommit — иначе долгий прогон ничего бы не показывал
до самого конца.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import psycopg2
import psycopg2.extras

from btcproc import config
from btcproc.db.session import connect, fetch_all, fetch_one

logger = logging.getLogger(__name__)

MAX_LOG_CHARS = 200_000


def start_run(kind: str, params: dict[str, Any] | None = None) -> int:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO runs (kind, params) VALUES (%s, %s) RETURNING run_id",
            (kind, psycopg2.extras.Json(params or {})),
        )
        return int(cur.fetchone()[0])


def _autocommit_execute(sql: str, params: tuple) -> None:
    conn = psycopg2.connect(config.db.url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET search_path TO {config.db.schema}, public")
            cur.execute(sql, params)
    finally:
        conn.close()


def update_run(
    run_id: int,
    *,
    stage: str | None = None,
    progress: float | None = None,
    status: str | None = None,
    stats: dict | None = None,
    error: str | None = None,
    log_line: str | None = None,
) -> None:
    """Точечное обновление прогона. Виден админке немедленно."""
    sets, params = [], []
    if stage is not None:
        sets.append("stage = %s")
        params.append(stage)
    if progress is not None:
        sets.append("progress = %s")
        params.append(round(float(progress), 4))
    if status is not None:
        sets.append("status = %s")
        params.append(status)
        if status in {"done", "failed", "cancelled"}:
            sets.append("finished_at = NOW()")
    if stats is not None:
        sets.append("stats = %s")
        params.append(psycopg2.extras.Json(stats))
    if error is not None:
        sets.append("error = %s")
        params.append(error[:8000])
    if log_line is not None:
        # Лог режем слева, чтобы страница прогона не разрасталась бесконечно.
        sets.append(f"log = right(log || %s, {MAX_LOG_CHARS})")
        params.append(log_line.rstrip() + "\n")
    if not sets:
        return
    params.append(run_id)
    _autocommit_execute(f"UPDATE runs SET {', '.join(sets)} WHERE run_id = %s", tuple(params))


def log(run_id: int | None, message: str, *, stage: str | None = None,
        progress: float | None = None) -> None:
    """Строка в лог прогона + в обычный логгер процесса."""
    logger.info("[run %s] %s", run_id, message)
    if run_id is not None:
        update_run(run_id, log_line=message, stage=stage, progress=progress)


def finish_run(run_id: int, stats: dict | None = None) -> None:
    update_run(run_id, status="done", progress=1.0, stats=stats, log_line="Прогон завершён")


def fail_run(run_id: int, error: str) -> None:
    update_run(run_id, status="failed", error=error, log_line=f"ОШИБКА: {error}")


def get_run(run_id: int) -> dict | None:
    return fetch_one("SELECT * FROM runs WHERE run_id = %s", (run_id,))


def list_runs(limit: int = 50) -> list[dict]:
    return fetch_all("SELECT * FROM runs ORDER BY started_at DESC LIMIT %s", (limit,))


def active_run(kind: str | None = None) -> dict | None:
    """Текущий незавершённый прогон — чтобы не запускать два одновременно."""
    if kind:
        return fetch_one(
            "SELECT * FROM runs WHERE status = 'running' AND kind = %s "
            "ORDER BY started_at DESC LIMIT 1",
            (kind,),
        )
    return fetch_one(
        "SELECT * FROM runs WHERE status = 'running' ORDER BY started_at DESC LIMIT 1"
    )


def latest_completed_run(kind: str = "train") -> dict | None:
    """Последний успешный прогон нужного типа — источник актуальной модели."""
    return fetch_one(
        "SELECT * FROM runs WHERE kind = %s AND status = 'done' "
        "ORDER BY finished_at DESC LIMIT 1",
        (kind,),
    )


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
