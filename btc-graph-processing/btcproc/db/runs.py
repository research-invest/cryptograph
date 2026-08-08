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


def start_run(
    kind: str,
    params: dict[str, Any] | None = None,
    symbol: str | None = None,
) -> int:
    """
    Заводит прогон. symbol пишется отдельной колонкой, а не только в params:
    по нему идут выборки «последняя модель монеты» и фильтр списка прогонов.
    """
    params = dict(params or {})
    symbol = symbol or params.get("symbol") or config.data.symbol
    params.setdefault("symbol", symbol)

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO runs (kind, symbol, params) VALUES (%s, %s, %s) RETURNING run_id",
            (kind, symbol, psycopg2.extras.Json(params)),
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


def list_runs(limit: int = 50, symbol: str | None = None) -> list[dict]:
    """Последние прогоны. symbol=None — по всем монетам."""
    sql = "SELECT * FROM runs"
    params: list[Any] = []
    if symbol:
        sql += " WHERE symbol = %s"
        params.append(symbol)
    sql += " ORDER BY started_at DESC LIMIT %s"
    params.append(limit)
    return fetch_all(sql, params)


def active_run(kind: str | None = None, symbol: str | None = None) -> dict | None:
    """
    Текущий незавершённый прогон.

    symbol=None означает «любой прогон вообще» — это нужно для общего лимита
    одновременных расчётов. Прогоны РАЗНЫХ монет друг другу не мешают
    (пишут в разные строки по symbol и в свой run_id), поэтому блокировать
    их взаимно нельзя: иначе мультимонетность сводится к очереди из одной
    монеты за раз.
    """
    conditions = ["status = 'running'"]
    params: list[Any] = []
    if kind:
        conditions.append("kind = %s")
        params.append(kind)
    if symbol:
        conditions.append("symbol = %s")
        params.append(symbol)
    return fetch_one(
        f"SELECT * FROM runs WHERE {' AND '.join(conditions)} "
        "ORDER BY started_at DESC LIMIT 1",
        tuple(params),
    )


def active_runs(kind: str | None = None) -> list[dict]:
    """Все идущие прогоны — для лимита одновременных расчётов в админке."""
    sql = "SELECT * FROM runs WHERE status = 'running'"
    params: list[Any] = []
    if kind:
        sql += " AND kind = %s"
        params.append(kind)
    return fetch_all(sql + " ORDER BY started_at DESC", params)


def latest_completed_run(kind: str = "train", symbol: str | None = None) -> dict | None:
    """
    Последний успешный прогон нужного типа — источник актуальной модели.

    Фильтр по монете обязателен по смыслу: модель состояний ETH нельзя
    применить к барам BTC, а без фильтра live взял бы модель последнего
    train'а вообще, чем бы он ни был.
    """
    sql = "SELECT * FROM runs WHERE kind = %s AND status = 'done'"
    params: list[Any] = [kind]
    if symbol:
        sql += " AND symbol = %s"
        params.append(symbol)
    return fetch_one(sql + " ORDER BY finished_at DESC LIMIT 1", tuple(params))


def symbols_with_runs() -> list[str]:
    """Монеты, по которым были прогоны — для селектора в админке."""
    rows = fetch_all(
        "SELECT DISTINCT symbol FROM runs WHERE symbol IS NOT NULL ORDER BY symbol"
    )
    return [row["symbol"] for row in rows]


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
