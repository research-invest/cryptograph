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


def list_runs(limit: int = 50, symbol: str | None = None,
              offset: int = 0, kind: str | None = None) -> list[dict]:
    """
    Прогоны от свежих к старым. symbol=None — по всем монетам.

    Сортировка по `started_at DESC`, а не по `run_id`: при `--all` монеты идут
    последовательно и номера растут вместе со временем, но прогон, запущенный
    из админки во время крон-прогона, может получить номер меньше при более
    позднем старте.
    """
    sql = "SELECT * FROM runs"
    conditions: list[str] = []
    params: list[Any] = []
    if symbol:
        conditions.append("symbol = %s")
        params.append(symbol)
    if kind:
        conditions.append("kind = %s")
        params.append(kind)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY started_at DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])
    return fetch_all(sql, params)


def count_runs(symbol: str | None = None, kind: str | None = None) -> int:
    """Сколько всего прогонов подходит под фильтр — для пагинации."""
    sql = "SELECT count(*) AS n FROM runs"
    conditions: list[str] = []
    params: list[Any] = []
    if symbol:
        conditions.append("symbol = %s")
        params.append(symbol)
    if kind:
        conditions.append("kind = %s")
        params.append(kind)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    row = fetch_one(sql, params)
    return int(row["n"]) if row else 0


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


def model_run_scope(run_id: int) -> tuple[str, list[Any]]:
    """
    SQL-условие «строки, посчитанные ЭТОЙ моделью состояний».

    Одна модель живёт во многих прогонах: сам `train`, который её обучил, и все
    последующие `live`, которые её загрузили. Поэтому фильтр `run_id = N`
    означает не «эта модель», а «этот запуск» — куда более узкое множество.

    Пренебрежение этим различием даёт две разные беды, и обе тихие:

    * **в админке** — страницы замирают на моменте последнего обучения, потому
      что всё выпущенное краном лежит под другими `run_id`. Выглядит как
      «кандидаты пропали», хотя они на месте и исправно оценены;
    * **в замерах** — если, наоборот, взять всех кандидатов монеты, в выборку
      попадут записи от прежних моделей. `group_id` и `transition_id` у них из
      чужой нумерации, а на базе разработчика с несколькими частичными
      прогонами такие записи ещё и скапливаются в свежем периоде, ломая
      разбиение на обучение и holdout.

    Связь live → модель хранится в `runs.params.model_run_id`; сравнение идёт
    с текстом, а не с числом, чтобы отсутствующий или нечисловой ключ не ронял
    запрос приведением типа.
    """
    return (
        "(run_id = %s OR run_id IN ("
        " SELECT run_id FROM runs WHERE params->>'model_run_id' = %s))",
        [run_id, str(run_id)],
    )


def symbols_with_runs() -> list[str]:
    """Монеты, по которым были прогоны — для селектора в админке."""
    rows = fetch_all(
        "SELECT DISTINCT symbol FROM runs WHERE symbol IS NOT NULL ORDER BY symbol"
    )
    return [row["symbol"] for row in rows]


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
