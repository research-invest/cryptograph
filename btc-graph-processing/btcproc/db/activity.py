"""
Что PostgreSQL делает прямо сейчас: активные запросы и их длительность.

Зачем отдельно от `repo.py`. Здесь не данные предметной области, а состояние
самой базы, и читается оно ради страницы «Сервер» — рядом с CPU, диском и
прогонами. Повод завести появился на практике: открытие узла графа уходило в
запрос на десять с лишним минут и клало базу, а увидеть это можно было только
руками через psql на сервере (журнал 43, 44).

Два правила, из которых следует всё остальное:

* **диагностика не имеет права стать причиной проблемы.** Свой потолок —
  две секунды: запрос к `pg_stat_activity` дешёвый, и если он не уложился,
  значит базе плохо ровно так, как страница и должна показать;
* **недоступная база — это факт для страницы, а не её ошибка.** Postgres
  живёт в том же docker-стеке, за которым следит монитор, и смотрят на эту
  страницу как раз тогда, когда стеку плохо. Поэтому исключения ловит
  вызывающий и показывает их как состояние, а не как 500.
"""
from __future__ import annotations

import logging

import psycopg2.extras

from btcproc.db.session import connect, fetch_one

logger = logging.getLogger(__name__)

#: Потолок на сами диагностические запросы. См. докстринг модуля.
TIMEOUT_MS = 2_000

#: Сколько строк показывать. Соединений в пуле btcproc шестнадцать, плюс
#: celery и api btc-graph — активных одновременно единицы, два десятка с
#: запасом покрывают даже всплеск.
LIMIT = 25

#: Сколько символов запроса отдавать наружу. `pg_stat_activity.query` и так
#: обрезан на `track_activity_query_size` (1024 по умолчанию), но в таблицу
#: столько не влезает.
QUERY_CHARS = 400

#: Состояния, в которых бэкенд держит транзакцию открытой, ничего не делая.
#: Отменять там нечего — `pg_cancel_backend` на них не действует, нужен
#: `pg_terminate_backend`. Это же самое вредное состояние: блокировки держатся,
#: autovacuum не может убрать мёртвые строки.
IDLE_IN_TRANSACTION = ("idle in transaction", "idle in transaction (aborted)")

#: Один запрос на весь снимок — и таблица, и сводка считаются из НЕГО.
#:
#: Отдельный SQL для сводки (`count(*) FILTER ...`) выглядел естественнее, но
#: врал: между двумя запросами проходят миллисекунды, и за них картина
#: меняется. На отладке это видно сразу — «активных 2» над таблицей с одной
#: строкой, потому что второй бэкенд появился уже после сбора строк. Строк
#: здесь максимум сотня (`max_connections`), так что фильтрация, сортировка и
#: подсчёт делаются в Python по одному согласованному срезу.
_ACTIVITY_SQL = """
SELECT pid,
       state,
       EXTRACT(EPOCH FROM (now() - query_start))::float AS query_seconds,
       EXTRACT(EPOCH FROM (now() - xact_start))::float  AS xact_seconds,
       wait_event_type,
       wait_event,
       application_name,
       backend_type,
       usename,
       host(client_addr) AS client,
       left(query, %s) AS query
FROM pg_stat_activity
WHERE datname = current_database()
  -- Своё соединение исключаем везде: активным запросом, который рисует эту
  -- таблицу, всегда является она сама.
  AND pid <> pg_backend_pid()
"""


def format_seconds(value: float | None) -> str:
    """
    Длительность запроса словами.

    Свой формат, а не общий `human_seconds` админки: тот округляет до минут
    («12 мин»), и здесь это врёт в обе стороны — быстрый запрос показывался бы
    как «0 с», а десятиминутный терял бы секунды ровно там, где за ними и
    следят.
    """
    if value is None:
        return "—"
    if value < 1:
        return f"{value:.2f} с"
    if value < 60:
        return f"{value:.1f} с"
    minutes, seconds = divmod(int(value), 60)
    if minutes < 60:
        return f"{minutes} м {seconds:02d} с"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} ч {minutes:02d} м"


def snapshot(slow_seconds: float, limit: int = LIMIT) -> dict:
    """
    Текущие запросы и сводка по соединениям.

    `slow_seconds` не фильтрует, а размечает: строка получает `slow=True`, и
    интерфейс её подсвечивает. Прятать короткие запросы нельзя — «активных
    три, все быстрые» это тоже ответ, причём чаще всего именно он и нужен.
    """
    with connect(timeout_ms=TIMEOUT_MS) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_ACTIVITY_SQL, (QUERY_CHARS,))
            backends = [dict(row) for row in cur.fetchall()]
            # Отдельным запросом только то, что не меняется между ними.
            cur.execute("SELECT current_setting('max_connections')::int AS n")
            max_connections = (cur.fetchone() or {}).get("n")

    summary = {
        "total": len(backends),
        "active": sum(1 for b in backends if b["state"] == "active"),
        "idle": sum(1 for b in backends if b["state"] == "idle"),
        "in_transaction": sum(1 for b in backends
                              if b["state"] in IDLE_IN_TRANSACTION),
        "waiting": sum(1 for b in backends
                       if b["state"] == "active" and b["wait_event_type"]),
        "max_connections": max_connections,
    }
    # Простаивающие соединения пула в таблицу не идут: их полтора десятка, они
    # ничего не делают и вытеснили бы всё осмысленное. В СВОДКЕ они есть — там
    # они и нужны, как расход лимита соединений.
    rows = [b for b in backends if b["state"] != "idle"]

    for row in rows:
        seconds = row.get("query_seconds")
        row["slow"] = seconds is not None and seconds >= slow_seconds
        row["duration"] = format_seconds(seconds)
        row["xact_duration"] = format_seconds(row.get("xact_seconds"))
        row["stuck_in_transaction"] = row.get("state") in IDLE_IN_TRANSACTION
        # Текст запроса схлопываем в одну строку: в таблице многострочный SQL
        # с отступами занимает пол-экрана и читается хуже, чем в одну.
        query = (row.get("query") or "").strip()
        row["query"] = " ".join(query.split())

    # Сортировка по длительности убыв. делается здесь, а не в SQL: NULL у
    # query_start (бэкенд без запроса) в SQL пришлось бы отдельно оговаривать
    # в обе стороны, а строк два десятка.
    rows.sort(key=lambda r: (r.get("query_seconds") or -1), reverse=True)
    return {"rows": rows[:limit], "summary": summary, "slow_seconds": slow_seconds}


def stop_query(pid: int) -> dict:
    """
    Снять один бэкенд. Возвращает {action, ok, note} — что было сделано.

    Способ выбирается по актуальному состоянию, а не по тому, что видел
    оператор на странице: между отрисовкой и нажатием проходят секунды.
    `cancel` отменяет запрос и оставляет соединение живым — это и есть
    штатное действие. `terminate` применяется ТОЛЬКО к 'idle in transaction',
    где отменять нечего: там нет запроса, а есть висящая транзакция.

    Прервать можно и запрос идущего прогона — это законное действие
    оператора, но прогон после него упадёт. Поэтому кнопка спрашивает
    подтверждение, а факт пишется в лог.
    """
    row = fetch_one(
        "SELECT pid, state, backend_type, left(query, 200) AS query "
        "FROM pg_stat_activity "
        "WHERE pid = %s AND datname = current_database() AND pid <> pg_backend_pid()",
        (pid,),
        timeout_ms=TIMEOUT_MS,
    )
    if row is None:
        # Не ошибка: запрос мог закончиться сам, пока оператор наводил мышь.
        return {"action": "none", "ok": False,
                "note": f"Бэкенда {pid} уже нет — запрос завершился сам."}

    terminate = row["state"] in IDLE_IN_TRANSACTION
    func = "pg_terminate_backend" if terminate else "pg_cancel_backend"
    logger.warning(
        "Оператор снимает бэкенд %s (%s, %s): %s",
        pid, row["state"], func, (row["query"] or "").strip()[:200],
    )
    result = fetch_one(f"SELECT {func}(%s) AS ok", (pid,), timeout_ms=TIMEOUT_MS) or {}
    ok = bool(result.get("ok"))
    action = "terminate" if terminate else "cancel"
    if ok:
        note = (f"Соединение {pid} закрыто (висящая транзакция)." if terminate
                else f"Запрос {pid} отменён.")
    else:
        # Чаще всего это чужой владелец: autovacuum и служебные процессы
        # принадлежат суперпользователю, а мы ходим под btc_user.
        note = (f"PostgreSQL отказал в сигнале бэкенду {pid} "
                f"({row['backend_type']}) — скорее всего он принадлежит "
                f"другой роли.")
    return {"action": action, "ok": ok, "note": note}
