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

from btcproc.db.session import connect, fetch_all, fetch_one

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


# ─── Профиль базы: настройки, нагрузка, таблицы, статистика запросов ────────
#
# Всё ниже отвечает на вопрос, на который `snapshot()` не отвечает в принципе:
# «из-за чего база столько занимает и что её грузило». `pg_stat_activity`
# показывает мгновенный срез — закончившийся запрос из неё исчезает бесследно.

#: Настройки, определяющие потребление памяти. Порядок — как их читать:
#: сначала то, что выделяется при старте целиком, потом то, что умножается
#: на число обслуживающих процессов.
_MEMORY_SETTINGS = (
    ("shared_buffers", "Общий пул страниц",
     "Выделяется при старте целиком и не зависит от нагрузки. Именно он "
     "составляет почти весь MEM USAGE контейнера в docker stats."),
    ("work_mem", "Память на операцию сортировки",
     "На КАЖДУЮ сортировку или хэш в запросе, а не на запрос. Что не влезло — "
     "уходит во временные файлы на диск."),
    ("maintenance_work_mem", "Память на обслуживание",
     "Потолок для CREATE INDEX, VACUUM и ALTER TABLE, запущенных вручную."),
    ("autovacuum_work_mem", "Память на автовакуум",
     "Потолок на КАЖДОГО воркера автовакуума. При -1 берётся значение "
     "предыдущей строки — на машине с десятью воркерами это и есть тихий OOM."),
    ("autovacuum_max_workers", "Воркеров автовакуума",
     "Сколько их может работать одновременно. Умножается на строку выше."),
    ("max_connections", "Потолок соединений",
     "Каждое соединение — процесс со своей памятью."),
    ("effective_cache_size", "Оценка кэша ОС",
     "Памяти не занимает: подсказка планировщику, сколько данных, по его "
     "мнению, найдётся в кэше."),
)

#: Сколько запросов показывать в топе. Больше двух десятков на странице никто
#: не читает, а `pg_stat_statements` хранит тысячи.
STATEMENTS_LIMIT = 15


def memory_settings() -> list[dict]:
    """
    Настройки памяти в человеческих единицах, с пояснением каждой.

    `pg_settings` отдаёт значение и единицу раздельно (`8kB`, `kB`, пусто),
    и складывать их приходится здесь: `setting` без `unit` — просто число,
    и показать «shared_buffers = 254080» оператору нельзя.
    """
    names = tuple(name for name, _, _ in _MEMORY_SETTINGS)
    rows = fetch_all(
        "SELECT name, setting, unit, source FROM pg_settings WHERE name = ANY(%s)",
        (list(names),),
        timeout_ms=TIMEOUT_MS,
    )
    found = {row["name"]: row for row in rows}

    result = []
    for name, label, note in _MEMORY_SETTINGS:
        row = found.get(name)
        if row is None:                 # версия PostgreSQL без этой настройки
            continue
        result.append({
            "name": name,
            "label": label,
            "note": note,
            "value": _format_setting(row["setting"], row["unit"]),
            "raw": row["setting"],
            # `source` показывает, откуда значение: 'default', 'configuration
            # file' (то есть postgresql.conf, куда пишет timescaledb-tune) или
            # 'command line' (наш docker-compose). Без этого не отличить
            # «мы так решили» от «так настроил тюнер при первом старте».
            "source": row["source"],
        })
    return result


def _format_setting(setting: str, unit: str | None) -> str:
    """Значение настройки словами: '254080' + '8kB' → '1985 МБ'."""
    if not unit:
        return setting
    try:
        value = int(setting)
    except (TypeError, ValueError):
        return f"{setting} {unit}"

    # -1 у настроек памяти означает «наследовать», а не «минус один байт».
    if value < 0:
        return str(value)

    multipliers = {"B": 1, "kB": 1024, "8kB": 8 * 1024, "16kB": 16 * 1024,
                   "MB": 1024 ** 2, "GB": 1024 ** 3}
    if unit not in multipliers:         # ms, s, min — не про память
        return f"{setting} {unit}"
    return human_bytes(value * multipliers[unit])


def human_bytes(value: float | None) -> str:
    """Байты словами. Свой, а не из админки: модуль не должен зависеть от неё."""
    if value is None:
        return "—"
    step = 1024.0
    for suffix in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if abs(value) < step or suffix == "ТБ":
            return f"{value:.0f} {suffix}" if suffix in ("Б", "КБ") \
                else f"{value:.1f} {suffix}"
        value /= step
    return f"{value:.1f} ТБ"


def database_stats() -> dict:
    """
    Накопленная нагрузка на базу: попадание в кэш, транзакции, временные файлы.

    Ключевая строка здесь — временные файлы: это запросы, которым не хватило
    `work_mem` и которые досортировывались на диске. Ровно они и грузят
    машину, не оставляя следа в `pg_stat_activity`.

    Все счётчики — с момента `stats_reset` (обычно с создания базы), поэтому
    отдаём и его: «3.5 ГБ временных файлов» читается совсем по-разному за
    сутки и за полгода.
    """
    row = fetch_one(
        """
        SELECT numbackends,
               xact_commit, xact_rollback,
               blks_read, blks_hit,
               tup_returned, tup_fetched,
               temp_files, temp_bytes,
               deadlocks, conflicts,
               stats_reset,
               pg_database_size(datname) AS db_bytes
        FROM pg_stat_database
        WHERE datname = current_database()
        """,
        timeout_ms=TIMEOUT_MS,
    ) or {}
    if not row:
        return {}

    hits, reads = row.get("blks_hit") or 0, row.get("blks_read") or 0
    total = hits + reads
    row["cache_hit_pct"] = round(100.0 * hits / total, 2) if total else None
    row["db_size"] = human_bytes(row.get("db_bytes"))
    row["temp_size"] = human_bytes(row.get("temp_bytes"))
    return row


def top_tables(limit: int = 10) -> list[dict]:
    """
    Самые крупные таблицы с их профилем чтения.

    `seq_scan` рядом с размером — не украшение: последовательное чтение
    крупной таблицы и есть типичная причина, по которой база вдруг начинает
    молотить диск. Мёртвые строки в соседней колонке объясняют, почему
    таблица занимает больше, чем в ней данных.
    """
    rows = fetch_all(
        """
        SELECT schemaname || '.' || relname AS table_name,
               pg_total_relation_size(relid) AS total_bytes,
               seq_scan, idx_scan, n_live_tup, n_dead_tup,
               last_autovacuum
        FROM pg_stat_user_tables
        ORDER BY pg_total_relation_size(relid) DESC
        LIMIT %s
        """,
        (limit,),
        timeout_ms=TIMEOUT_MS,
    )
    for row in rows:
        row["size"] = human_bytes(row.get("total_bytes"))
        live, dead = row.get("n_live_tup") or 0, row.get("n_dead_tup") or 0
        # Доля мёртвых строк, а не их число: 90 тысяч мёртвых при двух
        # миллионах живых — норма, при пяти тысячах живых — повод смотреть.
        row["dead_pct"] = round(100.0 * dead / (live + dead), 1) if live + dead else 0.0
    return rows


def top_statements(limit: int = STATEMENTS_LIMIT) -> dict:
    """
    Что грузило базу за период — из `pg_stat_statements`.

    Единственный источник, отвечающий на вопрос «из-за чего было плохо ночью»:
    `pg_stat_activity` хранит только идущее сейчас. Сортировка по СУММАРНОМУ
    времени, а не по среднему: запрос на 50 мс, вызванный миллион раз, грузит
    машину сильнее одного десятиминутного, и именно его обычно и не замечают.

    Отсутствие расширения — штатное состояние, а не ошибка: оно грузится
    только из `shared_preload_libraries`, то есть требует рестарта базы.
    Возвращаем `available=False` и текст того, что надо сделать.
    """
    installed = fetch_one(
        "SELECT 1 AS ok FROM pg_extension WHERE extname = 'pg_stat_statements'",
        timeout_ms=TIMEOUT_MS,
    )
    if not installed:
        return {"available": False, "rows": [], "error": None}

    try:
        rows = fetch_all(
            """
            SELECT queryid,
                   calls,
                   total_exec_time,
                   mean_exec_time,
                   max_exec_time,
                   rows AS row_count,
                   shared_blks_hit,
                   shared_blks_read,
                   left(query, %s) AS query
            FROM pg_stat_statements
            WHERE dbid = (SELECT oid FROM pg_database WHERE datname = current_database())
            ORDER BY total_exec_time DESC
            LIMIT %s
            """,
            (QUERY_CHARS, limit),
            timeout_ms=TIMEOUT_MS,
        )
    except Exception as exc:            # noqa: BLE001
        # Расширение создано, но библиотека не загружена — на такой паре
        # обращение к вьюхе падает. Для страницы это состояние, а не сбой.
        logger.warning("pg_stat_statements не читается: %s", exc)
        return {"available": False, "rows": [],
                "error": str(exc).strip().splitlines()[0][:200]}

    total_time = sum(row.get("total_exec_time") or 0.0 for row in rows) or 1.0
    for row in rows:
        row["query"] = " ".join((row.get("query") or "").split())
        row["total_duration"] = format_seconds((row.get("total_exec_time") or 0) / 1000)
        row["mean_ms"] = round(row.get("mean_exec_time") or 0.0, 1)
        row["max_ms"] = round(row.get("max_exec_time") or 0.0, 1)
        # Доля от показанного топа, а не от всей нагрузки базы: за пределами
        # лимита остаётся хвост, и называть это «процентом нагрузки» было бы
        # враньём. В шаблоне подписано именно так.
        row["share_pct"] = round(100.0 * (row.get("total_exec_time") or 0) / total_time, 1)
        reads = row.get("shared_blks_read") or 0
        hits = row.get("shared_blks_hit") or 0
        row["cache_hit_pct"] = round(100.0 * hits / (hits + reads), 1) if hits + reads else None
    return {"available": True, "rows": rows, "error": None}


def reset_statements() -> bool:
    """
    Обнулить накопленную статистику запросов.

    Нужна, чтобы мерить период: «что грузило базу последний час» иначе не
    отделить от того, что грузило её полгода назад. Права на
    `pg_stat_statements_reset()` есть не у всякой роли — отказ возвращается
    как False, а не как исключение.
    """
    try:
        fetch_one("SELECT pg_stat_statements_reset() AS ok", timeout_ms=TIMEOUT_MS)
        logger.warning("Оператор обнулил статистику pg_stat_statements")
        return True
    except Exception as exc:            # noqa: BLE001
        logger.warning("Не удалось обнулить pg_stat_statements: %s", exc)
        return False
