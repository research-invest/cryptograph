"""
Хранилище замеров нагрузки — SQLite-файл на хосте.

Почему не Postgres проекта — в docstring `config.HostmonConfig`: монитор нужен
именно в тот момент, когда стеку плохо, и зависеть от него не имеет права.

Один писатель (сэмплер) и много читателей (запросы админки), поэтому база
открывается в режиме WAL: без него чтение страницы блокировало бы вставку
замера, а вставка — чтение. Админка ходит сюда read-only через URI-параметр
`mode=ro`; так опечатка в коде страницы не может испортить ряд.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from btcproc import config

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    ts        INTEGER PRIMARY KEY,   -- unix-секунды UTC, выровнены по сетке
    cpu       REAL NOT NULL,         -- загрузка всех ядер, %
    cpu_iowait REAL,                 -- ожидание диска, % (нет на macOS)
    load1     REAL,
    load5     REAL,
    load15    REAL,
    mem_used  INTEGER,               -- байты, «занято» без кэша
    mem_total INTEGER,
    mem_pct   REAL,
    swap_used INTEGER,
    swap_total INTEGER,
    swap_pct  REAL,
    io_read   REAL,                  -- байт/с, среднее с прошлого замера
    io_write  REAL,
    net_recv  REAL,
    net_sent  REAL,
    procs     INTEGER,
    uptime    INTEGER                -- секунды с загрузки машины
);

-- Состояние алертов: по строке на правило. Живёт в базе, а не в памяти
-- сэмплера, ровно из-за антиспама: перезапуск сервиса (выкатка кода, ребут)
-- обнулял бы память и слал бы повторное сообщение о той же проблеме сразу.
CREATE TABLE IF NOT EXISTS alert_state (
    key           TEXT PRIMARY KEY,  -- правило: disk:/ , memory, swap, cpu, load
    firing        INTEGER NOT NULL,  -- 1 — порог нарушен сейчас
    since         INTEGER,           -- когда нарушение началось
    breaches      INTEGER NOT NULL,  -- подряд идущих замеров за порогом
    last_notified INTEGER,           -- когда последний раз слали сообщение
    value         REAL
);

-- Журнал отправок: что именно и когда ушло в Telegram. Нужен не для красоты —
-- без него «почему не пришло уведомление» неотличимо от «нечего было
-- присылать», а неудачную отправку видно только в логе сервиса.
CREATE TABLE IF NOT EXISTS alert_log (
    ts      INTEGER NOT NULL,
    key     TEXT    NOT NULL,
    kind    TEXT    NOT NULL,        -- fired | still | recovered
    value   REAL,
    message TEXT,
    sent    INTEGER NOT NULL,        -- 1 — Telegram принял
    error   TEXT,
    PRIMARY KEY (ts, key, kind)
);

-- Диски отдельной таблицей: точек монтирования может быть несколько, и
-- держать их колонками значило бы менять схему при каждом новом разделе.
CREATE TABLE IF NOT EXISTS disk_samples (
    ts    INTEGER NOT NULL,
    mount TEXT    NOT NULL,
    used  INTEGER,
    total INTEGER,
    pct   REAL,
    PRIMARY KEY (ts, mount)
);
"""

# Метрики, у которых сохраняется пик по бакету, а не только среднее. Сглаженный
# график нагрузки врёт в самую важную сторону: минутный всплеск памяти перед
# OOM в среднем за час не виден вообще.
PEAK_METRICS = ("cpu", "mem_pct", "swap_pct", "load1")


def connect(path: Path | None = None, read_only: bool = False) -> sqlite3.Connection:
    """
    Соединение с базой замеров.

    `read_only=True` — для админки: файл не создаётся, а отсутствие его
    означает «сэмплер ни разу не отработал», и это должно быть видно как
    ошибка, а не как пустой график.
    """
    path = Path(path or config.hostmon.db_path)
    if read_only:
        if not path.exists():
            raise FileNotFoundError(path)
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=10)
        conn.executescript(SCHEMA)
    conn.row_factory = sqlite3.Row
    # WAL — свойство файла, а не соединения: выставляется однажды при записи.
    if not read_only:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def write(conn: sqlite3.Connection, sample: dict) -> None:
    """
    Записать снимок. `INSERT OR REPLACE`, потому что метка времени выровнена по
    сетке: перезапуск сэмплера внутри той же минуты должен обновлять замер, а
    не падать на конфликте первичного ключа.
    """
    host = {k: sample.get(k) for k in (
        "ts", "cpu", "cpu_iowait", "load1", "load5", "load15",
        "mem_used", "mem_total", "mem_pct",
        "swap_used", "swap_total", "swap_pct",
        "io_read", "io_write", "net_recv", "net_sent", "procs", "uptime",
    )}
    columns = ", ".join(host)
    placeholders = ", ".join(f":{k}" for k in host)
    with conn:
        conn.execute(
            f"INSERT OR REPLACE INTO samples ({columns}) VALUES ({placeholders})",
            host,
        )
        conn.executemany(
            "INSERT OR REPLACE INTO disk_samples (ts, mount, used, total, pct) "
            "VALUES (:ts, :mount, :used, :total, :pct)",
            [{"ts": sample["ts"], **disk} for disk in sample.get("disks", [])],
        )


def latest(conn: sqlite3.Connection) -> dict | None:
    """Последний замер вместе с дисками. `None` — замеров ещё нет."""
    row = conn.execute("SELECT * FROM samples ORDER BY ts DESC LIMIT 1").fetchone()
    if row is None:
        return None
    out = dict(row)
    out["disks"] = [
        dict(d) for d in conn.execute(
            "SELECT mount, used, total, pct FROM disk_samples "
            "WHERE ts = ? ORDER BY mount", (row["ts"],),
        )
    ]
    return out


def series(
    conn: sqlite3.Connection,
    since_ts: int,
    until_ts: int | None = None,
    max_points: int = 720,
) -> dict:
    """
    Ряды для графиков за окно `[since_ts, until_ts]`.

    Точек в окне может быть сколько угодно (месяц минутных замеров — сорок
    тысяч), поэтому ряд прореживается бакетами. Бакет кратен шагу сетки и
    подбирается так, чтобы точек вышло не больше `max_points`: браузеру
    хватает, а форма графика сохраняется.

    Для нагрузочных метрик рядом со средним отдаётся пик по бакету
    (`*_peak`) — усреднение прячет короткие всплески, ради которых монитор и
    существует.
    """
    until_ts = until_ts or _now(conn)
    span = max(1, until_ts - since_ts)
    step = max(1, config.hostmon.interval)
    bucket = max(step, -(-span // max(1, max_points)))
    # Бакет кратен шагу: иначе в бакет попадало бы то два замера, то один, и
    # ряд «дышал» бы амплитудой на ровной нагрузке.
    bucket = -(-bucket // step) * step

    rows = conn.execute(
        f"""
        SELECT (ts / {bucket}) * {bucket} AS bucket,
               AVG(cpu) AS cpu, MAX(cpu) AS cpu_peak,
               AVG(cpu_iowait) AS cpu_iowait,
               AVG(mem_pct) AS mem_pct, MAX(mem_pct) AS mem_pct_peak,
               AVG(mem_used) AS mem_used,
               AVG(swap_pct) AS swap_pct, MAX(swap_pct) AS swap_pct_peak,
               AVG(load1) AS load1, MAX(load1) AS load1_peak,
               AVG(io_read) AS io_read, AVG(io_write) AS io_write,
               AVG(net_recv) AS net_recv, AVG(net_sent) AS net_sent
          FROM samples
         WHERE ts BETWEEN ? AND ?
         GROUP BY bucket
         ORDER BY bucket
        """,
        (since_ts, until_ts),
    ).fetchall()

    names = [k for k in rows[0].keys() if k != "bucket"] if rows else []
    out: dict[str, list[dict]] = {name: [] for name in names}
    for row in rows:
        for name in names:
            if row[name] is not None:
                out[name].append({"time": row["bucket"], "value": round(row[name], 3)})

    # Заполнение дисков — по каждой точке монтирования своим рядом: разделы
    # заполняются независимо, и общий график по ним был бы бессмысленным.
    #
    # Рядом с процентом отдаются занятые и общие байты: график рисуется в
    # процентах (разделы разного размера иначе несопоставимы), но «91%» без
    # «осталось 6 ГБ» не говорит, сколько времени есть на реакцию.
    disks: dict[str, list[dict]] = {}
    for row in conn.execute(
        f"""
        SELECT (ts / {bucket}) * {bucket} AS bucket, mount,
               AVG(pct) AS pct, AVG(used) AS used, AVG(total) AS total
          FROM disk_samples
         WHERE ts BETWEEN ? AND ?
         GROUP BY bucket, mount
         ORDER BY bucket
        """,
        (since_ts, until_ts),
    ):
        if row["pct"] is not None:
            disks.setdefault(row["mount"], []).append({
                "time": row["bucket"],
                "value": round(row["pct"], 3),
                "used": int(row["used"]) if row["used"] is not None else None,
                "total": int(row["total"]) if row["total"] is not None else None,
            })

    return {"bucket": bucket, "points": len(rows), "metrics": out, "disks": disks}


def coverage(conn: sqlite3.Connection) -> dict:
    """Сколько замеров и за какой период есть в базе — для подписи на странице."""
    row = conn.execute(
        "SELECT COUNT(*) AS n, MIN(ts) AS first_ts, MAX(ts) AS last_ts FROM samples"
    ).fetchone()
    path = Path(config.hostmon.db_path)
    return {
        "samples": row["n"],
        "first_ts": row["first_ts"],
        "last_ts": row["last_ts"],
        "db_bytes": path.stat().st_size if path.exists() else 0,
        "db_path": str(path),
    }


def prune(conn: sqlite3.Connection, keep_days: int | None = None) -> int:
    """
    Удалить замеры старше `keep_days`. Возвращает число удалённых строк.

    Без `VACUUM`: файл не отдаёт место обратно системе, но и не растёт —
    освободившиеся страницы переиспользуются. Монитор диска, регулярно
    занимающий диск целиком под собственный VACUUM, был бы дурной шуткой.
    """
    keep_days = keep_days if keep_days is not None else config.hostmon.keep_days
    if keep_days <= 0:
        return 0
    cutoff = _now(conn) - keep_days * 86400
    with conn:
        deleted = conn.execute("DELETE FROM samples WHERE ts < ?", (cutoff,)).rowcount
        conn.execute("DELETE FROM disk_samples WHERE ts < ?", (cutoff,))
        # Журнал алертов живёт по тому же правилу: он бесполезен без графика,
        # рядом с которым его читают.
        conn.execute("DELETE FROM alert_log WHERE ts < ?", (cutoff,))
    return deleted


# ─── Алерты ─────────────────────────────────────────────────────────────────
def alert_states(conn: sqlite3.Connection) -> dict[str, dict]:
    """Состояние всех правил разом: правил единицы, отдельные SELECT'ы излишни."""
    return {
        row["key"]: dict(row)
        for row in conn.execute("SELECT * FROM alert_state")
    }


def save_alert_state(conn: sqlite3.Connection, key: str, state: dict) -> None:
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO alert_state "
            "(key, firing, since, breaches, last_notified, value) "
            "VALUES (:key, :firing, :since, :breaches, :last_notified, :value)",
            {"key": key, **state},
        )


def log_alert(conn: sqlite3.Connection, entry: dict) -> None:
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO alert_log (ts, key, kind, value, message, sent, error) "
            "VALUES (:ts, :key, :kind, :value, :message, :sent, :error)",
            entry,
        )


def recent_alerts(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    return [
        dict(row) for row in conn.execute(
            "SELECT * FROM alert_log ORDER BY ts DESC LIMIT ?", (limit,)
        )
    ]


def _now(conn: sqlite3.Connection) -> int:
    """Текущее время в тех же единицах, что метки замеров."""
    return int(conn.execute("SELECT strftime('%s','now')").fetchone()[0])
