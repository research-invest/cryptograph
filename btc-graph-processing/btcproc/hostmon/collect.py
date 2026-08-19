"""
Снимок состояния машины.

Всё, что пишется в историю, собирает `Sampler`: он держит состояние между
замерами, потому что загрузка CPU и скорость диска — это дельты между двумя
опросами, а не мгновенные величины. Отсюда правило: один долгоживущий
`Sampler` в процессе-сэмплере, а не новый объект на каждый замер.

Процессы и контейнеры в историю не пишутся сознательно. Ряд «топ процессов по
минутам за месяц» — это уже своя база размером с сам монитор, а вопрос, на
который он отвечает («кто съел память сейчас»), решается живым снимком. Что
происходило ночью, видно по графикам общей нагрузки; кто именно это делал —
по журналу прогонов на той же странице.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time

import psutil

from btcproc import config

logger = logging.getLogger(__name__)

# Дольше этого `docker` не ждём. Опрос идёт внутри такта сэмплера, и зависший
# демон docker не имеет права остановить запись остальных метрик.
DOCKER_TIMEOUT = 15


class Sampler:
    """
    Источник замеров. Первый снимок после создания объекта отдаёт нулевую
    загрузку CPU и нулевую скорость диска — дельту ещё не с чем считать.
    Поэтому сэмплер создаётся один раз при старте процесса и делает первый
    замер уже на следующем такте.
    """

    def __init__(self) -> None:
        self._io = psutil.disk_io_counters()
        self._net = psutil.net_io_counters()
        self._at = time.monotonic()
        # Инициализация внутреннего счётчика psutil: без этого первый вызов
        # cpu_percent() вернёт загрузку с момента загрузки машины, то есть
        # среднее за сутки под видом «сейчас».
        psutil.cpu_percent(interval=None)
        psutil.cpu_times_percent(interval=None)
        self._missing_mounts: set[str] = set()

    def snapshot(self, ts: int | None = None) -> dict:
        """Замер для записи в историю."""
        now = time.monotonic()
        elapsed = max(0.001, now - self._at)

        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        times = psutil.cpu_times_percent(interval=None)
        io = psutil.disk_io_counters()
        net = psutil.net_io_counters()

        sample = {
            "ts": int(ts if ts is not None else time.time()),
            "cpu": round(psutil.cpu_percent(interval=None), 2),
            # iowait есть только на Linux. На macOS (машина разработки) поля
            # нет вовсе — пишем NULL, а не ноль: ноль означал бы «диск не
            # тормозит», чего мы не знаем.
            "cpu_iowait": round(getattr(times, "iowait", None), 2)
            if getattr(times, "iowait", None) is not None else None,
            "mem_used": int(vm.total - vm.available),
            "mem_total": int(vm.total),
            "mem_pct": round(vm.percent, 2),
            "swap_used": int(swap.used),
            "swap_total": int(swap.total),
            "swap_pct": round(swap.percent, 2),
            "procs": len(psutil.pids()),
            "uptime": int(time.time() - psutil.boot_time()),
            "disks": self._disks(),
        }
        sample.update(zip(("load1", "load5", "load15"), self._load()))
        sample.update(self._rates(io, net, elapsed))

        self._io, self._net, self._at = io, net, now
        return sample

    # ─── Внутренности ───────────────────────────────────────────────────────
    def _load(self) -> tuple[float, float, float]:
        try:
            return tuple(round(v, 2) for v in psutil.getloadavg())  # type: ignore[return-value]
        except (OSError, AttributeError):       # Windows и экзотика
            return (None, None, None)          # type: ignore[return-value]

    def _rates(self, io, net, elapsed: float) -> dict:
        """Скорости диска и сети — дельта счётчиков, делённая на прошедшее время."""
        rates = {"io_read": None, "io_write": None, "net_recv": None, "net_sent": None}
        if io and self._io:
            # Счётчики монотонны, но перезагрузка машины сбрасывает их в ноль:
            # без max(0, …) первый замер после ребута дал бы огромный
            # отрицательный выброс на графике.
            rates["io_read"] = round(max(0, io.read_bytes - self._io.read_bytes) / elapsed, 1)
            rates["io_write"] = round(max(0, io.write_bytes - self._io.write_bytes) / elapsed, 1)
        if net and self._net:
            rates["net_recv"] = round(max(0, net.bytes_recv - self._net.bytes_recv) / elapsed, 1)
            rates["net_sent"] = round(max(0, net.bytes_sent - self._net.bytes_sent) / elapsed, 1)
        return rates

    def _disks(self) -> list[dict]:
        out = []
        for mount in config.hostmon.mounts:
            try:
                usage = psutil.disk_usage(mount)
            except OSError as exc:
                # Ругаемся один раз на точку монтирования: замер идёт каждую
                # минуту, и повторение залило бы лог.
                if mount not in self._missing_mounts:
                    self._missing_mounts.add(mount)
                    logger.warning("Точка монтирования %s недоступна: %s", mount, exc)
                continue
            self._missing_mounts.discard(mount)
            out.append({
                "mount": mount,
                "used": int(usage.used),
                "total": int(usage.total),
                "pct": round(usage.percent, 2),
            })
        return out


def processes(limit: int | None = None, interval: float = 0.3) -> dict:
    """
    Живой снимок процессов: топ по CPU и топ по памяти.

    Загрузка процессора у процесса — тоже дельта, поэтому здесь два прохода с
    паузой `interval`. Пауза короткая и намеренно блокирующая: страница
    вызывает это в пуле потоков FastAPI, и полсекунды ожидания дешевле, чем
    хранить состояние по каждому pid между запросами.

    Проценты CPU — в единицах psutil: 100% означает одно полностью занятое
    ядро, на шести ядрах сумма доходит до 600%.
    """
    limit = limit or config.hostmon.top_processes
    procs = []
    for proc in psutil.process_iter():
        try:
            proc.cpu_percent(interval=None)      # первый проход: только затравка
            procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    time.sleep(interval)

    rows = []
    for proc in procs:
        try:
            with proc.oneshot():
                mem = proc.memory_info()
                rows.append({
                    "pid": proc.pid,
                    "name": proc.name(),
                    # Командная строка обрезается: у прогонов btcproc она
                    # длиннее, чем ширина таблицы, а различает их начало.
                    "cmdline": " ".join(proc.cmdline())[:160] or proc.name(),
                    "user": proc.username(),
                    "cpu": round(proc.cpu_percent(interval=None), 1),
                    "rss": int(mem.rss),
                    "mem_pct": round(proc.memory_percent(), 2),
                    "created": int(proc.create_time()),
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            # Процесс мог завершиться между проходами — это норма, а не сбой.
            continue

    return {
        "by_cpu": sorted(rows, key=lambda r: r["cpu"], reverse=True)[:limit],
        "by_memory": sorted(rows, key=lambda r: r["rss"], reverse=True)[:limit],
        "total": len(rows),
    }


def containers() -> dict:
    """
    Контейнеры docker: статус из `docker ps -a` плюс потребление из
    `docker stats`.

    Два вызова, а не один, потому что `stats` не показывает остановленные
    контейнеры вовсе — а именно упавший postgres и надо увидеть. `ps -a`
    остаётся источником списка, `stats` только дополняет его числами.

    Ошибка любого рода (нет docker, нет прав на сокет, демон не отвечает)
    возвращается полем `error`: страница обязана открываться и без docker.
    """
    if not config.hostmon.docker:
        return {"enabled": False, "rows": [], "error": None}
    if not shutil.which("docker"):
        return {"enabled": True, "rows": [], "error": "docker не найден в PATH"}

    listed, error = _docker_json(["ps", "-a", "--no-trunc", "--format", "{{json .}}"])
    if error:
        return {"enabled": True, "rows": [], "error": error}

    stats, stats_error = _docker_json(
        ["stats", "--no-stream", "--format", "{{json .}}"]
    )
    usage = {row.get("Name"): row for row in stats}

    rows = []
    for row in listed:
        name = row.get("Names") or row.get("Name") or "—"
        live = usage.get(name, {})
        rows.append({
            "name": name,
            "image": row.get("Image", ""),
            "state": row.get("State", ""),
            "status": row.get("Status", ""),
            "cpu": live.get("CPUPerc", ""),
            "mem": live.get("MemUsage", ""),
            "mem_pct": live.get("MemPerc", ""),
            "net": live.get("NetIO", ""),
            "block": live.get("BlockIO", ""),
        })
    rows.sort(key=lambda r: (r["state"] != "running", r["name"]))
    # Ключ `rows`, а не `items`: в шаблоне `containers.items` резолвится в метод
    # словаря, и страница падает с «object is not iterable» — Jinja сначала
    # пробует getattr, и только потом обращение по ключу.
    return {"enabled": True, "rows": rows, "error": stats_error}


def _docker_json(args: list[str]) -> tuple[list[dict], str | None]:
    """Вызов docker с форматом `{{json .}}`: по строке JSON на объект."""
    try:
        proc = subprocess.run(
            ["docker", *args],
            capture_output=True, text=True, timeout=DOCKER_TIMEOUT, check=False,
        )
    except subprocess.TimeoutExpired:
        return [], f"docker {args[0]} не ответил за {DOCKER_TIMEOUT} с"
    except OSError as exc:
        return [], f"docker {args[0]}: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        return [], f"docker {args[0]}: {detail[-1] if detail else 'код ' + str(proc.returncode)}"

    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Неразбираемая строка docker %s: %r", args[0], line[:120])
    return rows, None


# Кэш снимка контейнеров. `docker stats --no-stream` работает секунду-две (он
# ждёт собственный такт сбора), а страница обновляется каждые несколько секунд
# и может быть открыта в нескольких вкладках. Без кэша монитор сам стал бы
# заметной нагрузкой на машину, которую сторожит.
_containers_cache: dict = {"at": 0.0, "value": None}
CONTAINERS_TTL = 8.0


def containers_cached(ttl: float = CONTAINERS_TTL) -> dict:
    """`containers()` с кэшем на `ttl` секунд — для страницы админки."""
    now = time.monotonic()
    if _containers_cache["value"] is None or now - _containers_cache["at"] > ttl:
        _containers_cache.update(at=now, value=containers())
    return _containers_cache["value"]


#: Пути к memory.stat внутри контейнера: cgroup v2 (одна иерархия) и v1.
#: Порядок важен — на v2 второго пути нет, на v1 первого.
_CGROUP_STAT_PATHS = ("/sys/fs/cgroup/memory.stat",
                      "/sys/fs/cgroup/memory/memory.stat")

#: Что вытаскиваем из memory.stat и как это называется для оператора.
#: `shmem` первым не случайно: у Postgres именно он и есть весь ответ на
#: вопрос «почему контейнер съел два гигабайта».
CGROUP_MEMORY_KEYS = (
    ("shmem", "Разделяемая память",
     "Пул страниц, выделенный при старте (shared_buffers). Постоянен и от "
     "нагрузки не зависит."),
    ("anon", "Приватная память процессов",
     "Всё, что бэкенды заняли под себя: сортировки, соединения, планы. Растёт "
     "от запросов."),
    ("file", "Страничный кэш",
     "Файлы базы, поднятые ядром в память. Отдаётся под давлением и в "
     "MEM USAGE докера не входит."),
    ("kernel", "Ядро", "Служебные структуры cgroup."),
)


def container_memory(name: str) -> dict:
    """
    Разбор памяти одного контейнера по cgroup: сколько из показанного
    `docker stats` — разделяемая память, сколько приватная, сколько кэш.

    Зачем отдельно от `containers()`. `docker stats` даёт одно число
    `MemUsage`, и оно вводит в заблуждение ровно в том случае, ради которого
    на него смотрят: у Postgres в него входит `shared_buffers` — статический
    пул, выделенный при старте. Контейнер с 16 МБ реально занятой памяти
    выглядит как контейнер, съевший два гигабайта, и оператор идёт искать
    тяжёлые запросы там, где их нет. Разбор снимает вопрос за один взгляд.

    Читается изнутри контейнера (`docker exec`), а не с хоста: путь к cgroup
    контейнера на хосте зависит от драйвера cgroup, версии docker и systemd,
    а `/sys/fs/cgroup/memory.stat` внутри — один и тот же везде.

    Недоступный docker — состояние, а не исключение: вызывающий показывает
    `error` строкой, страница открывается в любом случае.
    """
    if not config.hostmon.docker:
        return {"enabled": False, "values": {}, "error": None}
    if not shutil.which("docker"):
        return {"enabled": True, "values": {}, "error": "docker не найден в PATH"}

    last_error = "не удалось прочитать cgroup контейнера"
    for path in _CGROUP_STAT_PATHS:
        try:
            proc = subprocess.run(
                ["docker", "exec", name, "cat", path],
                capture_output=True, text=True, timeout=DOCKER_TIMEOUT, check=False,
            )
        except subprocess.TimeoutExpired:
            return {"enabled": True, "values": {},
                    "error": f"docker exec не ответил за {DOCKER_TIMEOUT} с"}
        except OSError as exc:
            return {"enabled": True, "values": {}, "error": f"docker exec: {exc}"}

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip().splitlines()
            last_error = detail[-1] if detail else f"код {proc.returncode}"
            continue

        values: dict[str, int] = {}
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                values[parts[0]] = int(parts[1])
        if values:
            return {"enabled": True, "values": values, "error": None}
        last_error = f"{path} пуст"

    return {"enabled": True, "values": {}, "error": last_error}
