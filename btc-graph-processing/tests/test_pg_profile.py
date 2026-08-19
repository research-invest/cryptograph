"""
Страница «PostgreSQL»: разбор памяти контейнера и профиль нагрузки базы.

Появилась из конкретного вопроса: контейнер базы держит два гигабайта, когда
с системой никто не работает. Ответ оказался не в запросах — столько занимает
`shared_buffers`, пул страниц, выделенный при старте раз и навсегда. Таблица
контейнеров на «Сервере» показывает это одним числом и тем самым отправляет
искать нагрузку там, где её нет.

Отсюда и предмет проверок: что разбор памяти отделяет статику от того, что
действительно тратят процессы; что отсутствующее `pg_stat_statements` — это
состояние с внятной подсказкой, а не ошибка; и что страница открывается,
когда недоступны docker, база или оба сразу, — смотрят на неё как раз тогда.
"""
from __future__ import annotations

import pytest

from btcproc import config
from btcproc.admin import auth
from btcproc.db import activity
from btcproc.hostmon import collect


# ─── Разбор памяти контейнера ───────────────────────────────────────────────
#: Настоящий вывод `cat /sys/fs/cgroup/memory.stat` из btc_postgres на боевом
#: контуре. Числа не выдуманы: 1.93 ГиБ shmem против 15 МБ приватной памяти —
#: это и есть картина, ради которой страница написана.
CGROUP_SAMPLE = """anon 15888384
file 3547439104
kernel 24481792
kernel_stack 196608
pagetables 4841472
shmem 2073124864
file_mapped 638021632
inactive_anon 753664
"""


@pytest.fixture
def fake_docker(monkeypatch):
    """Подменяет вызов docker, оставляя разбор вывода настоящим."""
    def _install(stdout="", returncode=0, stderr="", exc=None):
        calls = []

        def fake_run(args, **kwargs):
            calls.append(args)
            if exc is not None:
                raise exc

            class Result:
                pass
            result = Result()
            result.returncode = returncode
            result.stdout = stdout
            result.stderr = stderr
            return result

        monkeypatch.setattr(collect.subprocess, "run", fake_run)
        monkeypatch.setattr(collect.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(config, "hostmon",
                            config.replace_hostmon(docker=True)
                            if hasattr(config, "replace_hostmon")
                            else config.hostmon)
        return calls
    return _install


def test_memory_split_separates_static_pool_from_real_usage(fake_docker):
    """
    Главное свойство разбора: shmem и anon не складываются в одно число.

    Иначе страница повторила бы ошибку `docker stats` — ту самую, из-за
    которой её и завели.
    """
    fake_docker(stdout=CGROUP_SAMPLE)

    data = collect.container_memory("btc_postgres")

    assert data["error"] is None
    assert data["values"]["shmem"] == 2073124864
    assert data["values"]["anon"] == 15888384
    # Приватной памяти в СТО с лишним раз меньше, чем разделяемой: вывод,
    # который страница обязана донести до оператора.
    assert data["values"]["anon"] * 100 < data["values"]["shmem"]


def test_memory_falls_back_to_cgroup_v1_path(fake_docker):
    """
    Путь к memory.stat различается между cgroup v1 и v2. Пробуются оба:
    хост с v1 не должен показывать «разбор недоступен».
    """
    attempts = {"n": 0}

    def fake_run(args, **kwargs):
        attempts["n"] += 1

        class Result:
            pass
        result = Result()
        # Первый путь (v2) не существует, второй (v1) отдаёт данные.
        if attempts["n"] == 1:
            result.returncode = 1
            result.stdout = ""
            result.stderr = "cat: /sys/fs/cgroup/memory.stat: No such file"
        else:
            result.returncode = 0
            result.stdout = CGROUP_SAMPLE
            result.stderr = ""
        return result

    fake_docker()
    collect.subprocess.run = fake_run

    data = collect.container_memory("btc_postgres")

    assert attempts["n"] == 2
    assert data["values"]["shmem"] == 2073124864


def test_unavailable_docker_is_a_state_not_an_exception(fake_docker):
    """
    Разбор памяти не имеет права бросить наружу: страницу открывают, когда
    машине плохо, и упавший docker — это её содержание, а не сбой.
    """
    fake_docker(returncode=126, stderr="permission denied on /var/run/docker.sock")

    data = collect.container_memory("btc_postgres")

    assert data["values"] == {}
    assert "permission denied" in data["error"]


# ─── Настройки памяти ───────────────────────────────────────────────────────
def test_settings_are_shown_in_human_units():
    """
    `pg_settings` отдаёт значение и единицу раздельно: 254080 в блоках по 8 кБ.
    Показать оператору «254080» — это не показать ничего.
    """
    assert activity._format_setting("254080", "8kB") == "1.9 ГБ"
    assert activity._format_setting("10587", "kB") == "10.3 МБ"


def test_inherited_setting_stays_minus_one():
    """
    -1 у autovacuum_work_mem означает «наследовать maintenance_work_mem», а не
    минус один байт. Формат обязан оставить его узнаваемым: именно эта пара
    настроек и даёт тихий OOM (десять воркеров по гигабайту).
    """
    assert activity._format_setting("-1", "kB") == "-1"


def test_non_memory_units_are_left_alone():
    """Настройки времени через тот же список проходят и портиться не должны."""
    assert activity._format_setting("2000", "ms") == "2000 ms"


# ─── Статистика запросов ────────────────────────────────────────────────────
@pytest.fixture
def fake_sql(monkeypatch):
    """Подменяет fetch_one/fetch_all по началу SQL-запроса."""
    def _install(answers):
        def pick(sql, params=None, timeout_ms=None):
            for probe, value in answers.items():
                if probe in " ".join(sql.split()):
                    return value
            return []

        monkeypatch.setattr(activity, "fetch_all", pick)
        monkeypatch.setattr(
            activity, "fetch_one",
            lambda sql, params=None, timeout_ms=None: (pick(sql, params) or [None])[0]
            if isinstance(pick(sql, params), list) else pick(sql, params),
        )
    return _install


def test_missing_extension_is_reported_as_a_state(fake_sql):
    """
    `pg_stat_statements` грузится только из shared_preload_libraries, то есть
    включается рестартом базы. Его отсутствие — штатное состояние с понятной
    инструкцией, а не ошибка страницы.
    """
    fake_sql({"FROM pg_extension": []})

    data = activity.top_statements()

    assert data["available"] is False
    assert data["rows"] == []
    assert data["error"] is None


def test_statements_are_ranked_by_total_time_not_average(fake_sql):
    """
    Запрос на 50 мс, вызванный миллион раз, грузит машину сильнее одного
    десятиминутного — и именно его обычно не замечают. Порядок задаёт SQL,
    здесь проверяется, что доля считается от суммарного времени и что частый
    дешёвый запрос её и получает.
    """
    rows = [
        {"queryid": 1, "calls": 1_000_000, "total_exec_time": 900_000.0,
         "mean_exec_time": 0.9, "max_exec_time": 5.0, "row_count": 1,
         "shared_blks_hit": 990, "shared_blks_read": 10, "query": "SELECT  1\n  FROM t"},
        {"queryid": 2, "calls": 1, "total_exec_time": 100_000.0,
         "mean_exec_time": 100_000.0, "max_exec_time": 100_000.0, "row_count": 9,
         "shared_blks_hit": 0, "shared_blks_read": 0, "query": "VACUUM"},
    ]
    fake_sql({"FROM pg_extension": [{"ok": 1}], "FROM pg_stat_statements": rows})

    data = activity.top_statements()

    assert data["available"] is True
    frequent, single = data["rows"]
    assert frequent["share_pct"] == 90.0
    assert single["share_pct"] == 10.0
    # Многострочный SQL схлопывается — в таблице он иначе занимает пол-экрана.
    assert frequent["query"] == "SELECT 1 FROM t"
    assert frequent["cache_hit_pct"] == 99.0
    # Запрос без обращений к страницам не должен показывать «0% попаданий»:
    # это не плохой кэш, это отсутствие данных.
    assert single["cache_hit_pct"] is None


def test_dead_tuple_share_is_relative(fake_sql):
    """
    Мёртвых строк 90 тысяч — это норма при двух миллионах живых и повод
    смотреть при пяти тысячах. Абсолютное число не отвечает ни на что.
    """
    fake_sql({"FROM pg_stat_user_tables": [
        {"table_name": "processing.bar_states", "total_bytes": 421527552,
         "seq_scan": 29, "idx_scan": 37287056, "n_live_tup": 1948686,
         "n_dead_tup": 92090, "last_autovacuum": None},
    ]})

    row = activity.top_tables()[0]

    assert row["dead_pct"] == 4.5
    assert row["size"] == "402.0 МБ"


# ─── Страница целиком ───────────────────────────────────────────────────────
@pytest.fixture
def client(monkeypatch):
    import fastapi.testclient as fastapi_testclient

    monkeypatch.setattr(
        config, "admin",
        config.AdminConfig(user="operator", password="very-long-password-42",
                           secret_key="k" * 64, ip_allowlist=[]),
    )
    from btcproc.admin import app as admin_app

    monkeypatch.setattr(auth, "current_user", lambda request: "operator")
    monkeypatch.setattr(admin_app, "init_schema", lambda: None, raising=False)
    with fastapi_testclient.TestClient(admin_app.app) as test_client:
        yield test_client


def test_page_opens_when_everything_is_down(client, monkeypatch):
    """
    Ровно тот случай, ради которого страницу открывают: база не отвечает,
    docker не отвечает. Ни один из четырёх блоков не имеет права унести с
    собой остальные, и 500 здесь — худший из исходов.
    """
    def boom(*args, **kwargs):
        raise RuntimeError("could not connect to server")

    for name in ("snapshot", "memory_settings", "database_stats",
                 "top_tables", "top_statements"):
        monkeypatch.setattr(activity, name, boom)
    monkeypatch.setattr(collect, "container_memory", boom)

    resp = client.get("/server/postgres")

    assert resp.status_code == 200
    assert "could not connect to server" in resp.text


def test_container_row_links_to_the_page(client, monkeypatch):
    """
    Кликабельность контейнера — вход в диагностику. Ссылка ведёт с той самой
    строки, чьё число вызывает вопрос.
    """
    monkeypatch.setattr(collect, "containers_cached", lambda *a, **k: {
        "enabled": True, "error": None,
        "rows": [{"name": "btc_postgres", "image": "timescale/timescaledb-ha:pg16",
                  "state": "running", "status": "Up 3 days", "cpu": "0.00%",
                  "mem": "1.971GiB / 7.754GiB", "mem_pct": "25.42%",
                  "net": "—", "block": "—"},
                 {"name": "btc_redis", "image": "redis:7-alpine",
                  "state": "running", "status": "Up 3 days", "cpu": "0.6%",
                  "mem": "10MiB / 7.754GiB", "mem_pct": "0.13%",
                  "net": "—", "block": "—"}],
    })

    resp = client.get("/server")

    assert resp.status_code == 200
    assert '<a href="/server/postgres">btc_postgres</a>' in resp.text
    # Остальные контейнеры остаются текстом: разбирать у них нечего.
    assert '<a href="/server/postgres">btc_redis</a>' not in resp.text
