"""
Монитор ресурсов хоста: хранилище замеров и пороговые уведомления.

Сети здесь нет: отправка в Telegram подменяется, и это единственный способ
проверять антиспам — иначе тест либо шлёт сообщения по-настоящему, либо не
проверяет ничего. Ходить в psutil тесты тоже не обязаны: правила считаются по
готовому словарю-замеру, ровно такому, какой пишет сэмплер.
"""
from __future__ import annotations

import contextlib

import pytest

from btcproc import config
from btcproc.hostmon import alerts, store, telegram


def sample(ts: int, *, cpu: float = 10.0, mem: float = 40.0, swap: float = 0.0,
           disk: float = 50.0, load: float = 0.5) -> dict:
    """Замер в том же виде, в каком его отдаёт `collect.Sampler.snapshot`."""
    return {
        "ts": ts, "cpu": cpu, "cpu_iowait": 0.5,
        "load1": load, "load5": load, "load15": load,
        "mem_used": int(8 * 1024 ** 3 * mem / 100), "mem_total": 8 * 1024 ** 3,
        "mem_pct": mem,
        "swap_used": int(2 * 1024 ** 3 * swap / 100), "swap_total": 2 * 1024 ** 3,
        "swap_pct": swap,
        "io_read": 1e6, "io_write": 2e6, "net_recv": 1e5, "net_sent": 1e5,
        "procs": 200, "uptime": 100000,
        "disks": [{"mount": "/", "used": int(80 * 1024 ** 3 * disk / 100),
                   "total": 80 * 1024 ** 3, "pct": disk}],
    }


@pytest.fixture()
def db(tmp_path):
    """Пустая база замеров в tmp: боевой файл трогать нельзя."""
    with contextlib.closing(store.connect(tmp_path / "hostmon.sqlite")) as conn:
        yield conn


@pytest.fixture()
def sent(monkeypatch):
    """Перехват отправки: список ушедших сообщений вместо походов в сеть."""
    outbox: list[str] = []
    monkeypatch.setattr(telegram, "send", lambda text, **kw: outbox.append(text))
    monkeypatch.setattr(
        config, "alerts",
        config.AlertsConfig(
            enabled=True, bot_token="токен", chat_id="42",
            cooldown_minutes=5, hysteresis=5.0, sustain=3,
            disk_pct=90.0, disk_critical_pct=96.0, mem_pct=90.0,
            swap_pct=60.0, cpu_pct=90.0, load_per_core=2.0,
        ),
    )
    return outbox


# ─── Хранилище ──────────────────────────────────────────────────────────────
def test_missing_database_is_an_explicit_error(tmp_path):
    """
    Отсутствие файла — это «сэмплер не запущен», и админка обязана узнать об
    этом ошибкой. Молчаливое создание пустой базы дало бы пустые графики без
    объяснения причины.
    """
    with pytest.raises(FileNotFoundError):
        store.connect(tmp_path / "нет.sqlite", read_only=True)


def test_write_and_read_roundtrip(db):
    store.write(db, sample(1_800_000_000, cpu=33.0, disk=71.5))
    latest = store.latest(db)
    assert latest["cpu"] == 33.0
    assert latest["disks"][0]["mount"] == "/"
    assert latest["disks"][0]["pct"] == 71.5


def test_repeated_timestamp_updates_instead_of_failing(db):
    """
    Метки выровнены по сетке, поэтому перезапуск сэмплера внутри такта даёт
    вторую запись с тем же ts. Это обновление, а не конфликт.
    """
    store.write(db, sample(1_800_000_000, cpu=10.0))
    store.write(db, sample(1_800_000_000, cpu=90.0))
    assert store.latest(db)["cpu"] == 90.0
    assert store.coverage(db)["samples"] == 1


def test_series_thinning_keeps_the_peak(db):
    """
    Главное свойство прореживания: пик в бакете сохраняется отдельным рядом.
    Минутный всплеск памяти перед OOM в среднем за час не виден вообще — ради
    этого `*_peak` и существует.
    """
    base = 1_800_000_000
    for i in range(60):
        store.write(db, sample(base + i * 60, cpu=99.0 if i == 17 else 5.0))

    data = store.series(db, base, base + 3600, max_points=6)
    assert data["bucket"] % config.hostmon.interval == 0, "бакет обязан быть кратен шагу сетки"
    assert data["points"] <= 6
    assert max(p["value"] for p in data["metrics"]["cpu_peak"]) == 99.0
    # Среднее всплеск размывает — именно поэтому одного его недостаточно.
    assert max(p["value"] for p in data["metrics"]["cpu"]) < 99.0


def test_prune_drops_old_samples_and_keeps_fresh(db):
    now = store._now(db)
    store.write(db, sample(now - 40 * 86400))
    store.write(db, sample(now - 60))
    assert store.prune(db, keep_days=30) == 1
    assert store.coverage(db)["samples"] == 1


# ─── Алерты: срабатывание и антиспам ────────────────────────────────────────
def test_disk_alert_fires_once_and_repeats_only_after_cooldown(db, sent):
    """
    Первое сообщение — сразу, дальше не чаще cooldown. Это и есть требование
    «уведомлять, но не спамить»: без окна каждый замер за порогом означал бы
    сообщение в минуту.
    """
    now = 1_800_000_000
    alerts.check(db, sample(now, disk=91.0), now=now)
    assert len(sent) == 1 and "диск /" in sent[0]

    # Минута спустя — проблема та же, сообщения быть не должно.
    alerts.check(db, sample(now + 60, disk=91.5), now=now + 60)
    assert len(sent) == 1

    # Ровно через cooldown уходит напоминание, помеченное как таковое.
    later = now + config.alerts.cooldown_minutes * 60
    alerts.check(db, sample(later, disk=92.0), now=later)
    assert len(sent) == 2 and "напоминание" in sent[1]


def test_recovery_needs_hysteresis(db, sent):
    """
    Возврат под порог не гасит алерт: гасит только уход ниже порога на
    гистерезис. Иначе метрика, топчущаяся вокруг 90%, слала бы пару
    «сработало/отпустило» на каждом замере.
    """
    now = 1_800_000_000
    alerts.check(db, sample(now, disk=91.0), now=now)
    assert len(sent) == 1

    alerts.check(db, sample(now + 60, disk=88.0), now=now + 60)   # под порогом, но в зоне гистерезиса
    assert len(sent) == 1, "в зоне гистерезиса сообщений быть не должно"

    alerts.check(db, sample(now + 120, disk=84.0), now=now + 120)
    assert len(sent) == 2 and "в норме" in sent[1]


def test_cpu_alert_waits_for_sustain(db, sent):
    """
    CPU на этой машине штатно уходит в потолок на время кластеризации, поэтому
    мгновенное значение поводом не является: нужна выдержка в несколько
    замеров подряд.
    """
    now = 1_800_000_000
    for i in range(config.alerts.sustain - 1):
        alerts.check(db, sample(now + i * 60, cpu=97.0), now=now + i * 60)
    assert sent == [], "до выдержки уведомлять нельзя"

    last = now + (config.alerts.sustain - 1) * 60
    alerts.check(db, sample(last, cpu=97.0), now=last)
    assert len(sent) == 1 and "процессор" in sent[0]


def test_breach_streak_resets_on_a_calm_sample(db, sent):
    """Выдержка требует ПОДРЯД идущих замеров, иначе она ничего не значит."""
    now = 1_800_000_000
    alerts.check(db, sample(now, cpu=97.0), now=now)
    alerts.check(db, sample(now + 60, cpu=20.0), now=now + 60)
    alerts.check(db, sample(now + 120, cpu=97.0), now=now + 120)
    assert sent == []


def test_alert_state_survives_a_restart(tmp_path, sent):
    """
    Состояние живёт в базе, а не в памяти процесса: перезапуск сервиса не
    имеет права превращаться в повторное сообщение о той же проблеме.
    """
    path = tmp_path / "hostmon.sqlite"
    now = 1_800_000_000
    with contextlib.closing(store.connect(path)) as conn:
        alerts.check(conn, sample(now, disk=91.0), now=now)
    assert len(sent) == 1

    # «Новый процесс» — новое соединение с тем же файлом.
    with contextlib.closing(store.connect(path)) as conn:
        alerts.check(conn, sample(now + 60, disk=91.0), now=now + 60)
    assert len(sent) == 1


def test_critical_disk_level_marks_the_message(db, sent):
    now = 1_800_000_000
    alerts.check(db, sample(now, disk=97.0), now=now)
    assert sent and sent[0].startswith("🔴")


def test_swap_rule_is_skipped_without_swap(db, sent):
    """
    Ноль из нуля на машине без swap — не «всё хорошо», а отсутствие метрики.
    Правило в этом случае не заводится вовсе.
    """
    payload = sample(1_800_000_000)
    payload.update(swap_total=0, swap_used=0, swap_pct=0.0)
    assert all(rule.key != "swap" for rule in alerts.rules_for(payload))


def test_events_are_journalled_with_delivery_result(db, sent):
    now = 1_800_000_000
    alerts.check(db, sample(now, disk=91.0), now=now)
    row = store.recent_alerts(db, 5)[0]
    assert (row["key"], row["kind"], row["sent"]) == ("disk:/", "fired", 1)


def test_failed_delivery_is_recorded_not_raised(db, monkeypatch, sent):
    """
    Неудача отправки не должна ронять такт сэмплера — она обязана быть видна
    как запись журнала с ошибкой.
    """
    def boom(text, **kwargs):
        raise telegram.TelegramError("chat not found")

    monkeypatch.setattr(telegram, "send", boom)
    now = 1_800_000_000
    events = alerts.check(db, sample(now, disk=91.0), now=now)
    assert events[0]["sent"] == 0 and "chat not found" in events[0]["error"]
    assert store.recent_alerts(db, 5)[0]["sent"] == 0


def test_alerts_are_silent_when_channel_is_not_configured(db, monkeypatch):
    """
    Без токена сообщения не уходят, но событие всё равно попадает в журнал с
    объяснением — иначе «уведомление не пришло» невозможно отличить от
    «повода не было».
    """
    monkeypatch.setattr(config, "alerts", config.AlertsConfig(
        enabled=True, bot_token="", chat_id="", disk_pct=90.0,
    ))
    now = 1_800_000_000
    events = alerts.check(db, sample(now, disk=95.0), now=now)
    assert events[0]["sent"] == 0
    assert "TELEGRAM_BOT_TOKEN" in events[0]["error"]


def test_message_never_leaks_the_token():
    """
    Токен входит в URL Bot API, поэтому любое сообщение об ошибке обязано его
    вымарывать: трейсбек с полным URL — это утёкший бот.
    """
    hidden = telegram._hide_token("connect to /bot123:ABC/sendMessage failed", "123:ABC")
    assert "123:ABC" not in hidden


# ─── Сбор ───────────────────────────────────────────────────────────────────
def test_snapshot_has_every_column_the_store_writes():
    """
    Схема таблицы и снимок сэмплера обязаны совпадать по составу. Разъезд был
    бы тихим: `store.write` берёт поля через `.get`, и забытое поле молча
    писалось бы как NULL — то есть метрика исчезала бы с графика без ошибки.
    """
    psutil = pytest.importorskip("psutil")  # noqa: F841 — нужен самому collect
    from btcproc.hostmon import collect

    snapshot = collect.Sampler().snapshot()
    columns = {
        line.split()[0]
        for line in store.SCHEMA.split("CREATE TABLE IF NOT EXISTS samples (")[1]
        .split(");")[0].splitlines()
        if line.strip() and not line.strip().startswith("--")
    }
    assert columns - {"ts"} <= set(snapshot), "в снимке нет поля, которое ждёт таблица"
    assert snapshot["disks"] and snapshot["disks"][0]["total"] > 0
