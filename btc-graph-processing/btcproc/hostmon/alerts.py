"""
Пороговые уведомления монитора: диск, память, swap, процессор, очередь.

Правила и антиспам описаны в `config.AlertsConfig`; здесь — их применение к
очередному замеру. Три состояния события:

* `fired`     — порог нарушен впервые (после выдержки), сообщение уходит сразу;
* `still`     — нарушение держится, прошёл cooldown, уходит напоминание;
* `recovered` — метрика вернулась ниже `порог − гистерезис`, одно сообщение.

Состояние правил лежит в SQLite рядом с замерами, а не в памяти процесса:
перезапуск сервиса не должен превращаться в повторную рассылку про ту же
проблему. По той же причине cooldown считается от `last_notified` из базы, а
не от старта процесса.

Неудача отправки не откладывается и не повторяется. Ретраев нет по той же
причине, что и у вебхуков о кандидатах: уведомление «диск заполнен», дошедшее
через час, вреднее недошедшего — за час либо всё встало, либо место нашлось.
Неудача видна в журнале `alert_log` на странице «Сервер» и в логе сервиса.
"""
from __future__ import annotations

import logging
import os
import socket
import sqlite3
import time
from dataclasses import dataclass

from btcproc import config
from btcproc.hostmon import store, telegram

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Rule:
    """Одно правило, уже вычисленное на конкретном замере."""
    key: str                 # идентификатор состояния: disk:/ , memory, swap, cpu, load
    title: str               # как называть ресурс в сообщении
    value: float             # текущее значение
    threshold: float         # порог срабатывания
    sustain: int             # сколько замеров подряд за порогом нужно
    unit: str = "%"
    detail: str = ""         # абсолютные числа для тела сообщения
    critical: bool = False   # красный уровень, а не жёлтый
    # Показывать ли в сообщении, кто именно съел ресурс, и по какому полю
    # сортировать процессы: "rss" для памяти, "cpu" для процессора.
    culprits: str | None = None


def rules_for(sample: dict) -> list[Rule]:
    """
    Правила, применимые к замеру. Метрика без значения (`load` на системе без
    getloadavg) правила не даёт вовсе — «нет данных» не должно молча читаться
    как «всё в порядке».
    """
    cfg = config.alerts
    out: list[Rule] = []

    for disk in sample.get("disks", []):
        if disk.get("pct") is None:
            continue
        free = (disk.get("total") or 0) - (disk.get("used") or 0)
        critical = disk["pct"] >= cfg.disk_critical_pct
        out.append(Rule(
            key=f"disk:{disk['mount']}",
            title=f"диск {disk['mount']}",
            value=disk["pct"],
            # У диска два порога, но одно состояние: критический уровень не
            # заводит второе правило, а повышает уровень того же. Иначе при
            # переходе 90% → 96% оператор получил бы два независимых события
            # об одном и том же разделе.
            threshold=cfg.disk_critical_pct if critical else cfg.disk_pct,
            sustain=1,       # диск заполняется монотонно, выдержка не нужна
            detail=f"свободно {_gb(free)} из {_gb(disk.get('total'))}",
            critical=critical,
        ))

    if sample.get("mem_pct") is not None:
        out.append(Rule(
            key="memory", title="память", value=sample["mem_pct"],
            threshold=cfg.mem_pct, sustain=2,
            detail=f"занято {_gb(sample.get('mem_used'))} из {_gb(sample.get('mem_total'))}",
            culprits="rss",
        ))

    # Swap проверяем только если он вообще есть: на машине без swap ноль из
    # нуля — это не «всё хорошо», это отсутствие метрики.
    if sample.get("swap_pct") is not None and (sample.get("swap_total") or 0) > 0:
        out.append(Rule(
            key="swap", title="swap", value=sample["swap_pct"],
            threshold=cfg.swap_pct, sustain=2,
            detail=f"занято {_gb(sample.get('swap_used'))} из {_gb(sample.get('swap_total'))}",
            culprits="rss",
        ))

    if sample.get("cpu") is not None:
        out.append(Rule(
            key="cpu", title="процессор", value=sample["cpu"],
            threshold=cfg.cpu_pct, sustain=cfg.sustain,
            detail=f"выдержка {cfg.sustain} замеров",
            culprits="cpu",
        ))

    # os.cpu_count(), а не psutil: правила не должны зависеть от библиотеки
    # сбора — так модуль проверяем и там, где psutil не установлен.
    cores = os.cpu_count() or 1
    if sample.get("load1") is not None:
        out.append(Rule(
            key="load", title="очередь к CPU (load1)", value=sample["load1"],
            threshold=cfg.load_per_core * cores, sustain=cfg.sustain,
            unit="", detail=f"ядер {cores}", culprits="cpu",
        ))

    return out


def check(conn: sqlite3.Connection, sample: dict, now: int | None = None) -> list[dict]:
    """
    Проверить замер и разослать то, что положено. Возвращает журнальные записи
    (они же пишутся в `alert_log`).

    Вся функция обязана быть безопасной для такта сэмплера: она вызывается
    после того, как замер уже записан, и любая её беда не должна мешать
    следующему замеру.
    """
    if not config.alerts.enabled:
        return []

    now = int(now if now is not None else time.time())
    states = store.alert_states(conn)
    cooldown = config.alerts.cooldown_minutes * 60
    events: list[dict] = []

    for rule in rules_for(sample):
        state = states.get(rule.key) or {
            "firing": 0, "since": None, "breaches": 0, "last_notified": None, "value": None,
        }
        firing = bool(state["firing"])
        breaches = int(state["breaches"] or 0)
        since = state["since"]
        last_notified = state["last_notified"]
        kind: str | None = None

        if rule.value >= rule.threshold:
            breaches += 1
            if breaches >= rule.sustain:
                if not firing:
                    kind, firing, since = "fired", True, now
                elif last_notified is None or now - last_notified >= cooldown:
                    kind = "still"
        else:
            breaches = 0
            # Гистерезис: гасим только при уверенном возврате. Метрика,
            # топчущаяся у порога, иначе слала бы «сработало/отпустило» парой
            # на каждый замер — самый неприятный вид спама, потому что каждое
            # сообщение по отдельности выглядит осмысленным.
            if firing and rule.value < rule.threshold - config.alerts.hysteresis:
                firing = False
                since = None
                if config.alerts.notify_recovery:
                    kind = "recovered"

        if kind:
            event = _deliver(rule, kind, sample, now)
            events.append(event)
            store.log_alert(conn, event)
            if kind != "recovered":
                last_notified = now

        store.save_alert_state(conn, rule.key, {
            "firing": int(firing), "since": since, "breaches": breaches,
            "last_notified": last_notified, "value": rule.value,
        })

    return events


def _deliver(rule: Rule, kind: str, sample: dict, now: int) -> dict:
    """Собрать текст и отправить. Неудача — это поле записи, а не исключение."""
    message = compose(rule, kind, sample)
    entry = {
        "ts": now, "key": rule.key, "kind": kind, "value": rule.value,
        "message": message, "sent": 0, "error": None,
    }
    if not config.alerts.configured:
        entry["error"] = "канал не настроен (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)"
        logger.warning("Алерт %s (%s) не отправлен: %s", rule.key, kind, entry["error"])
        return entry
    try:
        telegram.send(message)
        entry["sent"] = 1
        logger.info("Алерт %s (%s) отправлен в Telegram", rule.key, kind)
    except telegram.TelegramError as exc:
        entry["error"] = str(exc)[:500]
        logger.error("Алерт %s (%s) не отправлен: %s", rule.key, kind, entry["error"])
    return entry


def compose(rule: Rule, kind: str, sample: dict) -> str:
    """
    Текст сообщения.

    Сводка по остальным ресурсам идёт в каждом сообщении намеренно: алерт про
    диск приходит в тот момент, когда полезно сразу видеть, что при этом было
    с памятью, — иначе оператор всё равно полезет в админку за тем же самым.
    """
    esc = telegram.esc
    host = esc(socket.gethostname())
    value = _fmt(rule.value, rule.unit)
    threshold = _fmt(rule.threshold, rule.unit)

    if kind == "recovered":
        head = f"🟢 <b>{host}</b> · {esc(rule.title)} в норме: {value}"
        body = [f"порог {threshold}, гистерезис {config.alerts.hysteresis:g}"]
    else:
        icon = "🔴" if rule.critical else "⚠️"
        repeat = " (напоминание)" if kind == "still" else ""
        head = f"{icon} <b>{host}</b> · {esc(rule.title)}: {value}{repeat}"
        body = [f"порог {threshold}" + (f", {esc(rule.detail)}" if rule.detail else "")]

    body.append(
        "CPU {cpu} · RAM {mem} · swap {swap} · load {load}".format(
            cpu=_fmt(sample.get("cpu"), "%"),
            mem=_fmt(sample.get("mem_pct"), "%"),
            swap=_fmt(sample.get("swap_pct"), "%"),
            load=_fmt(sample.get("load1"), ""),
        )
    )
    for disk in sample.get("disks", []):
        body.append(
            f"{esc(disk['mount'])} — {_fmt(disk.get('pct'), '%')} "
            f"({_gb(disk.get('used'))} из {_gb(disk.get('total'))})"
        )

    if rule.culprits and kind != "recovered":
        top = _top_processes(rule.culprits)
        if top:
            body.append("Больше всех: " + top)

    return head + "\n" + "\n".join(body)


def _top_processes(by: str, limit: int = 3) -> str:
    """
    Кто съел ресурс — три процесса в одну строку.

    Считается только при отправке сообщения: полсекунды на два прохода psutil
    оправданы для алерта, но не для каждого замера.
    """
    from btcproc.hostmon import collect

    try:
        snapshot = collect.processes(limit=limit)
    except Exception:                                  # noqa: BLE001
        logger.warning("Не удалось собрать топ процессов для алерта", exc_info=True)
        return ""

    rows = snapshot["by_memory"] if by == "rss" else snapshot["by_cpu"]
    parts = []
    for row in rows[:limit]:
        weight = _gb(row["rss"]) if by == "rss" else f"{row['cpu']:g}%"
        parts.append(f"{telegram.esc(row['name'])} {weight}")
    return " · ".join(parts)


def _fmt(value, unit: str) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}{unit}" if unit else f"{value:.2f}"


def _gb(value) -> str:
    if not value:
        return "0 ГБ"
    return f"{value / 1024 ** 3:.1f} ГБ"
