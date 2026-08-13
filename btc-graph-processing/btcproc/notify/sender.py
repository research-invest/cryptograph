"""
Доставка уведомлений: очередь и фоновые потоки-отправители.

Почему не «просто httpx.post в месте события». Кандидаты появляются внутри
прогона, а прогон — это конвейер, у которого есть время выполнения и крон
раз в полчаса. Синхронная отправка означала бы, что недоступный получатель
удлиняет прогон на `таймаут × число кандидатов`, а совсем зависший адрес
блокирует его до таймаута каждого запроса. Поэтому прогон кладёт задание в
очередь и идёт дальше — это и есть «послать без ожидания ответа».

Ответ при этом ЧИТАЕТСЯ, просто не тем, кто отправлял: фоновый поток
дожидается HTTP-статуса и кладёт его в журнал доставок. Не ждёт прогон —
а не «никто не смотрит». Разница принципиальная: без записи в журнал вопрос
«почему получатель ничего не получил» не имеет ответа в принципе, потому что
отправка не оставляет следов нигде.

Ловушка, ради которой существует `flush`. Крон-процесс (`make live`) живёт
ровно столько, сколько идёт прогон. Потоки-отправители демонические, и на
выходе процесса они умирают вместе с ним — вместе с неразобранной очередью.
Поэтому прогон в конце ждёт разбора очереди с дедлайном. Внутри прогона это
ожидание нулевое: к финалу очередь обычно уже пуста.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from btcproc import config

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """Одно уведомление: правило, куда слать, и готовое тело запроса."""

    rule_id: int
    name: str
    url: str
    headers: dict[str, str]
    candidate_id: str
    body: dict


@dataclass
class Result:
    status: str            # sent | failed | dropped
    http_status: int | None = None
    error: str | None = None


class Dispatcher:
    """
    Очередь + пул потоков. Отправка и запись результата инжектируются —
    тесты подставляют свои и не ходят ни в сеть, ни в БД.
    """

    def __init__(
        self,
        send: Callable[[Job], Result] | None = None,
        record: Callable[[Job, Result], None] | None = None,
        workers: int | None = None,
        queue_size: int | None = None,
    ) -> None:
        self._send = send or http_send
        self._record = record or record_delivery
        self._workers = max(1, workers if workers is not None else config.notify.workers)
        self._queue: queue.Queue[Job] = queue.Queue(
            maxsize=queue_size if queue_size is not None else config.notify.queue_size
        )
        # Свой счётчик «в работе» вместо Queue.join(): join не умеет
        # дедлайна, а ждать вечно на выходе из крон-прогона нельзя.
        self._inflight = 0
        self._idle = threading.Condition()
        self._started = False
        self._lock = threading.Lock()
        self.sent = 0
        self.failed = 0
        self.dropped = 0

    # ── Приём заданий ───────────────────────────────────────────────────────
    def submit(self, job: Job) -> bool:
        """
        Ставит задание в очередь. Не блокирует и не ждёт ответа.

        Возвращает False, если очередь переполнена. Переполнение — это
        честная потеря с записью в журнал, а не молчаливое ожидание: копить
        задания в памяти прогона неограниченно хуже, чем потерять уведомление
        и увидеть `dropped` в админке.
        """
        self._ensure_started()
        with self._idle:
            self._inflight += 1
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            with self._idle:
                self._inflight -= 1
                self.dropped += 1
                self._idle.notify_all()
            self._safe_record(job, Result("dropped", error="очередь уведомлений переполнена"))
            logger.warning(
                "Уведомление по правилу «%s» отброшено: очередь переполнена", job.name
            )
            return False
        return True

    def flush(self, timeout: float | None = None) -> int:
        """
        Ждёт разбора очереди до дедлайна. Возвращает число неразобранных
        заданий (0 — всё отправлено).
        """
        deadline = time.monotonic() + (
            config.notify.flush_seconds if timeout is None else timeout
        )
        with self._idle:
            while self._inflight > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "Очередь уведомлений не разобрана за отведённое время: "
                        "осталось %d", self._inflight,
                    )
                    return self._inflight
                self._idle.wait(remaining)
        return 0

    # ── Потоки ──────────────────────────────────────────────────────────────
    def _ensure_started(self) -> None:
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            for i in range(self._workers):
                thread = threading.Thread(
                    target=self._loop, name=f"notify-{i}", daemon=True
                )
                thread.start()
            self._started = True

    def _loop(self) -> None:
        while True:
            job = self._queue.get()
            try:
                result = self._send(job)
            except Exception as exc:  # noqa: BLE001 — поток обязан пережить любой отказ
                result = Result("failed", error=f"{type(exc).__name__}: {exc}")
            if result.status != "sent":
                logger.warning(
                    "Уведомление по правилу «%s» не доставлено: %s",
                    job.name, result.error or result.http_status,
                )
            self._safe_record(job, result)
            # Счётчики под тем же замком, что и _inflight: потоков несколько,
            # а `self.sent += 1` из двух потоков теряет инкременты.
            with self._idle:
                if result.status == "sent":
                    self.sent += 1
                else:
                    self.failed += 1
                self._inflight -= 1
                self._idle.notify_all()

    def _safe_record(self, job: Job, result: Result) -> None:
        try:
            self._record(job, result)
        except Exception:  # noqa: BLE001 — журнал не должен ронять отправку
            logger.exception("Не записан результат доставки по правилу %s", job.rule_id)


# ── Транспорт и журнал по умолчанию ─────────────────────────────────────────
def http_send(job: Job) -> Result:
    """
    POST с телом уведомления.

    `Idempotency-Key` = candidate_id: у нас есть защита от повторов (журнал
    доставок), но она не переживает удаления строк по retention, а получатель
    может получить дубль и по другой причине — например, оператор нажал
    «отправить повторно». Ключ даёт ему возможность отбросить дубль самому.
    """
    import httpx

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "btcproc-notify/1",
        "Idempotency-Key": job.candidate_id,
        **(job.headers or {}),
    }
    try:
        with httpx.Client(timeout=config.notify.timeout) as client:
            response = client.post(job.url, json=job.body, headers=headers)
    except Exception as exc:  # noqa: BLE001 — диагноз нужен в журнале, а не в трейсбеке
        return Result("failed", error=f"{type(exc).__name__}: {exc}")

    if response.status_code >= 400:
        # Тело ответа обрезаем: у чужой системы там может оказаться HTML
        # страницы ошибки на десятки килобайт.
        return Result(
            "failed",
            http_status=response.status_code,
            error=f"HTTP {response.status_code}: {response.text[:500]}",
        )
    return Result("sent", http_status=response.status_code)


def record_delivery(job: Job, result: Result) -> None:
    """Отметка в журнале доставок. Строка уже создана при захвате (см. dispatch)."""
    from btcproc.db.session import execute

    execute(
        "UPDATE notification_deliveries SET status = %s, http_status = %s, "
        "error = %s, attempts = attempts + 1, "
        "sent_at = CASE WHEN %s = 'sent' THEN NOW() ELSE sent_at END "
        "WHERE rule_id = %s AND candidate_id = %s",
        (result.status, result.http_status, result.error[:2000] if result.error else None,
         result.status, job.rule_id, job.candidate_id),
    )


# ── Синглтон ────────────────────────────────────────────────────────────────
_dispatcher: Dispatcher | None = None
_dispatcher_lock = threading.Lock()


def dispatcher() -> Dispatcher:
    global _dispatcher
    if _dispatcher is None:
        with _dispatcher_lock:
            if _dispatcher is None:
                _dispatcher = Dispatcher()
    return _dispatcher


def set_dispatcher(instance: Dispatcher | None) -> None:
    """Подмена для тестов и для разовой отправки с другим транспортом."""
    global _dispatcher
    with _dispatcher_lock:
        _dispatcher = instance


def stats() -> dict[str, Any]:
    d = dispatcher()
    return {"sent": d.sent, "failed": d.failed, "dropped": d.dropped}
