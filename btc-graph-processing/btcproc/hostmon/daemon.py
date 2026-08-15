"""
Процесс-сэмплер: пишет снимок нагрузки в SQLite по сетке `HOSTMON_INTERVAL_SECONDS`.

Отдельный долгоживущий процесс, а не крон и не поток админки — по трём
причинам, каждая из которых уже стоила бы отладки:

* **не крон.** Минутный крон запускал бы `python -m btcproc.cli` шестьдесят
  раз в час, а импорт btcproc тянет pandas и sklearn: пара секунд и сотня
  мегабайт на каждый замер. Монитор съедал бы ощутимую долю того, что мерит.
* **не поток админки.** Прогоны идут `BackgroundTasks` в её процессе; замер,
  живущий там же, встаёт в ту же очередь событий и в момент пиковой нагрузки
  пропускает такты — то есть перестаёт писать именно тогда, когда нужен.
* **не в docker.** Внутри контейнера psutil показывает контейнер, а не хост,
  а следить надо за хостом — и за самим docker в том числе.

Сетка выровнена: метки замеров кратны интервалу. Это не косметика — на
выровненной сетке бакеты прореживания содержат одинаковое число замеров, иначе
график «дышит» амплитудой на ровной нагрузке.
"""
from __future__ import annotations

import contextlib
import logging
import signal
import time

from btcproc import config
from btcproc.hostmon import alerts, collect, store

logger = logging.getLogger(__name__)

# Как часто чистить старые замеры. Раз в час: удаление идёт по индексу и стоит
# миллисекунды, но делать его каждую минуту незачем.
PRUNE_EVERY = 3600


class _Stop:
    """Флаг остановки по SIGTERM/SIGINT: такт обязан дописаться до конца."""

    def __init__(self) -> None:
        self.requested = False
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._handle)

    def _handle(self, signum, frame) -> None:  # noqa: ARG002
        logger.info("Получен сигнал %s — останавливаюсь после текущего такта", signum)
        self.requested = True


def sample_once() -> dict:
    """Один замер с записью. Для отладки и тестов; демон им не пользуется."""
    sampler = collect.Sampler()
    # Пауза нужна, чтобы дельты CPU и диска были посчитаны хоть на каком-то
    # интервале: снимок сразу после создания сэмплера показал бы нули.
    time.sleep(1)
    sample = sampler.snapshot(ts=_aligned(time.time()))
    # closing, а не сам `with`: контекст-менеджер sqlite3 закрывает транзакцию,
    # но НЕ соединение — файл остался бы открытым до сборки мусора.
    with contextlib.closing(store.connect()) as conn:
        store.write(conn, sample)
        # Пороги проверяем и здесь: `--once` — это ещё и способ убедиться, что
        # канал уведомлений настроен верно, на реальных значениях машины.
        sample["alerts"] = alerts.check(conn, sample)
    return sample


def run(interval: int | None = None) -> None:
    """Бесконечный цикл замеров. Выходит по SIGTERM/SIGINT."""
    interval = interval or config.hostmon.interval
    stop = _Stop()
    sampler = collect.Sampler()
    conn = store.connect()
    logger.info(
        "Сэмплер запущен: шаг %d с, база %s, хранение %d сут",
        interval, config.hostmon.db_path, config.hostmon.keep_days,
    )
    # Про канал уведомлений говорим на старте и явно. Ненастроенный Telegram —
    # штатный режим (монитор полезен и без него), но узнавать об этом в момент
    # заполнения диска поздно.
    if not config.alerts.enabled:
        logger.info("Алерты выключены (HOSTMON_ALERTS_ENABLED=false)")
    elif not config.alerts.configured:
        logger.warning(
            "Алерты включены, но канал не настроен: задай TELEGRAM_BOT_TOKEN и "
            "TELEGRAM_CHAT_ID — иначе о заполненном диске никто не узнает"
        )
    else:
        logger.info(
            "Алерты в Telegram: диск %g%%/%g%%, память %g%%, swap %g%%, CPU %g%%, "
            "load %g на ядро; напоминание не чаще %d мин",
            config.alerts.disk_pct, config.alerts.disk_critical_pct,
            config.alerts.mem_pct, config.alerts.swap_pct, config.alerts.cpu_pct,
            config.alerts.load_per_core, config.alerts.cooldown_minutes,
        )

    last_prune = 0.0
    while not stop.requested:
        # Спим до ближайшей границы сетки, а не «интервал от конца замера»:
        # иначе метки медленно уползают, и шаг перестаёт быть постоянным.
        target = (int(time.time()) // interval + 1) * interval
        while not stop.requested and time.time() < target:
            time.sleep(min(1.0, target - time.time()))
        if stop.requested:
            break

        try:
            sample = sampler.snapshot(ts=target)
            store.write(conn, sample)
        except Exception:
            # Монитор не имеет права умереть от неудачного замера: диск мог
            # оказаться полон, точка монтирования — отвалиться. Пишем в лог и
            # идём на следующий такт.
            logger.exception("Замер не записан")
        else:
            # Алерты — строго после записи замера: отправка ходит в сеть, и
            # висящий Telegram не должен стоить пропущенной точки на графике.
            try:
                alerts.check(conn, sample)
            except Exception:
                logger.exception("Проверка порогов не удалась")

        if time.monotonic() - last_prune > PRUNE_EVERY:
            last_prune = time.monotonic()
            try:
                deleted = store.prune(conn)
                if deleted:
                    logger.info("Удалено старых замеров: %d", deleted)
            except Exception:
                logger.exception("Чистка старых замеров не удалась")

    conn.close()
    logger.info("Сэмплер остановлен")


def _aligned(ts: float, interval: int | None = None) -> int:
    interval = interval or config.hostmon.interval
    return int(ts) // interval * interval
