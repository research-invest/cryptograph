"""
Схлопывание одинаковых параллельных вычислений.

Зачем. Админка синхронная: uvicorn отдаёт обработчики в пул из сорока
потоков, и пять кликов по одному узлу графа — это пять потоков, каждый со
своим соединением, каждый со своим тяжёлым запросом. Кэш от этого не спасает
в принципе: он заполняется ПОСЛЕ первого ответа, а все пятеро уходят в БД
одновременно, до того. Именно так одно открытие узла укладывало базу
(журнал 43).

Правило простое: первый вызов с данным ключом считает, остальные ждут его
результат и получают тот же объект. Ошибку получают тоже все — иначе
четверо повторили бы поход, который только что не удался.

Ключ включает всё, что меняет ответ (монета, прогон): схлопывать разные
вопросы в один нельзя, это не кэш, а именно дедупликация одновременных.
"""
from __future__ import annotations

import threading
from typing import Any, Callable


class _Call:
    __slots__ = ("done", "value", "error", "waiters")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.value: Any = None
        self.error: BaseException | None = None
        #: Сколько потоков ждут этот результат. Считается под общим локом —
        #: это единственный способ снаружи достоверно узнать, что ждущий уже
        #: прицепился, а не только собирается.
        self.waiters = 0


class SingleFlight:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: dict[Any, _Call] = {}

    def run(self, key: Any, factory: Callable[[], Any]) -> Any:
        with self._lock:
            call = self._calls.get(key)
            leader = call is None
            if leader:
                call = _Call()
                self._calls[key] = call
            else:
                call.waiters += 1

        if not leader:
            # Без таймаута намеренно: сверху ограничивает statement_timeout
            # лидера, а «сдался и пошёл считать сам» — ровно то поведение,
            # которое здесь и убирается.
            call.done.wait()
            if call.error is not None:
                raise call.error
            return call.value

        try:
            call.value = factory()
        except BaseException as exc:  # noqa: BLE001 — ждущие обязаны узнать
            call.error = exc
            raise
        finally:
            # Из реестра убираем ДО того, как разбудить ждущих: следующий
            # вызов после провала должен начать заново, а не подхватить
            # чужую ошибку.
            with self._lock:
                self._calls.pop(key, None)
            call.done.set()
        return call.value

    def in_flight(self) -> int:
        """Сколько вычислений идёт прямо сейчас. Нужно тестам и диагностике."""
        with self._lock:
            return len(self._calls)

    def waiters(self, key: Any) -> int:
        """Сколько потоков ждут чужой результат по этому ключу."""
        with self._lock:
            call = self._calls.get(key)
            return call.waiters if call is not None else 0
