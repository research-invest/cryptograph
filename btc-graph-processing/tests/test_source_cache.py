"""
Кэш источников сигнала: пара «ключ+значение» неделима.

Регрессия на B2 (аудит 2026-08-15). Админка запускает до
`ADMIN_MAX_CONCURRENT_RUNS` прогонов в РАЗНЫХ ПОТОКАХ одного процесса, а
запись кэша была парой присваиваний — перемежение потоков оставляло в кэше
ключ одной монеты со значением другой. Не падало: бары всех монет лежат на
одной 15-минутной сетке, и джойн по совпавшим timestamp молча выравнивал
чужие величины.
"""
from __future__ import annotations

import threading

from btcproc.features._cache import SingleEntryCache


def test_hit_and_miss():
    cache = SingleEntryCache()
    assert cache.get("a") is None
    cache.put("a", 1)
    assert cache.get("a") == 1
    # Кэш на ОДНУ запись: вторая вытесняет первую.
    cache.put("b", 2)
    assert cache.get("b") == 2
    assert cache.get("a") is None


def test_concurrent_writers_never_mix_key_and_value():
    """
    Два потока молотят кэш своими парами. Читатель обязан видеть либо пару
    целиком, либо пару целиком — рассогласованной («ключ B, значение A»)
    не существует.
    """
    cache = SingleEntryCache()
    stop = threading.Event()
    mismatches: list[tuple] = []

    def writer(name: str) -> None:
        while not stop.is_set():
            cache.put(name, name * 3)

    def reader() -> None:
        while not stop.is_set():
            for name in ("alpha", "beta"):
                value = cache.get(name)
                if value is not None and value != name * 3:
                    mismatches.append((name, value))

    threads = [threading.Thread(target=writer, args=("alpha",)),
               threading.Thread(target=writer, args=("beta",)),
               threading.Thread(target=reader)]
    for t in threads:
        t.start()
    stop.wait(0.5)
    stop.set()
    for t in threads:
        t.join()

    assert not mismatches, f"ключ и значение разъехались: {mismatches[:5]}"


def test_sources_use_the_atomic_cache():
    """
    Все три источника обязаны пользоваться именно им: правка, вернувшая
    голый dict хотя бы в одном модуле, воспроизводит гонку целиком.
    """
    from btcproc.features import deriv, fear_greed, smc

    for module in (smc, fear_greed, deriv):
        assert isinstance(module._CACHE, SingleEntryCache), module.__name__
