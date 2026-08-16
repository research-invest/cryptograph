"""
Фон состояния: контекстные атомы рядом с узлом графа.

Контекстные атомы (16 из них — SMC) в вектор признаков не входят и в
кластеризации не участвуют, поэтому в имя состояния попасть не могут в
принципе. Но состояния по ним различаются, и это различие показывается
отдельным блоком. Здесь проверяется отбор: что показывается, что нет и
почему.

Сам агрегат считает `repo.save_state_context` (train, один раз на прогон) —
здесь он подменяется: проверяется решение админки о показе, а не SQL.
"""
from __future__ import annotations

import threading
import time

import pytest

from btcproc.admin import queries


def _row(group_id, atom, share, lift):
    return {"group_id": group_id, "atom": atom, "share": share, "lift": lift}


@pytest.fixture(autouse=True)
def clean_cache():
    queries._CONTEXT_CACHE.clear()
    yield
    queries._CONTEXT_CACHE.clear()


@pytest.fixture
def rows(monkeypatch):
    """Готовый фон в таблице: считать нечего, читаем."""
    def _install(data, ready=True):
        calls = []
        monkeypatch.setattr(queries.repo, "state_context_ready", lambda run_id: ready)
        monkeypatch.setattr(
            queries.repo, "load_state_context",
            lambda run_id: calls.append(run_id) or data,
        )
        monkeypatch.setattr(
            queries.repo, "save_state_context",
            lambda run_id, symbol=None, timeout_ms=None: pytest.fail(
                "готовый фон не должен пересчитываться"
            ),
        )
        return calls

    return _install


def test_rare_atom_does_not_qualify_on_lift_alone(rows):
    """
    Атом на десятке баров даёт огромный лифт и не значит ничего. Порог доли
    отсекает именно это — иначе фон состояния заполнился бы случайностями.
    """
    rows([_row(1.0, "sweep_high", 0.01, 8.0)])

    assert queries.state_context_atoms(7, "BTCUSDT") == {}


def test_ubiquitous_atom_does_not_qualify_either(rows):
    """
    Обратный случай: in_breaker сидит на 83% истории, и его доля в любом
    состоянии высока. Без порога лифта он был бы в каждом состоянии и не
    отличал бы их друг от друга.
    """
    rows([_row(1.0, "in_breaker", 0.83, 1.02)])

    assert queries.state_context_atoms(7, "BTCUSDT") == {}


def test_atom_that_is_both_present_and_lifted_qualifies(rows):
    rows([_row(3.0, "in_discount", 0.76, 1.41)])

    result = queries.state_context_atoms(7, "BTCUSDT")

    assert list(result) == [3.0]
    assert result[3.0][0]["atom"] == "in_discount"


def test_label_is_russian(rows):
    """Ради этого блок и делался: sweep_low_reclaim читается не лучше group_id."""
    rows([_row(3.0, "sweep_low_reclaim", 0.25, 1.65)])

    atom = queries.state_context_atoms(7, "BTCUSDT")[3.0][0]

    assert atom["label"] == "снятие снизу с возвратом"


def test_list_is_capped_and_ordered_by_lift(rows):
    """SQL отдаёт по убыванию лифта; берём верхушку, порядок сохраняем."""
    rows([
        _row(1.0, f"atom_{i}", 0.5, 3.0 - i * 0.1)
        for i in range(queries.CONTEXT_TOP_N + 4)
    ])

    picked = queries.state_context_atoms(7, "BTCUSDT")[1.0]

    assert len(picked) == queries.CONTEXT_TOP_N
    assert [a["lift"] for a in picked] == sorted(
        (a["lift"] for a in picked), reverse=True
    )


def test_result_is_cached_per_symbol_and_run(rows):
    """
    Запрос разворачивает массивы атомов по всей истории и идёт секундами.
    Разметка завершённого прогона не меняется, поэтому кэш безопасен —
    но ключ обязан включать и монету, и прогон.
    """
    calls = rows([_row(1.0, "in_premium", 0.4, 1.5)])

    queries.state_context_atoms(7, "BTCUSDT")
    queries.state_context_atoms(7, "BTCUSDT")
    assert len(calls) == 1, "повторный запрос должен идти из кэша"

    queries.state_context_atoms(7, "ETHUSDT")
    assert len(calls) == 2, "другая монета — другой ключ"

    queries.state_context_atoms(9, "BTCUSDT")
    assert len(calls) == 3, "другой прогон — другой ключ"


def test_cache_does_not_grow_without_bound(rows):
    """Админка живёт неделями; неограниченный словарь — это утечка."""
    rows([_row(1.0, "in_premium", 0.4, 1.5)])

    for run_id in range(queries._CONTEXT_CACHE_LIMIT + 5):
        queries.state_context_atoms(run_id, "BTCUSDT")

    assert len(queries._CONTEXT_CACHE) <= queries._CONTEXT_CACHE_LIMIT


def test_null_lift_does_not_crash(rows):
    """NULLIF в SQL может отдать None — деление на нулевую частоту."""
    rows([_row(1.0, "in_premium", 0.4, None)])

    assert queries.state_context_atoms(7, "BTCUSDT") == {}


def test_run_without_precomputed_context_is_computed_once(monkeypatch):
    """
    Прогоны, сделанные до появления таблицы, досчитываются по первому
    обращению — и ровно один раз: отметку ставит сам расчёт, а пустой
    результат от непосчитанного отличается только ею.
    """
    computed = []
    ready = {"value": False}

    def _save(run_id, symbol=None, timeout_ms=None):
        computed.append((run_id, symbol))
        ready["value"] = True
        return {"bars": 0, "rows": 0}

    monkeypatch.setattr(queries.repo, "state_context_ready",
                        lambda run_id: ready["value"])
    monkeypatch.setattr(queries.repo, "save_state_context", _save)
    monkeypatch.setattr(queries.repo, "load_state_context",
                        lambda run_id: [_row(1.0, "in_premium", 0.4, 1.5)])

    queries.state_context_atoms(7, "BTCUSDT")
    queries._CONTEXT_CACHE.clear()          # кэш не должен быть единственной защитой
    queries.state_context_atoms(7, "BTCUSDT")

    assert computed == [(7, "BTCUSDT")]


def test_parallel_requests_for_one_state_hit_the_db_once(monkeypatch):
    """
    Пять кликов по узлу — пять потоков uvicorn, и кэш их не ловит: он
    заполняется после первого ответа, а уходят они одновременно. Ровно это
    и укладывало базу, поэтому проверяется именно параллельный случай.
    """
    started = threading.Event()
    release = threading.Event()
    calls = []

    def _load(run_id):
        calls.append(run_id)
        started.set()
        release.wait(timeout=5)
        return [_row(1.0, "in_premium", 0.4, 1.5)]

    monkeypatch.setattr(queries.repo, "state_context_ready", lambda run_id: True)
    monkeypatch.setattr(queries.repo, "load_state_context", _load)

    results = []
    threads = [
        threading.Thread(target=lambda: results.append(
            queries.state_context_atoms(7, "BTCUSDT")))
        for _ in range(5)
    ]
    threads[0].start()
    started.wait(timeout=5)                 # первый гарантированно в БД
    for t in threads[1:]:
        t.start()
    # Ждём, пока все четверо реально прицепятся к чужому результату. Без
    # этого тест зеленел бы и на сломанном коде: опоздавший поток нашёл бы
    # реестр уже пустым и честно пошёл бы в БД вторым.
    deadline = time.time() + 5
    while queries._CONTEXT_FLIGHT.waiters(("BTCUSDT", 7)) < 4:
        assert time.time() < deadline, "ждущие не прицепились"
    release.set()
    for t in threads:
        t.join(timeout=5)

    assert calls == [7], "остальные обязаны дождаться первого, а не пойти своим путём"
    assert len(results) == 5
    assert all(r == results[0] for r in results)
