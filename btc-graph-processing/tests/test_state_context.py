"""
Фон состояния: контекстные атомы рядом с узлом графа.

Контекстные атомы (16 из них — SMC) в вектор признаков не входят и в
кластеризации не участвуют, поэтому в имя состояния попасть не могут в
принципе. Но состояния по ним различаются, и это различие показывается
отдельным блоком. Здесь проверяется отбор: что показывается, что нет и
почему.
"""
from __future__ import annotations

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
    def _install(data):
        calls = []
        monkeypatch.setattr(
            queries, "fetch_all",
            lambda sql, params=None: calls.append(params) or data,
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
