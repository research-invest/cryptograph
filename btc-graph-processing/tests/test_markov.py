"""
Граф как марковская модель: контроли, без которых числа замера нечитаемы.

Здесь два положительных контроля и два отрицательных, и каждый закрывает
конкретный способ получить осмысленно выглядящее, но бессмысленное число.

* **implied timescales** обязаны давать плато на честной марковской цепи — и
  обязаны молчать там, где процесс уже затух (иначе логарифм у нуля
  растягивает шум счётчиков в разы, и «плато не вышло» получается на любых
  данных, включая идеальные);
* **CMI** смещена вверх на конечной выборке, поэтому сравнивать её с нулём
  нельзя ни при каких обстоятельствах. Проверяется, что цепь второго порядка
  отличается от нулёвки, а первого — нет;
* **отбор точек смены BOCPD** обязан оставлять их разреженными. Плотная
  разметка убивает мощность теста совпадений полностью — случайный уровень
  подходит к единице, и «не подтвердилось» означает «не измерено».
"""
from __future__ import annotations

import numpy as np
import pytest

from btcproc.analysis import markov as mk


def _markov_chain(n: int = 60000, n_states: int = 5, stickiness: float = 8.0,
                  seed: int = 0) -> np.ndarray:
    """Честная цепь первого порядка с липкой диагональю."""
    rng = np.random.default_rng(seed)
    matrix = rng.random((n_states, n_states)) + np.eye(n_states) * stickiness
    matrix /= matrix.sum(axis=1, keepdims=True)
    cumulative = matrix.cumsum(axis=1)
    states = np.empty(n, dtype=np.int64)
    states[0] = 0
    draws = rng.random(n)
    for i in range(1, n):
        states[i] = np.searchsorted(cumulative[states[i - 1]], draws[i])
    return states


def _second_order_chain(n: int = 60000, n_states: int = 5,
                        seed: int = 1) -> np.ndarray:
    """Цепь, у которой следующее состояние зависит от ДВУХ предыдущих."""
    rng = np.random.default_rng(seed)
    states = np.empty(n, dtype=np.int64)
    states[:2] = 0
    for i in range(2, n):
        weights = np.ones(n_states)
        weights[(states[i - 1] + states[i - 2]) % n_states] += 12.0
        states[i] = rng.choice(n_states, p=weights / weights.sum())
    return states


def test_implied_timescales_plateau_on_a_true_markov_chain():
    """
    Положительный контроль. Без него «плато не вышло» на боевых данных
    нельзя отличить от «плато не выходит у этой реализации никогда».
    """
    states = _markov_chain()
    curve = mk.timescale_curve(states, [1, 2, 4, 8], 5, count=3)
    assert mk.plateau_deviation(curve, "t1", 1, 8) < 0.05
    # И само значение обязано быть осмысленным, а не любым стабильным числом.
    assert 3.0 < float(curve["t1"].iloc[0]) < 8.0


def test_decayed_process_gives_no_timescale_instead_of_a_wrong_one():
    """
    Регрессия на реальную ошибку разработки (2026-08-24).

    На лаге, где процесс затух на 99.8%, оценка `λ` — это оценка шума
    счётчиков, а логарифм у нуля растягивает её в разы: кривая честной
    марковской цепи подскакивала с 4.9 до 6.3 и «плато» переставало
    выполняться. Правильный ответ — NaN, а не число.
    """
    states = _markov_chain()
    times = mk.implied_timescales(
        mk.transition_matrix(mk.count_matrix(states, 64, 5)), 64, count=3)
    assert np.isnan(times).all()


def _hidden_regime_chain(n: int = 120000, n_states: int = 6,
                         mean_regime: int = 800, seed: int = 2) -> np.ndarray:
    """
    Наблюдаемая последовательность НЕ марковская, хотя выглядит как цепь.

    Устройство: медленный скрытый режим переключается раз в ~800 шагов, а
    наблюдаются только эмиссии — состояния, распределение которых зависит от
    режима. Это классический не-марковский случай и, что важнее, ровно тот,
    который даёт подпись боевых данных: implied timescales растут с лагом,
    потому что за дискретизацией спрятан процесс медленнее её самой.
    """
    rng = np.random.default_rng(seed)
    hidden = np.empty(n, dtype=int)
    hidden[0] = 0
    switch = rng.random(n) < 1.0 / mean_regime
    for i in range(1, n):
        hidden[i] = 1 - hidden[i - 1] if switch[i] else hidden[i - 1]
    weights = np.linspace(1.0, 0.05, n_states)
    emission = np.vstack([weights / weights.sum(),
                          weights[::-1] / weights.sum()])
    cumulative = emission.cumsum(axis=1)
    draws = rng.random(n)
    return np.array([np.searchsorted(cumulative[h], d)
                     for h, d in zip(hidden, draws)], dtype=np.int64)


def test_chapman_kolmogorov_separates_markov_from_hidden_regime():
    """
    CK-ошибка сама по себе порогом не является: она растёт с `k` просто
    потому, что предсказанное `λ^k` уменьшается, а шум счётчиков — нет.
    Поэтому тест сравнительный: у честной цепи ошибка единицы процентов, у
    наблюдаемой поверх скрытого медленного режима — в разы больше.
    """
    honest = mk.chapman_kolmogorov(_markov_chain(), 2, [2, 3, 4], 5, count=3)
    hidden = mk.chapman_kolmogorov(_hidden_regime_chain(), 2, [2, 3, 4], 6, count=3)
    assert not honest.empty and not hidden.empty
    assert float(honest["error"].max()) < 0.15
    assert float(hidden["error"].max()) > 10 * float(honest["error"].max())


def test_hidden_regime_makes_timescales_grow_with_lag():
    """
    Подпись боевых данных, воспроизведённая на управляемом примере.

    У всех шести монет время медленнейшего процесса растёт с лагом в 3–10 раз
    (замер 2026-08-24). Здесь показано, что это и есть подпись «за
    дискретизацией спрятан процесс медленнее её самой», а не артефакт оценки:
    на честной цепи той же длины кривая плоская.
    """
    curve = mk.timescale_curve(_hidden_regime_chain(), [1, 2, 4, 8, 16, 32], 6,
                               count=1)
    defined = curve.dropna(subset=["t1"])
    growth = float(defined["t1"].iloc[-1] / defined["t1"].iloc[0])
    assert growth > 5.0

    honest = mk.timescale_curve(_markov_chain(), [1, 2, 4, 8], 5, count=1)
    honest_defined = honest.dropna(subset=["t1"])
    assert float(honest_defined["t1"].iloc[-1] / honest_defined["t1"].iloc[0]) < 1.2


def test_cmi_separates_second_order_from_first():
    """
    Оценка CMI смещена вверх, поэтому вопрос не «больше ли она нуля», а
    «больше ли она СВОЕЙ нулёвки». Первый порядок обязан не отличаться,
    второй — отличаться в разы.
    """
    rng = np.random.default_rng(4)
    first = _markov_chain()
    second = _second_order_chain()

    value_first = mk.conditional_mutual_information(first[:-2], first[1:-1],
                                                    first[2:], 5)
    null_first = mk.cmi_null(first, 5, 96, 20, rng)
    assert value_first <= float(np.percentile(null_first, 95)) * 1.5

    value_second = mk.conditional_mutual_information(second[:-2], second[1:-1],
                                                     second[2:], 5)
    assert value_second > float(np.percentile(null_first, 95)) * 3


def test_jump_chain_removes_self_transitions():
    states = np.array([1, 1, 1, 2, 2, 3, 1, 1])
    assert list(mk.jump_chain(states)) == [1, 2, 3, 1]


def test_changepoints_stay_sparse():
    """
    Регрессия на потерю мощности (2026-08-24).

    Порог по квантилю выглядит естественно и не работает: ряд `P(runlength=0)`
    сильно автокоррелирован, всплеск растягивается на десятки баров, и
    «верхние 2%» давали 21% помеченных баров. При такой плотности случайное
    совпадение в пределах ±2 баров имеет вероятность 0.7, и тест не может
    отличить ничего.
    """
    rng = np.random.default_rng(0)
    n = 20000
    probability = np.abs(np.convolve(rng.random(n), np.ones(50) / 50, "same"))
    points = mk.pick_changepoints(probability, share=0.02, min_distance=24)
    assert len(points) <= int(n * 0.02) + 1
    assert int(np.diff(points).min()) > 24 // 2, "точки обязаны быть разнесены"


def test_overlap_rate_and_shift_null_agree_on_unrelated_series():
    """Два независимых ряда точек не обязаны совпадать чаще случайного."""
    rng = np.random.default_rng(3)
    length = 20000
    a = np.sort(rng.choice(length, 400, replace=False))
    b = np.sort(rng.choice(length, 400, replace=False))
    observed = mk.overlap_rate(a, b, 2)
    null = mk.shift_null(a, b, 2, length, 100, rng)
    assert abs(observed - float(np.median(null))) < 0.05


def test_empty_row_is_not_filled_with_a_uniform_distribution():
    """
    Состояние, ни разу не встретившееся источником, обязано остаться нулевой
    строкой: равномерная строка — выдуманное наблюдение, и в спектре она
    отзовётся собственным значением, которого в данных нет.
    """
    counts = np.array([[3.0, 1.0], [0.0, 0.0]])
    matrix = mk.transition_matrix(counts)
    assert matrix[1].sum() == 0.0


def test_bad_lag_fails_loudly():
    with pytest.raises(ValueError, match="недопустимый лаг"):
        mk.count_matrix(np.array([0, 1, 0]), 5, 2)
