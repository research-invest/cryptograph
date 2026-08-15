"""
Тесты btcproc/analysis/range_lift.py (гейт R).

Основное — корректность forward_range_ratio: ручной пример с известным
размахом и известным ATR, чтобы не полагаться только на то, что код не падает.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from btcproc.analysis.range_lift import (
    autocorr_block_rows,
    forward_range_ratio,
    paired_diff_p,
    partial_r2_gain,
    spearman_block_p,
)


def test_forward_range_ratio_manual_example():
    """
    5 баров, close=100 на баре 0, ATR14(0)=2. Горизонт H=2: бары 1..2 имеют
    high=[101, 104], low=[99, 96] → диапазон = 104-96=8, range_pct = 8%.
    atr_pct = 2/100*100*sqrt(2) = 2*sqrt(2). range_ratio = 8 / (2*sqrt(2)).
    """
    index = pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")
    base = pd.DataFrame({
        "high": [100, 101, 104, 100, 100],
        "low":  [100, 99, 96, 100, 100],
        "close": [100, 100, 100, 100, 100],
    }, index=index)
    atr14 = pd.Series(2.0, index=index)

    ratio = forward_range_ratio(base, atr14, horizon_bars=2)
    expected = 8.0 / (2.0 * math.sqrt(2))
    assert ratio.iloc[0] == pytest.approx(expected)


def test_forward_range_ratio_nan_at_tail():
    """Последние horizon_bars баров не имеют полного окна вперёд — NaN."""
    index = pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")
    base = pd.DataFrame({"high": [1, 2, 3, 4, 5], "low": [1, 2, 3, 4, 5],
                        "close": [1, 2, 3, 4, 5]}, index=index)
    atr14 = pd.Series(1.0, index=index)

    ratio = forward_range_ratio(base, atr14, horizon_bars=2)
    assert ratio.iloc[-2:].isna().all()


def test_spearman_block_p_finds_a_real_relationship():
    rng = np.random.default_rng(0)
    n = 500
    predictor = rng.normal(size=n)
    target = predictor * 2.0 + rng.normal(scale=0.1, size=n)  # сильная связь

    rho, p = spearman_block_p(predictor, target, block_length=5, n_boot=200,
                              rng=np.random.default_rng(1))
    assert rho > 0.9
    assert p < 0.05


def test_spearman_block_p_rejects_noise():
    rng = np.random.default_rng(2)
    n = 500
    predictor = rng.normal(size=n)
    target = rng.normal(size=n)  # независимы

    rho, p = spearman_block_p(predictor, target, block_length=5, n_boot=200,
                              rng=np.random.default_rng(3))
    assert p > 0.05


def test_paired_diff_p_prefers_the_stronger_predictor():
    rng = np.random.default_rng(4)
    n = 500
    target = rng.normal(size=n)
    strong = target * 3.0 + rng.normal(scale=0.2, size=n)
    weak = rng.normal(size=n)  # шум, никак не связан с target

    rho_a, rho_b, p = paired_diff_p(strong, weak, target, block_length=5,
                                    n_boot=200, rng=np.random.default_rng(5))
    assert abs(rho_a) > abs(rho_b)
    assert p < 0.05


def test_autocorr_block_rows_detects_daily_ar1_decay():
    """
    Синтетический AR(1) с известным коэффициентом — сверяем порядок величины.

    Длинный ряд (n=5000) и усреднение эффекта шума конечной выборки: на
    коротких сериях (n~200) эмпирическая автокорреляция гуляет вокруг
    теоретической кривой phi^k достаточно сильно, чтобы сравнение с узким
    диапазоном стало нестабильным тестом, а не проверкой логики.
    """
    rng = np.random.default_rng(6)
    n = 5000
    phi = 0.8
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = phi * values[i - 1] + rng.normal(scale=1.0)
    index = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    series = pd.Series(values, index=index)

    lag_days, block_rows = autocorr_block_rows(series, bars_per_day=96, floor=0.2)
    # phi^k < 0.2 при k > log(0.2)/log(0.8) ≈ 7.2 — оставляем запас на шум.
    assert 5 <= lag_days <= 15
    assert block_rows == lag_days * 96


def test_autocorr_block_rows_falls_back_to_ceiling_on_persistent_series():
    index = pd.date_range("2020-01-01", periods=50, freq="1D", tz="UTC")
    series = pd.Series(np.arange(50, dtype=float), index=index)  # линейный тренд, не затухает

    lag_days, block_rows = autocorr_block_rows(series, bars_per_day=96, floor=0.2, max_lag_days=30)
    assert lag_days == 30
    assert block_rows == 30 * 96


# ── partial_r2_gain: бенчмарк-решение §2.1 задачи по деривативам ─────────
#
# "Добавляет к rv", а не "сильнее rv" — источник может дойти до значимого
# СОБСТВЕННОГО rho и не должен штрафоваться за то, что rv конструктивно
# ближе к цели (тот же ATR в знаменателе).

def _correlated_series(n: int, seed: int):
    """
    benchmark и target коррелированы (как rv и range_ratio по построению);
    у target есть ДОПОЛНИТЕЛЬНАЯ компонента shared_signal, которой в
    benchmark нет вовсе — ровно то, что должен ловить partial_r2_gain.
    """
    rng = np.random.default_rng(seed)
    benchmark = rng.normal(size=n)
    shared_signal = rng.normal(size=n)
    target = benchmark + 0.8 * shared_signal + rng.normal(scale=0.3, size=n)
    return benchmark, shared_signal, target, rng


def test_partial_r2_gain_finds_informative_predictor():
    """
    Позитивный контроль (раздел 6 ТЗ): predictor несёт ту же информацию,
    что shared_signal (которой у benchmark нет) — приращение R² обязано
    быть большим и значимым.
    """
    benchmark, shared_signal, target, rng = _correlated_series(3000, seed=10)
    predictor = shared_signal + rng.normal(scale=0.2, size=len(target))

    r2_base, r2_full, r_partial, p = partial_r2_gain(
        predictor, benchmark, target, block_length=5, n_boot=500,
        rng=np.random.default_rng(11),
    )
    assert r2_full > r2_base + 0.1
    assert r_partial > 0.5
    assert p < 0.01


def test_partial_r2_gain_rejects_uninformative_predictor():
    """Шум, не связанный ни с target, ни с benchmark, — приращения быть не должно."""
    benchmark, _shared, target, rng = _correlated_series(3000, seed=20)
    predictor = rng.normal(size=len(target))  # чистый шум

    r2_base, r2_full, r_partial, p = partial_r2_gain(
        predictor, benchmark, target, block_length=5, n_boot=500,
        rng=np.random.default_rng(21),
    )
    assert r2_full == pytest.approx(r2_base, abs=0.02)
    assert abs(r_partial) < 0.1
    assert p > 0.05


def test_partial_r2_gain_does_not_penalize_predictor_stronger_than_benchmark():
    """
    §2.1: источник со значимым СОБСТВЕННЫМ rho, сильнее самого benchmark,
    не должен получать отказ только из-за конструктивного преимущества rv —
    здесь predictor вообще не зависит от benchmark, и это не мешает найти
    приращение (в отличие от paired_diff_p, где источник мог бы проиграть
    сопернику, сильному по конструкции цели).
    """
    rng = np.random.default_rng(30)
    n = 3000
    benchmark = rng.normal(size=n)  # слабая связь с target
    predictor = rng.normal(size=n)  # СИЛЬНАЯ, независимая от benchmark
    target = 0.2 * benchmark + 1.0 * predictor + rng.normal(scale=0.3, size=n)

    r2_base, r2_full, r_partial, p = partial_r2_gain(
        predictor, benchmark, target, block_length=5, n_boot=500,
        rng=np.random.default_rng(31),
    )
    assert r2_full > r2_base + 0.3
    assert p < 0.01
