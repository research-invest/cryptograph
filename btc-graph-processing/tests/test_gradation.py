"""
Квантильный лифт и тест на монотонность (поток A задачи по зонной SMC).

Проверяется ровно то, ради чего инструмент заводился: что монотонный тренд
отличается от «разошлись края», что зависимость наблюдений не превращается
в значимость, и что вырожденная градация даёт диагноз, а не молчаливый ноль.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcproc.analysis import gradation


def _frame(n: int = 2000, effect: float = 0.0, seed: int = 3,
           shape: str = "monotone") -> pd.DataFrame:
    """
    Кандидаты с непрерывным признаком `x` и бинарной метрикой.

    shape="monotone" — доля растёт с x линейно;
    shape="edges"    — доля выше только в крайних бинах (немонотонно).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    if shape == "monotone":
        p = 0.5 + effect * (x - 0.5)
    else:
        p = 0.5 + effect * (np.abs(x - 0.5) - 0.25)
    metric = (rng.uniform(0, 1, n) < np.clip(p, 0, 1)).astype(float)
    return pd.DataFrame({
        "ts": pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC"),
        "x": x,
        "metric": metric,
    })


def test_quantile_bins_are_equal_sized():
    labels, ranges, note = gradation.quantile_bins(np.linspace(0, 1, 1000), bins=5)
    assert len(ranges) == 5
    assert note is None
    counts = np.bincount(labels.astype(int))
    assert counts.min() >= 195 and counts.max() <= 205


def test_quantile_bins_report_degenerate_input():
    """
    «Зоны нет» = 0 у большинства строк — квантильные границы схлопываются.

    Это ровно случай признаков вида near_ob на монете с редкими зонами.
    Молча вернуть один бин нельзя: замер выглядел бы как «эффекта нет».
    """
    values = np.concatenate([np.zeros(900), np.linspace(0.1, 1.0, 100)])
    labels, ranges, note = gradation.quantile_bins(values, bins=5)
    assert len(ranges) < 5
    assert note is not None and "занимает" in note


def test_cochran_armitage_finds_a_monotone_trend():
    labels = np.repeat([0, 1, 2, 3, 4], 400)
    metric = np.concatenate([
        np.repeat([1.0, 0.0], [int(400 * p), 400 - int(400 * p)])
        for p in (0.40, 0.45, 0.50, 0.55, 0.60)
    ])
    z, p = gradation.cochran_armitage(labels, metric)
    assert z > 3.0 and p < 0.001


def test_cochran_armitage_ignores_a_symmetric_profile():
    """
    Края разошлись вверх одинаково — линейного тренда нет.

    Это главное отличие от двухвыборочного z-теста: тот на «верх против
    низа» дал бы значимость, а тренда здесь нет никакого.
    """
    labels = np.repeat([0, 1, 2, 3, 4], 400)
    metric = np.concatenate([
        np.repeat([1.0, 0.0], [int(400 * p), 400 - int(400 * p)])
        for p in (0.65, 0.50, 0.45, 0.50, 0.65)
    ])
    z, _ = gradation.cochran_armitage(labels, metric)
    assert abs(z) < 1.5


def test_cochran_armitage_is_degenerate_safe():
    assert gradation.cochran_armitage(np.array([]), np.array([])) == (0.0, 1.0)
    # Один бин — тренду не по чему идти.
    assert gradation.cochran_armitage(np.zeros(100), np.ones(100)) == (0.0, 1.0)
    # Метрика константна — дисперсии нет.
    assert gradation.cochran_armitage(np.repeat([0, 1], 50), np.ones(100)) == (0.0, 1.0)


def test_measure_gradation_finds_the_trend_and_confirms_it():
    results = gradation.measure_gradation(
        _frame(n=6000, effect=0.5), features=["x"], horizon_minutes=1440,
        n_boot=200, correction="none",
    )
    result = results[0]
    assert result.monotone
    assert result.spread > 0.2
    assert result.p_boot is not None and result.p_boot < 0.05
    assert result.confirmed


def test_non_monotone_profile_is_not_confirmed():
    """Значимость без монотонности — это разница двух групп, а не тренд."""
    results = gradation.measure_gradation(
        _frame(n=6000, effect=0.8, shape="edges"), features=["x"],
        horizon_minutes=1440, n_boot=200, correction="none",
    )
    assert not results[0].confirmed


def test_dependent_observations_do_not_become_a_trend():
    """
    Признак и метрика автокоррелированы, связи между ними нет.

    Наивный Cochran–Armitage «находит» тренд по той же причине, по которой
    его находит двухвыборочный z-тест, — блочный бутстрап обязан это снять.
    """
    rng = np.random.default_rng(5)
    n = 3000

    def ar(rho):
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = rho * x[i - 1] + rng.normal()
        return x

    frame = pd.DataFrame({
        "ts": pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC"),
        "x": ar(0.99),
        "metric": (ar(0.99) > 0).astype(float),
    })

    naive = gradation.measure_gradation(frame, ["x"], holdout=None,
                                        correction="none", horizon_minutes=None)
    assert naive[0].p_value < 0.05, "синтетика перестала быть зависимой"

    boot = gradation.measure_gradation(frame, ["x"], holdout=None,
                                       correction="none", horizon_minutes=1440,
                                       n_boot=300)
    assert boot[0].p_boot > 0.05
    assert not boot[0].significant


def test_degenerate_feature_is_reported_not_silently_zero():
    frame = _frame(n=1000)
    frame["x"] = 0.0
    results = gradation.measure_gradation(frame, ["x"], holdout=None,
                                          horizon_minutes=1440, n_boot=20)
    assert results[0].bins == []
    assert results[0].degenerate
    assert not results[0].significant
    assert "градация" in gradation.format_table(results) or \
           results[0].degenerate in gradation.format_table(results)


def test_table_marks_inversions_and_missing_bootstrap():
    results = gradation.measure_gradation(
        _frame(n=3000, effect=0.5), features=["x"], correction="none",
    )
    table = gradation.format_table(results)
    assert "монот." in table
    assert "ВНИМАНИЕ" in table          # бутстрап не считался


def test_unknown_correction_is_rejected():
    with pytest.raises(ValueError):
        gradation.measure_gradation(_frame(), ["x"], correction="wat")
