"""
FDR по конфигурациям: закрепляется то, из-за чего замер мог бы «найти»
значимые ключи там, где их нет.

Три места, где здесь легко ошибиться, и все три проверяются:

1. **раздутие дисперсии.** Наблюдения зависимы (перекрытие окон исходов), и
   `z`, посчитанный по биномиальной ошибке, анти-консервативен. Оценка
   раздутия обязана это ловить: на независимых данных она даёт около единицы,
   на зависимых — заметно больше;
2. **порог выборки** обязан отсекать ключи с парой наблюдений: при 12 тысячах
   ключей самые перекошенные — всегда самые мелкие;
3. **негативный контроль** обязан давать около нуля выживших. Если он даёт
   больше основного прогона, читать основной нельзя.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from btcproc.analysis import configs as cf
from btcproc.analysis.lift import benjamini_hochberg


def _independent(n: int = 30000, n_keys: int = 150, rate: float = 0.52,
                 seed: int = 0):
    rng = np.random.default_rng(seed)
    keys = np.array([f"k{i % n_keys}" for i in range(n)])
    outcomes = (rng.random(n) < rate).astype(float)
    return keys, outcomes


def _clustered(n: int = 30000, n_keys: int = 150, block: int = 200,
               seed: int = 1):
    """
    Исходы идут сериями, а ключи — блоками. Ровно та зависимость, что есть в
    данных: окна исходов соседних реализаций перекрываются, и ключ держится
    подряд.
    """
    rng = np.random.default_rng(seed)
    n_blocks = n // block + 1
    outcomes = np.repeat((rng.random(n_blocks) < 0.52).astype(float), block)[:n]
    keys = np.repeat([f"k{i % n_keys}" for i in range(n_blocks)], block)[:n]
    return keys, outcomes


def test_null_scale_is_near_one_on_independent_data():
    """На независимых данных раздувать нечего — иначе оценка сама смещена."""
    keys, outcomes = _independent()
    scale, table = cf.null_scale(keys, outcomes, block_length=2, n_boot=150,
                                 rng=np.random.default_rng(0))
    filled = table[table["n_null"] >= 30]
    assert not filled.empty
    assert (filled["sigma"].between(0.85, 1.2)).all()


def test_null_scale_grows_when_observations_are_clustered():
    """
    Смысл всей конструкции: зависимость обязана быть ВИДНА в числе, а не
    подразумеваться. Если σ остаётся около единицы на явно зависимых данных,
    поправка не работает и p-value анти-консервативны.
    """
    keys, outcomes = _clustered()
    scale, table = cf.null_scale(keys, outcomes, block_length=200, n_boot=150,
                                 rng=np.random.default_rng(0))
    filled = table[table["n_null"] >= 30]
    assert not filled.empty
    assert float(filled["sigma"].max()) > 1.3


def test_small_keys_are_not_tested_at_all():
    """
    При тысячах ключей самые перекошенные — всегда самые мелкие. Порог
    выборки — единственное, что стоит между этим фактом и «находками».
    """
    keys = np.array(["big"] * 500 + ["tiny"] * 5)
    outcomes = np.concatenate([np.repeat([1.0, 0.0], 250), np.ones(5)])
    table = cf.observed_z(keys, outcomes, min_rows=30)
    assert list(table.index) == ["big"]


def test_planted_effect_survives_and_noise_does_not():
    """
    Прямая проверка: подсаженный сильный ключ обязан выжить после BH, а
    полторы сотни шумовых — нет.
    """
    keys, outcomes = _independent()
    planted = keys == "k7"
    rng = np.random.default_rng(5)
    outcomes[planted] = (rng.random(planted.sum()) < 0.85).astype(float)

    table = cf.observed_z(keys, outcomes)
    scale, _ = cf.null_scale(keys, outcomes, 2, 150, np.random.default_rng(0))
    marks = benjamini_hochberg(list(cf.scaled_p_values(table, scale)), 0.10)
    survivors = [key for key, mark in zip(table.index, marks) if mark]
    assert survivors == ["k7"]


def test_permuted_control_kills_the_planted_effect():
    """
    Негативный контроль обязан давать около нуля. Это единственная проверка,
    которая ловит дефект в самой процедуре: если она находит эффект там, где
    его разрушили, читать основной прогон нельзя.
    """
    keys, outcomes = _independent()
    planted = keys == "k7"
    rng = np.random.default_rng(5)
    outcomes[planted] = (rng.random(planted.sum()) < 0.85).astype(float)

    control_keys, control_outcomes = cf.permuted_control(
        keys, outcomes, block_length=50, rng=np.random.default_rng(9))
    table = cf.observed_z(control_keys, control_outcomes)
    scale, _ = cf.null_scale(keys, outcomes, 2, 150, np.random.default_rng(0))
    marks = benjamini_hochberg(list(cf.scaled_p_values(table, scale)), 0.10)
    assert sum(marks) == 0


def test_p_values_are_two_sided():
    """
    Система выпускает кандидатов в ОБЕ стороны, значит и тест обязан быть
    двусторонним. Односторонний означал бы, что сторона известна заранее.
    """
    table = pd.DataFrame({"n": [100.0, 100.0], "z": [3.0, -3.0]},
                         index=["up", "down"])
    values = cf.scaled_p_values(table, {10 ** 9: 1.0})
    assert abs(values["up"] - values["down"]) < 1e-12
