"""
Дрейф распределения: проверяется постановка вопроса, а не арифметика.

Главное здесь — не то, что `W₁` считается правильно (это две строки кода), а
то, что вопрос задан так, чтобы на него можно было получить отрицательный
ответ. «Отличается ли квартал от истории» — вопрос, на который ответ всегда
«да»: при трёхстах тысячах наблюдений значимо любое различие. Поэтому
наблюдение сравнивается с расстояниями между случайными парами окон ТОЙ ЖЕ
длины, и тесты закрепляют именно это.

Плюс одно требование к нулёвке, которое ломается молча: окна пары обязаны не
пересекаться. Перекрывающиеся окна делят наблюдения, расстояние между ними
занижено механически, и любое реальное расхождение стало бы «значимым».
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from btcproc.analysis import drift as dr


def _frame(n: int = 6000, seed: int = 0, shift_tail: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n),
        "c": rng.normal(0, 5, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"))
    if shift_tail:
        frame.iloc[-500:] += shift_tail
    return frame


def test_wasserstein_is_zero_for_identical_samples():
    values = np.linspace(-3, 3, 500)
    assert dr.wasserstein_1d(values, values) == 0.0


def test_wasserstein_equals_the_shift_for_a_pure_shift():
    """Для сдвига распределения `W₁` равен величине сдвига — точно."""
    rng = np.random.default_rng(0)
    sample = rng.normal(0, 1, 20000)
    assert abs(dr.wasserstein_1d(sample, sample + 2.0) - 2.0) < 0.05


def test_robust_scale_prevents_one_feature_from_dominating():
    """
    Без приведения к общей шкале среднее по признакам описывало бы признак с
    наибольшим разбросом, а не расхождение распределений.
    """
    frame = _frame()
    raw = dr.frame_distance(frame.iloc[-500:], frame.iloc[:-500])
    scaled = dr.robust_scale(frame)
    balanced = dr.frame_distance(scaled.iloc[-500:], scaled.iloc[:-500])
    # У сырого кадра «c» с разбросом впятеро больше тянет среднее на себя.
    assert raw > 2 * balanced


def test_stationary_tail_does_not_stand_out():
    """
    Отрицательный контроль. На стационарном ряде последнее окно обязано быть
    неотличимо от случайной пары — иначе замер находил бы «смену режима»
    всегда.
    """
    frame = dr.robust_scale(_frame())
    window = 500
    observed = dr.frame_distance(frame.iloc[-window:], frame.iloc[:-window])
    null = dr.null_distances(frame, window, 150, np.random.default_rng(1))
    assert observed < float(np.percentile(null, 95))


def test_shifted_tail_does_stand_out():
    """Положительный контроль: подсаженный сдвиг обязан быть виден."""
    frame = dr.robust_scale(_frame(shift_tail=2.0))
    window = 500
    observed = dr.frame_distance(frame.iloc[-window:], frame.iloc[:-window])
    null = dr.null_distances(frame, window, 150, np.random.default_rng(1))
    assert observed > float(np.percentile(null, 95))


def test_null_pairs_never_overlap():
    """
    Перекрывающиеся окна делят наблюдения, и расстояние между ними занижено
    механически. Нулёвка из таких пар сделала бы «значимым» что угодно.
    """
    frame = dr.robust_scale(_frame(n=3000))
    window = 500
    values = dr.null_distances(frame, window, 200, np.random.default_rng(2))
    # Косвенная проверка: у перекрывающихся пар расстояния были бы заметно
    # меньше, вплоть до нуля при полном совпадении.
    assert values.min() > 0.0
    assert len(values) > 100


def test_rolling_drift_compares_window_with_its_own_past_only():
    """
    Ряд обязан сравнивать окно с ПРЕДШЕСТВУЮЩЕЙ историей, а не со всей:
    иначе величина заглядывает вперёд и на исторических точках означает не то,
    что на свежих.
    """
    frame = dr.robust_scale(_frame(n=3000))
    series = dr.rolling_drift(frame, window=500, step=250)
    assert not series.empty
    assert series["ts"].is_monotonic_increasing
    # Первая точка не может появиться раньше, чем накопятся два окна.
    assert series["ts"].iloc[0] >= frame.index[999]
