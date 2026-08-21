"""
Range-оценщики дисперсии из OHLC — задача A ТЗ
`crypto-graph/docs/tz_candle_geometry_20-08-26.md`, §2.6.

Проверяется не рынок, а формулы и их обвязка. Порядок тестов повторяет
порядок обязательных проверок ТЗ, и первым стоит **позитивный контроль**: без
него ноль в замере неотличим от опечатки в знаке или в константе
(§3.4 `extending_features.md`). Три оценщика построены из одних и тех же
H, L, O, C и коррелированы между собой — тесты это учитывают и не считают
согласие трёх формул тремя подтверждениями.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcproc.analysis import range_model as rm

#: Сколько внутрибарных шагов имитируем. Чем их больше, тем ближе бар к
#: непрерывной траектории, для которой формулы и выведены: при m = 1000
#: дискретизация занижает оценку на 4–7%, при m = 50 — уже на 20–26%.
STEPS = 1000


def synthetic_bars(n: int, steps: int = STEPS, sigma_step: float = 0.0005,
                   drift: float = 0.0, seed: int = 1) -> pd.DataFrame:
    """
    Бары, собранные из ЯВНОЙ внутрибарной траектории с известной дисперсией.

    Каждый бар — своя траектория от логарифма 0, без переноса уровня между
    барами. Так снос не накапливается по всему ряду: при переносе снос в
    0.001 на шаг взрывает цену до переполнения за пару тысяч баров, и тест
    проверял бы арифметику float, а не формулу.

    Дисперсия лог-доходности бара по построению — `steps · sigma_step²`.
    """
    rng = np.random.default_rng(seed)
    path = np.cumsum(rng.normal(drift, sigma_step, size=(n, steps)), axis=1)
    price = np.exp(path)
    return pd.DataFrame(
        {
            "open": np.ones(n),
            "high": np.maximum(price.max(axis=1), 1.0),
            "low": np.minimum(price.min(axis=1), 1.0),
            "close": price[:, -1],
        },
        index=pd.date_range("2024-01-01", periods=n, freq="15min"),
    )


def true_variance(steps: int = STEPS, sigma_step: float = 0.0005) -> float:
    return steps * sigma_step ** 2


# ─── Позитивный контроль ────────────────────────────────────────────────────
def test_estimators_recover_the_known_variance():
    """
    Все три оценщика восстанавливают ЗАДАННУЮ дисперсию с точностью до шума
    дискретизации.

    Верхняя граница 1.05 важнее нижней: завышение означало бы ошибку в
    константе формулы, а занижение объясняется тем, что бар из тысячи шагов
    всё-таки пропускает часть экстремумов непрерывной траектории. Ждать здесь
    единицы с точностью до процента нельзя — это свойство дискретизации, а не
    дефект.
    """
    bars = synthetic_bars(4000, seed=1)
    estimators = rm.range_estimators(bars)
    for name in rm.RANGE_ESTIMATORS:
        ratio = float(estimators[name].mean()) / true_variance()
        assert 0.90 <= ratio <= 1.05, f"{name}: оценка/истина = {ratio:.3f}"


def test_parkinson_is_the_least_biased_by_discretisation():
    """
    Порядок занижения известен и фиксируется: Паркинсон ближе всех к истине.

    Он строится из экстремумов, а они на дискретной сетке теряются медленнее,
    чем восстанавливается тело. Тест держит взаимный порядок трёх формул —
    если он перевернётся, значит перепутаны константы.
    """
    estimators = rm.range_estimators(synthetic_bars(4000, seed=2))
    means = {name: float(estimators[name].mean()) for name in rm.RANGE_ESTIMATORS}
    assert means["p"] > means["gk"]
    assert means["p"] > means["rs"]


def test_rogers_satchell_ignores_the_drift_inside_the_bar():
    """
    Свойство, ради которого RS третий в наборе: он не зависит от сноса.

    Сравнивается не близость к истине (у RS своё занижение от дискретизации),
    а ИЗМЕНЕНИЕ оценки при добавлении сноса величиной в два стандартных
    отклонения бара. У Паркинсона трендовый бар засчитывается как
    волатильность и оценка растёт в разы; у RS она почти не двигается.

    Это проверка того, что формула набрана правильно, а не того, что рынок
    такой.
    """
    sigma_bar = np.sqrt(true_variance())
    drift = 2.0 * sigma_bar / STEPS

    flat = rm.range_estimators(synthetic_bars(4000, seed=3))
    trend = rm.range_estimators(synthetic_bars(4000, drift=drift, seed=3))
    shift = {
        name: abs(float(trend[name].mean()) / float(flat[name].mean()) - 1.0)
        for name in rm.RANGE_ESTIMATORS
    }
    assert shift["rs"] < 0.25, f"RS сдвинулся на {shift['rs']:.2f} — снос на него влияет"
    assert shift["rs"] < shift["gk"] < shift["p"], shift


# ─── Инварианты формул ──────────────────────────────────────────────────────
def test_estimators_are_scale_invariant():
    """
    Тот же рынок, умноженный на 128, даёт те же значения.

    Степень двойки взята, чтобы тест не ловил разъезд округления: умножение на
    неё точно в двоичной арифметике, и любое расхождение будет означать
    ошибку в формуле, а не в последнем бите.
    """
    bars = synthetic_bars(500, steps=200, seed=4)
    scaled = bars * 128.0
    pd.testing.assert_frame_equal(
        rm.range_estimators(bars), rm.range_estimators(scaled), atol=1e-12
    )


def test_columns_do_not_look_ahead():
    """
    Колонки, посчитанные на префиксе истории, совпадают с посчитанными на
    полной. Скользящее среднее смотрит назад, и никакой `shift(-1)` в модуле
    появиться не должен.
    """
    bars = synthetic_bars(1200, steps=100, seed=5)
    windows = (16, 64)
    full = rm.range_estimator_columns(bars, windows=windows)
    prefix = rm.range_estimator_columns(bars.iloc[:800], windows=windows)
    pd.testing.assert_frame_equal(full.iloc[:800], prefix)


def test_flat_bar_gives_zero_not_nan():
    """
    Вырожденный бар (`high == low`) — это максимальное сжатие, а не отсутствие
    данных: оценщик обязан вернуть ноль КОНСТАНТОЙ.

    NaN здесь стоил бы дороже, чем кажется: он выпал бы из окна усреднения и
    сдвинул агрегат, а в векторе признаков финальный `dropna()` выбросил бы
    строку целиком — ровно дефект B8 аудита 2026-08-15. В боевой базе таких
    баров 147 у BTCUSDT, 329 у ETHUSDT, 62 у AAVEUSDT, 14 у SOLUSDT.
    """
    bars = synthetic_bars(50, steps=50, seed=6)
    bars.iloc[10] = [1.0, 1.0, 1.0, 1.0]
    estimators = rm.range_estimators(bars)
    row = estimators.iloc[10]
    assert not row.isna().any(), "вырожденный бар дал NaN"
    assert (row == 0.0).all(), f"вырожденный бар дал {row.to_dict()}"


def test_aggregate_of_a_flat_window_is_nan_not_minus_inf():
    """
    А вот НУЛЕВОЕ СРЕДНЕЕ по всему окну уходит в NaN — по той же дисциплине,
    что у `har_columns` и у цели. Разница с тестом выше принципиальная: ноль
    у одного бара это измерение, ноль у всего окна это отсутствие рынка, и
    −inf в регрессии не имеет смысла ни в каком виде.
    """
    n = 40
    flat = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0},
        index=pd.date_range("2024-01-01", periods=n, freq="15min"),
    )
    columns = rm.range_estimator_columns(flat, windows=(16,))
    values = columns.iloc[16:]
    assert values.isna().all().all()
    assert not np.isneginf(values.to_numpy()).any()


def test_column_names_and_count():
    """Три оценщика × три окна = девять колонок, имена фиксированы."""
    bars = synthetic_bars(120, steps=50, seed=7)
    columns = rm.range_estimator_columns(bars, windows=(16, 32))
    assert list(columns.columns) == [
        "log_p_16", "log_p_32", "log_gk_16", "log_gk_32", "log_rs_16", "log_rs_32",
    ]


# ─── Негативный контроль измерителя ─────────────────────────────────────────
def test_estimators_of_unrelated_bars_are_not_significant():
    """
    Сто прогонов: колонки оценщиков, посчитанные по барам, НЕ СВЯЗАННЫМ с
    целью, не дают значимого приращения сверх бенчмарка чаще номинала.

    Отличие от `test_false_positive_rate_is_near_nominal` в `test_range_model`
    существенное: там предиктором был белый шум, здесь — настоящие колонки
    этого модуля, липкие по построению (скользящее среднее по 16 барам). Если
    бы липкость предиктора ломала уровень процедуры, поймать это можно было бы
    только так.
    """
    trials, rejected = 100, 0
    rng = np.random.default_rng(2026)
    for trial in range(trials):
        bars = synthetic_bars(600, steps=50, seed=1000 + trial)
        predictor = rm.range_estimator_columns(bars, windows=(16,))["log_p_16"]
        benchmark = pd.Series(_ar1(rng, len(bars)), index=bars.index)
        target = 0.5 * benchmark + pd.Series(_ar1(rng, len(bars)), index=bars.index)
        frame = pd.concat(
            [predictor.rename("x"), benchmark.rename("b"), target.rename("y")], axis=1
        ).dropna()
        _, _, _, p_gain = rm.partial_r2_gain_matrix(
            frame["x"].to_numpy(), frame[["b"]].to_numpy(), frame["y"].to_numpy(),
            block_length=16, n_boot=200, rng=rng,
        )
        rejected += int(p_gain <= 0.05)
    assert rejected / trials <= 0.07, f"ложных срабатываний {rejected}/{trials}"


def _ar1(rng: np.random.Generator, n: int, phi: float = 0.9) -> np.ndarray:
    """Липкий ряд — модель волатильности, живущей кластерами."""
    noise = rng.normal(size=n)
    out = np.empty(n)
    out[0] = noise[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + noise[i]
    return out


# ─── Третья нормировка ──────────────────────────────────────────────────────
def test_rv_h_normalization_exists_and_differs_from_atr():
    """
    `rv_h` считается и даёт другую цель, чем ATR-нормировки.

    Смысл её в задаче A — контроль на артефакт знаменателя: `atr14` и `atr_h`
    построены из размахов баров, то есть из тех же величин, что Паркинсон и
    Гарман–Класс. Совпади она с ними численно — контролем бы не была.
    """
    bars = synthetic_bars(600, steps=50, seed=8)
    atr = rm.range_target(bars, 16, "atr14").dropna()
    rv = rm.range_target(bars, 16, "rv_h").dropna()
    assert len(rv) > 100
    common = atr.index.intersection(rv.index)
    assert float(np.corrcoef(atr[common], rv[common])[0, 1]) < 0.99


def test_extra_normalization_is_not_in_the_default_set():
    """
    Сторож против молчаливого расширения: `rv_h` НЕ входит в `NORMALIZATIONS`.

    Три стенда (`measure_range_horizons`, `validate_range_holdout`,
    `fit_range_forecast`) берут этот кортеж как значение по умолчанию, и
    добавленный в него элемент увеличил бы число их ячеек — а с ним и
    BH-поправку — при том что их критерии заявлены на двух нормировках.
    Ошибка была бы бесшумной: прогон отработал бы, числа сошлись бы, поехал
    бы только знаменатель поправки.
    """
    assert rm.NORMALIZATIONS == ("atr14", "atr_h")
    assert "rv_h" in rm.EXTRA_NORMALIZATIONS
    assert not set(rm.NORMALIZATIONS) & set(rm.EXTRA_NORMALIZATIONS)


def test_unknown_normalization_raises():
    bars = synthetic_bars(100, steps=50, seed=9)
    with pytest.raises(ValueError):
        rm.range_target(bars, 16, "atr_whatever")
