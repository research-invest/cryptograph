"""
Тесты стенда `btcproc/analysis/range_model.py` (задача A4 ТЗ
`crypto-graph/docs/tz_range_horizons_19-08-26.md`).

Стенд меряет приращение R² сверх бенчмарка, и вся его ценность в том, что
этому числу можно верить. Поэтому здесь не «код не падает», а пять проверок,
без которых задача не принимается:

* восстанавливает ли стенд ИЗВЕСТНОЕ приращение на синтетике;
* видит ли он внутрисуточный цикл (позитивный контроль на сезонность);
* не выдумывает ли он эффект на чистом шуме (негативный контроль, доля ложных
  срабатываний на 100 прогонах);
* считается ли out-of-sample R² от среднего ОБУЧАЮЩЕЙ части, а не проверочной;
* равен ли зазор purged CV горизонту ЗАМЕРА, а не `config.data.horizon_bars`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcproc.analysis import range_model as rm


# ─── Цель и колонки ─────────────────────────────────────────────────────────
def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC")


def _ar1(rng: np.random.Generator, n: int, rho: float = 0.95) -> np.ndarray:
    """Липкий ряд — модель уровня волатильности, у которого память в десятки баров."""
    noise = rng.normal(0, 1, n)
    out = np.empty(n)
    out[0] = noise[0]
    for i in range(1, n):
        out[i] = rho * out[i - 1] + noise[i]
    return out


def test_horizon_bars_matches_the_three_horizons():
    assert rm.horizon_bars("4h", 15) == 16
    assert rm.horizon_bars("12h", 15) == 48
    assert rm.horizon_bars("24h", 15) == 96


def test_range_target_normalizations_use_different_denominators():
    """
    `atr_h` обязан брать ATR по окну горизонта, а не 14 баров: иначе обе
    нормировки A3 совпали бы и проверка чувствительности к знаменателю стала
    бы формальностью.
    """
    rng = np.random.default_rng(3)
    n = 3000
    index = _index(n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.002, n))), index=index)
    base = pd.DataFrame({
        "high": close * 1.002, "low": close * 0.998, "close": close,
    }, index=index)

    atr14 = rm.range_target(base, 96, "atr14")
    atr_h = rm.range_target(base, 96, "atr_h")
    common = atr14.notna() & atr_h.notna()
    assert common.sum() > 1000
    assert not np.allclose(atr14[common], atr_h[common])

    with pytest.raises(ValueError):
        rm.range_target(base, 96, "atr7")


def test_log_target_drops_non_positive_instead_of_minus_infinity():
    values = pd.Series([1.0, 0.0, 2.0, -1.0])
    result = rm.log_target(values)
    assert np.isfinite(result.iloc[0]) and np.isfinite(result.iloc[2])
    assert result.isna().iloc[1] and result.isna().iloc[3]


def test_seasonal_columns_are_cyclic_and_continuous():
    """
    Час как циклическая координата, а не сессионные атомы: у тех два часа в
    сутках не принадлежат ни одной сессии (аудит 2026-08-15, B6).
    """
    columns = rm.seasonal_columns(_index(96 * 3))
    assert list(columns.columns) == ["hour_sin1", "hour_cos1", "hour_sin2",
                                     "hour_cos2", "is_weekend"]
    # Значение в 00:00 и в 24:00 следующего дня совпадает — разрыва нет.
    assert columns.iloc[0]["hour_sin1"] == pytest.approx(columns.iloc[96]["hour_sin1"])
    assert columns["is_weekend"].isin([0.0, 1.0]).all()


# ─── Восстановление известного приращения ───────────────────────────────────
def _linear_case(n: int, share_benchmark: float, share_predictor: float,
                 seed: int = 11):
    """
    Ряд, у которого приращение R² от предиктора известно ТОЧНО: три
    ортогональных по построению слагаемых с заданными долями дисперсии.
    """
    rng = np.random.default_rng(seed)
    benchmark = rng.normal(size=n)
    predictor = rng.normal(size=n)
    noise = rng.normal(size=n)
    target = (
        np.sqrt(share_benchmark) * benchmark
        + np.sqrt(share_predictor) * predictor
        + np.sqrt(1.0 - share_benchmark - share_predictor) * noise
    )
    return benchmark, predictor, target


def test_recovers_a_known_delta_r2():
    """Предиктор объясняет ровно 5% дисперсии — стенд обязан их увидеть."""
    n = 40000
    benchmark, predictor, target = _linear_case(n, 0.30, 0.05)
    ts = _index(n).to_numpy()

    base = rm.walk_forward(n, target, ts, gap=16, folds=5,
                           predict=rm.ols_predictor(benchmark, target))
    full = rm.walk_forward(n, target, ts, gap=16, folds=5,
                           predict=rm.ols_predictor(
                               np.column_stack([benchmark, predictor]), target))
    gain = rm.compare(base, full, block_length=16, n_boot=200,
                      rng=np.random.default_rng(0))

    assert gain.r2_base == pytest.approx(0.30, abs=0.02)
    assert gain.delta == pytest.approx(0.05, abs=0.01)
    assert gain.p_value <= 0.05


def test_in_sample_matrix_gain_recovers_a_known_delta():
    """
    То же для in-sample частного R² с матричным бенчмарком (задача B).
    На рангах доля объяснённой дисперсии слегка меньше — сравнение с запасом.
    """
    n = 20000
    benchmark, predictor, target = _linear_case(n, 0.30, 0.05, seed=5)
    matrix = np.column_stack([benchmark, benchmark ** 2])
    r2_base, r2_full, r_partial, p_gain = rm.partial_r2_gain_matrix(
        predictor, matrix, target, block_length=8, n_boot=200,
        rng=np.random.default_rng(1),
    )
    assert r2_full > r2_base
    assert (r2_full - r2_base) == pytest.approx(0.05, abs=0.015)
    assert r_partial > 0
    assert p_gain <= 0.05


# ─── Позитивный контроль на сезонность ──────────────────────────────────────
def test_seasonality_is_detected_over_the_har_benchmark():
    """
    Синтетика с внутрисуточным циклом: вклад B2 над B1 значимо положителен.

    Это ровно та величина, ради которой B1 оставлен в стеке, — «сколько
    эффекта съедает именно сезонность» (§0.3в ТЗ).
    """
    n = 40000
    index = _index(n)
    rng = np.random.default_rng(17)
    hour = (index.hour + index.minute / 60.0).to_numpy(dtype=float)
    seasonal = 0.6 * np.sin(2 * np.pi * hour / 24.0)
    volatility = np.cumsum(rng.normal(0, 0.01, n))       # липкий уровень
    target = seasonal + volatility + rng.normal(0, 0.5, n)

    har = np.column_stack([
        pd.Series(volatility, index=index).rolling(w, min_periods=w // 2).mean()
        .bfill().to_numpy()
        for w in rm.HAR_WINDOWS
    ])
    season = rm.seasonal_columns(index).to_numpy(dtype=float)
    ts = index.to_numpy()

    b1 = rm.walk_forward(n, target, ts, gap=16, folds=5,
                         predict=rm.ols_predictor(har, target))
    b2 = rm.walk_forward(n, target, ts, gap=16, folds=5,
                         predict=rm.ols_predictor(np.column_stack([har, season]), target))
    gain = rm.compare(b1, b2, block_length=16, n_boot=200,
                      rng=np.random.default_rng(2))
    assert gain.delta > 0.05
    assert gain.p_value <= 0.05


# ─── Негативный контроль ────────────────────────────────────────────────────
def test_pure_noise_predictor_is_not_significant():
    """
    Один прогон на чистом шуме: приращение крошечное и незначимое.
    Массовая проверка уровня — в тесте ниже.
    """
    n = 20000
    rng = np.random.default_rng(23)
    benchmark = rng.normal(size=n)
    target = 0.5 * benchmark + rng.normal(size=n)
    noise = rng.normal(size=n)
    ts = _index(n).to_numpy()

    base = rm.walk_forward(n, target, ts, gap=16, folds=5,
                           predict=rm.ols_predictor(benchmark, target))
    full = rm.walk_forward(n, target, ts, gap=16, folds=5,
                           predict=rm.ols_predictor(
                               np.column_stack([benchmark, noise]), target))
    gain = rm.compare(base, full, block_length=16, n_boot=400,
                      rng=np.random.default_rng(3))
    assert abs(gain.delta) < 0.005
    assert gain.p_value > 0.05


def test_false_positive_rate_is_near_nominal():
    """
    Сто прогонов с предиктором из ЧИСТОГО ШУМА поверх зависимых цели и
    бенчмарка: доля ложных срабатываний ≤ 0.07 при номинале 0.05 (требование
    A4 ТЗ).

    Зависимость оставлена там, где она есть в данных, — в цели и бенчмарке
    (AR(1) — модель липкой волатильности). Проверяемый предиктор независим,
    то есть нулевая гипотеза верна буквально, и любое превышение уровня было
    бы дефектом процедуры, а не свойством входа.

    Зерно фиксировано, поэтому тест детерминирован: он проверяет уровень
    процедуры, а не удачу конкретного запуска.
    """
    n = 2000
    rng = np.random.default_rng(2026)
    rejected = 0
    trials = 100
    for _ in range(trials):
        benchmark = _ar1(rng, n)
        target = 0.5 * benchmark + _ar1(rng, n)
        predictor = rng.normal(0, 1, n)
        _, _, _, p_gain = rm.partial_r2_gain_matrix(
            predictor, benchmark[:, None], target, block_length=16,
            n_boot=200, rng=rng,
        )
        rejected += int(p_gain <= 0.05)
    assert rejected / trials <= 0.07, (
        f"ложных срабатываний {rejected}/{trials} — блочная пересборка "
        f"анти-консервативна"
    )


def test_block_too_short_for_a_sticky_predictor_is_anti_conservative():
    """
    Известное ограничение, зафиксированное здесь намеренно.

    Если ЛИПОК сам предиктор, длина блока по правилу «первый лаг с
    автокорреляцией ниже 0.2» (`autocorr_block_rows`, то самое правило, что
    использует `measure_deriv_range.py`) оказывается вдвое короче нужного, и
    уровень теста уезжает примерно к 9% при номинале 5%. Более строгий порог
    (0.05) даёт блок в полтора раза длиннее, и уровень возвращается.

    Правило НЕ меняется: оно то же, что в разделе 36, и смена сломала бы
    сравнимость перепроверки с исходным замером. Тест нужен, чтобы
    направление смещения было известно при чтении результата: оно делает
    значимость дешевле, то есть УКРЕПЛЯЕТ отрицательный вывод и подрывает
    положительный. Отсюда же требование ТЗ иметь рядом со значимостью порог
    практической величины.
    """
    from btcproc.analysis.range_lift import autocorr_block_rows

    n = 2000
    trials = 60
    rates = {}
    for floor in (0.2, 0.05):
        rng = np.random.default_rng(99)
        rejected = 0
        for _ in range(trials):
            benchmark = _ar1(rng, n)
            target = 0.5 * benchmark + _ar1(rng, n)
            predictor = _ar1(rng, n)
            _, block = autocorr_block_rows(pd.Series(predictor), bars_per_day=1,
                                           floor=floor, max_lag_days=500)
            _, _, _, p_gain = rm.partial_r2_gain_matrix(
                predictor, benchmark[:, None], target, block_length=block,
                n_boot=200, rng=rng,
            )
            rejected += int(p_gain <= 0.05)
        rates[floor] = rejected / trials

    assert rates[0.2] > 0.05, "ограничение исчезло — перечитать шапку теста"
    assert rates[0.05] <= rates[0.2]


# ─── Знаменатель out-of-sample R² ───────────────────────────────────────────
def test_out_of_sample_r2_is_measured_against_the_train_mean():
    """
    Проверочное окно смещено относительно обучающего. Прогноз-константа,
    равная среднему ПРОВЕРОЧНОГО окна, обязана дать положительный R² при
    честном знаменателе (это и есть подглядывание) и ровно ноль — при
    знаменателе от среднего test.
    """
    actual = np.array([10.0, 11.0, 9.0, 10.0])
    train_mean = np.zeros(4)
    test_mean = np.full(4, actual.mean())

    assert rm.out_of_sample_r2(actual, test_mean, train_mean) > 0.9
    assert rm.out_of_sample_r2(actual, test_mean, test_mean) == pytest.approx(0.0)


def test_fold_predictions_report_the_drift_of_the_mean():
    """`drift` — цена ошибки «считать R² от среднего test», отдельным числом."""
    predictions = rm.FoldPredictions(
        ts=np.arange(4), actual=np.array([10.0, 11.0, 9.0, 10.0]),
        predicted=np.full(4, 10.0), train_mean=np.zeros(4), fold=np.ones(4),
    )
    assert predictions.drift > 0.9
    assert predictions.r2 > predictions.r2_versus_test_mean


def test_comparing_models_on_different_rows_is_refused():
    left = rm.FoldPredictions(ts=np.arange(3), actual=np.zeros(3),
                              predicted=np.zeros(3), train_mean=np.zeros(3),
                              fold=np.array([1, 1, 2]))
    right = rm.FoldPredictions(ts=np.arange(3), actual=np.zeros(3),
                               predicted=np.zeros(3), train_mean=np.zeros(3),
                               fold=np.array([1, 2, 2]))
    with pytest.raises(ValueError):
        left.assert_aligned(right)


# ─── Зазор purged CV ────────────────────────────────────────────────────────
def test_gap_equals_the_measured_horizon_not_the_config_horizon():
    """
    Зазор берётся из горизонта ЗАМЕРА. Захардкоженные 96 баров для 4h-замера
    выбросили бы вшестеро больше нужного и незаметно съели мощность
    (ловушка 4 ТЗ).
    """
    from btcproc import config
    from btcproc.analysis.control import purged_splits

    n = 40000
    target = np.zeros(n)
    ts = _index(n).to_numpy()
    seen: list[tuple[slice, slice]] = []

    def predict(train: slice, test: slice) -> np.ndarray:
        seen.append((train, test))
        return np.zeros(test.stop - test.start)

    rm.walk_forward(n, target, ts, gap=16, folds=5, predict=predict)

    assert seen == purged_splits(n, 5, 16)
    assert seen != purged_splits(n, 5, config.data.horizon_bars)
    for train, test in seen:
        assert test.start - train.stop == 16


def test_walk_forward_refuses_a_zero_gap():
    with pytest.raises(ValueError):
        rm.walk_forward(1000, np.zeros(1000), _index(1000).to_numpy(), gap=0,
                        folds=5, predict=lambda a, b: np.zeros(0))


# ─── Целевое кодирование ────────────────────────────────────────────────────
def test_target_encoding_uses_only_the_training_part_of_the_fold():
    """
    Код категории обязан зависеть ТОЛЬКО от обучающей части фолда: иначе
    среднее состояния содержит будущее, и это самый лёгкий способ получить
    впечатляющий и ложный R² (ловушка 6 ТЗ).
    """
    codes = np.array([0, 0, 1, 1, 0, 0, 1, 1] * 50)
    target = np.tile([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0], 50)
    train = slice(0, 300)
    test = slice(300, 400)

    encoded = rm.target_encode(codes, target, train, prior_weight=1.0)
    moved = target.copy()
    moved[test] += 100.0            # проверочное окно поменялось до неузнаваемости
    encoded_after = rm.target_encode(codes, moved, train, prior_weight=1.0)

    assert np.allclose(encoded, encoded_after)
    assert encoded[codes == 0].max() > encoded[codes == 1].max()


def test_unknown_category_falls_back_to_the_training_mean():
    """
    Категория, впервые встретившаяся в проверочном окне: «сведений на тот
    момент не было» — это среднее обучающей части, а не ноль.
    """
    codes = np.array([0] * 100 + [7] * 20)
    target = np.concatenate([np.full(100, 3.0), np.full(20, -50.0)])
    encoded = rm.target_encode(codes, target, slice(0, 100), prior_weight=0.0)
    assert encoded[100:] == pytest.approx(3.0)
