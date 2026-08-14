"""
Замер гейта R (docs/task_fear_greed.md §2, §5.3): предсказывает ли FGI РАЗМАХ.

Отдельный модуль, а не расширение `lift.py`/`gradation.py`: те работают с
бинарным исходом (доля роста) на выборке КАНДИДАТОВ, а здесь непрерывная
цель (`range_ratio`) на выборке ВСЕХ баров — кандидаты и модель состояний
тут ни при чём (§5.3: «range_ratio считается автономно, в самом скрипте, из
баров и ATR, без правки outcomes.py и без правки candidates/builder.py»).

Зависимость наблюдений — та же природа (перекрытие горизонта), поэтому
переиспользуется `stationary_block_indices` из `lift.py`, но статистика
другая: Spearman вместо разницы долей, а нулевая гипотеза воспроизводится
тем же приёмом — блоками пересобирается ПРЕДИКТОР, цель остаётся на месте.

Бенчмарк — не ноль, а скользящая `rv` (§2, урок раздела 26.4 журнала):
абсолютное «FGI коррелирует с размахом» — наполовину тавтология, потому что
волатильность входит в сам индекс с весом 25%. Отличимость даёт только
парное сравнение с тем, что уже есть в системе.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from btcproc.analysis.lift import CHUNK_CELLS, stationary_block_indices


def forward_range_ratio(base: pd.DataFrame, atr14: pd.Series, horizon_bars: int) -> pd.Series:
    """
    range_ratio(t) = размах цены за (t, t+H] баров, нормированный на ATR14(t)
    и на √H (размах растёт примерно как корень из времени — без нормировки
    величина мерила бы длину горизонта, а не рынок).

    NaN на последних horizon_bars барах истории (окно не помещается) и там,
    где ATR14 недоступен (прогрев).
    """
    high, low, close = base["high"], base["low"], base["close"]
    fwd_high = high.shift(-1).rolling(horizon_bars, min_periods=horizon_bars).max().shift(-(horizon_bars - 1))
    fwd_low = low.shift(-1).rolling(horizon_bars, min_periods=horizon_bars).min().shift(-(horizon_bars - 1))
    range_pct = (fwd_high - fwd_low) / close * 100.0
    atr_pct = (atr14 / close * 100.0) * math.sqrt(horizon_bars)
    return range_pct / atr_pct.replace(0.0, np.nan)


def _aligned(*series: pd.Series) -> list[np.ndarray]:
    """Общие непустые строки по всем сериям, в исходном (хронологическом) порядке."""
    frame = pd.concat(series, axis=1).dropna()
    return [frame.iloc[:, i].to_numpy(dtype=float) for i in range(len(series))]


def _rank(values: np.ndarray) -> np.ndarray:
    """Ранги (со средним рангом на связках) — Spearman = Pearson(rank, rank)."""
    from scipy.stats import rankdata

    return rankdata(values, method="average")


def _row_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Pearson-корреляция построчно между двумя матрицами (reps, n) — векторно,
    одним проходом на все реплики разом (BLAS-умножение вместо Python-цикла).
    """
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    num = np.sum(xc * yc, axis=1)
    den = np.sqrt(np.sum(xc * xc, axis=1) * np.sum(yc * yc, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def spearman_block_p(
    predictor: np.ndarray, target: np.ndarray, block_length: int,
    n_boot: int, rng: np.random.Generator,
) -> tuple[float, float]:
    """
    Spearman(predictor, target) и его блочный p-value.

    Нулевая гипотеза «связи нет» воспроизводится блочной пересборкой
    ПРЕДИКТОРА (та же логика, что в `lift.block_bootstrap_p`): сохраняет
    автокорреляцию обоих рядов, ломает только их взаимное выравнивание.

    Векторизовано по образцу `lift.block_bootstrap_p`: ранг предиктора
    пересобирается блоками СРАЗУ на весь чанк реплик (умножение матриц),
    а не циклом Python со `scipy.stats.spearmanr` на каждую реплику —
    на выборках в сотни тысяч строк цикл занимал бы часы.
    Приближение (стандартное для блочного бутстрапа): ранг блочно
    пересобранного ряда берётся как блочная пересборка ИСХОДНЫХ рангов, а не
    пересчитывается заново на реплике. Для непрерывных величин без связок
    (наш случай) это точно; на больших n расхождение с пересчётом
    пренебрежимо.
    """
    n = len(target)
    rank_x = _rank(predictor)
    rank_y = _rank(target)
    observed = float(_row_pearson(rank_x[None, :], rank_y[None, :])[0])
    if n < 20 or not np.isfinite(observed):
        return observed, 1.0

    chunk = max(1, min(n_boot, CHUNK_CELLS // max(n, 1)))
    extreme, done = 0, 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        idx = stationary_block_indices(n, block_length, rng, size)
        rho = _row_pearson(rank_x[idx], np.broadcast_to(rank_y, idx.shape))
        extreme += int(np.sum(np.isfinite(rho) & (np.abs(rho) >= abs(observed))))
        done += size
    return observed, float((1 + extreme) / (1 + n_boot))


def paired_diff_p(
    pred_a: np.ndarray, pred_b: np.ndarray, target: np.ndarray, block_length: int,
    n_boot: int, rng: np.random.Generator,
) -> tuple[float, float, float]:
    """
    Одностороннее сравнение |rho_a| против |rho_b| на ОДНОЙ и той же цели —
    отвечает на вопрос гейта R: значимо ли FGI сильнее бенчмарка `rv`, а не
    просто «оба коррелируют с размахом» (то самое «FGI ~= tautology через rv»,
    против которого поставлен бенчмарк, §2).

    Блоками пересобираются СТРОКИ (a, b и target вместе — пара предикторов
    привязана к одному и тому же бару), поэтому это оценка распределения
    разности статистик под сохранённой структурой зависимости, а не тест
    нулевой гипотезы через шаффл. Возвращает (rho_a, rho_b, p_a_stronger) —
    p — доля реплик, где |rho_a| реплики НЕ превысила |rho_b| реплики
    (типичный однохвостый bootstrap p-value для «a лучше b»).

    Векторизовано так же, как `spearman_block_p` — см. её докстринг про
    приближение рангов.
    """
    n = len(target)
    rank_a, rank_b, rank_t = _rank(pred_a), _rank(pred_b), _rank(target)
    rho_a = float(_row_pearson(rank_a[None, :], rank_t[None, :])[0])
    rho_b = float(_row_pearson(rank_b[None, :], rank_t[None, :])[0])
    if n < 20:
        return rho_a, rho_b, 1.0

    chunk = max(1, min(n_boot, CHUNK_CELLS // max(n, 1)))
    not_better, done = 0, 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        idx = stationary_block_indices(n, block_length, rng, size)
        ra = _row_pearson(rank_a[idx], rank_t[idx])
        rb = _row_pearson(rank_b[idx], rank_t[idx])
        valid = np.isfinite(ra) & np.isfinite(rb)
        not_better += int(np.sum(~valid | (np.abs(ra) <= np.abs(rb))))
        done += size
    return rho_a, rho_b, float((1 + not_better) / (1 + n_boot))


def autocorr_block_rows(
    daily_values: pd.Series, bars_per_day: int,
    floor: float = 0.2, max_lag_days: int = 400,
) -> tuple[int, int]:
    """
    Первый лаг (в ДНЯХ), на котором автокорреляция дневного ряда падает
    ниже `floor`, и его перевод в базовые бары (docs/task_fear_greed.md §1.3).
    Используется и как block_rows для measure_atom_lift.py --block-rows, и
    здесь — как нижняя граница длины блока по FGI.

    Возвращает (lag_days, block_rows); если автокорреляция не упала за
    max_lag_days — берётся потолок (с явным допущением, а не тихим NaN).
    """
    series = daily_values.dropna()
    for lag in range(1, max_lag_days + 1):
        r = series.autocorr(lag=lag)
        if r is not None and not np.isnan(r) and abs(r) < floor:
            return lag, lag * bars_per_day
    return max_lag_days, max_lag_days * bars_per_day


@dataclass
class RangeGateResult:
    symbol: str
    horizon: str
    epoch: str
    n: int
    rho_fgi: float
    p_fgi: float
    rho_rv: float
    p_fgi_beats_rv: float

    @property
    def fgi_significant(self) -> bool:
        return self.p_fgi <= 0.05

    @property
    def beats_benchmark(self) -> bool:
        return self.p_fgi_beats_rv <= 0.05 and abs(self.rho_fgi) > abs(self.rho_rv)
