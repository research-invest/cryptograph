"""
Лифт по ГРАДАЦИИ непрерывного признака, а не по бинарному флагу.

Зачем отдельный модуль. `lift.py` отвечает на вопрос «отличается ли исход
там, где атом сработал». Это самая бедная возможная формулировка гипотезы:
`in_unfilled_fvg` — булево «цена внутри незакрытого имбаланса», и вся
градация (свежесть зоны, её размер в ATR, доля уже закрытого разрыва, номер
касания) в этот один бит не влезает. Она живёт в непрерывных признаках
`near_*`, `fvg_fill_pct`, `ob_age_norm`, а те в замере лифта не участвовали
вовсе — признаки выключены флагом.

Отсюда постановка: разбить непрерывный признак на квантильные бины и
проверить не разницу двух групп, а **монотонность тренда** по бинам
(Cochran–Armitage). «Чем свежее зона, тем выше доля роста» — гораздо более
сильное свидетельство, чем «за порогом доля выше»: один удачно попавший
порог объясняется случайностью легко, монотонность по пяти бинам — трудно.

Признаки для этого включать в модель НЕ нужно: значения считаются на барах
кандидатов и приджойниваются. Поэтому замер не зависит от калибровки
дробления состояний и запускается на работающей системе.

Зависимость наблюдений здесь ровно та же, что в `lift.py` (перекрытие
горизонтов), и лечится тем же блочным бутстрапом — переиспользуются его
`block_length_rows` и `stationary_block_indices`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from btcproc.analysis.lift import (
    CHUNK_CELLS,
    DEFAULT_N_BOOT,
    atom_seed,
    benjamini_hochberg,
    block_length_rows,
    bonferroni,
    resolution_is_sufficient,
    stationary_block_indices,
    two_sided_p,
)

#: Сколько квантильных бинов по умолчанию. Пять — компромисс: терцилей мало,
#: чтобы увидеть монотонность, а децили на выборке в 30 тыс. кандидатов с
#: учётом перекрытия горизонтов дают меньше сотни независимых наблюдений
#: на бин.
DEFAULT_BINS = 5

#: Ниже этого числа бинов говорить о тренде бессмысленно.
MIN_BINS = 3


@dataclass
class Bin:
    """Один квантильный бин: границы, объём и доля успеха."""

    index: int
    lo: float
    hi: float
    n: int
    p: float


@dataclass
class TrendResult:
    """Результат проверки монотонного тренда по одному признаку."""

    feature: str
    bins: list[Bin]
    z: float
    p_value: float
    p_boot: float | None = None
    significant: bool = False
    holdout: "TrendResult | None" = field(default=None, repr=False)
    #: Заполняется, если бинов не набралось: доля вырожденного значения.
    degenerate: str | None = None

    @property
    def spread(self) -> float:
        """Разница долей между верхним и нижним бином."""
        if len(self.bins) < 2:
            return 0.0
        return self.bins[-1].p - self.bins[0].p

    @property
    def monotone(self) -> bool:
        """Строго ли доли идут в одну сторону по бинам."""
        if len(self.bins) < MIN_BINS:
            return False
        shares = [b.p for b in self.bins]
        ups = all(b >= a for a, b in zip(shares, shares[1:]))
        downs = all(b <= a for a, b in zip(shares, shares[1:]))
        return ups or downs

    @property
    def inversions(self) -> int:
        """Сколько раз доля пошла против общего направления."""
        if len(self.bins) < 2:
            return 0
        shares = [b.p for b in self.bins]
        steps = [b - a for a, b in zip(shares, shares[1:])]
        direction = 1 if sum(steps) >= 0 else -1
        return sum(1 for s in steps if s * direction < 0)

    @property
    def effective_p(self) -> float:
        return self.p_value if self.p_boot is None else self.p_boot

    @property
    def confirmed(self) -> bool:
        """
        Засчитывается тренд, который значим после поправки, монотонен и
        сохранил направление на holdout при сопоставимой величине.

        Требование монотонности здесь не украшательство: Cochran–Armitage
        значим и на немонотонном профиле, если крайние бины разошлись, —
        а это уже обычная разница двух групп, ради которой отдельный
        инструмент не нужен.
        """
        if not self.significant or not self.monotone or self.holdout is None:
            return False
        if self.spread == 0 or self.holdout.spread == 0:
            return False
        if (self.spread > 0) != (self.holdout.spread > 0):
            return False
        ratio = abs(self.holdout.spread) / abs(self.spread)
        return 0.5 <= ratio <= 2.0


def quantile_bins(values: np.ndarray, bins: int = DEFAULT_BINS) -> tuple[np.ndarray, list[tuple[float, float]], str | None]:
    """
    Номер бина для каждого значения плюс границы бинов.

    Границы квантильные, а не равномерные: у SMC-величин распределения
    сильно скошены (близость `1/(1+d)` жмётся к нулю), и равномерная сетка
    сложила бы 90% наблюдений в один бин.

    Вырождение обрабатывается явно. У признаков вида «расстояние до
    ближайшей зоны» ноль означает «зоны нет вовсе», и таких строк бывает
    больше, чем ширина бина, — тогда квантильные границы совпадают, бины
    схлопываются, и `pd.qcut` без `duplicates="drop"` просто падает.
    Возвращаем сколько получилось и текст диагноза: замер по трём бинам
    вместо пяти — результат, а не ошибка, а вот замер по одному бину надо
    увидеть, а не молча получить ноль.
    """
    series = pd.Series(values, dtype=float)
    labels, edges = pd.qcut(series, q=bins, labels=False, retbins=True,
                            duplicates="drop")
    labels = labels.to_numpy()
    n_bins = int(np.nanmax(labels)) + 1 if len(labels) and not np.all(np.isnan(labels)) else 0

    note = None
    if n_bins < bins:
        modal = series.mode()
        modal_value = float(modal.iat[0]) if len(modal) else float("nan")
        share = float((series == modal_value).mean())
        note = (f"бинов {n_bins} вместо {bins}: значение {modal_value:.3g} "
                f"занимает {share:.0%} строк")
    ranges = [(float(edges[i]), float(edges[i + 1])) for i in range(n_bins)]
    return labels, ranges, note


def cochran_armitage(labels: np.ndarray, metric: np.ndarray) -> tuple[float, float]:
    """
    Тест на линейный тренд долей по упорядоченным группам.

        T   = Σ tᵢ (rᵢ − nᵢ·p̄)
        Var = p̄(1−p̄) · [ Σ nᵢtᵢ² − (Σ nᵢtᵢ)² / N ]
        z   = T / √Var

    где tᵢ — оценка группы (берём номер бина: бины равнонаполненные, и
    любая монотонная шкала даёт тот же знак), nᵢ — объём, rᵢ — число
    «успехов», p̄ — общая доля.

    Отличие от двухвыборочного z-теста принципиальное: тот спрашивает
    «разошлись ли две группы», этот — «идут ли доли по возрастанию оценки».
    Немонотонный профиль с разошедшимися краями первый тест находит, второй
    штрафует.

    Возвращает (z, p_value); при вырожденных входах — (0.0, 1.0).
    """
    labels = np.asarray(labels, dtype=float)
    metric = np.asarray(metric, dtype=float)
    ok = ~np.isnan(labels)
    labels, metric = labels[ok].astype(int), metric[ok]
    if labels.size == 0:
        return 0.0, 1.0

    groups = np.unique(labels)
    if len(groups) < 2:
        return 0.0, 1.0

    n = np.array([int((labels == g).sum()) for g in groups], dtype=float)
    r = np.array([float(metric[labels == g].sum()) for g in groups])
    total, successes = n.sum(), r.sum()
    p_bar = successes / total
    if p_bar <= 0.0 or p_bar >= 1.0:
        return 0.0, 1.0

    t = groups.astype(float)
    statistic = float(np.sum(t * (r - n * p_bar)))
    variance = p_bar * (1 - p_bar) * (float(np.sum(n * t * t))
                                      - float(np.sum(n * t)) ** 2 / total)
    if variance <= 0.0:
        return 0.0, 1.0
    z = statistic / math.sqrt(variance)
    return z, two_sided_p(z)


def trend_bootstrap_p(
    labels: np.ndarray,
    metric: np.ndarray,
    block_length: int,
    n_boot: int = DEFAULT_N_BOOT,
    rng: np.random.Generator | None = None,
) -> float:
    """
    P-value тренда через stationary block bootstrap.

    Логика та же, что у `lift.block_bootstrap_p`: блоками пересобирается
    ПРИЗНАК (здесь — номер бина), метрика остаётся на месте. Так сохраняются
    автокорреляция обоих рядов и рушится только их взаимное выравнивание.

    Векторизации по репликам, как в `lift`, здесь нет: Cochran–Armitage
    считается по группам, а групп на реплику пять. Цикл по репликам с
    группировкой через `np.bincount` — на 2000 реплик это единицы секунд,
    и городить трёхмерные массивы ради этого незачем.
    """
    rng = rng or np.random.default_rng(42)
    labels = np.asarray(labels, dtype=float)
    metric = np.asarray(metric, dtype=float)
    ok = ~np.isnan(labels)
    labels, metric = labels[ok].astype(np.int64), metric[ok]

    n = labels.size
    n_groups = int(labels.max()) + 1 if n else 0
    if n < 10 or n_groups < 2:
        return 1.0

    observed = abs(cochran_armitage(labels, metric)[0])
    total = float(metric.sum())
    p_bar = total / n
    if p_bar <= 0.0 or p_bar >= 1.0:
        return 1.0

    t = np.arange(n_groups, dtype=float)
    chunk = max(1, min(n_boot, CHUNK_CELLS // max(n, 1)))
    extreme, done = 0, 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        idx = stationary_block_indices(n, block_length, rng, size)
        for row in idx:
            shuffled = labels[row]
            counts = np.bincount(shuffled, minlength=n_groups).astype(float)
            sums = np.bincount(shuffled, weights=metric, minlength=n_groups)
            statistic = float(np.sum(t * (sums - counts * p_bar)))
            variance = p_bar * (1 - p_bar) * (
                float(np.sum(counts * t * t)) - float(np.sum(counts * t)) ** 2 / n
            )
            if variance <= 0.0:
                continue
            if abs(statistic / math.sqrt(variance)) >= observed:
                extreme += 1
        done += size

    return float((1 + extreme) / (1 + n_boot))


def _measure_one(frame: pd.DataFrame, feature: str, metric_column: str,
                 bins: int) -> TrendResult | None:
    values = frame[feature].to_numpy(dtype=float)
    metric = frame[metric_column].to_numpy(dtype=float)
    if np.all(np.isnan(values)):
        return None

    labels, ranges, note = quantile_bins(values, bins)
    if len(ranges) < MIN_BINS:
        return TrendResult(feature=feature, bins=[], z=0.0, p_value=1.0,
                           degenerate=note or "градация невозможна")

    z, p_value = cochran_armitage(labels, metric)
    described = [
        Bin(index=i, lo=lo, hi=hi,
            n=int((labels == i).sum()),
            p=float(metric[labels == i].mean()) if (labels == i).any() else float("nan"))
        for i, (lo, hi) in enumerate(ranges)
    ]
    return TrendResult(feature=feature, bins=described, z=z, p_value=p_value,
                       degenerate=note)


def measure_gradation(
    frame: pd.DataFrame,
    features: list[str],
    metric_column: str = "metric",
    ts_column: str = "ts",
    bins: int = DEFAULT_BINS,
    alpha: float = 0.05,
    correction: str = "bonferroni",
    holdout: float | None = 0.3,
    horizon_minutes: int | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 42,
) -> list[TrendResult]:
    """
    Квантильный лифт с тестом на монотонность по каждому признаку.

    frame — строка на кандидата: `ts`, метрика в [0, 1] и колонка на каждый
    проверяемый признак. Holdout и поправка на множественные сравнения —
    те же, что в `lift.measure_lift`, и по той же причине.
    """
    if correction not in ("bonferroni", "bh", "none"):
        raise ValueError(f"неизвестная поправка: {correction}")

    ordered = frame.sort_values(ts_column)
    if holdout is None:
        train, test = ordered, None
    else:
        cut = int(len(ordered) * (1.0 - holdout))
        train, test = ordered.iloc[:cut], ordered.iloc[cut:]

    block = (block_length_rows(train[ts_column], horizon_minutes)
             if horizon_minutes and not train.empty else None)
    block_test = (block_length_rows(test[ts_column], horizon_minutes)
                  if horizon_minutes and test is not None and not test.empty else None)

    results: list[TrendResult] = []
    for feature in features:
        if feature not in train.columns:
            continue
        result = _measure_one(train, feature, metric_column, bins)
        if result is None:
            continue
        if result.bins and block:
            rng = np.random.default_rng([seed, atom_seed(feature)])
            labels, _, _ = quantile_bins(train[feature].to_numpy(dtype=float), bins)
            result.p_boot = trend_bootstrap_p(
                labels, train[metric_column].to_numpy(dtype=float),
                block, n_boot=n_boot, rng=rng,
            )
        if result.bins and test is not None and not test.empty:
            result.holdout = _measure_one(test, feature, metric_column, bins)
            if block_test and result.holdout is not None and result.holdout.bins:
                rng = np.random.default_rng([seed + 1, atom_seed(feature)])
                labels, _, _ = quantile_bins(test[feature].to_numpy(dtype=float), bins)
                result.holdout.p_boot = trend_bootstrap_p(
                    labels, test[metric_column].to_numpy(dtype=float),
                    block_test, n_boot=n_boot, rng=rng,
                )
        results.append(result)

    testable = [r for r in results if r.bins]
    if testable:
        p_values = [r.effective_p for r in testable]
        if correction == "bonferroni":
            flags = bonferroni(p_values, alpha)
        elif correction == "bh":
            flags = benjamini_hochberg(p_values, alpha)
        else:
            flags = [p <= alpha for p in p_values]
        for result, flag in zip(testable, flags):
            result.significant = flag

    return sorted(results, key=lambda r: abs(r.z), reverse=True)


def format_table(results: list[TrendResult], correction: str = "bonferroni",
                 alpha: float = 0.05, n_boot: int | None = None) -> str:
    """Таблица: доли по бинам, разброс, z тренда, оба p-value, монотонность."""
    if not results:
        return "Нет признаков с достаточной градацией."

    testable = [r for r in results if r.bins]
    width = max((len(r.bins) for r in testable), default=0)
    bootstrapped = any(r.p_boot is not None for r in testable)

    header = f"{'признак':<24}" + "".join(f"{'бин ' + str(i + 1):>8}" for i in range(width))
    header += f" {'разброс':>9} {'z':>7} {'p наив.':>9}"
    if bootstrapped:
        header += f" {'p блок.':>9}"
    header += f" {'монот.':>7} {'знач.':>6} {'итог':>5}"

    lines = [header, "─" * len(header)]
    for r in results:
        if not r.bins:
            lines.append(f"{r.feature:<24} {r.degenerate}")
            continue
        row = f"{r.feature:<24}" + "".join(f"{b.p:>8.3f}" for b in r.bins)
        row += " " * (8 * (width - len(r.bins)))
        row += f" {r.spread:>+9.4f} {r.z:>7.2f} {r.p_value:>9.5f}"
        if bootstrapped:
            row += f" {'—':>9}" if r.p_boot is None else f" {r.p_boot:>9.5f}"
        monotone = "да" if r.monotone else f"нет({r.inversions})"
        row += (f" {monotone:>7} {'да' if r.significant else 'нет':>6} "
                f"{'ДА' if r.confirmed else '—':>5}")
        lines.append(row)
        if r.degenerate:
            lines.append(f"{'':<24} ⚠ {r.degenerate}")

    lines.append("")
    lines.append(f"Тестов: {len(testable)}. Поправка: {correction}, α = {alpha}")
    if bootstrapped and n_boot and not resolution_is_sufficient(
            n_boot, alpha, len(testable), correction):
        need = int(4 * len(testable) / alpha) if correction == "bonferroni" \
            else int(4 / alpha)
        lines.append(
            f"⚠ РЕПЛИК МАЛО: при {n_boot} минимальный достижимый p-value "
            f"{1 / (1 + n_boot):.5f} сопоставим с порогом поправки. "
            f"Нужно --n-boot ≥ {need}."
        )
    lines.append(
        "«итог» = значим И монотонен И направление удержалось на holdout. "
        "Монотонность обязательна: без неё это обычная разница двух групп."
    )
    if not bootstrapped:
        lines.append(
            "ВНИМАНИЕ: блочный бутстрап не считался — наивный p-value "
            "анти-консервативен (шапка btcproc/analysis/lift.py)."
        )
    return "\n".join(lines)
