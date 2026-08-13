"""
Валидация системы на отложенной части истории (Ш0 плана 2026-08-12).

Отвечает на один вопрос, который до сих пор ни разу не задавали целиком:
**реализуется ли исторический перекос на данных, которых модель не видела.**
Система утверждает «после такой конфигурации в 62% случаев было вверх»;
здесь это утверждение проверяется на holdout как есть, агрегатом по всем
кандидатам, а не по отдельному атому.

Функции чистые: на вход DataFrame кандидатов holdout'а, на выход числа.
В БД и в модель состояний ходит `scripts/validate_holdout.py`.

## Что считается и почему именно так

**Directional accuracy** — доля кандидатов, у которых `research_side` совпал
с фактическим исходом бара. Нулевая гипотеза — 0.5.

**Но 0.5 недостаточно.** Рынок за период holdout'а мог расти, и тогда
стратегия «всегда long» даёт accuracy заметно выше половины, ничего не зная.
Поэтому рядом считается ПАРНОЕ сравнение с этим бенчмарком: разница
попаданий системы и «всегда long» на тех же строках. Для long-кандидатов
разница нулевая по построению, то есть весь вклад дают short-кандидаты —
ровно те, где система утверждает нечто против дрейфа. Критерий плана
(«accuracy значимо выше 50%») этим не подменяется, а дополняется: пройденный
первый тест при проваленном втором означает, что система переоткрыла тренд
рынка, а не конфигурацию.

**Brier** — среднеквадратичная ошибка вероятностного прогноза
`long_outcome_share` против факта `is_up`. Рядом печатается Brier
константного прогноза (базовая частота holdout'а) и skill score относительно
него: сам по себе Brier не читается, читается только разница.

**Калибровка по децилям** — предсказанная доля против фактической. Идеал
диагональ. Отдельно считается ECE (взвешенное по объёму среднее отклонение).

## Зависимость наблюдений

Та же, что в `lift.py`, и по той же причине: горизонт 24h — это 96 баров,
кандидаты идут чаще, плюс снимки офсетов дают до четырёх строк на одну
реализацию перехода. Наивный биномиальный интервал вокруг accuracy занижен
примерно втрое. Поэтому значимость здесь считается **только** блочным
бутстрапом (инвариант 11), и длина блока берётся тем же
`block_length_rows` — общий код, а не копия: расхождение методик между двумя
замерами было бы худшим из возможных исходов.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from btcproc.analysis.lift import (
    CHUNK_CELLS,
    DEFAULT_N_BOOT,
    block_bootstrap_p,
    block_length_rows,
    stationary_block_indices,
)

#: Ниже этого числа наблюдений разрез не печатается как результат: блочный
#: бутстрап на двух десятках зависимых строк меряет собственный шум.
MIN_BUCKET_ROWS = 100


# ─── Разбиение истории ──────────────────────────────────────────────────────
def split_bar(index: pd.DatetimeIndex, frac: float,
              min_part: int = 1000) -> pd.Timestamp:
    """
    Бар, на котором заканчиваются первые `frac` истории.

    Граница берётся по числу БАРОВ, а не по календарю: история монеты
    неравномерна (у BTC ранние годы разрежены дырами), и календарная
    середина дала бы не ту долю данных, которую заявляем.

    Живёт здесь, а не в скрипте, ровно по одной причине: контрольная модель
    без графа (D1) обязана делить историю ТЕМ ЖЕ бором, что и валидация
    графа, иначе сравнивать их числа нельзя. Две реализации одной формулы
    разъехались бы молча.
    """
    total = len(index)
    cut = int(total * frac)
    if cut < min_part or total - cut < min_part:
        raise ValueError(
            f"разбиение {frac} на {total} барах даёт слишком короткую часть "
            f"({cut} / {total - cut})"
        )
    return index[cut]


# ─── Карты редкости без заглядывания вперёд ─────────────────────────────────
def prefix_maps(states: pd.DataFrame, events: pd.DataFrame,
                outcomes: pd.DataFrame, split_ts: pd.Timestamp
                ) -> tuple[dict[str, str], dict[str, dict]]:
    """
    Редкость переходов и статистика блоков, посчитанные ТОЛЬКО по данным до
    `split_ts`.

    В штатном прогоне обе величины считаются по всей истории — незакрытый
    look-ahead (Ш2.1 плана 2026-08-12). Внутри валидации на отложенной части
    его нельзя оставлять ни в каком виде: `transition_rarity` входит в
    `research_score` (вес 0.10) и отдельной осью в скоринг btc-graph, то есть
    кандидат holdout'а получал бы оценку, зависящую от того, чем этот holdout
    закончился.

    Блок или переход, впервые встретившийся после границы, в картах
    отсутствует. Вызывающий код обязан трактовать это как «на тот момент
    сведений не было» (`rarity='common'`, нулевая доля), а не как ошибку.
    """
    from btcproc.features import events as ev
    from btcproc.states import graph

    past_states = states[states.index < split_ts]
    past_events = events[events.index < split_ts]
    transitions = graph.transition_stats(
        past_states, outcomes.reindex(past_states.index)
    )
    blocks = ev.block_statistics(past_events)
    rarity_map = dict(zip(transitions["transition_id"], transitions["rarity"]))
    block_map = blocks.set_index("event_block_id").to_dict("index") \
        if not blocks.empty else {}
    return rarity_map, block_map


@dataclass
class Estimate:
    """Оценка доли с блочным доверительным интервалом и p-value к H0."""

    n: int
    value: float
    ci_low: float
    ci_high: float
    p_value: float
    null_value: float

    @property
    def significant(self) -> bool:
        return self.p_value <= 0.05

    def __str__(self) -> str:
        return (f"{self.value:.4f} [{self.ci_low:.4f}; {self.ci_high:.4f}] "
                f"p={self.p_value:.4f} n={self.n}")


@dataclass
class HoldoutReport:
    """Полный результат валидации одной монеты на одной модели."""

    symbol: str
    model_run: int
    split_ts: pd.Timestamp
    n_candidates: int
    n_valid: int

    accuracy: Estimate                      # H0: 0.5
    versus_always_long: Estimate            # H0: 0 (парная разница)
    base_rate: float                        # доля is_up на holdout
    brier: float
    brier_reference: float                  # прогноз = base_rate на всё
    calibration: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    ece: float = 0.0

    by_rating: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    by_sample_size: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    by_research_score: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    #: STRONG против WEAK: разница accuracy и её блочный p-value.
    strong_vs_weak: dict = field(default_factory=dict)

    @property
    def brier_skill(self) -> float:
        """1 − Brier / Brier_reference. Больше нуля — прогноз лучше константы."""
        if self.brier_reference <= 0:
            return 0.0
        return 1.0 - self.brier / self.brier_reference

    @property
    def passes(self) -> bool:
        """
        Критерий плана для ОДНОЙ монеты: accuracy значимо выше 50% и STRONG
        значимо лучше WEAK. Итоговый вердикт требует ещё двух монет из трёх и
        двух моделей — это считает вызывающий код, здесь про одну монету.
        """
        return bool(
            self.accuracy.significant
            and self.accuracy.value > 0.5
            and self.strong_vs_weak.get("significant")
            and self.strong_vs_weak.get("delta", 0.0) > 0
        )


# ─── Метрики ────────────────────────────────────────────────────────────────
def hit_flags(side: pd.Series, is_up: pd.Series) -> np.ndarray:
    """
    Попал ли кандидат: long → бар закрылся вверх, short → вниз.

    `is_up` приходит из `outcomes` и на невалидных строках равен None —
    такие строки вызывающий код обязан отсеять ДО этой функции, иначе None
    молча станет False, то есть «система ошиблась» вместо «факта нет».
    """
    up = is_up.to_numpy(dtype=bool)
    long_side = (side == "long").to_numpy(dtype=bool)
    return np.where(long_side, up, ~up).astype(float)


def brier(predicted_up: np.ndarray, is_up: np.ndarray) -> float:
    """Средний квадрат ошибки вероятностного прогноза «вверх»."""
    p = np.asarray(predicted_up, dtype=float)
    y = np.asarray(is_up, dtype=float)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def calibration_bins(predicted_up: np.ndarray, is_up: np.ndarray,
                     bins: int = 10) -> pd.DataFrame:
    """
    Калибровочная таблица по децилям предсказанной доли.

    Бины строятся по КВАНТИЛЯМ, а не по равным интервалам [0..1]: предсказания
    системы сосредоточены вокруг 0.5 (кандидат выпускается при перекосе от
    6%), и равные интервалы дали бы восемь пустых строк из десяти.

    Возвращает: bin, n, predicted (среднее предсказание), actual (фактическая
    доля up), gap = actual − predicted.
    """
    p = np.asarray(predicted_up, dtype=float)
    y = np.asarray(is_up, dtype=float)
    if p.size == 0:
        return pd.DataFrame(columns=["bin", "n", "predicted", "actual", "gap"])

    frame = pd.DataFrame({"p": p, "y": y})
    # duplicates="drop": при узком разбросе предсказаний часть границ квантилей
    # совпадает, и десять бинов честно схлопываются в меньшее число.
    frame["bin"] = pd.qcut(frame["p"], q=bins, labels=False, duplicates="drop")
    grouped = frame.groupby("bin").agg(
        n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")
    ).reset_index()
    grouped["gap"] = grouped["actual"] - grouped["predicted"]
    return grouped


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Средневзвешенное |actual − predicted| по бинам (ECE)."""
    if table.empty:
        return float("nan")
    weights = table["n"].to_numpy(dtype=float)
    return float(np.average(np.abs(table["gap"].to_numpy(dtype=float)),
                            weights=weights))


# ─── Значимость ─────────────────────────────────────────────────────────────
def block_means(values: np.ndarray, block_length: int,
                n_boot: int = DEFAULT_N_BOOT,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Средние по репликам стационарного блочного бутстрапа.

    Здесь пересобирается сам ряд значений (в отличие от `block_bootstrap_p`,
    где пересобирается признак при неподвижной метрике): нулевая гипотеза
    касается уровня одной величины, а не связи двух. Блоки геометрической
    длины сохраняют автокорреляцию ряда, из-за которой наивный SE занижен.
    """
    rng = rng or np.random.default_rng(42)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.array([])

    chunk = max(1, min(n_boot, CHUNK_CELLS // max(n, 1)))
    means: list[np.ndarray] = []
    done = 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        idx = stationary_block_indices(n, block_length, rng, size)
        means.append(values[idx].mean(axis=1))
        done += size
    return np.concatenate(means)


def estimate(values: np.ndarray, null_value: float, block_length: int,
             n_boot: int = DEFAULT_N_BOOT, alpha: float = 0.05,
             rng: np.random.Generator | None = None) -> Estimate:
    """
    Наблюдённое среднее, блочный перцентильный доверительный интервал и
    двусторонний p-value к гипотезе «среднее равно null_value».

    P-value считается инверсией интервала: доля реплик, отклонившихся от
    наблюдённого не меньше, чем наблюдённое отклонилось от нулевой гипотезы,
    с поправкой на единицу — `(1 + k) / (1 + B)`. Без поправки нулевой
    p-value читался бы как «достоверно абсолютно», хотя означает лишь
    «реплик не хватило».
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return Estimate(0, float("nan"), float("nan"), float("nan"), 1.0, null_value)

    observed = float(values.mean())
    means = block_means(values, block_length, n_boot=n_boot, rng=rng)
    if means.size == 0:
        return Estimate(n, observed, float("nan"), float("nan"), 1.0, null_value)

    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    deviation = abs(observed - null_value)
    extreme = int(np.sum(np.abs(means - observed) >= deviation))
    p_value = (1 + extreme) / (1 + means.size)
    return Estimate(n, observed, float(low), float(high), float(p_value), null_value)


def contrast(frame: pd.DataFrame, column: str, high: str, low: str,
             hit_column: str = "hit", ts_column: str = "ts",
             horizon_minutes: int = 1440, n_boot: int = DEFAULT_N_BOOT,
             rng: np.random.Generator | None = None) -> dict:
    """
    Различает ли категориальная ось хоть что-нибудь: accuracy в группе `high`
    против группы `low`.

    Двухвыборочная задача, поэтому берётся `block_bootstrap_p` из `lift.py`:
    там пересобирается признак при неподвижной метрике, что и есть нулевая
    гипотеза «связи между группой и попаданием нет».

    Значимым считается только УЛУЧШЕНИЕ: p-value двусторонний, поэтому знак
    проверяется отдельно. «STRONG значимо отличается от WEAK в худшую
    сторону» — это провал критерия, а не его прохождение.
    """
    subset = frame[frame[column].isin([high, low])]
    if subset.empty:
        return {"n_high": 0, "n_low": 0, "delta": 0.0, "p_value": 1.0,
                "significant": False}

    subset = subset.sort_values(ts_column)
    flags = (subset[column] == high).to_numpy(dtype=bool)
    metric = subset[hit_column].to_numpy(dtype=float)
    n_high, n_low = int(flags.sum()), int((~flags).sum())
    if n_high == 0 or n_low == 0:
        return {"n_high": n_high, "n_low": n_low, "delta": 0.0, "p_value": 1.0,
                "significant": False}

    acc_high = float(metric[flags].mean())
    acc_low = float(metric[~flags].mean())
    block = block_length_rows(subset[ts_column], horizon_minutes)
    p_value = block_bootstrap_p(flags, metric, block, n_boot=n_boot, rng=rng)
    delta = acc_high - acc_low
    return {
        "n_high": n_high, "n_low": n_low,
        "acc_high": acc_high, "acc_low": acc_low,
        "delta": delta, "p_value": float(p_value),
        "significant": bool(p_value <= 0.05 and delta > 0),
        "block": block,
    }


def by_bucket(frame: pd.DataFrame, column: str, hit_column: str = "hit",
              ts_column: str = "ts", horizon_minutes: int = 1440,
              n_boot: int = DEFAULT_N_BOOT, order: list[str] | None = None,
              rng: np.random.Generator | None = None) -> pd.DataFrame:
    """
    Accuracy в разрезе категориальной колонки, с блочным p-value к 0.5 внутри
    каждой группы.

    Длина блока считается ВНУТРИ группы: у редкой группы кандидаты разрежены,
    и блок, посчитанный по всей выборке, был бы для неё бессмысленно длинным.
    """
    rows = []
    values = order or sorted(frame[column].dropna().unique())
    for value in values:
        subset = frame[frame[column] == value].sort_values(ts_column)
        if subset.empty:
            continue
        hits = subset[hit_column].to_numpy(dtype=float)
        block = block_length_rows(subset[ts_column], horizon_minutes)
        est = estimate(hits, 0.5, block, n_boot=n_boot, rng=rng)
        rows.append({
            "bucket": value, "n": est.n, "accuracy": est.value,
            "ci_low": est.ci_low, "ci_high": est.ci_high,
            "p_value": est.p_value,
            "significant": est.significant and est.value > 0.5,
            "thin": est.n < MIN_BUCKET_ROWS,
        })
    return pd.DataFrame(rows)


def quantile_buckets(values: pd.Series, labels: tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4")
                     ) -> pd.Series:
    """
    Квартили непрерывной величины как категории.

    Квантили, а не фиксированные пороги: `sample_size` растёт с историей и у
    трёх монет разный на порядок, фиксированные границы дали бы у SOL пустые
    верхние группы.
    """
    try:
        return pd.qcut(values, q=len(labels), labels=list(labels), duplicates="drop")
    except ValueError:
        # Вырожденное распределение (все значения равны) — квартилей нет.
        return pd.Series(["Q1"] * len(values), index=values.index)


# ─── Сборка ─────────────────────────────────────────────────────────────────
def measure(frame: pd.DataFrame, symbol: str, model_run: int,
            split_ts: pd.Timestamp, horizon_minutes: int,
            n_candidates: int | None = None,
            n_boot: int = DEFAULT_N_BOOT, seed: int = 42) -> HoldoutReport:
    """
    Полный отчёт по holdout-выборке одной монеты.

    Ожидаемые колонки frame: ts, side, is_up (bool, только валидные строки),
    long_outcome_share, rating, sample_size, research_score.

    `n_candidates` — сколько кандидатов было ДО отсева по созревшим исходам;
    нужно только для отчёта, метрики считаются по frame.
    """
    frame = frame.sort_values("ts").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    block = block_length_rows(frame["ts"], horizon_minutes) if len(frame) else 2

    hits = frame["hit"].to_numpy(dtype=float)
    up = frame["is_up"].to_numpy(dtype=float)
    predicted = frame["long_outcome_share"].to_numpy(dtype=float)

    accuracy = estimate(hits, 0.5, block, n_boot=n_boot, rng=rng)
    # Парная разница с бенчмарком «всегда long»: на long-кандидатах ноль по
    # построению, весь сигнал дают short'ы.
    versus = estimate(hits - up, 0.0, block, n_boot=n_boot, rng=rng)

    base_rate = float(up.mean()) if up.size else float("nan")
    table = calibration_bins(predicted, up)

    report = HoldoutReport(
        symbol=symbol,
        model_run=model_run,
        split_ts=split_ts,
        n_candidates=n_candidates if n_candidates is not None else len(frame),
        n_valid=len(frame),
        accuracy=accuracy,
        versus_always_long=versus,
        base_rate=base_rate,
        brier=brier(predicted, up),
        brier_reference=brier(np.full_like(up, base_rate), up),
        calibration=table,
        ece=expected_calibration_error(table),
    )

    common = dict(hit_column="hit", ts_column="ts", horizon_minutes=horizon_minutes,
                  n_boot=n_boot, rng=rng)
    if "rating" in frame:
        report.by_rating = by_bucket(frame, "rating", order=["STRONG", "MODERATE", "WEAK"],
                                     **common)
        report.strong_vs_weak = contrast(frame, "rating", "STRONG", "WEAK", **common)

    work = frame.copy()
    work["sample_bucket"] = quantile_buckets(work["sample_size"])
    work["score_bucket"] = quantile_buckets(work["research_score"])
    report.by_sample_size = by_bucket(work, "sample_bucket", **common)
    report.by_research_score = by_bucket(work, "score_bucket", **common)
    return report


# ─── Печать ─────────────────────────────────────────────────────────────────
def format_report(report: HoldoutReport) -> str:
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add(f"HOLDOUT {report.symbol} — модель прогона #{report.model_run}, "
        f"граница {report.split_ts:%Y-%m-%d}")
    add("=" * 78)
    add(f"Кандидатов на holdout: {report.n_candidates}, "
        f"с созревшим исходом: {report.n_valid}")
    add("")
    add("ОСНОВНОЕ")
    add(f"  directional accuracy   {report.accuracy}")
    add(f"  «всегда long» даёт     {report.base_rate:.4f} "
        f"(парная разница {report.versus_always_long})")
    add(f"  Brier                  {report.brier:.4f} "
        f"против константы {report.brier_reference:.4f} "
        f"(skill {report.brier_skill:+.4f})")
    add(f"  ECE калибровки         {report.ece:.4f}")
    add("")

    add("КАЛИБРОВКА ПО ДЕЦИЛЯМ long_outcome_share")
    add(f"  {'дециль':>7} {'n':>7} {'предсказано':>12} {'факт':>8} {'разрыв':>8}")
    for _, row in report.calibration.iterrows():
        add(f"  {int(row['bin']):>7} {int(row['n']):>7} {row['predicted']:>12.4f} "
            f"{row['actual']:>8.4f} {row['gap']:>+8.4f}")
    add("")

    for title, table in (
        ("РАЗРЕЗ ПО RATING (btc-graph)", report.by_rating),
        ("РАЗРЕЗ ПО sample_size (квартили)", report.by_sample_size),
        ("РАЗРЕЗ ПО research_score (квартили)", report.by_research_score),
    ):
        if table.empty:
            continue
        add(title)
        add(f"  {'группа':<10} {'n':>7} {'accuracy':>9} {'95% ДИ':>19} "
            f"{'p':>8} {'знач.':>6}")
        for _, row in table.iterrows():
            mark = "да" if row["significant"] else "нет"
            thin = "  (мало данных)" if row["thin"] else ""
            add(f"  {str(row['bucket']):<10} {int(row['n']):>7} "
                f"{row['accuracy']:>9.4f} "
                f"[{row['ci_low']:.4f}; {row['ci_high']:.4f}] "
                f"{row['p_value']:>8.4f} {mark:>6}{thin}")
        add("")

    if report.strong_vs_weak:
        c = report.strong_vs_weak
        add("STRONG ПРОТИВ WEAK")
        if c.get("n_high") and c.get("n_low"):
            add(f"  STRONG {c['acc_high']:.4f} (n={c['n_high']}) − "
                f"WEAK {c['acc_low']:.4f} (n={c['n_low']}) = {c['delta']:+.4f}, "
                f"p={c['p_value']:.4f} → "
                f"{'различает' if c['significant'] else 'НЕ различает'}")
        else:
            add(f"  недостаточно данных: STRONG {c.get('n_high', 0)}, "
                f"WEAK {c.get('n_low', 0)}")
        add("")

    add(f"ИТОГ ПО МОНЕТЕ: {'критерий пройден' if report.passes else 'критерий НЕ пройден'}")
    add("  (accuracy значимо > 50% И STRONG значимо лучше WEAK)")
    return "\n".join(lines)


def format_verdict(reports: list[HoldoutReport]) -> str:
    """
    Сводный вердикт по монетам — вторая половина критерия плана.

    Требование «две модели состояний» здесь не проверяется: это второй запуск
    скрипта с другой границей или другим зерном, и сравнивать их должен
    человек, глядя на два отчёта. Автоматическая проверка создала бы
    впечатление, что красная линия закрыта одним прогоном.
    """
    if not reports:
        return "Нет отчётов."
    passed = [r for r in reports if r.passes]
    lines = ["", "=" * 78, "СВОДКА ПО МОНЕТАМ", "=" * 78]
    header = (f"{'монета':<10} {'n':>7} {'accuracy':>9} {'p':>8} "
              f"{'vs long':>9} {'STRONG−WEAK':>12} {'p':>8} {'итог':>6}")
    lines.append(header)
    lines.append("─" * len(header))
    for r in reports:
        c = r.strong_vs_weak or {}
        lines.append(
            f"{r.symbol:<10} {r.n_valid:>7} {r.accuracy.value:>9.4f} "
            f"{r.accuracy.p_value:>8.4f} {r.versus_always_long.value:>+9.4f} "
            f"{c.get('delta', float('nan')):>+12.4f} "
            f"{c.get('p_value', float('nan')):>8.4f} "
            f"{'ДА' if r.passes else '—':>6}"
        )
    lines.append("")
    lines.append(
        f"Прошло монет: {len(passed)} из {len(reports)}. Критерий плана — "
        f"две из трёх, и это же должно повториться на второй модели состояний."
    )
    return "\n".join(lines)
