"""
Контрольная модель без графа состояний — задача D1 внешнего аудита.

## Вопрос, на который отвечает

Ни один из существующих замеров не спрашивал: **дело в графе состояний или в
самих признаках?** Граф — это дискретизация 32-мерного вектора в несколько
десятков состояний, а дискретизация всегда теряет информацию. Раздел 26
журнала показал, что система не воспроизводит своё утверждение на отложенной
части. Из этого не следует, где потолок.

Здесь тот же вопрос задаётся напрямую: градиентный бустинг на ТЕХ ЖЕ
признаках, с ТОЙ ЖЕ целевой, на ТОМ ЖЕ разбиении истории. Интерпретация
объявлена заранее (таблица из ТЗ):

| GBM на признаках | Граф состояний | Вывод |
|---|---|---|
| нет сигнала | нет сигнала | потолок в признаках; граф ни при чём, доводить его бессмысленно — менять источники данных |
| есть сигнал | нет сигнала | потеря на дискретизации; граф — узкое место, менять представление |
| нет сигнала | есть сигнал | почти невозможно; если так — искать протечку в графе |

## Критерий, заявленный ДО запуска (инвариант 5)

> **«Сигнал есть»** означает: на отложенной части directional accuracy
> значимо выше 0.5 по блочному бутстрапу **И** парная разница с бенчмарком
> «всегда long» значимо положительна, **И** Brier skill относительно
> константного прогноза положителен, — на двух монетах из трёх и на двух
> разных зёрнах бустинга (инвариант 10).
>
> Дополнительно, но не как часть критерия: различает ли модель собственную
> уверенность — accuracy в верхнем квартиле |p − 0.5| против нижнего. Это
> прямой аналог вопроса «STRONG против WEAK», на котором граф провалился
> шесть раз из шести.

Слабее этого критерий делать нельзя: ровно на нём проверялся граф, и
контрольный замер обязан играть по тем же правилам.

## Как устроена честность замера

**Purged walk-forward CV.** Между обучающим и проверочным окном каждого фолда
выбрасывается `horizon_bars` строк с обеих сторон. Без этого зазора последние
бары обучения имеют окна исходов, накрывающие начало проверки: метка «вверх»
для бара t считается по барам t+1…t+96, и без пропуска модель училась бы на
исходах, которые она же потом предсказывает. Это самая частая ошибка в
бэктестах на перекрывающихся метках, и она даёт впечатляющие и ложные числа.

**Нормировка признаков не нужна вовсе.** Деревьям безразличен масштаб,
поэтому у контрольной модели нет look-ahead'а `robust_scale_params`, который
остаётся в боевом расчёте графа (A7 аудита). Контроль в этом смысле ЧИЩЕ
проверяемой системы — и это правильная сторона для ошибки: если сигнала нет
даже у более честной модели, вывод крепче.

**Калибровка изотонической регрессией** на хвосте обучающей части, отрезанном
тем же зазором. Сырые вероятности бустинга смещены, а Brier и калибровочная
таблица без этого мерили бы смещение, а не информативность.

**Значимость — только блочным бутстрапом**, общим кодом с `lift.py` и
`holdout.py` (инвариант 11, копировать нельзя). Наблюдения зависимы: горизонт
24h — это 96 баров, и соседние строки почти дублируют друг друга. Здесь это
даже острее, чем у кандидатов: строка на КАЖДОМ баре, а не раз в ~11.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from btcproc.analysis import holdout as ho
from btcproc.analysis.lift import DEFAULT_N_BOOT, block_length_rows

logger = logging.getLogger(__name__)

#: Сколько фолдов в purged walk-forward CV на обучающей части.
DEFAULT_FOLDS = 5

#: Какая доля обучающей части уходит под изотоническую калибровку.
CALIBRATION_TAIL = 0.2


@dataclass
class Fold:
    """Один фолд walk-forward: где учились, где проверялись, что вышло."""

    index: int
    n_train: int
    n_test: int
    accuracy: float
    auc: float
    base_rate: float
    brier: float
    brier_reference: float

    @property
    def brier_skill(self) -> float:
        if self.brier_reference <= 0:
            return 0.0
        return 1.0 - self.brier / self.brier_reference


@dataclass
class ControlReport:
    """Результат контрольной модели по одной монете и одному зерну."""

    symbol: str
    seed: int
    split_ts: pd.Timestamp
    n_features: int
    n_train: int
    n_holdout: int

    folds: list[Fold] = field(default_factory=list)

    accuracy: ho.Estimate | None = None          # H0: 0.5
    versus_always_long: ho.Estimate | None = None  # H0: 0
    #: Ранжирование по сырым скорам: AUC и accuracy по порогу 0.5 до калибровки.
    auc: float = float("nan")
    accuracy_raw: ho.Estimate | None = None
    #: Схлопнула ли изотоника прогноз в константу (см. fit_predict).
    calibration_collapsed: bool = False
    base_rate: float = float("nan")
    brier: float = float("nan")
    brier_reference: float = float("nan")
    calibration: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    ece: float = 0.0
    by_confidence: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    confident_vs_unsure: dict = field(default_factory=dict)
    feature_importance: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    @property
    def brier_skill(self) -> float:
        if self.brier_reference <= 0:
            return 0.0
        return 1.0 - self.brier / self.brier_reference

    @property
    def passes(self) -> bool:
        """Критерий из шапки модуля, часть про одну монету и одно зерно."""
        if self.accuracy is None or self.versus_always_long is None:
            return False
        return bool(
            self.accuracy.significant and self.accuracy.value > 0.5
            and self.versus_always_long.significant
            and self.versus_always_long.value > 0
            and self.brier_skill > 0
        )


def roc_auc(scores: np.ndarray, actual: np.ndarray) -> float:
    """
    AUC по СЫРЫМ скорам — различающая способность в чистом виде.

    Нужна отдельно от accuracy по двум причинам. Во-первых, accuracy зависит
    от порога 0.5, а он у некалиброванной модели произволен; AUC от порога не
    зависит вовсе. Во-вторых, при вырожденной калибровке accuracy становится
    равной базовой частоте («всегда long»), и по ней уже не отличить «модель
    ничего не нашла» от «нашла, но сместилась». AUC отвечает на этот вопрос
    прямо: 0.5 — ранжирование не лучше монетки.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(actual, dtype=float)
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores, dtype=float)))


# ─── Разбиение ──────────────────────────────────────────────────────────────
def purged_splits(n: int, folds: int, gap: int) -> list[tuple[slice, slice]]:
    """
    Расширяющееся окно с зазором: [0, a) учим, [a + gap, b) проверяем.

    Зазор выбрасывается ТОЛЬКО между окнами и равен горизонту в барах. Метка
    бара t считается по барам t+1…t+gap, поэтому последние `gap` строк
    обучения «видят» начало проверочного окна. Без пропуска это протечка, и
    она не выглядит ошибкой: числа просто становятся лучше.

    Возвращает список пар срезов; фолды с пустым проверочным окном
    отбрасываются — на короткой истории их может не набраться пять.
    """
    if folds < 2 or n <= gap * (folds + 2):
        return []

    # Первое обучающее окно — половина истории: на четверти данных бустинг
    # меряет не рынок, а собственную недоученность.
    start = n // 2
    edges = np.linspace(start, n, folds + 1, dtype=int)
    result = []
    for i in range(folds):
        train_end = int(edges[i])
        test_start = train_end + gap
        test_end = int(edges[i + 1])
        if test_end - test_start < gap:
            continue
        result.append((slice(0, train_end), slice(test_start, test_end)))
    return result


# ─── Модель ─────────────────────────────────────────────────────────────────
def _make_model(seed: int):
    """
    Бустинг с параметрами по умолчанию, кроме глубины и регуляризации.

    Подбирать гиперпараметры под метрику нельзя: это ровно тот способ найти
    эффект, которого нет (раздел 8 ТЗ — «не подбирать пороги под красивую
    accuracy»). Значения ниже — консервативные типовые, объявленные заранее.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=False,   # ранняя остановка по случайному срезу — протечка
        random_state=seed,
    )


def fit_predict(features: pd.DataFrame, target: np.ndarray,
                train: slice, test: slice, seed: int,
                gap: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Учит модель на `train`; возвращает пару (сырые, калиброванные) на `test`.

    Калибровка изотонической регрессией на хвосте обучающей части: последние
    `CALIBRATION_TAIL` строк, отделённые от остатка тем же зазором. Модель
    учится на голове, изотоника — на хвосте, и ни одна не видит проверочного
    окна.

    **Возвращаются оба вектора, и это не удобство, а необходимость.**
    Изотоника монотонна и ранга обычно не меняет, но при отсутствии
    монотонной связи на калибровочном хвосте она схлопывает прогноз в
    КОНСТАНТУ. Прогноз-константа не ошибка калибровки — это её честный ответ
    «связи нет», — но по нему уже нельзя отличить «модель не различает» от
    «различает, но смещена»: все стороны становятся одной, а разрез по
    уверенности вырождается в одну группу. Поэтому калиброванные числа идут в
    Brier и в калибровочную таблицу, а различающая способность (AUC, разрез
    по уверенности) считается по сырым.
    """
    from sklearn.isotonic import IsotonicRegression

    x = features.to_numpy(dtype=float)
    x_train, y_train = x[train], target[train]

    n = len(y_train)
    tail = int(n * CALIBRATION_TAIL)
    head_end = max(1, n - tail - gap)
    model = _make_model(seed)
    model.fit(x_train[:head_end], y_train[:head_end])

    raw = model.predict_proba(x[test])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(model.predict_proba(x_train[n - tail:])[:, 1], y_train[n - tail:])
    return raw, iso.predict(raw)


def permutation_importance(features: pd.DataFrame, target: np.ndarray,
                           train: slice, test: slice, seed: int,
                           top: int = 10) -> pd.DataFrame:
    """
    Насколько падает accuracy, если перемешать один признак.

    Не часть критерия: важность признаков осмысленна, только если модель
    вообще что-то нашла. Печатается для разбора — если сигнала нет, таблица
    покажет шум вокруг нуля, и это тоже информация.
    """
    x = features.to_numpy(dtype=float)
    model = _make_model(seed)
    model.fit(x[train], target[train])

    x_test, y_test = x[test].copy(), target[test]
    base = float(((model.predict_proba(x_test)[:, 1] > 0.5) == y_test).mean())

    rng = np.random.default_rng(seed)
    rows = []
    for i, name in enumerate(features.columns):
        saved = x_test[:, i].copy()
        x_test[:, i] = rng.permutation(saved)
        shuffled = float(((model.predict_proba(x_test)[:, 1] > 0.5) == y_test).mean())
        x_test[:, i] = saved
        rows.append({"feature": name, "drop": base - shuffled})
    frame = pd.DataFrame(rows).sort_values("drop", ascending=False)
    frame.attrs["baseline_accuracy"] = base
    return frame.head(top).reset_index(drop=True)


# ─── Замер ──────────────────────────────────────────────────────────────────
def cross_validate(features: pd.DataFrame, target: np.ndarray, gap: int,
                   folds: int, seed: int) -> list[Fold]:
    """Purged walk-forward на обучающей части — честная оценка «внутри train»."""
    result = []
    for i, (train, test) in enumerate(purged_splits(len(target), folds, gap), start=1):
        raw, predicted = fit_predict(features, target, train, test, seed, gap=gap)
        actual = target[test].astype(float)
        base_rate = float(actual.mean())
        result.append(Fold(
            index=i,
            n_train=train.stop - train.start,
            n_test=test.stop - test.start,
            accuracy=float(((predicted > 0.5) == actual.astype(bool)).mean()),
            auc=roc_auc(raw, actual),
            base_rate=base_rate,
            brier=ho.brier(predicted, actual),
            brier_reference=ho.brier(np.full_like(actual, base_rate), actual),
        ))
    return result


def measure(ts: pd.Series, raw: np.ndarray, predicted: np.ndarray,
            actual: np.ndarray, symbol: str, seed: int, split_ts: pd.Timestamp,
            horizon_minutes: int, n_features: int, n_train: int,
            n_boot: int = DEFAULT_N_BOOT) -> ControlReport:
    """
    Метрики на отложенной части — те же, что у валидации графа.

    `predicted` — калиброванная вероятность «вверх», `raw` — она же до
    изотоники, `actual` — факт. Сторона модели: long при p > 0.5, иначе short.
    Это ровно то, что делает кандидат со своим `research_side`, поэтому
    accuracy сравнима с валидацией графа напрямую.

    Разрез по уверенности и AUC считаются по СЫРЫМ скорам: изотоника
    монотонна, ранга не меняет, а при вырождении обнуляет его вовсе — и разрез
    по калиброванным превратился бы в одну группу (см. fit_predict).
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"ts": pd.to_datetime(ts), "p": predicted, "raw": raw,
                          "is_up": actual.astype(bool)}).sort_values("ts")
    frame = frame.reset_index(drop=True)
    frame["side"] = np.where(frame["p"] > 0.5, "long", "short")
    frame["hit"] = ho.hit_flags(frame["side"], frame["is_up"])
    frame["side_raw"] = np.where(frame["raw"] > 0.5, "long", "short")
    frame["hit_raw"] = ho.hit_flags(frame["side_raw"], frame["is_up"])

    block = block_length_rows(frame["ts"], horizon_minutes) if len(frame) else 2
    hits = frame["hit"].to_numpy(dtype=float)
    up = frame["is_up"].to_numpy(dtype=float)
    p = frame["p"].to_numpy(dtype=float)

    base_rate = float(up.mean()) if up.size else float("nan")
    table = ho.calibration_bins(p, up)

    report = ControlReport(
        symbol=symbol, seed=seed, split_ts=split_ts,
        n_features=n_features, n_train=n_train, n_holdout=len(frame),
        accuracy=ho.estimate(hits, 0.5, block, n_boot=n_boot, rng=rng),
        versus_always_long=ho.estimate(hits - up, 0.0, block, n_boot=n_boot, rng=rng),
        auc=roc_auc(frame["raw"].to_numpy(), up),
        accuracy_raw=ho.estimate(frame["hit_raw"].to_numpy(dtype=float), 0.5,
                                 block, n_boot=n_boot, rng=rng),
        calibration_collapsed=bool(np.ptp(p) < 1e-9),
        base_rate=base_rate,
        brier=ho.brier(p, up),
        brier_reference=ho.brier(np.full_like(up, base_rate), up),
        calibration=table,
        ece=ho.expected_calibration_error(table),
    )

    # Аналог «STRONG против WEAK»: различает ли модель собственную уверенность.
    frame["confidence"] = (frame["raw"] - 0.5).abs()
    frame["confidence_bucket"] = ho.quantile_buckets(frame["confidence"])
    common = dict(hit_column="hit", ts_column="ts",
                  horizon_minutes=horizon_minutes, n_boot=n_boot, rng=rng)
    report.by_confidence = ho.by_bucket(frame, "confidence_bucket", **common)
    report.confident_vs_unsure = ho.contrast(frame, "confidence_bucket", "Q4", "Q1",
                                             **common)
    return report


# ─── Печать ─────────────────────────────────────────────────────────────────
def format_report(report: ControlReport) -> str:
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add(f"КОНТРОЛЬНАЯ МОДЕЛЬ БЕЗ ГРАФА — {report.symbol}, зерно {report.seed}")
    add("=" * 78)
    add(f"Признаков {report.n_features}, обучающая часть {report.n_train} баров, "
        f"holdout {report.n_holdout} (граница {report.split_ts:%Y-%m-%d})")
    add("")

    if report.folds:
        add("PURGED WALK-FORWARD НА ОБУЧАЮЩЕЙ ЧАСТИ")
        add(f"  {'фолд':>5} {'train':>9} {'test':>8} {'accuracy':>9} {'AUC':>7} "
            f"{'база':>7} {'Brier skill':>12}")
        for fold in report.folds:
            add(f"  {fold.index:>5} {fold.n_train:>9} {fold.n_test:>8} "
                f"{fold.accuracy:>9.4f} {fold.auc:>7.4f} {fold.base_rate:>7.4f} "
                f"{fold.brier_skill:>+12.4f}")
        mean_acc = float(np.mean([f.accuracy for f in report.folds]))
        mean_auc = float(np.nanmean([f.auc for f in report.folds]))
        add(f"  средние по фолдам: accuracy {mean_acc:.4f}, AUC {mean_auc:.4f}")
        add("")

    add("НА ОТЛОЖЕННОЙ ЧАСТИ")
    add(f"  directional accuracy   {report.accuracy}")
    add(f"  «всегда long» даёт     {report.base_rate:.4f} "
        f"(парная разница {report.versus_always_long})")
    add(f"  Brier                  {report.brier:.4f} "
        f"против константы {report.brier_reference:.4f} "
        f"(skill {report.brier_skill:+.4f})")
    add(f"  ECE калибровки         {report.ece:.4f}")
    add(f"  AUC по сырым скорам    {report.auc:.4f} "
        f"(0.5 — ранжирование не лучше монетки)")
    if report.accuracy_raw is not None:
        add(f"  accuracy до калибровки {report.accuracy_raw}")
    if report.calibration_collapsed:
        add("  ВНИМАНИЕ: изотоника схлопнула прогноз в константу — монотонной "
            "связи")
        add("  между сырым скором и исходом на калибровочном хвосте нет. "
            "Сторона у всех")
        add("  строк одна, поэтому accuracy равна базовой частоте, а разница с "
            "«всегда")
        add("  long» — нулю по построению. Различающую способность смотреть по AUC.")
    add("")

    if not report.calibration.empty:
        add("КАЛИБРОВКА ПО ДЕЦИЛЯМ ПРЕДСКАЗАННОЙ ВЕРОЯТНОСТИ")
        add(f"  {'дециль':>7} {'n':>8} {'предсказано':>12} {'факт':>8} {'разрыв':>8}")
        for _, row in report.calibration.iterrows():
            add(f"  {int(row['bin']):>7} {int(row['n']):>8} "
                f"{row['predicted']:>12.4f} {row['actual']:>8.4f} "
                f"{row['gap']:>+8.4f}")
        add("")

    if not report.by_confidence.empty:
        add("РАЗРЕЗ ПО УВЕРЕННОСТИ МОДЕЛИ |p − 0.5| (квартили)")
        add(f"  {'группа':<8} {'n':>8} {'accuracy':>9} {'95% ДИ':>19} {'p':>8}")
        for _, row in report.by_confidence.iterrows():
            add(f"  {str(row['bucket']):<8} {int(row['n']):>8} "
                f"{row['accuracy']:>9.4f} "
                f"[{row['ci_low']:.4f}; {row['ci_high']:.4f}] "
                f"{row['p_value']:>8.4f}")
        add("")

    if report.confident_vs_unsure:
        c = report.confident_vs_unsure
        add("УВЕРЕННЫЕ ПРОТИВ НЕУВЕРЕННЫХ (аналог STRONG против WEAK)")
        if c.get("n_high") and c.get("n_low"):
            add(f"  Q4 {c['acc_high']:.4f} (n={c['n_high']}) − "
                f"Q1 {c['acc_low']:.4f} (n={c['n_low']}) = {c['delta']:+.4f}, "
                f"p={c['p_value']:.4f} → "
                f"{'различает' if c['significant'] else 'НЕ различает'}")
        add("")

    if not report.feature_importance.empty:
        base = report.feature_importance.attrs.get("baseline_accuracy")
        add("ВАЖНОСТЬ ПРИЗНАКОВ (падение accuracy при перемешивании)")
        if base is not None:
            add(f"  базовая accuracy модели на этой части: {base:.4f}")
        for _, row in report.feature_importance.iterrows():
            add(f"  {row['feature']:<28} {row['drop']:>+8.4f}")
        add("")

    add(f"ИТОГ: {'критерий пройден' if report.passes else 'критерий НЕ пройден'}")
    add("  (accuracy значимо > 0.5 И разница с «всегда long» значимо > 0 "
        "И Brier skill > 0)")
    return "\n".join(lines)


def format_verdict(reports: list[ControlReport]) -> str:
    """Сводка по всем монетам и зёрнам — вторая половина критерия."""
    if not reports:
        return "Нет отчётов."

    lines = ["", "=" * 78, "СВОДКА: КОНТРОЛЬНАЯ МОДЕЛЬ БЕЗ ГРАФА", "=" * 78]
    header = (f"{'монета':<10} {'зерно':>6} {'n':>8} {'accuracy':>9} {'p':>8} "
              f"{'vs long':>9} {'p':>8} {'skill':>8} {'AUC':>7} {'итог':>6}")
    lines.append(header)
    lines.append("─" * len(header))
    for r in reports:
        lines.append(
            f"{r.symbol:<10} {r.seed:>6} {r.accuracy.n:>8} "
            f"{r.accuracy.value:>9.4f} {r.accuracy.p_value:>8.4f} "
            f"{r.versus_always_long.value:>+9.4f} "
            f"{r.versus_always_long.p_value:>8.4f} "
            f"{r.brier_skill:>+8.4f} {r.auc:>7.4f} "
            f"{'да' if r.passes else 'нет':>6}"
        )

    symbols_passed = {r.symbol for r in reports if r.passes}
    seeds = {r.seed for r in reports}
    lines.append("")
    lines.append(f"Монет прошло критерий: {len(symbols_passed)} "
                 f"из {len({r.symbol for r in reports})}; зёрен в замере: {len(seeds)}")
    verdict = (
        "СИГНАЛ ЕСТЬ" if len(symbols_passed) >= 2 and len(seeds) >= 2
        and all(
            any(r.passes for r in reports if r.symbol == s and r.seed == seed)
            for s in symbols_passed for seed in seeds
        )
        else "СИГНАЛА НЕТ"
    )
    lines.append(f"ВЕРДИКТ ПО ЗАЯВЛЕННОМУ КРИТЕРИЮ: {verdict}")
    lines.append("")
    lines.append("Читать вместе с валидацией графа (раздел 26 журнала):")
    lines.append("  нет сигнала здесь и нет там → потолок в ПРИЗНАКАХ, граф ни при чём;")
    lines.append("  есть здесь и нет там       → потеря на дискретизации, узкое место — граф;")
    lines.append("  нет здесь и есть там       → искать протечку в графе.")
    return "\n".join(lines)
