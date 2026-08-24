"""
Квантильный регрессор размаха на признаках.

## Зачем он и чем отличается от всего, что было

Раздел 47 журнала измерил две вещи, которые вместе задают эту конструкцию:

* **признаки предсказывают размах** — 32 признака дают +0.03…+0.10
  out-of-sample R² сверх тривиального бенчмарка «липкая волатильность плюс
  время суток» (47.6);
* **разметка графа этого почти не сохраняет** — на 4h `group_id` и
  `event_block_id` дают 0–10% того же приращения (47.8), а перцентиль
  конфигурации проиграл бенчмарку во всех 36 ячейках замера C3 (47.10).

Отсюда конструкция: считать размах **прямо из вектора признаков**, минуя
дискретизацию в состояния. Это не «второй тип кандидата» из блока C аудита —
там предсказание собиралось из исторической выборки по ключу конфигурации, и
именно это измерено как неработающее.

## Что на входе и что на выходе

Вход — матрица из двух частей, и обе обязательны:

* **бенчмарк B2** (HAR-RV на окнах 96/672/2880 плюс sin/cos часа и флаг
  выходных). Он здесь не для сравнения, а как ЧАСТЬ модели: без него бустинг
  тратил бы деревья на переоткрытие кластеризации волатильности и формы суток;
* **32 признака** `features/builder.py` — тот самый вектор, что кормит
  кластеризацию, но здесь он идёт в модель целиком, без потери на
  дискретизации.

Выход — не одно число, а четыре квантиля `range_ratio` (p25/p50/p75/p90) плюс
две относительные величины. Квантили, а не среднее, по двум причинам.
Во-первых, распределение размаха скошено вправо (у BTC p50 = 1.10, p90 = 2.28
при p25 = 0.79), и среднее в нём не описывает ни один типичный случай.
Во-вторых, вопрос, ради которого всё делалось, звучит как «насколько широко
может сходить цена», а это вопрос про хвост, а не про центр.

Модель учится на `log(range_ratio)`, а квантили возвращаются в исходной шкале
простым экспонированием: квантиль монотонного преобразования — это
преобразование квантиля, поэтому здесь нет ни приближения, ни поправки.

## Почему ответ обязан быть относительным

`range_lift` = p50 модели / p50 бенчмарка, посчитанного ТЕМ ЖЕ классом модели
на ТОЙ ЖЕ выборке, но только по колонкам B2. Это прямой урок раздела 26.4 и
47.10: абсолютное «ожидаемый размах 3.2%» неотличимо от «сейчас волатильно и
сейчас американская сессия», и ровно на этом C3 провалился. Относительная
величина отвечает на единственный вопрос, на который у системы есть право
отвечать: **шире или уже, чем следует из времени суток и недавней
волатильности.**

Бенчмарк обучается тем же квантильным бустингом, а не линейной регрессией,
намеренно: сравнивать надо наборы признаков, а не классы моделей. Иначе
«модель лучше» означало бы «бустинг гибче МНК», что известно и без замера.

## Чего этот модуль НЕ делает

Не пишет в БД, не трогает схему кандидата и не участвует в `train`/`live`.
Он живёт в `analysis/` по тому же праву, что и `control.py`: это модель,
существующая ради замера, пока не измерено, что её стоит вносить в конвейер.
Порог для такого решения объявлен в шапке `scripts/fit_range_forecast.py`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from btcproc.analysis import range_model as rm
from btcproc.features import indicators as ind

logger = logging.getLogger(__name__)

#: Квантили, которые считает модель. Те же четыре, что перечислял блок C
#: аудита для `Accumulator`, — чтобы новую конструкцию можно было сравнивать
#: со старой напрямую.
QUANTILES: tuple[float, ...] = (0.25, 0.50, 0.75, 0.90)

#: Границы режима по `range_lift`. **Объявлены, а не подобраны**: подбор
#: границ под красивое распределение режимов — тот же способ найти эффект,
#: которого нет, что и подбор гиперпараметров под метрику. ±15% выбраны как
#: величина, заметная на глаз оператору; если модель почти никогда за них не
#: выходит, это факт про модель, и его надо печатать, а не чинить границами.
REGIME_EDGES: tuple[float, float] = (0.85, 1.15)

REGIME_NAMES = ("compressed", "normal", "expanded")

#: Пороги гейта годности, объявленные в разделе 49.3 журнала и НЕ
#: пересматриваемые. Живут здесь, а не в скрипте: с 2026-08-19 тот же гейт
#: считает `train`, и два списка порогов разъехались бы молча.
MIN_DELTA_R2 = 0.02
MAX_COVERAGE_ERROR = 0.05

#: Метка формата артефакта. Меняется при любой правке, после которой старые
#: сохранённые модели читать нельзя.
ARTIFACT_VERSION = "rf1"


def denominator_column(base: pd.DataFrame, horizon: str, normalization: str,
                       base_minutes: int) -> pd.DataFrame:
    """
    Логарифм ЗНАМЕНАТЕЛЯ цели, известного на баре `t`, — диагностическая
    колонка, а не часть модели.

    Зачем она понадобилась (2026-08-24, суррогатный прогон задачи S). Цель —
    `log(range_ratio) = log(будущий размах) − log(знаменатель)`, и знаменатель
    известен УЖЕ на баре `t`. При нормировке `atr14` это окно в 14 баров, а
    самое короткое окно бенчмарка B2 — 96. Значит модель, видящая 32 признака
    (среди них `atr_rank` и прочие короткие), знает вычитаемое лучше
    бенчмарка — и приращение R² частично меряет ЭТО, а не способность
    предсказывать будущее.

    На суррогате, где будущее непредсказуемо в принципе, механизм виден в
    чистом виде: ΔR² = +0.21 при `atr14` и +0.013 при `atr_h`, где окно
    знаменателя совпадает с горизонтом.

    Колонка добавляется в бенчмарк ТОЛЬКО диагностическим флагом. Она не
    входит ни в боевую модель, ни в штатный замер: её задача — ответить,
    сколько от измеренного приращения остаётся, когда информационная
    асимметрия убрана, а не улучшить прогноз.
    """
    h_bars = rm.horizon_bars(horizon, base_minutes)
    if normalization == "atr14":
        denominator = ind.atr(base, 14)
    elif normalization == "atr_h":
        denominator = ind.atr(base, h_bars)
    elif normalization == "rv_h":
        rv = ind.realized_vol(base["close"], h_bars)
        denominator = rv.where(rv > 0) * base["close"]
    else:
        raise ValueError(f"неизвестная нормировка {normalization!r}")
    # В долях цены, как и всё остальное: уровень цены за девять лет меняется
    # на два порядка, и колонка в абсолютных долларах была бы прежде всего
    # календарём.
    scaled = (denominator / base["close"]).where(lambda x: x > 0)
    return pd.DataFrame({"log_denominator": np.log(scaled)}, index=base.index)


def input_matrix(base: pd.DataFrame, features: pd.DataFrame,
                 extra_benchmark: pd.DataFrame | None = None
                 ) -> tuple[pd.DataFrame, list[str]]:
    """
    Матрица «B2 + признаки» БЕЗ цели, для предсказания.

    Отдельно от `design_matrix`, и это принципиально. Та режет строки, у
    которых нет цели, — а цель считается по горизонту ВПЕРЁД, то есть её нет
    ровно у последних баров истории. Предсказывать надо именно на них:
    воспользуйся `live` обучающей матрицей, он молча получил бы пустой прогноз
    на свежем баре и заполненный на баре суточной давности.

    `extra_benchmark` — диагностическая добавка к колонкам бенчмарка
    (`denominator_column`). По умолчанию None, то есть поведение боевого
    контура и всех замеров до 2026-08-24 не меняется ни на бит.
    """
    parts = [rm.har_columns(base["close"]), rm.seasonal_columns(base.index)]
    if extra_benchmark is not None:
        parts.append(extra_benchmark.reindex(base.index))
    benchmark = pd.concat(parts, axis=1)
    frame = pd.concat([benchmark, features.reindex(base.index)], axis=1).dropna()
    return frame, list(benchmark.columns)


def design_matrix(base: pd.DataFrame, features: pd.DataFrame, horizon: str,
                  normalization: str, base_minutes: int,
                  augment_benchmark: bool = False
                  ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Матрица «B2 + признаки», цель `log(range_ratio)` и список колонок бенчмарка.

    Пропуски режутся один раз и здесь: модель и её бенчмарк обязаны считаться
    на ОДНИХ И ТЕХ ЖЕ строках, иначе их квантили несравнимы, а `range_lift`
    сравнивает разные периоды под видом одного. Для ПРЕДСКАЗАНИЯ эта функция
    не годится — см. `input_matrix`.
    """
    h_bars = rm.horizon_bars(horizon, base_minutes)
    target = rm.log_target(rm.range_target(base, h_bars, normalization))
    extra = (denominator_column(base, horizon, normalization, base_minutes)
             if augment_benchmark else None)
    inputs, benchmark_names = input_matrix(base, features, extra)
    frame = pd.concat([target.rename("_target"), inputs], axis=1).dropna()
    columns = [c for c in frame.columns if c != "_target"]
    return frame[columns], frame["_target"], benchmark_names


def _make_model(quantile: float, seed: int):
    """
    Квантильный бустинг. Гиперпараметры — те же, что у контрольной модели
    (`control._make_model`) и у стенда 47.6, и меняется ровно одно: функция
    потерь.

    Это не экономия мысли, а условие сравнимости. Приращение R², измеренное в
    47.6, получено на ЭТИХ значениях; подбери мы здесь другие, и «регрессор
    работает лучше замера» нельзя было бы отличить от «модель другая».
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=False,   # ранняя остановка по случайному срезу — протечка
        random_state=seed,
    )


@dataclass
class RangeForecast:
    """
    Обученная пара «модель на признаках + её бенчмарк», по одной на квантиль.

    Хранятся обе, и это не удобство. `range_lift` определён как отношение к
    бенчмарку, посчитанному на той же выборке тем же классом модели, — значит
    бенчмарк является частью обученного объекта, а не внешней справкой. Отдать
    наружу только полную модель означало бы отдать абсолютное число, то есть
    ровно то, на чём провалился замер C3.
    """

    symbol: str
    horizon: str
    normalization: str
    seed: int
    feature_names: list[str]
    benchmark_names: list[str]
    models: dict[float, object] = field(default_factory=dict, repr=False)
    benchmark_models: dict[float, object] = field(default_factory=dict, repr=False)
    n_train: int = 0
    train_end: pd.Timestamp | None = None

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Квантили `range_ratio`, их бенчмарк, `range_lift` и режим.

        Возвращается в исходной шкале, а не в логарифмах: квантиль монотонного
        преобразования — это преобразование квантиля, поэтому `exp` здесь
        точен и никакой поправки на смещение не требует (в отличие от среднего,
        которому она была бы нужна).
        """
        x_full = frame[self.feature_names].to_numpy(dtype=float)
        x_bench = frame[self.benchmark_names].to_numpy(dtype=float)

        out = pd.DataFrame(index=frame.index)
        for q in QUANTILES:
            out[f"p{int(q * 100)}"] = np.exp(self.models[q].predict(x_full))
            out[f"bench_p{int(q * 100)}"] = np.exp(
                self.benchmark_models[q].predict(x_bench)
            )
        # Квантильные модели обучаются независимо и на хвостах могут
        # пересечься (p75 ниже p50). Это известное свойство пофункционального
        # подхода, а не ошибка данных; чиним монотонностью по строке, а не
        # общей моделью — общая потребовала бы другого класса и сломала бы
        # сравнимость с 47.6.
        heads = [f"p{int(q * 100)}" for q in QUANTILES]
        out[heads] = np.maximum.accumulate(out[heads].to_numpy(), axis=1)
        bench_heads = [f"bench_p{int(q * 100)}" for q in QUANTILES]
        out[bench_heads] = np.maximum.accumulate(out[bench_heads].to_numpy(), axis=1)

        out["range_lift"] = out["p50"] / out["bench_p50"].replace(0.0, np.nan)
        low, high = REGIME_EDGES
        out["range_regime"] = np.select(
            [out["range_lift"] < low, out["range_lift"] > high],
            [REGIME_NAMES[0], REGIME_NAMES[2]],
            default=REGIME_NAMES[1],
        )
        return out


def fit(symbol: str, frame: pd.DataFrame, target: pd.Series,
        benchmark_names: list[str], train: slice, seed: int,
        horizon: str, normalization: str,
        log=logger.info) -> RangeForecast:
    """
    Учит по паре моделей на каждый квантиль: полную и бенчмарк.

    `train` — срез обучающей части. Зазор перед проверочной частью вырезает
    вызывающий код: горизонт исхода накрывает следующие H баров, и последние H
    строк обучения «видят» начало проверки.
    """
    x_full = frame.to_numpy(dtype=float)[train]
    x_bench = frame[benchmark_names].to_numpy(dtype=float)[train]
    y = target.to_numpy(dtype=float)[train]

    models, benchmark_models = {}, {}
    for q in QUANTILES:
        log(f"  квантиль {q:.2f}: полная модель…")
        models[q] = _make_model(q, seed).fit(x_full, y)
        log(f"  квантиль {q:.2f}: бенчмарк B2…")
        benchmark_models[q] = _make_model(q, seed).fit(x_bench, y)

    return RangeForecast(
        symbol=symbol, horizon=horizon, normalization=normalization, seed=seed,
        feature_names=list(frame.columns), benchmark_names=list(benchmark_names),
        models=models, benchmark_models=benchmark_models,
        n_train=len(y), train_end=target.index[train.stop - 1],
    )


# ─── Метрики ────────────────────────────────────────────────────────────────
def pinball_loss(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    """
    Средняя потеря пинбола — единственная метрика, которую квантильная модель
    оптимизирует напрямую.

    Читается только в сравнении: сама по себе величина ничего не значит,
    значимо только отношение к бенчмарку на тех же строках.
    """
    diff = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def coverage(actual: np.ndarray, predicted: pd.DataFrame) -> dict[float, float]:
    """
    Доля фактических исходов ниже каждого предсказанного квантиля.

    Идеал — сам квантиль: под p90 обязаны оказаться 90% случаев. Это проверка
    КАЛИБРОВКИ, и она отдельна от точности: модель может быть идеально
    откалибрована и при этом бесполезна (безусловное распределение калибровано
    по построению). Поэтому в критерии она стоит рядом с приращением R², а не
    вместо него.
    """
    actual = np.asarray(actual, dtype=float)
    return {
        q: float((actual <= predicted[f"p{int(q * 100)}"].to_numpy(float)).mean())
        for q in QUANTILES
    }


def calibration_error(observed: dict[float, float]) -> float:
    """Среднее абсолютное отклонение покрытия от номинала по четырём квантилям."""
    return float(np.mean([abs(observed[q] - q) for q in QUANTILES]))


# ─── Гейт годности ──────────────────────────────────────────────────────────
@dataclass
class Gate:
    """
    Результат проверки модели на отложенной части — три условия раздела 49.3.

    Считается ВНУТРИ обучения, а не отдельным скриптом, ровно по одной
    причине: модель, не прошедшую гейт, нельзя пускать в кандидата, и решать
    это должен тот же прогон, который её обучил. Иначе между «обучили» и
    «проверили» появляется зазор, в котором непроверенная модель уже отдаёт
    числа.
    """

    delta_r2: float
    p_delta: float
    rho: float
    rho_bench: float
    p_better: float
    coverage: dict = field(default_factory=dict)
    coverage_error: float = float("nan")
    n_train: int = 0
    n_test: int = 0
    split: str = ""
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict:
        return {
            "delta_r2": self.delta_r2, "p_delta": self.p_delta,
            "rho": self.rho, "rho_bench": self.rho_bench,
            "p_better": self.p_better,
            "coverage": {str(k): v for k, v in self.coverage.items()},
            "coverage_error": self.coverage_error,
            "n_train": self.n_train, "n_test": self.n_test, "split": self.split,
            "failures": self.failures, "passed": self.passed,
        }


def evaluate(model: "RangeForecast", frame: pd.DataFrame, target: pd.Series,
             test: slice, train_mean: float, horizon: str,
             n_boot: int, rng: np.random.Generator) -> Gate:
    """Три условия гейта на отложенной части. Пороги — константы модуля."""
    from btcproc.analysis.range_lift import paired_diff_p

    predicted = model.predict(frame.iloc[test])
    actual_log = target.to_numpy(dtype=float)[test]
    actual = np.exp(actual_log)
    ts = frame.index.to_numpy()[test]

    means = np.full(len(actual), train_mean)
    fold = np.ones(len(actual))
    base_fit = rm.FoldPredictions(
        ts=ts, actual=actual_log,
        predicted=np.log(predicted["bench_p50"].to_numpy(dtype=float)),
        train_mean=means, fold=fold,
    )
    full_fit = rm.FoldPredictions(
        ts=ts, actual=actual_log,
        predicted=np.log(predicted["p50"].to_numpy(dtype=float)),
        train_mean=means, fold=fold,
    )
    block = rm.bootstrap_block(pd.Series(ts), horizon)
    gain = rm.compare(base_fit, full_fit, block, n_boot, rng)
    rho, rho_bench, p_better = paired_diff_p(
        predicted["p50"].to_numpy(float), predicted["bench_p50"].to_numpy(float),
        actual, block, n_boot, rng,
    )
    observed = coverage(actual, predicted)
    worst = max(abs(observed[q] - q) for q in QUANTILES)

    failures = []
    if not (gain.delta >= MIN_DELTA_R2 and gain.p_value <= 0.05):
        failures.append(
            f"приращение R² {gain.delta:+.4f} (p={gain.p_value:.4f}) "
            f"не достигло {MIN_DELTA_R2} со значимостью"
        )
    if not (p_better <= 0.05 and rho > rho_bench):
        failures.append(
            f"не обыгрывает бенчмарк по ранжированию: rho {rho:+.3f} против "
            f"{rho_bench:+.3f}, p={p_better:.4f}"
        )
    if not (worst <= MAX_COVERAGE_ERROR):
        failures.append(
            f"покрытие квантилей уходит на {worst:.3f} при пороге "
            f"{MAX_COVERAGE_ERROR}"
        )

    return Gate(
        delta_r2=gain.delta, p_delta=gain.p_value, rho=rho, rho_bench=rho_bench,
        p_better=p_better, coverage=observed, coverage_error=worst,
        n_train=test.start, n_test=len(actual),
        split=str(frame.index[test.start]), failures=failures,
    )


def fit_production(symbol: str, frame: pd.DataFrame, target: pd.Series,
                   benchmark_names: list[str], horizon: str, normalization: str,
                   seed: int, gap: int, train_frac: float = 0.7,
                   n_boot: int = 2000, log=logger.info
                   ) -> tuple["RangeForecast", Gate]:
    """
    Боевое обучение: проверка на отложенной части, затем переобучение на ВСЕЙ
    истории.

    Порядок именно такой, и он не взаимозаменяем с обратным. Гейт обязан быть
    посчитан на данных, которых модель не видела, — иначе он меряет
    запоминание. Боевая же модель обязана видеть всю доступную историю: она
    применяется к барам ПОСЛЕ конца обучения, и отдавать ей на треть меньше
    данных значило бы платить за замер качеством прогноза.

    Из этого следует правило, которое обязан соблюдать вызывающий код:
    **числа боевой модели годны только для баров после `train_end`.** На
    исторических барах она даёт in-sample прогноз, и это не прогноз вовсе.
    """
    from btcproc.analysis import holdout as ho

    split_ts = ho.split_bar(frame.index, train_frac)
    n_train = int((frame.index < split_ts).sum())
    # Зазор на границе: последние `gap` строк обучения имеют окна исходов,
    # накрывающие начало отложенной части.
    train = slice(0, max(1, n_train - gap))
    test = slice(n_train, len(frame))

    log(f"[{symbol}] гейт: обучение {train.stop} строк до {split_ts:%Y-%m-%d}, "
        f"проверка на {test.stop - test.start}")
    probe = fit(symbol, frame, target, benchmark_names, train, seed,
                horizon, normalization, log=lambda m: None)
    rng = np.random.default_rng(seed)
    gate = evaluate(probe, frame, target, test,
                    float(target.to_numpy(dtype=float)[train].mean()),
                    horizon, n_boot, rng)
    if gate.passed:
        log(f"[{symbol}] гейт пройден: ΔR²={gate.delta_r2:+.4f}, "
            f"rho {gate.rho:+.3f} против {gate.rho_bench:+.3f}, "
            f"покрытие ±{gate.coverage_error:.3f}")
    else:
        log(f"[{symbol}] гейт НЕ пройден: " + "; ".join(gate.failures))

    log(f"[{symbol}] переобучение на всей истории ({len(frame)} строк)…")
    final = fit(symbol, frame, target, benchmark_names, slice(0, len(frame)),
                seed, horizon, normalization, log=lambda m: None)
    return final, gate


# ─── Сериализация ───────────────────────────────────────────────────────────
def dumps(model: "RangeForecast") -> bytes:
    """
    Артефакт для хранения в БД. joblib, а не JSON: внутри модели живут
    обученные бустинги, и в JSON они не сериализуются в принципе.

    Версия sklearn кладётся рядом намеренно. Пиклы моделей несовместимы между
    версиями библиотеки, и после её обновления артефакт может подняться
    молча-неправильным. Проверка — в `loads`.
    """
    import io

    import joblib
    import sklearn

    buffer = io.BytesIO()
    joblib.dump(
        {"version": ARTIFACT_VERSION, "sklearn": sklearn.__version__, "model": model},
        buffer,
    )
    return buffer.getvalue()


def loads(payload: bytes, log=logger.warning) -> "RangeForecast | None":
    """
    Обратно. Возвращает None при любой несовместимости — и ГРОМКО пишет
    причину.

    Отказ, а не падение: модель размаха украшает кандидата, но не является
    его частью, и прогон из-за неё останавливаться не должен. Отказ молча —
    худший вариант из трёх, поэтому причина логируется всегда.
    """
    import io

    import joblib
    import sklearn

    try:
        blob = joblib.load(io.BytesIO(payload))
    except Exception as exc:  # noqa: BLE001 — причин много, важна любая
        log("Модель размаха не читается (%s) — кандидаты пойдут без неё", exc)
        return None
    if blob.get("version") != ARTIFACT_VERSION:
        log("Модель размаха версии %s, ожидается %s — пропущена",
            blob.get("version"), ARTIFACT_VERSION)
        return None
    if blob.get("sklearn") != sklearn.__version__:
        log("Модель размаха обучена на sklearn %s, здесь %s — пропущена: "
            "пиклы между версиями несовместимы", blob.get("sklearn"),
            sklearn.__version__)
        return None
    return blob["model"]


def view(model: "RangeForecast", base: pd.DataFrame, features: pd.DataFrame,
         log=logger.info) -> pd.DataFrame:
    """
    Прогноз размаха по барам — в том виде, в каком его берёт сборка кандидата.

    **Отдаются только бары СТРОГО ПОСЛЕ конца обучения модели.** На
    исторических барах боевая модель обучалась, и её квантили там —
    запоминание, а не прогноз. Разница не видна ни по одному числу: кандидат
    выглядел бы одинаково хорошо оценённым по обе стороны границы, а
    исторический был бы фикцией. Ровно этот класс ошибки инвариант 4 закрывает
    для признаков; здесь он закрывается для модели.

    Набор колонок сверяется с тем, на котором модель обучалась, и расхождение
    — отказ, а не подгонка: молча предсказывать по другому вектору хуже, чем
    не предсказывать вовсе.
    """
    frame, _ = input_matrix(base, features)
    missing = set(model.feature_names) - set(frame.columns)
    if missing:
        log("Набор признаков разошёлся с моделью размаха — нет %s; "
            "кандидаты пойдут без размаха", sorted(missing))
        return pd.DataFrame()
    if model.train_end is not None:
        frame = frame[frame.index > model.train_end]
    if frame.empty:
        return pd.DataFrame()
    predicted = model.predict(frame[model.feature_names])
    return predicted[["p50", "p90", "range_lift", "range_regime"]]
