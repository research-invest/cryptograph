"""
Стенд «размах × горизонт × бенчмарк» — задача A ТЗ
`crypto-graph/docs/tz_range_horizons_19-08-26.md`.

## На какой вопрос отвечает

Один-единственный: **сколько дисперсии `range_ratio` предиктор X объясняет
СВЕРХ тривиального бенчмарка** — честно (out-of-sample), на трёх горизонтах
и с правильной значимостью. Направление здесь не меряется вовсе: оно закрыто
двенадцатью замерами раздела 26.4, и никакая функция этого модуля к нему не
применима.

Отдельный модуль рядом с `range_lift.py`, а не внутри него, потому что это
разные предметы: `range_lift` — одиночный предиктор, Spearman и in-sample
частный R²; здесь — многомерная модель, walk-forward и out-of-sample R².
Общее (`forward_range_ratio`, блочный бутстрап, поправки) импортируется, а не
копируется — инвариант 11.

## Стек бенчмарков: три уровня, каждый включает предыдущий

| Уровень | Состав | Прохождение сверх него означает |
|---|---|---|
| B0 | константа (среднее обучающей части) | «хоть что-то» |
| B1 | HAR-RV: `rv` на окнах 96 / 672 / 2880 баров | «сверх кластеризации волатильности» |
| B2 | B1 + сезонность: sin/cos часа (гармоники 1 и 2) + флаг выходных | **основной бенчмарк** |

B2 — обязательный знаменатель для любого вывода. Причина в §0.3в ТЗ: на
горизонте 4h час дня в одиночку объясняет вчетверо больше дисперсии, чем
`rv`, и всё, что коррелирует с часом (объёмы, поток тейкеров, открытый
интерес, активность состояний), получит сверх бенчмарка БЕЗ часа ложное
приращение. Бенчмарк без сезонности в этой задаче бенчмарком не является.

B1 сохранён не для симметрии: разность `R²(B2) − R²(B1)` — это и есть вклад
сезонности, число, которое до сих пор не мерил никто, и оно само по себе
результат.

## Три решения, которые легко принять неправильно

**Цель моделируется в логарифме, а не в рангах.** `range_ratio`
тяжелохвостый, и Pearson на сырых значениях обманывается выбросами — поэтому
весь `range_lift` работает на рангах. Но ранг НЕ переносится через границу
фолда: ранг проверочной строки зависит от остальных проверочных строк, то
есть от будущего внутри окна. Логарифм — фиксированное монотонное
преобразование, считается построчно и протечки не создаёт вовсе. Все
out-of-sample R² этого модуля — R² величины `log(range_ratio)`, и это
проговаривается в печати, а не подразумевается.

**In-sample часть (задача B) остаётся на рангах** — `partial_r2_gain_matrix`
буквально повторяет `range_lift.partial_r2_gain`, только бенчмарк у неё
матрица, а не вектор. Иначе перепроверка гейта R сравнивала бы своё число не
с разделом 36, а с другой методикой, и «эффект уменьшился» нельзя было бы
отличить от «сменили шкалу».

**Out-of-sample R² считается относительно среднего ОБУЧАЮЩЕЙ части.**
Относительно среднего проверочного окна — тихое подглядывание: на
нестационарной цели знание её собственного уровня уже даёт R² > 0. Разница
двух знаменателей печатается отдельной строкой («дрейф среднего») — она
показывает, насколько цель нестационарна, и заодно доказывает, что
подглядывание было бы не безобидным.

## Значимость

Только блочный бутстрап, общим кодом (`lift.stationary_block_indices`).
Наблюдения зависимы по построению: окна горизонта соседних баров
перекрываются почти целиком. Длина блока —
`max(block_length_rows(ts, horizon_minutes), autocorr_block_rows(predictor))`,
как в `measure_deriv_range.py`.

**И этого мало.** При n_эфф в тысячи минимальный детектируемый `ΔR²` — порядка
1e-4, то есть значимость здесь достигается на практически ничтожных
величинах. Урок раздела 36 (`ΔR² = 0.011` при `p ≤ 0.002` — «надёжно
значимый» эффект размером в 1% дисперсии) закреплён здесь тем, что ни один
критерий ТЗ не состоит из одного порога p: рядом всегда стоит порог
практической величины.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from btcproc.analysis.control import purged_splits  # noqa: F401 — переиспользуется скриптом
from btcproc.analysis.lift import (
    DEFAULT_N_BOOT,
    CHUNK_CELLS,
    block_length_rows,
    stationary_block_indices,
)
from btcproc.analysis.range_lift import _rank, forward_range_ratio
from btcproc.features import indicators as ind

#: Три горизонта ТЗ. 48h и длиннее не меряются намеренно (§1.2): мощность
#: вдвое хуже 24h, а режим успевает смениться внутри окна.
HORIZONS: tuple[str, ...] = ("4h", "12h", "24h")

#: Окна HAR-RV в базовых барах: сутки / неделя / месяц при 15-минутном баре.
HAR_WINDOWS: tuple[int, ...] = (96, 672, 2880)

#: Гармоники внутрисуточного цикла: 24 часа и 12 часов.
SEASONAL_HARMONICS: tuple[int, ...] = (1, 2)

#: Две нормировки цели (A3). Все выводы засчитываются только при совпадении
#: знака и порядка величины на обеих.
#:
#: **Кортеж не расширять.** Три стенда берут его как значение по умолчанию
#: (`args.norm or list(rm.NORMALIZATIONS)`), и добавленный сюда элемент молча
#: увеличил бы число их ячеек, а с ним и BH-поправку, — при том что их
#: критерии заявлены на двух нормировках. Нормировка, нужная одной задаче,
#: заводится ниже и запрашивается явно.
NORMALIZATIONS: tuple[str, ...] = ("atr14", "atr_h")

#: Нормировки сверх основных, каждая — под конкретную задачу.
#:
#: `rv_h` — знаменатель `realized_vol` по окну горизонта, то есть величина,
#: посчитанная ПО ЗАКРЫТИЯМ. Нужна задаче A ТЗ по геометрии свечи
#: (`crypto-graph/docs/tz_candle_geometry_20-08-26.md`, §2.4) и там же
#: объяснена: цель `range_ratio` делится на ATR, а ATR построен из размахов
#: баров — из тех же величин, что оценщики Паркинсона и Гармана–Класса. Общая
#: конструкция между предиктором и знаменателем цели способна дать
#: ЛОЖНОПОЛОЖИТЕЛЬНЫЙ результат, и обе нормировки A3 от неё не защищают: они
#: обе range-основанные. `rv_h` конструкцию не разделяет.
EXTRA_NORMALIZATIONS: tuple[str, ...] = ("rv_h",)

#: Оценщики дисперсии из полного OHLC — задача A ТЗ по геометрии свечи.
#: Порядок здесь же задаёт порядок колонок `range_estimator_columns`.
RANGE_ESTIMATORS: tuple[str, ...] = ("p", "gk", "rs")

#: Уровни бенчмарка, от тривиального к основному.
BENCHMARK_LEVELS: tuple[str, ...] = ("B0", "B1", "B2")

#: Потолок поиска лага автокорреляции — в БАРАХ (см. autocorr_block_rows,
#: которую здесь зовут с bars_per_day=1). Та же величина, что в
#: measure_deriv_range.py, ради одинаковой длины блока у задач B и C.
AUTOCORR_MAX_LAG_BARS = 5000

#: Минимум строк в каждой части разреза 70/30 (`holdout_forward`). Тот же
#: порядок, что у `holdout.split_bar`: на меньшем блочный бутстрап меряет
#: собственный шум, а не эффект.
MIN_HOLDOUT_PART = 1000


# ─── Горизонт и цель ────────────────────────────────────────────────────────
def horizon_bars(label: str, base_minutes: int) -> int:
    """«4h» → число базовых баров в горизонте."""
    unit = label[-1]
    value = int(label[:-1])
    minutes = value * {"m": 1, "h": 60, "d": 1440}[unit]
    return minutes // base_minutes


def horizon_minutes(label: str) -> int:
    unit = label[-1]
    return int(label[:-1]) * {"m": 1, "h": 60, "d": 1440}[unit]


def range_target(base: pd.DataFrame, h_bars: int, normalization: str) -> pd.Series:
    """
    `range_ratio` в одной из двух нормировок (A3).

    * `atr14` — знаменатель ATR14, как в разделе 36. Сохранён ради
      сравнимости, но он короче числителя: ATR14 на 15-минутном баре — это
      3.5 часа, а горизонт 24h — сутки. После всплеска волатильности будущий
      размах ОТНОСИТЕЛЬНО свежего ATR меньше, и разведка 2026-08-19 видела
      именно это: Spearman(`rv`, `range_ratio`) отрицателен и по модулю
      растёт с горизонтом. Это возврат к среднему внутри самой нормировки, а
      не свойство рынка.
    * `atr_h` — знаменатель ATR по окну, равному горизонту. Числитель и
      знаменатель меряют одну и ту же длину времени.
    * `rv_h` — знаменатель `realized_vol` по окну горизонта, переведённый в
      цену. НЕ входит в `NORMALIZATIONS` и запрашивается явно: это контроль
      задачи A ТЗ по геометрии свечи, где предиктор построен из тех же
      размахов баров, что и ATR. Единственная из трёх, не разделяющая
      конструкцию с range-оценщиками.

    Расхождение выводов между нормировками — это результат ПРО НОРМИРОВКУ, и
    записывать его надо именно так, а не как результат про рынок.
    """
    if normalization == "atr14":
        atr = ind.atr(base, 14)
    elif normalization == "atr_h":
        atr = ind.atr(base, h_bars)
    elif normalization == "rv_h":
        # `realized_vol` — доля, `forward_range_ratio` ждёт величину в цене
        # (как ATR): умножаем на закрытие того же бара. Никакого сдвига здесь
        # нет и быть не должно — знаменатель обязан быть известен на баре t.
        rv = ind.realized_vol(base["close"], h_bars)
        atr = rv.where(rv > 0) * base["close"]
    else:
        raise ValueError(f"неизвестная нормировка {normalization!r}")
    return forward_range_ratio(base, atr, h_bars)


def log_target(range_ratio: pd.Series) -> pd.Series:
    """
    `log(range_ratio)` — шкала, в которой считаются все OOS-числа модуля.

    Неположительные значения (плоское окно на неликвидном участке ранней
    истории) уходят в NaN, а не в −inf: строка без размаха — это отсутствие
    наблюдения, и молча тянуть её в регрессию нельзя.
    """
    values = range_ratio.where(range_ratio > 0)
    return np.log(values)


# ─── Колонки бенчмарка ──────────────────────────────────────────────────────
def har_columns(close: pd.Series) -> pd.DataFrame:
    """
    HAR-RV: логарифм реализованной волатильности на трёх окнах.

    Логарифм — по той же причине, что у цели: `rv` тяжелохвостая, и линейная
    модель на сырой величине описывала бы несколько всплесков вместо ряда.
    HAR (Corsi) в исходной форме и работает с логарифмами.
    """
    columns = {}
    for window in HAR_WINDOWS:
        rv = ind.realized_vol(close, window)
        columns[f"log_rv_{window}"] = np.log(rv.where(rv > 0))
    return pd.DataFrame(columns, index=close.index)


def range_estimators(base: pd.DataFrame) -> pd.DataFrame:
    """
    Три оценщика дисперсии ОДНОГО бара из полного OHLC — задача A ТЗ
    `crypto-graph/docs/tz_candle_geometry_20-08-26.md`.

        Паркинсон       σ²_P  = (1 / (4·ln2)) · (ln(H/L))²
        Гарман–Класс    σ²_GK = 0.5·(ln(H/L))² − (2·ln2 − 1)·(ln(C/O))²
        Роджерс–Сатчелл σ²_RS = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)

    Все три безразмерны и масштабно инвариантны по построению: умножение всех
    цен на константу не меняет ни одного отношения под логарифмом.

    **Почему их три, а не один.** Паркинсон видит только размах и завышает
    оценку на трендовом баре — снос внутри бара он засчитывает как
    волатильность. Гарман–Класс добавляет тело, но от сноса не свободен.
    Роджерс–Сатчелл от сноса не зависит вовсе, и он же наименее избыточен к
    HAR-RV по разведке перед прогоном (R² на `har_columns` 0.93–0.98 против
    0.97–0.99 у Паркинсона).

    **Yang–Zhang не считается сознательно** (§1.6 ТЗ). Его овернайт-компонента
    построена на разрыве `open[t]` против `close[t−1]`, а у 15-минутных баров
    круглосуточной биржи разрыв — это один тик: медиана
    `|open − close[-1]| / (high − low)` по пяти монетам Binance 0.0006–0.046
    при p90 ≤ 0.17 (замер 2026-08-21). Компонента вырождается в шум округления.

    **Ни один из трёх не считается для HYPEUSDT** — точнее, считать их можно,
    но толковать нельзя: `ingest/bybit.py` строит `open` как `close.shift(1)`,
    то есть `ln(C/O)` там равен полной close-to-close доходности бара, и GK с
    RS меряют на этой монете другую величину. Ответственность вызывающего:
    модуль не знает символа и молча посчитает что угодно.

    Вырожденный бар (`high == low`) даёт ноль у всех трёх, и это ПРАВИЛЬНОЕ
    значение, а не пропуск: нулевой размах — максимальное сжатие. NaN здесь
    выбросил бы строку из окна усреднения и сдвинул агрегат (тот же довод,
    что у `range_exp` в `features/builder.py`, аудит B8).
    """
    high, low = base["high"], base["low"]
    open_, close = base["open"], base["close"]
    hl = np.log(high / low)
    co = np.log(close / open_)
    ho, lo = np.log(high / open_), np.log(low / open_)
    hc, lc = np.log(high / close), np.log(low / close)
    return pd.DataFrame(
        {
            "p": hl ** 2 / (4.0 * math.log(2.0)),
            "gk": 0.5 * hl ** 2 - (2.0 * math.log(2.0) - 1.0) * co ** 2,
            "rs": hc * ho + lc * lo,
        },
        index=base.index,
    )


def range_estimator_columns(base: pd.DataFrame,
                            windows: tuple[int, ...] = HAR_WINDOWS) -> pd.DataFrame:
    """
    Колонки бенчмарка из range-оценщиков: среднее σ² по окну, затем логарифм.

    Шесть… точнее, `len(RANGE_ESTIMATORS) × len(windows)` колонок вида
    `log_p_96`, `log_gk_672`, `log_rs_2880`. Устроены буква в букву как
    `har_columns` и живут рядом с ними намеренно: разведи их по модулям — и
    разъедется дисциплина логарифмирования, из-за которой одна ветка отдаёт
    NaN, а другая −inf.

    Усреднение ДО логарифма, а не после: среднее логарифмов — это среднее
    геометрическое дисперсий, оно занижено на всплесках, а HAR по построению
    моделирует среднее арифметическое. Неположительное среднее (плоское окно
    на неликвидном участке ранней истории) уходит в NaN — строка без
    волатильности это отсутствие наблюдения, а не −inf.
    """
    estimators = range_estimators(base)
    columns: dict[str, pd.Series] = {}
    for name in RANGE_ESTIMATORS:
        for window in windows:
            mean = estimators[name].rolling(window, min_periods=window).mean()
            columns[f"log_{name}_{window}"] = np.log(mean.where(mean > 0))
    return pd.DataFrame(columns, index=base.index)


def seasonal_columns(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Час дня как непрерывная ЦИКЛИЧЕСКАЯ координата плюс флаг выходных.

    Не сессионные атомы (C3 ТЗ): они покрывают 00:00–22:00 UTC, и два часа в
    сутках не принадлежат ни одной сессии — известный дефект
    (`features/events.py`, аудит 2026-08-15 B6). Сдвигать границы сессий
    здесь нельзя: это потребовало бы `backfill_context_atoms --force --all`
    и является отдельным решением владельца. Циклический час свободен от
    дефекта целиком и вдобавок не имеет разрывов на границах.

    Гармоника 1 — суточная, гармоника 2 — полусуточная: рынок видит и смену
    суток, и пару «американская сессия / азиатская сессия».
    """
    hour = index.hour + index.minute / 60.0
    columns: dict[str, np.ndarray] = {}
    for k in SEASONAL_HARMONICS:
        angle = 2.0 * math.pi * k * hour / 24.0
        columns[f"hour_sin{k}"] = np.sin(angle)
        columns[f"hour_cos{k}"] = np.cos(angle)
    columns["is_weekend"] = (index.dayofweek >= 5).astype(float)
    return pd.DataFrame(columns, index=index)


def weekday_columns(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    День недели дамми-кодированием, понедельник — опорный уровень.

    Нужен только задаче C3 («час дня И день недели как самостоятельный
    ответ»); в бенчмарк B2 не входит — там дня недели хватает флагом
    выходных, а семь колонок в знаменателе всех выводов удорожали бы его без
    нужды.
    """
    dow = index.dayofweek
    return pd.DataFrame(
        {f"dow_{d}": (dow == d).astype(float) for d in range(1, 7)}, index=index
    )


def benchmark_columns(base: pd.DataFrame, level: str) -> pd.DataFrame:
    """Колонки уровня бенчмарка. B0 — пустая матрица (остаётся один свободный член)."""
    if level == "B0":
        return pd.DataFrame(index=base.index)
    if level == "B1":
        return har_columns(base["close"])
    if level == "B2":
        return pd.concat(
            [har_columns(base["close"]), seasonal_columns(base.index)], axis=1
        )
    raise ValueError(f"неизвестный уровень бенчмарка {level!r}")


# ─── Out-of-sample R² ───────────────────────────────────────────────────────
def out_of_sample_r2(actual: np.ndarray, predicted: np.ndarray,
                     reference: np.ndarray) -> float:
    """
    R² на проверочных строках относительно ЗАДАННОГО опорного прогноза.

    `reference` — среднее обучающей части (в штатном использовании). Считать
    относительно среднего проверочного окна — подглядывание: на
    нестационарной цели само знание её текущего уровня уже даёт положительный
    R², и модель получает его даром.

    Величина может быть отрицательной, и это не ошибка, а честный ответ
    «модель хуже константы, известной до начала окна».
    """
    actual = np.asarray(actual, dtype=float)
    sse = float(np.sum((actual - np.asarray(predicted, dtype=float)) ** 2))
    sst = float(np.sum((actual - np.asarray(reference, dtype=float)) ** 2))
    if sst <= 0:
        return float("nan")
    return 1.0 - sse / sst


@dataclass
class FoldPredictions:
    """
    Склеенные по фолдам проверочные строки walk-forward.

    Хранится всё, что нужно и для R², и для сравнения двух моделей на ОДНИХ И
    ТЕХ ЖЕ строках: сравнивать приращение по разным наборам строк нельзя, и
    равенство проверяется явно (`assert_aligned`).
    """

    ts: np.ndarray
    actual: np.ndarray
    predicted: np.ndarray
    train_mean: np.ndarray
    fold: np.ndarray
    n_train: list[int] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.actual)

    @property
    def r2(self) -> float:
        """OOS R² относительно среднего обучающей части своего фолда."""
        return out_of_sample_r2(self.actual, self.predicted, self.train_mean)

    @property
    def r2_versus_test_mean(self) -> float:
        """
        То же, но относительно среднего ПРОВЕРОЧНОГО окна — только ради
        диагностики дрейфа. Как метрику качества использовать нельзя.
        """
        return out_of_sample_r2(
            self.actual, self.predicted, np.full_like(self.actual, self.actual.mean())
        )

    @property
    def drift(self) -> float:
        """
        Дрейф среднего: сколько R² даёт ЗНАНИЕ среднего проверочного окна
        само по себе (прогноз-константа test-mean, знаменатель — train-mean).

        Строго положителен при нестационарной цели. Это и есть цена ошибки
        «считать R² от среднего test».
        """
        return out_of_sample_r2(
            self.actual, np.full_like(self.actual, self.actual.mean()), self.train_mean
        )

    @property
    def squared_errors(self) -> np.ndarray:
        return (self.actual - self.predicted) ** 2

    @property
    def total_sum_of_squares(self) -> float:
        return float(np.sum((self.actual - self.train_mean) ** 2))

    def assert_aligned(self, other: "FoldPredictions") -> None:
        if self.n != other.n or not np.array_equal(self.fold, other.fold):
            raise ValueError(
                "две модели посчитаны на разных строках — приращение R² между "
                "ними не имеет смысла"
            )


def walk_forward(n: int, target: np.ndarray, ts: np.ndarray, gap: int, folds: int,
                 predict) -> FoldPredictions:
    """
    Purged walk-forward: разбиение берётся у `control.purged_splits`, прогноз
    даёт переданная функция `predict(train_slice, test_slice) -> np.ndarray`.

    Зазор `gap` обязан равняться горизонту ЗАМЕРА в барах (16 / 48 / 96), а не
    `config.data.horizon_bars`. Захардкоженные 96 баров для 4h-замера
    выбросили бы вшестеро больше нужного и незаметно съели мощность
    (ловушка 4 ТЗ).

    Функция, а не матрица признаков, на входе — потому что задаче C2 нужно
    считать целевое кодирование категорий ВНУТРИ фолда, только по его
    обучающей части.
    """
    if gap <= 0:
        raise ValueError("зазор должен быть положительным и равным горизонту замера")
    splits = purged_splits(n, folds, gap)
    if not splits:
        raise ValueError(f"на {n} строках при зазоре {gap} фолдов не набирается")

    chunks_ts, chunks_actual, chunks_pred, chunks_mean, chunks_fold = [], [], [], [], []
    n_train: list[int] = []
    for i, (train, test) in enumerate(splits, start=1):
        predicted = np.asarray(predict(train, test), dtype=float)
        actual = target[test]
        if len(predicted) != len(actual):
            raise ValueError("прогноз и факт разной длины — ошибка в predict")
        chunks_ts.append(ts[test])
        chunks_actual.append(actual)
        chunks_pred.append(predicted)
        chunks_mean.append(np.full(len(actual), float(target[train].mean())))
        chunks_fold.append(np.full(len(actual), i))
        n_train.append(train.stop - train.start)

    return FoldPredictions(
        ts=np.concatenate(chunks_ts),
        actual=np.concatenate(chunks_actual),
        predicted=np.concatenate(chunks_pred),
        train_mean=np.concatenate(chunks_mean),
        fold=np.concatenate(chunks_fold),
        n_train=n_train,
    )


# ─── Модели ─────────────────────────────────────────────────────────────────
def holdout_forward(n: int, target: np.ndarray, ts: np.ndarray, gap: int,
                    predict, frac: float = 0.7) -> FoldPredictions:
    """
    Один разрез вместо walk-forward: учим на первых `frac` истории, проверяем
    на остатке, между ними — зазор в горизонт.

    Зачем рядом с `walk_forward`, а не вместо него. Walk-forward отвечает на
    вопрос «работает ли модель в среднем по истории» и усредняет пять эпох;
    разрез 70/30 отвечает на другой — «работает ли она на данных, которых не
    видел ни один этап подгонки», — и это единственная проверка, которую в
    проекте признают решающей (раздел 26, `holdout.split_bar`). Задача A ТЗ по
    геометрии свечи требует именно её.

    Возвращается тот же `FoldPredictions` с одним фолдом, поэтому `compare`,
    `Gain` и блочный бутстрап работают без изменений — приращение считается
    той же арифметикой, что и в остальных задачах модуля (инвариант 11).

    Зазор обязателен и здесь: метка бара `t` считается по барам `t+1…t+gap`,
    и без пропуска последние строки обучения видят начало проверочного окна.
    """
    if gap <= 0:
        raise ValueError("зазор должен быть положительным и равным горизонту замера")
    cut = int(n * frac)
    train, test = slice(0, cut), slice(cut + gap, n)
    if train.stop < MIN_HOLDOUT_PART or test.stop - test.start < MIN_HOLDOUT_PART:
        raise ValueError(
            f"разрез {frac} на {n} строках при зазоре {gap} даёт слишком "
            f"короткую часть ({train.stop} / {test.stop - test.start})"
        )
    predicted = np.asarray(predict(train, test), dtype=float)
    actual = target[test]
    if len(predicted) != len(actual):
        raise ValueError("прогноз и факт разной длины — ошибка в predict")
    return FoldPredictions(
        ts=ts[test], actual=actual, predicted=predicted,
        train_mean=np.full(len(actual), float(target[train].mean())),
        fold=np.ones(len(actual), dtype=int),
        n_train=[train.stop - train.start],
    )


def ols_predictor(columns: np.ndarray, target: np.ndarray):
    """
    Обычный МНК со свободным членом; пустая матрица колонок даёт прогноз-константу.

    Решается через `lstsq`: колонки бенчмарка коррелированы (три окна `rv` —
    почти вложенные), и нормальные уравнения на них плохо обусловлены.
    """
    columns = np.asarray(columns, dtype=float)
    if columns.ndim == 1:
        columns = columns[:, None]

    def predict(train: slice, test: slice) -> np.ndarray:
        x_train = np.column_stack([np.ones(train.stop - train.start), columns[train]])
        x_test = np.column_stack([np.ones(test.stop - test.start), columns[test]])
        beta, *_ = np.linalg.lstsq(x_train, target[train], rcond=None)
        return x_test @ beta

    return predict


def make_regressor(seed: int):
    """
    Бустинг-регрессор с теми же гиперпараметрами, что у классификатора
    контрольной модели (`control._make_model`): глубина 4, `learning_rate`
    0.05, `min_samples_leaf` 200, `l2` 1.0, 300 итераций, без ранней
    остановки.

    Значения объявлены заранее и под метрику НЕ подбираются (инвариант 5 ТЗ
    аудита): подбор гиперпараметров под целевую метрику — ровно тот способ
    найти эффект, которого нет. Ранняя остановка выключена по той же причине,
    что у классификатора: её случайный внутренний срез на временном ряде —
    протечка.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def boosted_predictor(columns: np.ndarray, target: np.ndarray, seed: int):
    """Бустинг на готовой матрице; учится на обучающей части фолда."""
    columns = np.asarray(columns, dtype=float)
    if columns.ndim == 1:
        columns = columns[:, None]

    def predict(train: slice, test: slice) -> np.ndarray:
        model = make_regressor(seed)
        model.fit(columns[train], target[train])
        return model.predict(columns[test])

    return predict


def target_encode(codes: np.ndarray, target: np.ndarray, train: slice,
                  prior_weight: float = 200.0) -> np.ndarray:
    """
    Целевое кодирование категорий по ОБУЧАЮЩЕЙ части фолда, сглаженное к
    общему среднему.

    Только по обучающей части — иначе среднее состояния содержит будущее, и
    это самый лёгкий способ получить впечатляющий и ложный R² (ловушка 6 ТЗ).
    Категория, впервые встретившаяся в проверочном окне, получает общее
    среднее обучающей части: «сведений на тот момент не было» — это не ошибка
    и не ноль.

    Сглаживание к среднему с весом `prior_weight` наблюдений — та же величина,
    что `min_samples_leaf` бустинга: редкое состояние из десятка баров иначе
    получило бы кодом собственный шум.

    Возвращает вектор длиной во ВСЮ выборку: значения посчитаны по train, но
    расставлены и по train, и по test — обучать модель на этом коде можно, он
    не смотрит вперёд.
    """
    codes = np.asarray(codes)
    train_codes = codes[train]
    train_target = np.asarray(target, dtype=float)[train]
    prior = float(train_target.mean())

    frame = pd.DataFrame({"code": train_codes, "y": train_target})
    stats = frame.groupby("code")["y"].agg(["sum", "count"])
    smoothed = (stats["sum"] + prior * prior_weight) / (stats["count"] + prior_weight)
    return pd.Series(codes).map(smoothed).fillna(prior).to_numpy(dtype=float)


# ─── Значимость приращения ──────────────────────────────────────────────────
@dataclass
class Gain:
    """Приращение R² одной модели над другой на одних и тех же строках."""

    r2_base: float
    r2_full: float
    delta: float
    p_value: float
    n: int
    block: int

    @property
    def n_effective(self) -> int:
        return max(1, self.n // max(self.block, 1))

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value <= alpha


def block_mean_p(values: np.ndarray, block_length: int, n_boot: int,
                 rng: np.random.Generator) -> float:
    """
    Односторонний блочный p-value для «среднее ряда положительно».

    Ряд центрируется, после чего считается доля реплик, у которых среднее
    центрированной пересборки достигло наблюдённого. Это стандартная
    бутстрапная проверка H0: «среднее равно нулю» против «больше нуля», и она
    сохраняет автокорреляцию ряда — здесь она велика по построению (окна
    горизонта соседних строк перекрываются почти целиком).

    Индексы блоков берутся общим кодом `lift.stationary_block_indices`
    (инвариант 11): своя копия пересборки разъехалась бы с остальными
    замерами молча.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    observed = float(values.mean())
    if n < 20 or not np.isfinite(observed) or observed <= 0:
        return 1.0

    centered = values - observed
    chunk = max(1, min(n_boot, CHUNK_CELLS // max(n, 1)))
    extreme, done = 0, 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        idx = stationary_block_indices(n, block_length, rng, size)
        means = centered[idx].mean(axis=1)
        extreme += int(np.sum(np.isfinite(means) & (means >= observed)))
        done += size
    return float((1 + extreme) / (1 + n_boot))


def compare(base: FoldPredictions, full: FoldPredictions, block_length: int,
            n_boot: int = DEFAULT_N_BOOT,
            rng: np.random.Generator | None = None) -> Gain:
    """
    Приращение out-of-sample R² модели `full` над `base`.

    Знаменатель у обеих один и тот же (сумма квадратов относительно среднего
    обучающей части), поэтому приращение равно среднему построчной разности
    квадратов ошибок, делённому на этот знаменатель. Отсюда и способ
    значимости: блочный бутстрап по ряду построчных разностей — H0 «модель
    `full` в среднем не точнее».

    Пересобирается именно РАЗНОСТЬ, а не предиктор: обе модели уже посчитаны
    на одних и тех же строках, и вопрос стоит про их сравнение, а не про связь
    величины с целью.
    """
    base.assert_aligned(full)
    rng = rng or np.random.default_rng(42)
    sst = base.total_sum_of_squares
    diff = base.squared_errors - full.squared_errors
    delta = float(diff.sum() / sst) if sst > 0 else float("nan")
    p_value = block_mean_p(diff, block_length, n_boot, rng)
    return Gain(
        r2_base=base.r2, r2_full=full.r2, delta=delta,
        p_value=p_value, n=base.n, block=block_length,
    )


# ─── In-sample частный R² с матричным бенчмарком (задача B) ─────────────────
def _orthonormal_basis(benchmark: np.ndarray) -> np.ndarray:
    """Ортонормированный базис колонок бенчмарка вместе со свободным членом."""
    n = len(benchmark) if benchmark.ndim else 0
    design = np.column_stack([np.ones(n), benchmark]) if benchmark.size else np.ones((n, 1))
    q, _ = np.linalg.qr(design)
    return q


def _residualize(rows: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Остатки строк (reps, n) после проекции на базис бенчмарка."""
    return rows - (rows @ basis) @ basis.T


def _row_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Корреляция построчно между двумя матрицами уже центрированных остатков."""
    num = np.sum(x * y, axis=1)
    den = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def partial_r2_gain_matrix(
    predictor: np.ndarray, benchmark: np.ndarray, target: np.ndarray,
    block_length: int, n_boot: int, rng: np.random.Generator,
    rank_benchmark_columns: int | None = None,
) -> tuple[float, float, float, float]:
    """
    То же, что `range_lift.partial_r2_gain`, но бенчмарк — МАТРИЦА.

    Ровно та статистика, ради которой ставилась задача B ТЗ: раздел 36 мерил
    `ΔR²` величины сверх одиночной `rv`, здесь тот же `ΔR²` меряется сверх
    B2 (HAR + сезонность). Методика не меняется ни в чём другом — иначе
    «эффект уменьшился» нельзя было бы отличить от «сменили шкалу».

    Что переводится в ранги. Цель и проверяемый предиктор — обязательно
    (тяжёлые хвосты, дисциплина всего `range_lift`). Из бенчмарка — первые
    `rank_benchmark_columns` колонок, то есть колонки `rv`: они тяжелохвостые
    той же природы. Гармоники часа и флаг выходных остаются как есть — ранг
    синуса не синус, и ранжирование сломало бы именно ту конструкцию, ради
    которой колонки заведены. По умолчанию ранжируются все колонки (случай
    одномерного бенчмарка `rv`, полностью совпадающий с разделом 36).

    Приращение считается через частную корреляцию остатков:

        ΔR² = (1 − R²_base) · corr(y⊥, p⊥)²

    где ⊥ — остаток после проекции на бенчмарк со свободным членом. Это то же
    самое, что разность R² двух регрессий, но пересчитывать регрессию на
    каждой реплике не нужно: остаток цели считается один раз, а остаток
    предиктора — одним умножением матриц на весь чанк реплик.

    Значимость односторонняя: бенчмарк и цель остаются на месте, блоками
    пересобирается только предиктор — H0 «предиктор не добавляет ничего
    сверх бенчмарка».

    Возвращает (r2_base, r2_full, r_partial, p_gain).
    """
    benchmark = np.asarray(benchmark, dtype=float)
    if benchmark.ndim == 1:
        benchmark = benchmark[:, None]
    n = len(target)
    limit = benchmark.shape[1] if rank_benchmark_columns is None else rank_benchmark_columns
    ranked_benchmark = np.column_stack([
        _rank(benchmark[:, j]) if j < limit else benchmark[:, j]
        for j in range(benchmark.shape[1])
    ])

    rank_y = _rank(target)
    rank_p = _rank(predictor)
    basis = _orthonormal_basis(ranked_benchmark)

    y_perp = _residualize(rank_y[None, :], basis)
    ss_total = float(np.sum((rank_y - rank_y.mean()) ** 2))
    r2_base = 1.0 - float(np.sum(y_perp ** 2)) / ss_total if ss_total > 0 else float("nan")

    p_perp = _residualize(rank_p[None, :], basis)
    r_partial = float(_row_corr(y_perp, p_perp)[0])
    delta_obs = (1.0 - r2_base) * r_partial ** 2
    r2_full = r2_base + delta_obs

    if n < 20 or not np.isfinite(delta_obs):
        return r2_base, r2_full, r_partial, 1.0

    chunk = max(1, min(n_boot, CHUNK_CELLS // max(n, 1)))
    extreme, done = 0, 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        idx = stationary_block_indices(n, block_length, rng, size)
        rep_perp = _residualize(rank_p[idx], basis)
        rep_corr = _row_corr(np.broadcast_to(y_perp, rep_perp.shape), rep_perp)
        rep_delta = (1.0 - r2_base) * rep_corr ** 2
        extreme += int(np.sum(np.isfinite(rep_delta) & (rep_delta >= delta_obs)))
        done += size
    return r2_base, r2_full, r_partial, float((1 + extreme) / (1 + n_boot))


# ─── Служебное ──────────────────────────────────────────────────────────────
def predictor_autocorr_rows(predictor: pd.Series) -> int:
    """
    Длина блока по автокорреляции самого предиктора, в барах.

    **Считать это ОДИН РАЗ на предиктор.** `autocorr_block_rows` перебирает
    лаги питон-циклом, вызывая `Series.autocorr` на каждом; у величины, чья
    автокорреляция не умирает (`top_vs_retail`), цикл доходит до потолка в
    пять тысяч лагов, и на выборке в двести тысяч строк это десятки секунд.
    Внутри цикла по горизонтам и нормировкам тот же расчёт повторился бы
    шесть раз без единого нового числа — замер вставал на четверть часа на
    ровном месте.
    """
    from btcproc.analysis.range_lift import autocorr_block_rows

    series = predictor.dropna()
    if len(series) < 200:
        return 0
    _, rows = autocorr_block_rows(series, bars_per_day=1, floor=0.2,
                                  max_lag_days=AUTOCORR_MAX_LAG_BARS)
    return int(rows)


def bootstrap_block(ts: pd.Series | np.ndarray, label: str,
                    predictor_rows: int = 0) -> int:
    """
    Длина блока бутстрапа для ячейки замера — как в `measure_deriv_range.py`:
    максимум из перекрытия окон горизонта и автокорреляции самого предиктора
    (её даёт `predictor_autocorr_rows`, посчитанная заранее).

    Оба входа обязательны по разным причинам. Перекрытие окон есть всегда:
    соседние бары делят почти весь горизонт. Автокорреляция предиктора бывает
    сильнее: у `top_vs_retail` она не умирает вовсе в пределах разумного
    потолка, и блок по горизонту оказался бы для неё коротким, то есть тест —
    анти-консервативным.
    """
    by_horizon = block_length_rows(pd.Series(ts), horizon_minutes(label))
    return max(by_horizon, int(predictor_rows))
