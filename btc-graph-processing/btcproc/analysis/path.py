"""
Асимметрия пути: то, чего замер направления не мерил ни разу.

ТЗ — `crypto-graph/docs/tz_wave_a_24-08-26.md`, задача P; постановка —
`crypto-graph/docs/ideas_math_2026-08-18.md`, §4.2 и дешёвая половина §4.3.

## Почему это не повтор раздела 26

Раздел 26 мерил **знак `ret` на фиксированном горизонте** — величину, к
которой асимметрия пути почти не сводится. Рынок может систематически ходить
«сначала вниз на 1%, потом вверх на 3%» и давать при этом ровно 50% по знаку
на 24h. Это не спекуляция: именно такая асимметрия — то, ради чего практики
смотрят на MFE/MAE, и в системе она посчитана (`outcomes.mfe_pct` и
`mae_pct` лежат по всей истории), но никогда не проверялась.

Разметка тройным барьером (López de Prado) нормирует событие на
волатильность: барьеры `±k·σ_t`, предел по времени — горизонт, метка — какой
барьер задет первым. Одно и то же движение в тихом и в бурном рынке
перестаёт быть одним событием.

## Нулёвка, и почему она обязана быть именно такой

**Аналитический якорь.** Для процесса без сноса вероятность достичь `+a`
раньше `−b` равна `b/(a+b)` — **независимо от волатильности** (замена
времени). При симметричных барьерах это ровно 0.5, и никакая сезонность на
это не влияет. Отсюда важное следствие: доля `up` среди РАЗРЕШЁННЫХ случаев
сравнивается с 0.5, а σ влияет только на долю неразрешённых. Якорь нужен как
проверка кода, а не рынка.

**Эмпирическая нулёвка (главная).** Наивная нулёвка «броуновское движение с
постоянной σ» некорректна, и это прямой урок отзыва гейта R (47.4): σ
меняется по часам суток вместе с вероятностью дойти до барьера, и нулёвка без
часа дня даёт ложную асимметрию тем же механизмом. Поэтому суррогатные пути
собираются перестановкой РЕАЛЬНЫХ БАРОВ внутри бина «час дня × день недели»
(`surrogate.seasonal_permutation`): сезонный профиль сохраняется полностью,
асимметрия пути разрушается.

Что нулёвка НЕ сохраняет — кластеризацию волатильности. Это осознанно: она
влияет на долю неразрешённых случаев, а не на сторону среди разрешённых, и
доля неразрешённых печатается отдельно.

## Одновременное касание

Внутри одного бара оба барьера могут быть задеты: `high` выше верхнего,
`low` ниже нижнего. Порядок внутри бара неизвестен — на 15-минутных барах
его нет в данных вообще. Такие случаи идут в отдельную метку `ambiguous` и
НЕ распределяются по сторонам ни по какому правилу: любое правило («считаем,
что сначала был low») внесло бы систематический перекос ровно в измеряемую
величину.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Метки разметки. Порядок фиксирован — по нему считаются доли.
LABELS = ("up", "down", "none", "ambiguous")


def sigma_series(base: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    σ бара в ЛОГАРИФМИЧЕСКОЙ шкале — та, в которой ставятся барьеры.

    ATR, делённый на цену: барьеры обязаны быть относительными, иначе за
    девять лет истории (цена меняется на два порядка) фиксированный барьер
    означал бы в начале истории одно, а в конце другое.

    Известен на баре `t` и в будущее не заглядывает — это ровно то же
    требование, что к знаменателю цели размаха.
    """
    from btcproc.features import indicators as ind

    return (ind.atr(base, window) / base["close"]).replace(0.0, np.nan)


def triple_barrier(base: pd.DataFrame, sigma: pd.Series, k: float,
                   horizon_bars: int) -> np.ndarray:
    """
    Разметка тройным барьером. Возвращает массив кодов меток (индексы LABELS).

    Реализация — `horizon_bars` векторных проходов по массиву, а не окно на
    каждый бар. Скользящее окно (n × h) на трёхстах тысячах баров при h = 96
    — это 230 МБ на каждую из двух колонок; проходов же ровно h, и каждый
    работает только с ещё не разрешёнными барами.

    Барьеры ставятся от `close` бара `t`, а проверка начинается с бара `t+1`:
    касание внутри самого бара входа — это прошлое, а не исход.
    """
    n = len(base)
    high = base["high"].to_numpy(dtype=float)
    low = base["low"].to_numpy(dtype=float)
    close = base["close"].to_numpy(dtype=float)
    width = k * sigma.to_numpy(dtype=float)

    up_level = close * np.exp(width)
    down_level = close * np.exp(-width)

    label = np.full(n, LABELS.index("none"), dtype=np.int8)
    resolved = ~np.isfinite(width)          # бары без σ разрешению не подлежат
    label[resolved] = -1                    # −1 = не размечен вовсе

    for step in range(1, horizon_bars + 1):
        if step >= n:
            break
        source = np.arange(n - step)
        open_bars = source[~resolved[:n - step]]
        if open_bars.size == 0:
            break
        hit_up = high[open_bars + step] >= up_level[open_bars]
        hit_down = low[open_bars + step] <= down_level[open_bars]

        both = open_bars[hit_up & hit_down]
        only_up = open_bars[hit_up & ~hit_down]
        only_down = open_bars[hit_down & ~hit_up]

        label[both] = LABELS.index("ambiguous")
        label[only_up] = LABELS.index("up")
        label[only_down] = LABELS.index("down")
        resolved[both] = True
        resolved[only_up] = True
        resolved[only_down] = True

    # Бары, у которых горизонт не помещается в историю, исходом не обладают.
    label[max(0, n - horizon_bars):] = -1
    return label


def label_shares(labels: np.ndarray) -> dict:
    """Доли меток. `up_share` считается среди РАЗРЕШЁННЫХ (up + down)."""
    valid = labels[labels >= 0]
    total = len(valid)
    if total == 0:
        return {"n": 0}
    counts = {name: int((valid == LABELS.index(name)).sum()) for name in LABELS}
    decided = counts["up"] + counts["down"]
    return {
        "n": total,
        "n_decided": decided,
        "up_share": counts["up"] / decided if decided else float("nan"),
        "none_share": counts["none"] / total,
        "ambiguous_share": counts["ambiguous"] / total,
        **{f"n_{name}": counts[name] for name in LABELS},
    }


def analytic_anchor(k: float) -> float:
    """
    Доля `up` для процесса без сноса при симметричных барьерах.

    Ровно 0.5 — из замены времени: вероятность достичь `+a` раньше `−b` равна
    `b/(a+b)` и от волатильности не зависит вовсе. Функция существует, чтобы
    это число стояло в отчёте явно, а не подразумевалось.
    """
    return 0.5


def surrogate_shares(base: pd.DataFrame, k: float, horizon_bars: int,
                     n_draws: int, rng: np.random.Generator,
                     sigma_window: int = 14) -> np.ndarray:
    """
    Распределение `up_share` по суррогатам с сохранённой сезонностью.

    σ пересчитывается НА СУРРОГАТЕ, а не берётся с реальных данных: барьер,
    поставленный по чужой волатильности, разметил бы суррогат систематически
    иначе, и сравнение перестало бы быть сравнением.
    """
    from btcproc.analysis import surrogate as sg

    out = []
    for _ in range(n_draws):
        fake = sg.surrogate_bars(base, "seasonal", rng)
        labels = triple_barrier(fake, sigma_series(fake, sigma_window), k,
                                horizon_bars)
        shares = label_shares(labels)
        out.append(shares.get("up_share", float("nan")))
    return np.array([v for v in out if np.isfinite(v)])


def surrogate_state_spread(base: pd.DataFrame, groups: np.ndarray, k: float,
                           horizon_bars: int, n_draws: int,
                           rng: np.random.Generator, sigma_window: int = 14,
                           min_rows: int = 200) -> np.ndarray:
    """
    Нулёвочное распределение РАЗМАХА доли `up` между состояниями.

    Без него наблюдённый размах нечитаем, и это не педантизм. Размах — это
    максимум минус минимум по полусотне состояний, то есть статистика
    экстремума: даже при полном отсутствии эффекта она заметно больше нуля и
    растёт с числом состояний. Сравнивать её с нулём — значит находить
    «различие между состояниями» на любых данных, включая случайные метки.

    Устройство нулёвки: бары переставляются сезонным суррогатом, а МЕТКИ
    СОСТОЯНИЙ остаются на своих позициях. Так рвётся ровно проверяемая связь
    («это состояние → такой путь дальше») и сохраняется всё остальное —
    сезонность, доли занятости состояний, их число и размеры выборок.
    """
    from btcproc.analysis import surrogate as sg

    out = []
    for _ in range(n_draws):
        fake = sg.surrogate_bars(base, "seasonal", rng)
        labels = triple_barrier(fake, sigma_series(fake, sigma_window), k,
                                horizon_bars)
        table = by_state(labels, groups, min_rows)
        if len(table) >= 3:
            out.append(float(table["up"].max() - table["up"].min()))
    return np.array(out)


def by_state(labels: np.ndarray, groups: np.ndarray, min_rows: int = 200
             ) -> pd.DataFrame:
    """
    Доля `up` по состояниям графа — вторая половина задачи P.

    Если общая асимметрия отсутствует, а по состояниям различается, это
    результат ПРО ГРАФ. Если не различается — граф и здесь ничего не
    добавляет.
    """
    mask = labels >= 0
    frame = pd.DataFrame({"label": labels[mask], "group": groups[mask]})
    decided = frame[frame["label"].isin(
        [LABELS.index("up"), LABELS.index("down")])]
    grouped = decided.groupby("group")["label"].agg(
        n="size", up=lambda x: float((x == LABELS.index("up")).mean()))
    return grouped[grouped["n"] >= min_rows].sort_values("up")


def state_holdout(labels: np.ndarray, groups: np.ndarray, train_frac: float = 0.7,
                  min_rows: int = 100) -> dict:
    """
    Переживает ли различие по состояниям переход на невиданные данные.

    Обязательная проверка, а не украшение. Разметка состояний обучена на ВСЕЙ
    истории, и доля `up` по состоянию, посчитанная там же, — величина
    in-sample со всеми вытекающими. Раздел 26 журнала — история ровно про
    это: перекос, который система приписывала конфигурации, на отложенной
    части не воспроизвёлся.

    Порядок: доли считаются отдельно на первых `train_frac` и на остальных,
    сравниваются рангами (Spearman) по состояниям, присутствующим в обеих
    частях. Плюс печатается размах на отложенной части — сам по себе, потому
    что высокая корреляция при схлопнувшемся размахе означала бы «порядок
    сохранился, величина исчезла».

    Здесь СОЗНАТЕЛЬНО нет зазора в горизонт между частями: на 300 тысячах
    баров 96 строк на границе меняют оценку в четвёртом знаке, а введение
    зазора потребовало бы объяснять, почему он есть здесь и отсутствует в
    `by_state`. Влияние оценено и признано пренебрежимым — это не забывчивость.
    """
    mask = labels >= 0
    index = np.flatnonzero(mask)
    if len(index) < 4 * min_rows:
        return {}
    cut = int(len(index) * train_frac)
    parts = []
    for piece in (index[:cut], index[cut:]):
        table = by_state(labels[piece], groups[piece], min_rows)
        parts.append(table)
    train, test = parts
    common = train.index.intersection(test.index)
    if len(common) < 4:
        return {"n_common": len(common)}
    rho = float(pd.Series(train.loc[common, "up"]).corr(
        pd.Series(test.loc[common, "up"]), method="spearman"))
    return {
        "n_common": len(common),
        "rho": rho,
        "spread_train": float(train.loc[common, "up"].max()
                              - train.loc[common, "up"].min()),
        "spread_test": float(test.loc[common, "up"].max()
                             - test.loc[common, "up"].min()),
    }


def cochran_armitage(counts_up: np.ndarray, counts_total: np.ndarray,
                     scores: np.ndarray) -> tuple[float, float]:
    """
    Тренд-тест Кохрана — Армитеджа: растёт ли доля монотонно по шкале.

    Используется здесь для проверки «различается ли доля между состояниями
    сильнее случайного», где шкала — ранг состояния по самой доле. Такой
    порядок делает тест **анти-консервативным по построению** (шкала выбрана
    по данным), поэтому его число печатается как справочное, а решение
    принимается по перестановочной нулёвке.
    """
    total = counts_total.sum()
    if total == 0:
        return 0.0, 1.0
    p = counts_up.sum() / total
    mean_score = float((counts_total * scores).sum() / total)
    numerator = float((counts_up * (scores - mean_score)).sum())
    variance = p * (1 - p) * float((counts_total * (scores - mean_score) ** 2).sum())
    if variance <= 0:
        return 0.0, 1.0
    import math

    z = numerator / math.sqrt(variance)
    return z, math.erfc(abs(z) / math.sqrt(2.0))
