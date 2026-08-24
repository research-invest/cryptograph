"""
Контроль FDR по ВСЕМ конфигурациям: есть ли хоть одна значимая.

ТЗ — `crypto-graph/docs/tz_wave_a_24-08-26.md`, задача F; постановка —
`crypto-graph/docs/ideas_math_2026-08-18.md`, §3.2.

## Вопрос, который ни разу не задавали

Кандидат выпускается, если перекос доли исходов по ключу конфигурации
превышает `min_abs_skew`. Ни разу не проверялось, **сколько конфигураций
отличаются от базовой ставки монеты значимо после поправки на
множественность**. У BTC таких ключей 12.5 тысячи; при α = 0.05 без поправки
«значимыми» окажутся шестьсот штук из чистого шума, и именно они, будучи
самыми перекошенными, и попадают в кандидаты.

Исход читается в обе стороны и обе стороны содержательны:

* выжили единицы — вот они, их можно разбирать штучно, вручную, как отдельные
  находки, а не как поток;
* ноль — фильтр кандидатов честнее заменить на разметку: система показывает,
  что было, и не делает вида, что отбирает.

## Одна строка = одна реализация, а не один снимок

Снимки офсетов (`SNAPSHOT_OFFSETS_MIN`) дают до четырёх строк на одну
реализацию перехода, с перекрытием окон исходов минимум на 87.5%. Считать по
ним значимость — значит вчетверо завысить `n` там, где и без того зависимость
недооценена. Поэтому здесь берётся бар самого перехода, по одному на
реализацию.

## Нулёвка и почему p-value не берётся прямо из бутстрапа

Наблюдаемая величина ключа — отклонение доли от базовой ставки монеты,
стандартизованное биномиальной ошибкой. Нулёвка — **блочная перестановка
принадлежности ключу при неподвижных исходах**, тем же приёмом, что в
`lift.block_bootstrap_p`: пересобирается вектор ключей, исходы остаются на
месте. Так сохраняются и автокорреляция исходов, и временная кластеризация
ключей, а ломается только их выравнивание.

Прямой эмпирический p из реплик здесь не годится, и это арифметика, а не
вкус: минимальное достижимое значение — `1/(1+B)`, то есть 0.0005 при
B = 2000, а порог BH для сильнейшего из 12 тысяч ключей при α = 0.10 — порядка
`1e-5`. `lift.resolution_is_sufficient` отвечает на этот вопрос «нет» задолго
до того, как результат будет получен.

Поэтому из реплик оценивается не сам p, а **коэффициент раздутия дисперсии**
`σ_null(n)`: стандартное отклонение null-статистики в бинах по `n`. P-value
считается по нормальному приближению от `z / σ_null(n)`. Величина `σ_null`
печатается: если она около 1, зависимость несущественна, и это тоже результат.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from btcproc.analysis.lift import stationary_block_indices

logger = logging.getLogger(__name__)

#: Ключи с выборкой меньше этой не тестируются. Значение взято равным
#: `candidates.min_effective_sample_size` не для красоты: тестировать надо
#: ровно те конфигурации, по которым система выпускает кандидатов, иначе
#: замер отвечает на вопрос про другое множество.
MIN_ROWS = 30

#: Границы бинов по `n` для оценки раздутия дисперсии — степени двойки.
#: Раздутие зависит от того, насколько тесно во времени сидят реализации
#: ключа, а это, в свою очередь, связано с их числом.
BIN_EDGES = (30, 60, 120, 240, 480, 10 ** 9)


def base_rate(outcomes: np.ndarray) -> float:
    """Базовая ставка монеты — доля роста по всей выборке реализаций."""
    return float(np.mean(outcomes))


def observed_z(keys: np.ndarray, outcomes: np.ndarray, min_rows: int = MIN_ROWS
               ) -> pd.DataFrame:
    """
    Таблица по ключам: `n`, доля, отклонение от базовой ставки, `z`.

    `z` — биномиальная стандартизация относительно ОБЩЕЙ ставки, а не
    относительно «всех остальных ключей». Разница невелика численно (ключ
    занимает доли процента выборки) и существенна логически: сравнение с
    остатком делает тесты зависимыми между собой сильнее, чем они уже есть.
    """
    frame = pd.DataFrame({"key": keys, "outcome": outcomes})
    grouped = frame.groupby("key")["outcome"].agg(["size", "mean"])
    grouped = grouped[grouped["size"] >= min_rows].copy()
    p0 = base_rate(outcomes)
    grouped.columns = ["n", "share"]
    se = np.sqrt(p0 * (1.0 - p0) / grouped["n"].to_numpy(dtype=float))
    grouped["diff"] = grouped["share"] - p0
    grouped["z"] = grouped["diff"] / se
    return grouped.sort_values("z", key=np.abs, ascending=False)


def null_scale(keys: np.ndarray, outcomes: np.ndarray, block_length: int,
               n_boot: int, rng: np.random.Generator,
               min_rows: int = MIN_ROWS) -> tuple[dict, pd.DataFrame]:
    """
    Раздутие дисперсии null-`z` по бинам `n`, из блочных перестановок ключей.

    Возвращает (словарь «правая граница бина → σ», таблицу для печати).
    Бин, в котором не набралось хотя бы 30 наблюдений null-`z`, наследует
    ближайший заполненный слева: пустой бин означает «ключей такого размера
    почти нет», и придумывать им отдельную оценку не на чем.
    """
    n = len(keys)
    p0 = base_rate(outcomes)
    codes, _ = pd.factorize(keys)
    n_keys = int(codes.max()) + 1

    collected: dict[int, list[float]] = {edge: [] for edge in BIN_EDGES}
    for _ in range(n_boot):
        idx = stationary_block_indices(n, block_length, rng, 1)[0]
        shuffled = codes[idx]
        counts = np.bincount(shuffled, minlength=n_keys).astype(float)
        sums = np.bincount(shuffled, weights=outcomes, minlength=n_keys)
        keep = counts >= min_rows
        if not keep.any():
            continue
        share = sums[keep] / counts[keep]
        se = np.sqrt(p0 * (1.0 - p0) / counts[keep])
        z = (share - p0) / se
        sizes = counts[keep]
        for edge, low in zip(BIN_EDGES, (0,) + BIN_EDGES[:-1]):
            mask = (sizes > low) & (sizes <= edge)
            if mask.any():
                collected[edge].extend(z[mask].tolist())

    scale: dict[int, float] = {}
    rows = []
    previous = 1.0
    for edge in BIN_EDGES:
        values = collected[edge]
        if len(values) >= 30:
            sigma = float(np.std(values, ddof=1))
            previous = sigma
        else:
            sigma = previous
        scale[edge] = max(sigma, 1e-6)
        rows.append({"bin": edge, "n_null": len(values), "sigma": scale[edge]})
    return scale, pd.DataFrame(rows)


def scaled_p_values(table: pd.DataFrame, scale: dict) -> pd.Series:
    """
    P-value по нормальному приближению от `z`, поделённого на раздутие бина.

    Двусторонний: вопрос звучит «отличается ли конфигурация от базовой
    ставки», а не «выше ли она». Односторонний тест здесь означал бы, что мы
    заранее знаем сторону, — а система выпускает кандидатов в обе.
    """
    edges = sorted(scale)
    sigmas = []
    for size in table["n"].to_numpy(dtype=float):
        edge = next(e for e in edges if size <= e)
        sigmas.append(scale[edge])
    z_adj = table["z"].to_numpy(dtype=float) / np.array(sigmas)
    return pd.Series([math.erfc(abs(v) / math.sqrt(2.0)) for v in z_adj],
                     index=table.index, name="p")


def permuted_control(keys: np.ndarray, outcomes: np.ndarray, block_length: int,
                     rng: np.random.Generator, min_rows: int = MIN_ROWS
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Негативный контроль: те же ключи, исходы блочно переставлены.

    Прогоняется через ту же процедуру целиком — от `observed_z` до BH.
    Обязан дать около нуля выживших; если не даёт, дефект в процедуре, а не в
    рынке, и результат основного прогона читать нельзя.
    """
    n = len(outcomes)
    idx = stationary_block_indices(n, block_length, rng, 1)[0]
    return keys, outcomes[idx]
