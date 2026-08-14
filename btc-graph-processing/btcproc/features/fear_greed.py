"""
Fear & Greed Index (alternative.me) как внешний источник сигнала.

Постановка и обоснования — `docs/task_fear_greed.md`. Коротко: индекс
заводится не для предсказания направления (это закрыто замерами, разделы 26
и 31 журнала btcproc), а как предиктор РАЗМАХА и описательный контекст.

## Чем этот источник отличается от `smc.py`

SMC считает величины ИЗ баров монеты — каждая монета получает свой расчёт.
Здесь наоборот: ряд общерыночный, один на все монеты (`external_daily`,
`btcproc/ingest/external.py`), и «расчёт» — это джойн внешнего ряда на индекс
баров, а не проход по свечам. Отсюда два следствия:

* стационарность и масштабная инвариантность выполняются тривиально —
  величина не зависит от цены вовсе (докстринг ниже поясняет, тесты всё
  равно пишутся, они дешёвые);
* сетевого похода здесь нет: `external_daily` заполняет отдельная команда
  CLI (`fetch-external`), этот модуль только читает таблицу.

## Look-ahead — единственное место, где легко ошибиться

Значение за сутки D характеризует их целиком и известно не раньше конца дня.
Правило: **значение с датой D применяется к барам, начинающимся с D+1 00:00
UTC** — тот же `shift(1)`, что для старших ТФ в `features/builder.py`, только
с шагом в сутки, а не в бар. Сдвиг делается ОДИН раз, для всей `_daily_metrics`
разом (значение, атомы экстремума и флип — все производные одного и того же
дневного факта), поэтому шаг look-ahead не может разойтись между колонками.

## Фаза 3 оставлена нерешённой сознательно

`fgi_residual` (остаток OLS-регрессии `fgi_level` на весь вектор из 32
признаков, `docs/task_fear_greed.md` §1.2 и §4.1) в `FEATURE_CANDIDATES`
здесь НЕТ. Причина архитектурная: `FeatureSource.compute(base)` получает
только сырые бары, а для остатка нужен уже посчитанный вектор признаков —
которого на этом шаге ещё не существует (сам источник — часть его сборки).
Считать его здесь через собственный, упрощённый набор регрессоров означало
бы измерять гейт N (§1.2 и `scripts/fgi_frequencies.py`) на одном наборе
предикторов, а поставлять в продакшен — на другом: ровно то расхождение
измерения и того, что едет в проде, которого проект избегает везде. Замер
гейта N и градация по остатку (§5.1, §5.4) — задача измерительных скриптов,
у них есть полный вектор признаков через `features/builder.py`. Провести
`fgi_residual` в продакшен-вектор (фаза 3) — отдельное архитектурное решение
(как передать источнику остальной вектор), и оно намеренно не принимается
здесь: фаза 3 не стартует автоматически.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from btcproc import config

logger = logging.getLogger(__name__)

# Контекстные атомы — по правилу раздела 1 extending_features.md все четыре
# состояния/флаги, не происшествия: активны заметную долю времени, в
# signature не годятся и не должны туда попадать.
CONTEXT_CANDIDATES = [
    "fear_extreme",
    "greed_extreme",
    "sentiment_flip_up",
    "sentiment_flip_down",
]

# Непрерывные величины. fgi_residual сюда сознательно не входит — см. докстринг.
FEATURE_CANDIDATES = [
    "fgi_level",
    "fgi_change_7d",
]

BOOLEAN_COLUMNS = CONTEXT_CANDIDATES
ALL_COLUMNS = BOOLEAN_COLUMNS + FEATURE_CANDIDATES


def _daily_metrics(raw: pd.Series, cfg: config.FearGreedConfig) -> pd.DataFrame:
    """
    Величины на СВОИХ сутках D — до сдвига на день вперёд.

    raw — дневной ряд индекса, индекс DatetimeIndex по полуночи UTC, без
    дыр (пропуски внутри истории намеренно не заполняются: NaN здесь честно
    означает «поставщик не отдал значение за этот день», а не «данных нет
    вовсе», как для периода до 2018-02-01).
    """
    out = pd.DataFrame(index=raw.index)
    out["value"] = raw
    out["fgi_level"] = (raw - 50.0) / 50.0
    prev_week = raw.shift(cfg.change_window_days)
    out["fgi_change_7d"] = (raw - prev_week) / 50.0

    out["fear_extreme"] = raw < cfg.fear_extreme_below
    out["greed_extreme"] = raw > cfg.greed_extreme_above

    mid = cfg.flip_midpoint
    prev_day = raw.shift(1)
    out["sentiment_flip_up"] = (prev_day < mid) & (raw >= mid)
    out["sentiment_flip_down"] = (prev_day >= mid) & (raw < mid)
    return out


def _empty_like(base: pd.DataFrame) -> pd.DataFrame:
    """Атом = False, признак = NULL — период без данных (docs §3.1)."""
    out = pd.DataFrame(index=base.index)
    for name in BOOLEAN_COLUMNS:
        out[name] = False
    for name in FEATURE_CANDIDATES:
        out[name] = np.nan
    return out[ALL_COLUMNS]


def build_fear_greed(
    base: pd.DataFrame,
    daily: pd.DataFrame | None = None,
    cfg: config.FearGreedConfig | None = None,
) -> pd.DataFrame:
    """
    FGI-величины на каждый бар: 4 контекстных атома, 2 признака.

    base  — бары базового ТФ (DatetimeIndex UTC).
    daily — дневной ряд индекса (индекс по суткам UTC, колонка `value`);
            по умолчанию читается из `external_daily` через
            `ingest.external.load_external_daily`. Параметр существует ради
            тестов и скриптов замера — они не должны ходить в БД, чтобы
            проверить джойн.

    Возвращает DataFrame с индексом base и колонками ALL_COLUMNS. Булевы —
    bool (без атома, если данных нет), непрерывные — float, с NaN там, где
    сутки D-1 не имеют значения (до 2018-02-01 или дыра в истории).
    """
    cfg = cfg or config.fgi
    if base.empty:
        return pd.DataFrame(columns=ALL_COLUMNS)

    if daily is None:
        from btcproc.ingest import external

        daily = external.load_external_daily(external.FEAR_GREED_SERIES)

    if daily.empty:
        return _empty_like(base)

    raw = daily["value"].astype(float)
    index = pd.DatetimeIndex(raw.index)
    # Зона обязана быть UTC, и приводится явно. Из `external_daily` ряд
    # приходит tz-aware, но параметр `daily` существует ради скриптов и
    # тестов — а ряд, прочитанный из JSON или CSV, приходит НАИВНЫМ, и тогда
    # `reindex` ниже не совпал бы ни с одним баром и вернул бы сплошной NaN.
    # Молча: ни ошибки, ни предупреждения, просто «индекса как будто нет».
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    raw.index = index.normalize()
    if raw.index.duplicated().any():
        # Не должно происходить (PK в external_daily по (series, day)), но
        # тихо ронять весь расчёт из-за одной дублирующей строки — хуже,
        # чем оставить последнее значение.
        raw = raw[~raw.index.duplicated(keep="last")]
    raw = raw.sort_index()

    # Календарь без дыр. `shift()` в `_daily_metrics` двигает по ПОЗИЦИЯМ, а в
    # истории индекса дыры есть (2018-04-14…16 и 2024-10-26 — 3113 записей при
    # 3117 сутках). Без выравнивания «вчера» для 2018-04-17 оказалось бы
    # 04-13, а окно `fgi_change_7d` — одиннадцатью календарными сутками вместо
    # семи. Пропуск становится NaN: атом на нём False, признак пуст — то есть
    # «мы не знаем», а не «перехода не было».
    if len(raw) > 1:
        raw = raw.reindex(pd.date_range(raw.index[0], raw.index[-1], freq="D"))

    metrics = _daily_metrics(raw, cfg)

    # Look-ahead: значение суток D известно не раньше их конца → применяется
    # к барам, начинающимся с D+1 00:00 UTC.
    shifted = metrics.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)

    bar_days = base.index.normalize()
    broadcast = shifted.reindex(bar_days)
    broadcast.index = base.index

    out = pd.DataFrame(index=base.index)
    for name in BOOLEAN_COLUMNS:
        column = broadcast[name]
        out[name] = np.where(column.notna(), column, False).astype(bool)
    for name in FEATURE_CANDIDATES:
        out[name] = broadcast[name].astype(float)

    return out[ALL_COLUMNS]


_CACHE: dict = {}


def _fingerprint(base: pd.DataFrame, daily: pd.DataFrame | None,
                 cfg: config.FearGreedConfig) -> tuple:
    """Дешёвый отпечаток входа — по образцу smc._fingerprint."""
    daily_print: tuple = ()
    if daily is not None and not daily.empty:
        daily_print = (len(daily), daily.index[0], daily.index[-1],
                       float(daily["value"].iat[-1]))
    if base.empty:
        return (0, None, None, daily_print, cfg)
    return (
        len(base), base.index[0], base.index[-1],
        float(base["close"].iat[0]), float(base["close"].iat[-1]),
        daily_print, cfg,
    )


def build_fear_greed_cached(
    base: pd.DataFrame,
    daily: pd.DataFrame | None = None,
    cfg: config.FearGreedConfig | None = None,
) -> pd.DataFrame:
    """
    То же, что build_fear_greed, но помнит последний результат.

    За один прогон FGI-величины нужны дважды — в признаках и в атомах, — и
    оба раза по одним и тем же барам. Кэш на одну запись: прогон обрабатывает
    монеты последовательно, вторая пара обращений вытесняет первую.
    """
    cfg = cfg or config.fgi
    if base.empty:
        return build_fear_greed(base, daily, cfg)

    key = _fingerprint(base, daily, cfg)
    cached = _CACHE.get("key")
    if cached == key:
        return _CACHE["value"]

    value = build_fear_greed(base, daily, cfg)
    _CACHE["key"], _CACHE["value"] = key, value
    return value
