"""
Деривативные метрики Binance USD-M (открытый интерес, long/short ratio,
давление тейкеров) как источник сигнала.

Постановка, гейты и обоснования — `docs/tz_deriv_ingest_14-08-26.md`. Разбор
проекта — раздел 31 `development_log.md` (D1: потолок в признаках →
расширять источники, а не структуру графа) — и это первый источник, который
меряет НЕ цену: открытый интерес — позиционирование, а не движение.

## Чем этот источник отличается и от SMC, и от FGI

Величина не считается из баров монеты (как SMC), но и не требует сдвига на
сутки (как FGI). Таблица `deriv_metrics` уже ПОМОНЕТНАЯ и уже на сетке
базового ТФ (`btcproc/ingest/metrics.py`, `aggregate`): строка для бара `T`
построена окном `[T, T+15m)` и известна к его закрытию ПО ПОСТРОЕНИЮ —
доказательство перенесено в докстринг `metrics.aggregate` целиком. Поэтому
джойн здесь — простой `reindex` по `ts`, без дополнительного `shift`.

## Три канала (§1.1 ТЗ)

* B2 — непрерывные величины (`oi_chg_1h`, `oi_chg_1d`, `oi_rank`,
  `ls_retail_z`, `top_vs_retail`, `taker_z`) → кандидаты в ПРИЗНАКИ, но не
  сейчас: фаза 3 не стартует автоматически, как и у SMC/FGI;
* B3 — `oi_vs_price` (`sign(ret_1h) * sign(oi_chg_1h)`) — величина с ТРЕМЯ
  значениями, в вектор признаков не годится (`robust_scale_params` считает
  IQR, а у трёхзначной величины он вырождается, и после `clip(±5)` она
  превращается в доминирующее направление кластеризации — тот же класс
  ошибки, что «признак-расстояние с потолком» в таблице граблей
  `extending_features.md`) → четыре взаимно исключающих КОНТЕКСТНЫХ атома;
* остальные кандидаты в контекстные атомы (`oi_spike`, `oi_flush`,
  `retail_long_extreme`, `retail_short_extreme`, `top_vs_retail_divergence`)
  здесь СОЗНАТЕЛЬНО не заводятся — их пороги калибрует фаза 1
  (`scripts/deriv_frequencies.py`), а не код (§B3 ТЗ: «пороги — предмет
  фазы 1, не выдумывать здесь»).

## Смешение площадок (§1.2 ТЗ)

Бары BTC/ETH/SOL — Binance СПОТ, метрики — Binance USD-M ФЬЮЧЕРС (у HYPE ещё
и биржи разные: Bybit спот / Binance фьючерс). Спот и перп одной биржи —
разные рынки, поэтому кросс-рыночных отношений («ОИ делить на спотовый
объём») здесь избегается там, где есть однорыночная замена: `oi_rank` —
ранг в скользящем окне, стационарен по построению и знаменателя не требует
вовсе. Это заменило `oi_norm_rank = rolling_rank(oi / volume, …)` из первой
редакции ТЗ именно по этой причине.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from btcproc import config
from btcproc.features import _cache
from btcproc.features import indicators as ind

logger = logging.getLogger(__name__)

# Контекстные атомы — по правилу раздела 1 extending_features.md: величина с
# механизмом (не корреляцией), но НЕ непрерывная и НЕ редкое происшествие →
# контекст. Взаимно исключающие квадранты «цена/ОИ» (§B3 ТЗ):
#   цена вверх + ОИ вверх   = набирают новые лонги
#   цена вверх + ОИ вниз    = закрывают шорты
#   цена вниз  + ОИ вверх   = набирают новые шорты
#   цена вниз  + ОИ вниз    = закрывают лонги
CONTEXT_CANDIDATES = [
    "oi_up_price_up",
    "oi_up_price_down",
    "oi_down_price_up",
    "oi_down_price_down",
]

# Непрерывные величины-кандидаты в признаки (B2 ТЗ). В вектор кластеризации
# не попадают, пока DERIV_FEATURES_ENABLED выключен (фаза 3).
FEATURE_CANDIDATES = [
    "oi_chg_1h",
    "oi_chg_1d",
    "oi_rank",
    "ls_retail_z",
    "top_vs_retail",
    "taker_z",
]

BOOLEAN_COLUMNS = CONTEXT_CANDIDATES
ALL_COLUMNS = BOOLEAN_COLUMNS + FEATURE_CANDIDATES


def _windows() -> dict[str, int]:
    per = config.data.bars_per
    return {"1h": per("1h"), "1d": per("1d"), "1m": per("1d") * 28}


def _empty_like(base: pd.DataFrame) -> pd.DataFrame:
    """Атом = False, признак = NaN — метрик нет вовсе (до metrics_start или ingest-metrics ещё не запускался)."""
    out = pd.DataFrame(index=base.index)
    for name in BOOLEAN_COLUMNS:
        out[name] = False
    for name in FEATURE_CANDIDATES:
        out[name] = np.nan
    return out[ALL_COLUMNS]


def build_deriv(
    base: pd.DataFrame,
    metrics_frame: pd.DataFrame | None = None,
    symbol: str | None = None,
    cfg: config.DerivConfig | None = None,
) -> pd.DataFrame:
    """
    Деривативные величины на каждый бар: 4 контекстных атома-квадранта (B3),
    6 непрерывных величин-кандидатов в признаки (B2).

    base          — бары базового ТФ (DatetimeIndex UTC).
    metrics_frame — `deriv_metrics` монеты, индекс по `ts`; по умолчанию
                    читается из БД (`ingest.metrics.load_deriv_metrics`).
                    Параметр существует ради тестов и скриптов замера — они
                    не должны ходить в БД, чтобы проверить джойн.

    Возвращает DataFrame с индексом base и колонками ALL_COLUMNS. Булевы —
    bool (False там, где данных нет), непрерывные — float, с NaN там, где
    метрик для бара нет (до metrics_start монеты, дыра в архиве или колонка
    пуста у поставщика, §0.9).
    """
    cfg = cfg or config.deriv
    if base.empty:
        return pd.DataFrame(columns=ALL_COLUMNS)

    if metrics_frame is None:
        from btcproc.ingest import metrics as metrics_ingest

        symbol = symbol or config.data.symbol
        metrics_frame = metrics_ingest.load_deriv_metrics(symbol, config.data.base_tf)

    if metrics_frame.empty:
        return _empty_like(base)

    # Строка metrics для бара T построена окном [T, T+15m) и известна к его
    # закрытию ПО ПОСТРОЕНИЮ (ingest.metrics.aggregate) — reindex БЕЗ сдвига,
    # в отличие от FGI, где дневной агрегат сдвигается на сутки вперёд.
    aligned = metrics_frame.reindex(base.index)

    w = _windows()
    oi = aligned["oi"].astype(float)
    ls_top_pos = aligned["ls_top_pos"].astype(float)
    ls_global = aligned["ls_global"].astype(float)
    taker_ratio = aligned["taker_ratio"].astype(float)

    # 0.0 в oi/ls_global — не легитимное значение (открытый интерес активного
    # перпа не бывает нулевым; ls_global — отношение чисел аккаунтов, тоже
    # строго положительное), а глитч фида Binance: на реальной истории
    # BTCUSDT встречается 116 баров с oi=0.0, разбросанных по 2021–2025,
    # src_rows=3 (то есть это не дыра, а испорченное значение). log(x/0) даёт
    # -inf молча — та же дисциплина, что atr14.replace(0.0, np.nan) в
    # builder.py: 0 в знаменателе логарифмируемого отношения превращается в
    # NaN («не доверяем»), а не в бесконечность.
    oi_safe = oi.replace(0.0, np.nan)
    ls_global_safe = ls_global.replace(0.0, np.nan)

    out = pd.DataFrame(index=base.index)

    # ── B2: непрерывные величины ─────────────────────────────────────────
    out["oi_chg_1h"] = np.log(oi_safe / oi_safe.shift(w["1h"]))
    out["oi_chg_1d"] = np.log(oi_safe / oi_safe.shift(w["1d"]))
    # Ранг и z-score считаются по ОЧИЩЕННЫМ рядам, а не по сырым: иначе
    # решение выше («0.0 — глитч фида, а не значение») действовало бы только
    # на логарифмы. Глитч-ноль в сыром ряду получает ранг ~0.0 —
    # «экстремально низкий открытый интерес» — и тащит за собой ранги соседей
    # по всему месячному окну; в z-score он точно так же смещает и среднее, и
    # разброс на сутки вперёд и назад (аудит 2026-08-15, B5). NaN здесь
    # безопасен: rolling.rank/mean/std его пропускают, а на самом баре
    # остаётся NaN — то самое «не доверяем».
    out["oi_rank"] = ind.rolling_rank(oi_safe, w["1m"])
    out["ls_retail_z"] = ind.zscore(ls_global_safe, w["1d"])
    out["top_vs_retail"] = np.log(ls_top_pos / ls_global_safe)
    out["taker_z"] = ind.zscore(taker_ratio, w["1d"])
    for name in FEATURE_CANDIDATES:
        out[name] = out[name].replace([np.inf, -np.inf], np.nan)

    # ── B3: квадранты oi_vs_price — ТОЛЬКО атомы, не признак (см. докстринг
    # модуля). NaN во входе (метрики/цена недоступны) даёт False на всех
    # четырёх — сравнение NaN > 0 / NaN < 0 в pandas само возвращает False,
    # и это ровно нужное поведение: «неизвестно» не должно выглядеть как
    # сработавший атом.
    ret_1h = np.log(base["close"] / base["close"].shift(w["1h"]))
    price_up, price_down = ret_1h > 0, ret_1h < 0
    oi_up, oi_down = out["oi_chg_1h"] > 0, out["oi_chg_1h"] < 0

    out["oi_up_price_up"] = price_up & oi_up
    out["oi_up_price_down"] = price_down & oi_up
    out["oi_down_price_up"] = price_up & oi_down
    out["oi_down_price_down"] = price_down & oi_down
    for name in BOOLEAN_COLUMNS:
        out[name] = out[name].fillna(False).astype(bool)

    return out[ALL_COLUMNS]


# Пара «ключ+значение» пишется одним присваиванием — иначе параллельные
# прогоны админки подмешивают монете чужие данные, см. _cache.py.
_CACHE = _cache.SingleEntryCache()


def _fingerprint(base: pd.DataFrame, metrics_frame: pd.DataFrame | None,
                 symbol: str | None, cfg: config.DerivConfig) -> tuple:
    """Дешёвый отпечаток входа — по образцу smc._fingerprint / fear_greed._fingerprint."""
    metrics_print: tuple = ()
    if metrics_frame is not None and not metrics_frame.empty:
        metrics_print = (
            len(metrics_frame), metrics_frame.index[0], metrics_frame.index[-1],
            float(metrics_frame["oi"].iat[-1]) if "oi" in metrics_frame else None,
        )
    if base.empty:
        return (0, None, None, symbol, metrics_print, cfg)
    return (
        len(base), base.index[0], base.index[-1],
        float(base["close"].iat[0]), float(base["close"].iat[-1]),
        symbol, metrics_print, cfg,
    )


def build_deriv_cached(
    base: pd.DataFrame,
    metrics_frame: pd.DataFrame | None = None,
    symbol: str | None = None,
    cfg: config.DerivConfig | None = None,
) -> pd.DataFrame:
    """
    То же, что build_deriv, но помнит последний результат.

    За один прогон величины нужны дважды — в признаках и в атомах — по одним
    и тем же барам. Кэш на одну запись: прогон обрабатывает монеты
    последовательно, вторая пара обращений вытесняет первую.
    """
    cfg = cfg or config.deriv
    if base.empty:
        return build_deriv(base, metrics_frame, symbol, cfg)

    key = _fingerprint(base, metrics_frame, symbol, cfg)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    value = build_deriv(base, metrics_frame, symbol, cfg)
    _CACHE.put(key, value)
    return value
