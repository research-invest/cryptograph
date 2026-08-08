"""
Вектор признаков, описывающий конфигурацию рынка в момент времени.

Требования к признакам:
  * стационарность — уровень 2018 и 2024 годов должен быть сравним, иначе
    кластеры превратятся в «эпохи цены», а не в состояния рынка;
  * только прошлое — любое значение доступно на закрытии своего бара;
  * мультитаймфреймовость — состояние определяется не только 15-минуткой,
    поэтому старшие ТФ входят в тот же вектор отдельными колонками.

Старший ТФ подмешивается со сдвигом на один бар: пока текущий 4h-бар не
закрыт, используется предыдущий закрытый. Без сдвига в признак попадала бы
информация из будущего относительно 15m-бара.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from btcproc import config
from btcproc.features import indicators as ind

logger = logging.getLogger(__name__)

FEATURE_VERSION = "v1"

# Окна в базовых барах при base_tf=15m.
W_1H, W_4H, W_1D, W_1W, W_1M = 4, 16, 96, 672, 2688


def _windows() -> dict[str, int]:
    """Окна пересчитываются под фактический базовый ТФ из конфига."""
    per = config.data.bars_per
    return {
        "1h": per("1h"),
        "4h": per("4h"),
        "1d": per("1d"),
        "1w": per("1d") * 7,
        "1m": per("1d") * 28,
    }


def _context_features(base_index: pd.DatetimeIndex, ctx: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Три признака со старшего ТФ: RSI, положение в диапазоне, удалённость от EMA.

    shift(1) — обязателен: значение старшего бара становится известно только
    после его закрытия.
    """
    # Префикс tf_ обязателен: базовые признаки уже занимают pos_1d,
    # dist_ema_1d и т.п., а имена колонок должны остаться уникальными.
    out = pd.DataFrame(index=ctx.index)
    atr_tf = ind.atr(ctx, 14)
    out[f"tf{tf}_rsi"] = ind.rsi(ctx["close"], 14) / 100.0
    out[f"tf{tf}_pos"] = ind.position_in_range(ctx, 20)
    out[f"tf{tf}_dist_ema"] = (ctx["close"] - ind.ema(ctx["close"], 20)) / atr_tf
    return out.shift(1).reindex(base_index, method="ffill")


def build_features(
    base: pd.DataFrame,
    context: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    base    — бары базового ТФ (DatetimeIndex UTC).
    context — бары старших ТФ по ключу таймфрейма.

    Возвращает DataFrame признаков без NaN: первые ~4 недели истории уходят
    на прогрев окон.
    """
    if base.empty:
        return pd.DataFrame()

    w = _windows()
    close, high, low = base["close"], base["high"], base["low"]
    log_close = np.log(close)
    atr14 = ind.atr(base, 14).replace(0.0, np.nan)
    rv_d = ind.realized_vol(close, w["1d"])
    rv_w = ind.realized_vol(close, w["1w"])

    f = pd.DataFrame(index=base.index)

    # ── Доходности, нормированные на текущую волатильность ──────────────────
    # Деление на rv·√n приводит доходности разных горизонтов к одной шкале.
    for name, span in (("1h", w["1h"]), ("4h", w["4h"]), ("1d", w["1d"]), ("1w", w["1w"])):
        f[f"ret_{name}"] = (log_close - log_close.shift(span)) / (rv_d * np.sqrt(span))

    # ── Волатильность ───────────────────────────────────────────────────────
    f["rv_ratio"] = np.log(rv_d / rv_w)
    f["rv_rank"] = ind.rolling_rank(rv_d, w["1m"])
    f["atr_rank"] = ind.rolling_rank(atr14 / close, w["1m"])
    f["range_exp"] = np.log(
        ((high - low) / (high - low).rolling(w["1d"], min_periods=w["1h"]).mean()).replace(0, np.nan)
    )

    # ── Положение в диапазоне ───────────────────────────────────────────────
    f["pos_1d"] = ind.position_in_range(base, w["1d"])
    f["pos_1w"] = ind.position_in_range(base, w["1w"])
    f["pos_1m"] = ind.position_in_range(base, w["1m"])
    f["dd_from_high"] = (close - high.rolling(w["1m"], min_periods=w["1d"]).max()) / atr14

    # ── Тренд ───────────────────────────────────────────────────────────────
    ema_d = ind.ema(close, w["1d"])
    ema_w = ind.ema(close, w["1w"])
    f["dist_ema_1d"] = (close - ema_d) / atr14
    f["dist_ema_1w"] = (close - ema_w) / atr14
    f["slope_ema_1d"] = (ema_d - ema_d.shift(w["1d"])) / atr14
    f["slope_ema_1w"] = (ema_w - ema_w.shift(w["1w"])) / atr14
    f["trend_align"] = (ema_d - ema_w) / atr14

    # ── Объём и поток ───────────────────────────────────────────────────────
    log_vol = np.log1p(base["volume"])
    f["vol_z"] = ind.zscore(log_vol, w["1d"])
    f["vol_ratio"] = np.log(
        (base["volume"].rolling(w["1d"], min_periods=w["1h"]).mean()
         / base["volume"].rolling(w["1w"], min_periods=w["1d"]).mean()).replace(0, np.nan)
    )
    if "taker_buy_base" in base:
        buy_share = (base["taker_buy_base"] / base["volume"].replace(0, np.nan)).clip(0, 1)
        f["taker_bias"] = buy_share.rolling(w["1d"], min_periods=w["1h"]).mean() - 0.5

    # ── Моментум и форма распределения ──────────────────────────────────────
    f["rsi"] = ind.rsi(close, 14) / 100.0
    f["up_bar_share"] = (close > base["open"]).rolling(w["1d"], min_periods=w["1h"]).mean()
    f["ret_skew"] = np.log(close).diff().rolling(w["1d"], min_periods=w["1h"]).skew()

    # ── Старшие таймфреймы ──────────────────────────────────────────────────
    for tf, ctx in (context or {}).items():
        if ctx is None or ctx.empty:
            continue
        f = f.join(_context_features(base.index, ctx, tf))

    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    logger.info("Признаков: %d, строк после прогрева: %d", f.shape[1], len(f))
    return f


def feature_names(df: pd.DataFrame) -> list[str]:
    return list(df.columns)


def robust_scale_params(f: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Медиана и IQR по каждому признаку.

    Robust, а не z-score: у доходностей и объёмов тяжёлые хвосты, среднее и
    std по ним смещаются одним флеш-крэшем.
    """
    # copy=True обязателен: на свежих pandas/numpy to_numpy() может вернуть
    # представление поверх данных Series, и оно read-only — присваивание
    # в iqr падало бы с «assignment destination is read-only».
    median = f.median().to_numpy(dtype=float, copy=True)
    iqr = (f.quantile(0.75) - f.quantile(0.25)).to_numpy(dtype=float, copy=True)
    iqr[iqr == 0] = 1.0
    return {"median": median, "iqr": iqr}


def apply_scale(f: pd.DataFrame, params: dict[str, np.ndarray], clip: float = 5.0) -> np.ndarray:
    """Нормировка + обрезка хвостов: единичный выброс не должен тянуть кластер."""
    values = (f.to_numpy(dtype=float) - params["median"]) / params["iqr"]
    return np.clip(values, -clip, clip)
