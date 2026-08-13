"""
Базовые индикаторы. Считаются только по прошлым барам — никакого заглядывания
вперёд, иначе вся историческая статистика кандидатов становится фикцией.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span // 2).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    # avg_loss == 0 означает движение только вверх → RSI = 100.
    return out.fillna(100.0).where(avg_gain.notna(), np.nan)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    """Стандартное отклонение лог-доходностей, в долях (не годовое)."""
    return np.log(close).diff().rolling(window, min_periods=window // 2).std()


def position_in_range(df: pd.DataFrame, window: int) -> pd.Series:
    """0 — на минимуме окна, 1 — на максимуме."""
    high = df["high"].rolling(window, min_periods=window // 2).max()
    low = df["low"].rolling(window, min_periods=window // 2).min()
    span = (high - low).replace(0.0, np.nan)
    return ((df["close"] - low) / span).clip(0.0, 1.0)


def rolling_rank(series: pd.Series, window: int) -> pd.Series:
    """
    Перцентиль текущего значения в своём окне [0..1].

    Даёт признаку стационарность: абсолютный уровень волатильности 2018 и 2024
    несравним, а ранг внутри окна — вполне.
    """
    return series.rolling(window, min_periods=window // 4).rank(pct=True)


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std().replace(0.0, np.nan)
    return (series - mean) / std
