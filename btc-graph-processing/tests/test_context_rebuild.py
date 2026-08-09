"""
Инкрементальная пересборка старших ТФ.

`live` каждые полчаса пересобирал старшие таймфреймы из ВСЕЙ базовой истории
и перезаписывал их upsert'ом целиком — сотни тысяч строк ради нескольких
свежих. Срезать это можно только при одном условии: результат обязан совпадать
с полным пересчётом бит в бит.

Опора — метка бара равна времени его ОТКРЫТИЯ (label=closed=left). Значит
последний сохранённый бар старшего ТФ и есть граница окна: агрегация от неё
видит ровно те же базовые бары, что и агрегация всей истории.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from btcproc.ingest.binance import resample


def _base(periods: int = 4000) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=periods, freq="15min", tz="UTC")
    rng = np.random.default_rng(42)
    close = pd.Series(100 + rng.standard_normal(periods).cumsum(), index=index)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.uniform(1, 100, periods),
            "quote_volume": rng.uniform(1, 100, periods),
            "trades": rng.integers(1, 100, periods),
            "taker_buy_base": rng.uniform(1, 50, periods),
        },
        index=index,
    )


def _incremental(base: pd.DataFrame, tf: str, boundary: pd.Timestamp) -> pd.DataFrame:
    return resample(base[base.index >= boundary], tf)


def test_tail_matches_full_rebuild():
    """Хвост, посчитанный от границы, совпадает с посчитанным от 2026 года."""
    base = _base()
    for tf in ("1h", "4h", "1d"):
        full = resample(base, tf)
        boundary = full.index[-5]  # «последний сохранённый бар» этого ТФ

        tail = _incremental(base, tf, boundary)

        pd.testing.assert_frame_equal(full.loc[boundary:], tail)


def test_boundary_bar_itself_is_recomputed():
    """
    Граничный бар должен пересчитываться, а не пропускаться: на прошлом
    запуске его окно ещё не закрылось и он был собран частично.
    """
    base = _base()
    full = resample(base, "4h")
    boundary = full.index[-1]

    tail = _incremental(base, "4h", boundary)

    assert tail.index[0] == boundary
    pd.testing.assert_series_equal(full.loc[boundary], tail.loc[boundary])


def test_partial_window_would_differ_if_boundary_were_wrong():
    """
    Проверка, что тест не вырожден: срез НЕ по границе бара действительно
    даёт другой результат — значит совпадение выше содержательно.
    """
    base = _base()
    full = resample(base, "4h")
    boundary = full.index[-3]
    misaligned = boundary + pd.Timedelta(minutes=15)

    tail = resample(base[base.index >= misaligned], "4h")

    assert not full.loc[boundary].equals(tail.loc[boundary])
