"""
Общие фикстуры. Тесты не ходят ни в БД, ни в сеть: всё считается на
синтетических барах, у которых заранее известна структура.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_bars(
    n: int = 12000,
    start: str = "2020-01-01",
    freq: str = "15min",
    seed: int = 7,
    regimes: bool = True,
) -> pd.DataFrame:
    """
    Синтетические свечи.

    regimes=True добавляет чередующиеся режимы (тихий рост / волатильное
    падение / боковик) — тогда у кластеризации есть что находить, и тест на
    «граф не выродился в один узел» имеет смысл.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n, freq=freq, tz="UTC")

    if regimes:
        block = max(n // 12, 50)
        drift, vol = [], []
        for i in range(0, n, block):
            phase = (i // block) % 3
            drift.append(np.full(min(block, n - i), [0.0004, -0.0006, 0.0][phase]))
            vol.append(np.full(min(block, n - i), [0.002, 0.006, 0.001][phase]))
        drift = np.concatenate(drift)[:n]
        vol = np.concatenate(vol)[:n]
    else:
        drift = np.zeros(n)
        vol = np.full(n, 0.003)

    returns = rng.normal(drift, vol)
    close = 10000 * np.exp(np.cumsum(returns))
    spread = close * vol

    high = close + np.abs(rng.normal(0, spread))
    low = close - np.abs(rng.normal(0, spread))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = np.abs(rng.normal(100, 30, n)) * (1 + 5 * vol)

    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum.reduce([high, open_, close]),
            "low": np.minimum.reduce([low, open_, close]),
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "trades": rng.integers(50, 500, n),
            "taker_buy_base": volume * rng.uniform(0.35, 0.65, n),
        },
        index=index,
    )


@pytest.fixture(scope="session")
def bars() -> pd.DataFrame:
    return make_bars()


@pytest.fixture(scope="session")
def context(bars) -> dict:
    from btcproc.ingest.binance import resample

    return {tf: resample(bars, tf) for tf in ("1h", "4h", "1d")}


@pytest.fixture(scope="session")
def features(bars, context) -> pd.DataFrame:
    from btcproc.features.builder import build_features

    return build_features(bars, context)
