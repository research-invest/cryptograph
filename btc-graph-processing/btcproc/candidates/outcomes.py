"""
Разметка исходов на горизонте.

Для каждого бара смотрим вперёд ровно на горизонт (24h = 96 баров при 15m) и
считаем три числа:

  ret_pct — куда пришла цена к концу горизонта;
  mfe_pct — максимальное движение «за» (Maximum Favorable Excursion);
  mae_pct — максимальное движение «против» (Maximum Adverse Excursion).

`valid` = у бара есть полный горизонт вперёд и в нём нет пропусков баров.
Именно это разделение даёт `valid_label_count` / `invalid_label_count`
в кандидате: последние сутки истории и участки с дырами в данных
статистику портить не должны.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from btcproc import config

logger = logging.getLogger(__name__)


def compute_outcomes(base: pd.DataFrame, horizon_bars: int | None = None) -> pd.DataFrame:
    """
    base — бары базового ТФ. Возвращает DataFrame с тем же индексом:
    ret_pct, mfe_pct, mae_pct, is_up, valid.
    """
    horizon_bars = horizon_bars or config.data.horizon_bars
    n = len(base)
    out = pd.DataFrame(index=base.index)
    if n == 0:
        return out.assign(ret_pct=[], mfe_pct=[], mae_pct=[], is_up=[], valid=[])

    close = base["close"].to_numpy(dtype=float)
    high = base["high"].to_numpy(dtype=float)
    low = base["low"].to_numpy(dtype=float)

    ret = np.full(n, np.nan)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)

    if n > horizon_bars:
        # Окно [i+1 .. i+H] — исход считается по барам строго после текущего.
        high_windows = np.lib.stride_tricks.sliding_window_view(high, horizon_bars)
        low_windows = np.lib.stride_tricks.sliding_window_view(low, horizon_bars)
        last = n - horizon_bars  # для i >= last горизонта не хватает

        entry = close[:last]
        ret[:last] = (close[horizon_bars:] / entry - 1.0) * 100.0
        mfe[:last] = (high_windows[1:last + 1].max(axis=1) / entry - 1.0) * 100.0
        mae[:last] = (low_windows[1:last + 1].min(axis=1) / entry - 1.0) * 100.0

    out["ret_pct"] = ret
    out["mfe_pct"] = mfe
    out["mae_pct"] = mae
    out["is_up"] = np.where(np.isnan(ret), None, ret > 0)

    # Пропуск баров внутри горизонта делает разметку недостоверной.
    bar_delta = pd.Timedelta(minutes=config.data.base_minutes)
    expected = bar_delta * horizon_bars
    ts = base.index.to_series()
    actual = ts.shift(-horizon_bars) - ts
    out["valid"] = (~np.isnan(ret)) & (actual == expected).to_numpy()

    logger.info(
        "Исходы: %d валидных из %d (горизонт %d баров)",
        int(out["valid"].sum()), n, horizon_bars,
    )
    return out
