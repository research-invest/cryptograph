"""
Присвоение состояний барам: сглаживание, возраст состояния, переходы,
энтропия траектории.

Сырые метки ближайшего центроида дребезжат на границе кластеров: рынок
формально «переходит» туда-обратно каждые пару баров. Такие переходы не
события, а шум, поэтому смена состояния засчитывается только если новое
состояние продержалось `smoothing_bars` баров подряд.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from btcproc import config

logger = logging.getLogger(__name__)

AGE_BUCKETS = ("age_lt_30", "age_30_60", "age_60_120", "age_gt_120")


def smooth_labels(labels: np.ndarray, min_run: int) -> np.ndarray:
    """
    Гасит дребезг: новое состояние принимается, только если оно удержалось
    min_run баров. Первый бар серии задним числом относится к новому
    состоянию — момент перехода не сдвигается.
    """
    if min_run <= 1 or len(labels) == 0:
        return labels.copy()

    out = np.empty_like(labels)
    current = labels[0]
    pending, pending_start = None, 0

    for i, value in enumerate(labels):
        if value == current:
            pending = None
        else:
            if pending is None or value != pending:
                pending, pending_start = value, i
            if i - pending_start + 1 >= min_run:
                current = value
                # Задним числом переписываем начало подтвердившейся серии.
                out[pending_start:i] = value
                pending = None
        out[i] = current
    return out


def _age_bucket(minutes: np.ndarray) -> np.ndarray:
    return np.select(
        [minutes < 30, minutes < 60, minutes < 120],
        ["age_lt_30", "age_30_60", "age_60_120"],
        default="age_gt_120",
    )


def _trajectory_entropy(labels: np.ndarray, window: int, chunk: int = 50_000) -> np.ndarray:
    """
    Нормированная энтропия Шеннона по окну последних `window` состояний.

    Низкая — рынок шёл в текущее состояние прямо; высокая — метался между
    состояниями. Считается чанками: sliding_window_view на всей истории
    сразу съедает слишком много памяти.
    """
    n = len(labels)
    if n == 0:
        return np.array([], dtype=float)

    uniq = np.unique(labels)
    codes = np.searchsorted(uniq, labels)
    k = len(uniq)
    padded = np.concatenate([np.full(window - 1, codes[0]), codes])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)

    out = np.empty(n, dtype=float)
    log_norm = np.log(min(window, k)) if min(window, k) > 1 else 1.0
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = windows[start:stop]
        counts = np.zeros((stop - start, k), dtype=np.int32)
        rows = np.repeat(np.arange(stop - start), window)
        np.add.at(counts, (rows, block.ravel()), 1)
        p = counts / window
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, -p * np.log(p), 0.0)
        out[start:stop] = terms.sum(axis=1) / log_norm
    return out


def _entropy_bucket(values: np.ndarray) -> np.ndarray:
    return np.select([values < 0.33, values < 0.66], ["low", "medium"], default="high")


def assign_states(
    index: pd.DatetimeIndex,
    labels: np.ndarray,
    smoothing_bars: int | None = None,
    trajectory_window: int | None = None,
) -> pd.DataFrame:
    """
    Разметка всей истории.

    Колонки: group_id, prev_group_id, state_seq, age_minutes, age_bucket,
    entropy, is_transition, transition_id.
    """
    cfg = config.states
    smoothing_bars = cfg.smoothing_bars if smoothing_bars is None else smoothing_bars
    trajectory_window = cfg.trajectory_window if trajectory_window is None else trajectory_window
    bar_minutes = config.data.base_minutes

    smoothed = smooth_labels(np.asarray(labels, dtype=float), smoothing_bars)

    df = pd.DataFrame({"group_id": smoothed}, index=index)
    changed = df["group_id"].ne(df["group_id"].shift(1))
    # Первый бар истории — не переход: предыдущего состояния просто нет.
    changed.iloc[0] = False

    df["state_seq"] = changed.cumsum().astype("int64")
    df["is_transition"] = changed
    df["prev_group_id"] = df["group_id"].shift(1).where(changed).ffill()

    # Возраст = сколько баров прошло с начала текущей серии состояния.
    bars_in_state = df.groupby("state_seq").cumcount()
    df["age_minutes"] = (bars_in_state * bar_minutes).astype(int)
    df["age_bucket"] = _age_bucket(df["age_minutes"].to_numpy())

    entropy = _trajectory_entropy(smoothed, trajectory_window)
    df["entropy_value"] = entropy
    df["entropy"] = _entropy_bucket(entropy)

    df["transition_id"] = np.where(
        df["is_transition"] & df["prev_group_id"].notna(),
        pd.Series(df["prev_group_id"]).map(_fmt).astype(str)
        + "->"
        + pd.Series(df["group_id"], index=df.index).map(_fmt).astype(str),
        None,
    )

    logger.info(
        "Разметка: %d баров, %d переходов, %d состояний",
        len(df), int(df["is_transition"].sum()), df["group_id"].nunique(),
    )
    return df


def _fmt(value: float) -> str:
    """group_id в transition_id пишется целым: «42->1», а не «42.0->1.0»."""
    if pd.isna(value):
        return "na"
    return str(int(value)) if float(value).is_integer() else str(value)
