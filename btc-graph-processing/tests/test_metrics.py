"""
Тесты загрузчика деривативных метрик (btcproc/ingest/metrics.py).

Без сети и без БД — образующая часть данных синтетическая, ровно как у
test_fear_greed.py. Сетевая сверка карты конвенций — отдельный файл,
tests/test_metrics_convention.py, помеченный `network`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcproc.ingest import metrics


def make_metrics_df(start: str, periods: int, freq: str = "5min", **overrides) -> pd.DataFrame:
    """DataFrame в формате уже распарсенного CSV (после _parse_csv)."""
    index = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    create_time = index.tz_localize(None).astype(str)
    base = {
        "create_time": create_time,
        "symbol": ["BTCUSDT"] * periods,
        "sum_open_interest": np.linspace(70_000, 71_000, periods),
        "sum_open_interest_value": np.linspace(3.0e9, 3.1e9, periods),
        "count_toptrader_long_short_ratio": np.full(periods, 2.4),
        "sum_toptrader_long_short_ratio": np.full(periods, 1.4),
        "count_long_short_ratio": np.full(periods, 2.6),
        "sum_taker_long_short_vol_ratio": np.full(periods, 1.05),
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ── Заголовок ────────────────────────────────────────────────────────────
def test_header_mismatch_raises():
    bad_csv = b"time,symbol,oi\n2024-01-01 00:00:00,BTCUSDT,100\n"
    with pytest.raises(metrics.HeaderMismatch):
        metrics._parse_csv(bad_csv, "BTCUSDT", "2024-01-01")


def test_correct_header_parses():
    header = ",".join(metrics.METRICS_COLUMNS)
    row = "2024-01-15 00:00:00,BTCUSDT,77082.018,3217372598.5128,2.40274115,1.39548418,2.65375410,1.05703456"
    csv = f"{header}\n{row}\n".encode()
    df = metrics._parse_csv(csv, "BTCUSDT", "2024-01-15")
    assert list(df.columns) == metrics.METRICS_COLUMNS
    assert len(df) == 1


# ── Дедупликация (§0.9) ─────────────────────────────────────────────────
def test_dedupe_removes_duplicated_rows():
    df = make_metrics_df("2020-09-01", 288)
    doubled = pd.concat([df, df], ignore_index=True)
    assert len(doubled) == 576

    deduped = metrics.dedupe(doubled)
    assert len(deduped) == 288
    assert deduped["create_time"].nunique() == 288


def test_dedupe_empty_is_noop():
    empty = pd.DataFrame(columns=metrics.METRICS_COLUMNS)
    assert metrics.dedupe(empty).empty


# ── Агрегация 5m → 15m ────────────────────────────────────────────────────
def test_aggregate_window_and_src_rows():
    """3 строки 5m → 1 бар 15m, src_rows=3."""
    df = make_metrics_df("2024-01-01 00:00", 3)
    agg = metrics.aggregate(df, "BTCUSDT", "15m")

    assert len(agg) == 1
    assert agg["ts"].iloc[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert agg["src_rows"].iloc[0] == 3


def test_aggregate_last_vs_mean():
    """oi/ratio — last бара (снимок), taker_ratio — mean (поток)."""
    df = make_metrics_df(
        "2024-01-01 00:00", 3,
        sum_open_interest=[100.0, 200.0, 300.0],
        sum_taker_long_short_vol_ratio=[1.0, 2.0, 3.0],
    )
    agg = metrics.aggregate(df, "BTCUSDT", "15m")
    assert agg["oi"].iloc[0] == pytest.approx(300.0)   # last
    assert agg["taker_ratio"].iloc[0] == pytest.approx(2.0)  # mean


def test_aggregate_partial_bar_has_lower_src_rows():
    """2 из 3 строк в окне → src_rows=2, полный бар не заявляется."""
    df = make_metrics_df("2024-01-01 00:00", 2)  # только T и T+5m, T+10m нет
    agg = metrics.aggregate(df, "BTCUSDT", "15m")
    assert agg["src_rows"].iloc[0] == 2


def test_aggregate_empty_columns_stay_nan_not_zero():
    """
    §0.9: ratio-колонки пустые строки в CSV → NaN после разбора, и агрегация
    обязана пронести NaN дальше, а не подставить 0 или предыдущее значение.
    """
    df = make_metrics_df("2024-01-01 00:00", 3)
    df["count_toptrader_long_short_ratio"] = ""
    agg = metrics.aggregate(df, "BTCUSDT", "15m")
    assert agg["ls_top_acc"].isna().all()
    # ОИ в этот же день остаётся в порядке — дыра по КОЛОНКЕ, не по бару целиком.
    assert agg["oi"].notna().all()
    assert agg["src_rows"].iloc[0] == 3  # строки были, значения в них — нет


def test_aggregate_empty_input():
    empty = pd.DataFrame(columns=metrics.METRICS_COLUMNS)
    agg = metrics.aggregate(empty, "BTCUSDT", "15m")
    assert agg.empty
    assert list(agg.columns) == metrics.DERIV_METRICS_COLUMNS


def test_aggregate_unknown_tf_raises():
    df = make_metrics_df("2024-01-01 00:00", 3)
    with pytest.raises(ValueError):
        metrics.aggregate(df, "BTCUSDT", "2h")


# ── Look-ahead: префикс = полная история ─────────────────────────────────
def test_prefix_matches_full_history():
    """
    Величина на префиксе истории обязана совпасть с величиной на полной —
    инвариант 4 проекта (test_features_do_not_look_ahead), применённый к
    агрегации метрик: окно [T, T+15m) не должно тянуть данные из будущих
    файлов.
    """
    df = make_metrics_df("2024-01-01 00:00", 288)  # ровно сутки, 96 баров 15m
    cut = 200  # обрезаем на середине бара — неполный последний бар префикса

    full = metrics.aggregate(df, "BTCUSDT", "15m")
    prefix = metrics.aggregate(df.iloc[:cut], "BTCUSDT", "15m")

    common = prefix["ts"]
    full_common = full[full["ts"].isin(common)].reset_index(drop=True)
    prefix_common = prefix[prefix["src_rows"] == 3].reset_index(drop=True)
    full_common = full_common[full_common["ts"].isin(prefix_common["ts"])].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        full_common.sort_values("ts").reset_index(drop=True),
        prefix_common.sort_values("ts").reset_index(drop=True),
    )


# ── Хранение: NaN → NULL, не число (см. docstring _store) ────────────────
def test_store_converts_nan_to_none(monkeypatch):
    captured = {}

    def fake_bulk_upsert(table, columns, rows, conflict_columns, **kw):
        captured["rows"] = list(rows)
        return len(captured["rows"])

    monkeypatch.setattr(metrics, "bulk_upsert", fake_bulk_upsert)

    df = make_metrics_df("2024-01-01 00:00", 3)
    df["count_toptrader_long_short_ratio"] = ""
    agg = metrics.aggregate(df, "BTCUSDT", "15m")
    metrics._store(agg)

    row = captured["rows"][0]
    idx = metrics.DERIV_METRICS_COLUMNS.index("ls_top_acc")
    assert row[idx] is None, "NaN обязан стать SQL NULL (None), а не остаться float('nan')"


# ── Конвенция create_time (карта, без сети) ───────────────────────────────
def test_convention_map_covers_full_range():
    for day in ["2020-09-01", "2022-06-15", "2024-07-01", "2025-01-01", "2026-08-14"]:
        assert metrics.create_time_convention(day) in ("start", "end")


def test_convention_map_raises_outside_range():
    with pytest.raises(ValueError):
        metrics.create_time_convention("2019-01-01")
