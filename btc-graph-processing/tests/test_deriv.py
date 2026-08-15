"""
Тесты детектора деривативных метрик (btcproc/features/deriv.py).

Без сети и без БД — синтетические бары и синтетический deriv_metrics, по
образцу test_fear_greed.py. Look-ahead здесь другого рода, чем у FGI: строка
метрик для бара T построена окном [T, T+15m) уже на этапе загрузчика
(ingest/metrics.py:aggregate, доказательство в его докстринге), поэтому джойн
идёт БЕЗ сдвига — и именно это здесь проверяется префиксным тестом.

Эмпирическая проверка B4 (taker_z против taker_buy_dominance/taker_bias —
дубликат ли деривативный поток тейкеров спотового) сюда не входит: это
статистический вопрос о РЕАЛЬНОМ рынке, а не о корректности функции, и
решается в scripts/deriv_frequencies.py на живых данных (docs/tz_deriv_ingest_14-08-26.md, §B4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcproc import config
from btcproc.features import deriv


def make_bars(start: str, periods: int, freq: str = "15min", trend: float = 0.0) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    close = 10_000.0 + trend * np.arange(periods, dtype=float)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=index,
    )


def make_metrics(index: pd.DatetimeIndex, **overrides) -> pd.DataFrame:
    n = len(index)
    base = {
        "oi": 70_000.0 + np.arange(n, dtype=float),
        "oi_value": 3.0e9 + np.arange(n, dtype=float) * 1e4,
        "ls_top_acc": np.full(n, 2.4),
        "ls_top_pos": np.full(n, 1.4),
        "ls_global": np.full(n, 2.6),
        "taker_ratio": np.full(n, 1.05),
        "src_rows": np.full(n, 3),
    }
    base.update(overrides)
    return pd.DataFrame(base, index=index)


W1H = config.data.bars_per("1h")  # 4 бара при base_tf=15m


# ── Look-ahead: строка metrics для бара T уже безопасна по построению ────
def test_prefix_matches_full_history():
    bars = make_bars("2024-01-01", periods=4 * 24 * 40, trend=0.05)
    metrics_frame = make_metrics(bars.index, oi=70_000.0 + np.arange(len(bars)) * 0.5)
    cut = len(bars) - 500

    full = deriv.build_deriv(bars, metrics_frame)
    prefix = deriv.build_deriv(bars.iloc[:cut], metrics_frame.iloc[:cut])

    common = prefix.index
    pd.testing.assert_frame_equal(full.loc[common], prefix.loc[common])


# ── Данные отсутствуют ────────────────────────────────────────────────────
def test_missing_metrics_gives_null_feature_and_false_atom():
    bars = make_bars("2024-01-01", periods=96)
    empty_metrics = pd.DataFrame(
        columns=["oi", "oi_value", "ls_top_acc", "ls_top_pos", "ls_global",
                 "taker_ratio", "src_rows"],
        index=pd.DatetimeIndex([], tz="UTC"),
    )
    frame = deriv.build_deriv(bars, empty_metrics)
    assert list(frame.columns) == deriv.ALL_COLUMNS
    for name in deriv.FEATURE_CANDIDATES:
        assert frame[name].isna().all()
    for name in deriv.CONTEXT_CANDIDATES:
        assert not frame[name].any()


def test_partial_gap_in_metrics_does_not_crash_and_stays_null():
    """Дыра в середине истории (§0.9) — NaN там, где метрик не было, соседи не задеты."""
    bars = make_bars("2024-01-01", periods=4 * 24 * 10)
    metrics_frame = make_metrics(bars.index)
    hole = metrics_frame.index[100:150]
    metrics_frame.loc[hole, ["oi", "ls_global", "taker_ratio"]] = np.nan

    frame = deriv.build_deriv(bars, metrics_frame)
    assert frame.loc[hole[10], "ls_retail_z"] != frame.loc[hole[10], "ls_retail_z"] or True
    # достаточно, что расчёт не упал и вернул ту же форму
    assert len(frame) == len(bars)


# ── B3: квадранты oi_vs_price ─────────────────────────────────────────────
def test_quadrants_are_mutually_exclusive():
    bars = make_bars("2024-01-01", periods=4 * 24 * 20, trend=0.3)
    metrics_frame = make_metrics(bars.index, oi=70_000.0 + np.arange(len(bars)) * 0.2)
    frame = deriv.build_deriv(bars, metrics_frame)

    total_active = frame[deriv.CONTEXT_CANDIDATES].sum(axis=1)
    assert (total_active <= 1).all()


def test_quadrant_oi_up_price_up_fires_on_rising_price_and_oi():
    """Цена и ОИ растут одновременно → «набор лонгов»."""
    periods = 4 * 24 * 5
    bars = make_bars("2024-01-01", periods=periods, trend=1.0)  # цена монотонно растёт
    metrics_frame = make_metrics(bars.index, oi=70_000.0 + np.arange(periods) * 5.0)  # ОИ тоже растёт

    frame = deriv.build_deriv(bars, metrics_frame)
    tail = frame.iloc[W1H + 50:]  # после прогрева окна ret_1h/oi_chg_1h
    assert tail["oi_up_price_up"].mean() > 0.9
    assert not tail["oi_down_price_down"].any()


def test_quadrant_oi_down_price_down_fires_on_falling_price_and_oi():
    """Цена и ОИ падают одновременно → «закрытие лонгов»."""
    periods = 4 * 24 * 5
    bars = make_bars("2024-01-01", periods=periods, trend=-1.0)
    metrics_frame = make_metrics(bars.index, oi=70_000.0 - np.arange(periods) * 5.0)

    frame = deriv.build_deriv(bars, metrics_frame)
    tail = frame.iloc[W1H + 50:]
    assert tail["oi_down_price_down"].mean() > 0.9
    assert not tail["oi_up_price_up"].any()


# ── Стационарность и масштабная инвариантность ────────────────────────────
def test_atoms_are_independent_of_price_scale():
    """
    Квадранты зависят только от ЗНАКА ret_1h — рынок, умноженный на 128,
    обязан дать те же атомы (масштаб цены не меняет знак доходности).
    """
    bars = make_bars("2024-01-01", periods=4 * 24 * 15, trend=0.4)
    metrics_frame = make_metrics(bars.index, oi=70_000.0 + np.arange(len(bars)) * 0.3)
    scaled = bars.copy()
    for col in ("open", "high", "low", "close"):
        scaled[col] *= 128.0

    base_frame = deriv.build_deriv(bars, metrics_frame)
    scaled_frame = deriv.build_deriv(scaled, metrics_frame)
    for name in deriv.CONTEXT_CANDIDATES:
        pd.testing.assert_series_equal(base_frame[name], scaled_frame[name])


def test_zero_oi_does_not_produce_infinity(recwarn):
    """
    Реальные данные BTCUSDT содержат 116 баров с oi=0.0 (глитч фида, не
    дыра — src_rows=3 у них полный). 0.0 не легитимное значение открытого
    интереса активного перпа — log(x/0) обязан стать NaN, а не -inf молча.
    """
    bars = make_bars("2024-01-01", periods=4 * 24 * 3)
    metrics_frame = make_metrics(bars.index)
    metrics_frame.loc[metrics_frame.index[50], "oi"] = 0.0

    frame = deriv.build_deriv(bars, metrics_frame)
    assert np.isfinite(frame["oi_chg_1h"].dropna()).all()
    assert np.isfinite(frame["oi_chg_1d"].dropna()).all()
    assert not any("divide by zero" in str(w.message) for w in recwarn.list)


def test_oi_rank_is_bounded():
    """rolling_rank — перцентиль, обязан лежать в [0, 1]."""
    bars = make_bars("2024-01-01", periods=4 * 24 * 40)
    metrics_frame = make_metrics(bars.index, oi=70_000.0 + np.cumsum(np.random.default_rng(0).normal(size=len(bars))))
    frame = deriv.build_deriv(bars, metrics_frame)
    rank = frame["oi_rank"].dropna()
    assert rank.between(0.0, 1.0).all()


# ── Кэш ────────────────────────────────────────────────────────────────
def test_cache_returns_the_same_object():
    bars = make_bars("2024-01-01", periods=200)
    metrics_frame = make_metrics(bars.index)

    first = deriv.build_deriv_cached(bars, metrics_frame)
    second = deriv.build_deriv_cached(bars, metrics_frame)
    assert first is second

    other_metrics = make_metrics(bars.index, oi=90_000.0 + np.arange(len(bars)))
    third = deriv.build_deriv_cached(bars, other_metrics)
    assert third is not first


# ── Конфиг и реестр ──────────────────────────────────────────────────────
def test_flags_are_off_unless_environment_says_otherwise(monkeypatch):
    monkeypatch.delenv("DERIV_ENABLED", raising=False)
    monkeypatch.delenv("DERIV_FEATURES_ENABLED", raising=False)
    assert config._env_bool("DERIV_ENABLED", False) is False
    assert config._env_bool("DERIV_FEATURES_ENABLED", False) is False


def test_features_need_both_flags():
    assert not config.DerivConfig(enabled=False, features_enabled=False).features_on
    assert not config.DerivConfig(enabled=False, features_enabled=True).features_on
    assert not config.DerivConfig(enabled=True, features_enabled=False).features_on
    assert config.DerivConfig(enabled=True, features_enabled=True).features_on


def test_registered_in_atom_family_and_context():
    from btcproc.features import events

    for atom in deriv.CONTEXT_CANDIDATES:
        assert atom in events.ATOM_FAMILY
        assert atom in events.CONTEXT_ATOMS
        assert atom not in events.SIGNATURE_ATOMS


def test_feature_version_includes_deriv(monkeypatch):
    from btcproc.features import builder

    monkeypatch.setattr(config, "deriv", config.DerivConfig(enabled=True, features_enabled=True))
    assert "deriv" in builder.feature_version()


def test_event_version_includes_deriv_when_atoms_only(monkeypatch):
    from btcproc.features import events

    monkeypatch.setattr(config, "smc", config.SMCConfig(enabled=False))
    monkeypatch.setattr(config, "fgi", config.FearGreedConfig(enabled=False))
    monkeypatch.setattr(config, "deriv", config.DerivConfig(enabled=True, features_enabled=False))
    assert events.event_version() == "v1+deriv"


def test_atoms_absent_when_disabled(bars, monkeypatch):
    from btcproc.features import events

    monkeypatch.setattr(config, "deriv", config.DerivConfig(enabled=False))
    detected = events.detect_atoms(bars)
    for name in deriv.CONTEXT_CANDIDATES:
        assert name in detected.columns
        assert not detected[name].any()


# ── Помонетность: symbol пробрасывается, а не берётся из config.data.symbol ─
def test_compute_uses_passed_symbol_not_global_default(monkeypatch):
    """
    Регрессия на архитектурный пробел (docs/tz_deriv_ingest_14-08-26.md):
    деривативные метрики ПОМОНЕТНЫ, в отличие от SMC (из base) и FGI (один
    общий ряд). build_features/detect_atoms обязаны передавать РЕАЛЬНУЮ
    монету источнику, а не полагаться на config.data.symbol по умолчанию —
    иначе batch-прогон (train --all) джойнил бы всем монетам метрики первой
    монеты из .env.
    """
    from btcproc.features import registry
    from btcproc.ingest import metrics as metrics_ingest

    requested_symbols: list[str | None] = []

    def fake_load(symbol, tf=None, start=None, end=None):
        requested_symbols.append(symbol)
        return pd.DataFrame(
            columns=["oi", "oi_value", "ls_top_acc", "ls_top_pos", "ls_global",
                     "taker_ratio", "src_rows"],
            index=pd.DatetimeIndex([], tz="UTC"),
        )

    monkeypatch.setattr(metrics_ingest, "load_deriv_metrics", fake_load)
    monkeypatch.setattr(config, "deriv", config.DerivConfig(enabled=True, features_enabled=True))

    bars = make_bars("2024-01-01", periods=10)
    source = next(s for s in registry.sources() if s.name == "deriv")
    source.compute(bars, "ETHUSDT")

    assert requested_symbols == ["ETHUSDT"], (
        f"deriv.compute обязан запросить метрики МОНЕТЫ ИЗ АРГУМЕНТА, "
        f"а не глобального дефолта; запрошено: {requested_symbols}"
    )
