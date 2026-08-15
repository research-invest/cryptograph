"""
Тесты детектора Fear & Greed Index (btcproc/features/fear_greed.py).

Главное здесь — look-ahead (docs/task_fear_greed.md, §3.4): значение суток D
известно не раньше их конца и обязано применяться к барам, начинающимся с
D+1 00:00 UTC. Ошибка на одни сутки не роняет ничего и не выглядит странно —
она просто делает всю статистику фикцией, поэтому тестов на это два:
префиксный (общий рецепт проекта) и точечный, с захардкоженной датой.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcproc import config
from btcproc.features import fear_greed as fg
from tests.conftest import disable_all_sources


def make_bars(start: str, periods: int, freq: str = "15min") -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    close = 10_000.0 + np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "open": close, "high": close + 1, "low": close - 1, "close": close,
            "volume": 100.0,
        },
        index=index,
    )


def make_daily(start: str, values: list[float]) -> pd.DataFrame:
    index = pd.date_range(start, periods=len(values), freq="1D", tz="UTC")
    return pd.DataFrame({"value": values}, index=index)


# ── Look-ahead ────────────────────────────────────────────────────────────
def test_point_shift_on_the_crash_day():
    """
    На баре 2021-05-19 00:15 UTC значение обязано быть значением за
    2021-05-18, а не за 19-е — образец из ТЗ, с захардкоженными числами.
    """
    daily = make_daily("2021-05-15", [40.0, 35.0, 30.0, 20.0, 25.0, 45.0])
    #                                  05-15 05-16 05-17 05-18 05-19 05-20
    bars = make_bars("2021-05-18 00:00", periods=4 * 24 * 3, freq="15min")

    frame = fg.build_fear_greed(bars, daily)

    bar_19 = frame.loc["2021-05-19 00:15"]
    assert bar_19["fgi_level"] == pytest.approx((20.0 - 50.0) / 50.0)  # значение 05-18

    bar_18 = frame.loc["2021-05-18 00:15"]
    assert bar_18["fgi_level"] == pytest.approx((30.0 - 50.0) / 50.0)  # значение 05-17


def test_prefix_matches_full_history():
    """
    Величина на префиксе истории обязана совпасть с величиной на полной —
    иначе где-то подглядывается будущее.
    """
    daily = make_daily("2021-01-01", list(np.linspace(10, 90, 120)))
    bars = make_bars("2021-01-10", periods=4 * 24 * 60, freq="15min")
    cut = len(bars) - 500

    full = fg.build_fear_greed(bars, daily)
    prefix = fg.build_fear_greed(bars.iloc[:cut], daily)

    common = prefix.index
    pd.testing.assert_frame_equal(full.loc[common], prefix.loc[common])


def test_value_holds_flat_across_the_day():
    """96 баров одних суток (15m) обязаны нести одно и то же значение."""
    daily = make_daily("2021-01-01", [30.0, 70.0, 30.0])
    bars = make_bars("2021-01-02 00:00", periods=96, freq="15min")  # весь день 01-02

    frame = fg.build_fear_greed(bars, daily)
    assert frame["fgi_level"].nunique() == 1
    assert frame["fgi_level"].iloc[0] == pytest.approx((30.0 - 50.0) / 50.0)  # значение 01-01


# ── Данные отсутствуют ───────────────────────────────────────────────────
def test_missing_history_gives_null_feature_and_false_atom():
    """
    Период до начала истории индекса (или пробел в ней) не заполняется НИЧЕМ:
    атом = False, признак = NaN (docs §3.1).
    """
    daily = make_daily("2021-02-01", [50.0] * 5)
    bars = make_bars("2021-01-01", periods=96, freq="15min")  # раньше истории

    frame = fg.build_fear_greed(bars, daily)
    assert frame["fgi_level"].isna().all()
    assert not frame["fear_extreme"].any()
    assert not frame["greed_extreme"].any()


def test_naive_daily_index_is_localized_not_silently_dropped():
    """
    Ряд без часовой зоны (из JSON/CSV, а не из БД) обязан приджойниться, а не
    дать сплошной NaN. До 2026-08-14 `reindex` не совпадал ни с одним баром и
    возвращал пустоту молча — ни ошибки, ни предупреждения.
    """
    daily = make_daily("2021-01-01", [30.0, 70.0, 30.0])
    naive = daily.copy()
    naive.index = naive.index.tz_localize(None)
    bars = make_bars("2021-01-02 00:00", periods=96)

    pd.testing.assert_frame_equal(
        fg.build_fear_greed(bars, naive), fg.build_fear_greed(bars, daily)
    )


def test_gap_in_the_daily_series_does_not_shift_yesterday():
    """
    Пропущенные сутки внутри истории (в реальном ряду их четыре) не должны
    делать «вчера» позавчерашним: `shift` идёт по позициям, поэтому календарь
    выравнивается заранее. На дне после дыры флип неизвестен — значит False,
    а не посчитанный через пропуск.
    """
    daily = make_daily("2021-01-01", [40.0, 60.0, 45.0, 80.0])
    daily = daily.drop(index=daily.index[2])  # выпали сутки 01-03

    # Бары 01-05 получают значение суток 01-04 (=80), «вчера» для которых —
    # выпавшие 01-03, а вовсе не 01-02 со значением 60.
    bars = make_bars("2021-01-05 00:00", periods=1)
    frame = fg.build_fear_greed(bars, daily)

    assert frame["fgi_level"].iloc[0] == pytest.approx((80.0 - 50.0) / 50.0)
    assert not frame["sentiment_flip_up"].iloc[0]

    # Сами выпавшие сутки не заполняются ничем: бар 01-04 берёт 01-03 = дыра.
    gap_bar = fg.build_fear_greed(make_bars("2021-01-04 00:00", periods=1), daily)
    assert gap_bar["fgi_level"].isna().all()
    assert not gap_bar["fear_extreme"].iloc[0]


def test_change_window_counts_calendar_days_not_rows():
    """`fgi_change_7d` — семь КАЛЕНДАРНЫХ суток, даже если строк меньше."""
    values = [50.0] * 7 + [90.0]
    daily = make_daily("2021-01-01", values)
    daily = daily.drop(index=daily.index[3])  # дыра внутри окна

    bars = make_bars("2021-01-09 00:00", periods=1)  # значение суток 01-08 = 90
    frame = fg.build_fear_greed(bars, daily)
    # Семь суток назад — 01-01 со значением 50, а не «седьмая строка назад».
    assert frame["fgi_change_7d"].iloc[0] == pytest.approx((90.0 - 50.0) / 50.0)


def test_empty_daily_series_does_not_crash():
    bars = make_bars("2021-01-01", periods=10)
    empty_daily = pd.DataFrame(columns=["value"], index=pd.DatetimeIndex([], tz="UTC"))

    frame = fg.build_fear_greed(bars, empty_daily)
    assert list(frame.columns) == fg.ALL_COLUMNS
    assert frame["fgi_level"].isna().all()
    assert not frame["fear_extreme"].any()


# ── Атомы ─────────────────────────────────────────────────────────────────
def test_extreme_atoms_use_configured_thresholds():
    daily = make_daily("2021-01-01", [10.0, 50.0, 90.0])
    bars = pd.concat([
        make_bars("2021-01-02 00:00", periods=1),   # получает значение 01-01 = 10
        make_bars("2021-01-03 00:00", periods=1),   # значение 01-02 = 50
        make_bars("2021-01-04 00:00", periods=1),   # значение 01-03 = 90
    ])

    frame = fg.build_fear_greed(bars, daily)
    assert frame["fear_extreme"].tolist() == [True, False, False]
    assert frame["greed_extreme"].tolist() == [False, False, True]


def test_sentiment_flip_atoms():
    daily = make_daily("2021-01-01", [40.0, 60.0, 45.0])  # up, down
    bars = pd.concat([
        make_bars("2021-01-02 00:00", periods=1),   # 01-01=40 (низ) → флипа нет
        make_bars("2021-01-03 00:00", periods=1),   # 01-02=60, вчера 40<50 → flip_up
        make_bars("2021-01-04 00:00", periods=1),   # 01-03=45, вчера 60>=50 → flip_down
    ])

    frame = fg.build_fear_greed(bars, daily)
    assert frame["sentiment_flip_up"].tolist() == [False, True, False]
    assert frame["sentiment_flip_down"].tolist() == [False, False, True]


# ── Стационарность и масштабная инвариантность (тривиальны, но проверяются) ─
def test_output_is_independent_of_price_scale():
    """
    Величина безразмерна и не зависит от цены (docs §4.2): рынок, умноженный
    на 128, обязан дать те же значения.
    """
    daily = make_daily("2021-01-01", list(np.linspace(5, 95, 40)))
    bars = make_bars("2021-01-05", periods=4 * 24 * 20, freq="15min")
    scaled = bars.copy()
    for column in ("open", "high", "low", "close"):
        scaled[column] *= 128.0

    pd.testing.assert_frame_equal(
        fg.build_fear_greed(bars, daily), fg.build_fear_greed(scaled, daily)
    )


def test_output_is_stationary_across_years():
    """Уровень 2018 и 2025 года сравним — величина одна и та же шкала 0..100."""
    daily = make_daily("2018-01-01", [15.0] * 200 + [85.0] * 600)
    bars_early = make_bars("2018-02-01", periods=4)
    bars_late = make_bars("2019-02-01", periods=4)

    early = fg.build_fear_greed(bars_early, daily)
    late = fg.build_fear_greed(bars_late, daily)
    assert early["fgi_level"].iloc[0] == pytest.approx(-0.7)
    assert late["fgi_level"].iloc[0] == pytest.approx(0.7)


def test_features_are_bounded():
    daily = make_daily("2021-01-01", list(np.linspace(0, 100, 60)))
    bars = make_bars("2021-01-05", periods=4 * 24 * 50, freq="15min")
    frame = fg.build_fear_greed(bars, daily)
    level = frame["fgi_level"].dropna()
    assert level.between(-1.0, 1.0).all()


# ── Кэш ───────────────────────────────────────────────────────────────────
def test_cache_returns_the_same_object():
    daily = make_daily("2021-01-01", list(np.linspace(0, 100, 30)))
    bars = make_bars("2021-01-05", periods=200)

    first = fg.build_fear_greed_cached(bars, daily)
    second = fg.build_fear_greed_cached(bars, daily)
    assert first is second

    other_daily = make_daily("2021-01-01", list(np.linspace(100, 0, 30)))
    third = fg.build_fear_greed_cached(bars, other_daily)
    assert third is not first


# ── Конфиг и реестр ──────────────────────────────────────────────────────
def test_flags_are_off_unless_environment_says_otherwise(monkeypatch):
    """
    Включение источника требует ЯВНОГО действия — переменной в окружении.

    Проверяется `_env_bool`, а не `FearGreedConfig()`: дефолты полей dataclass
    вычисляются один раз, на импорте `config`, из окружения процесса. На
    машине с `FGI_ENABLED=true` в `.env` (боевой VPS с 2026-08-14)
    «конфигурация по умолчанию» означает «включён», и тест, сверявшийся с
    конструктором, зеленел локально и падал на сервере — ровно та же ловушка,
    что описана у `test_event_version_includes_fgi_when_atoms_only`, только
    обнаруженная выкаткой.
    """
    monkeypatch.delenv("FGI_ENABLED", raising=False)
    monkeypatch.delenv("FGI_FEATURES_ENABLED", raising=False)
    assert config._env_bool("FGI_ENABLED", False) is False
    assert config._env_bool("FGI_FEATURES_ENABLED", False) is False


def test_features_need_both_flags():
    assert not config.FearGreedConfig(enabled=False, features_enabled=False).features_on
    assert not config.FearGreedConfig(enabled=False, features_enabled=True).features_on
    assert not config.FearGreedConfig(enabled=True, features_enabled=False).features_on
    assert config.FearGreedConfig(enabled=True, features_enabled=True).features_on


def test_registered_in_atom_family_and_context():
    from btcproc.features import events

    for atom in fg.CONTEXT_CANDIDATES:
        assert atom in events.ATOM_FAMILY
        assert atom in events.CONTEXT_ATOMS
        assert atom not in events.SIGNATURE_ATOMS


def test_feature_version_includes_fgi(monkeypatch):
    from btcproc.features import builder

    disable_all_sources(monkeypatch)
    monkeypatch.setattr(config, "fgi", config.FearGreedConfig(enabled=True, features_enabled=True))
    assert builder.feature_version() == "v1+fgi"


def test_event_version_includes_fgi_when_atoms_only(monkeypatch):
    """
    Чужие источники гасятся ЦЕЛИКОМ: их дефолты читаются из окружения, и на
    машине, где SMC_ENABLED=true (боевой .env), «умолчание» означает
    «включён» — тест, полагающийся на дефолт, зеленел бы у одних и падал бы
    у других. Перечислять их поимённо тоже нельзя: третий источник (deriv)
    в перечисление не попал и уронил бы тест ровно при выкатке
    (аудит 2026-08-15, B1).
    """
    from btcproc.features import events

    disable_all_sources(monkeypatch)
    monkeypatch.setattr(config, "fgi", config.FearGreedConfig(enabled=True, features_enabled=False))
    assert events.event_version() == "v1+fgi"


def test_atoms_absent_when_disabled(bars, monkeypatch):
    """При выключенном флаге имена существуют, но не срабатывают никогда."""
    from btcproc.features import events

    monkeypatch.setattr(config, "fgi", config.FearGreedConfig(enabled=False))
    detected = events.detect_atoms(bars)
    for name in fg.CONTEXT_CANDIDATES:
        assert name in detected.columns
        assert not detected[name].any()
