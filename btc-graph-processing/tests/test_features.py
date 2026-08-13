from __future__ import annotations

import numpy as np
import pandas as pd

from btcproc.features import builder, events, indicators


def test_rsi_bounds(bars):
    rsi = indicators.rsi(bars["close"], 14).dropna()
    assert rsi.between(0, 100).all()


def test_atr_positive(bars):
    atr = indicators.atr(bars, 14).dropna()
    assert (atr > 0).all()


def test_position_in_range_is_normalized(bars):
    pos = indicators.position_in_range(bars, 96).dropna()
    assert pos.between(0, 1).all()


def test_features_have_no_nan_and_are_finite(features):
    assert not features.empty
    assert features.notna().all().all()
    assert np.isfinite(features.to_numpy()).all()


def test_features_do_not_look_ahead(bars, context):
    """
    Признак на баре t не должен меняться от того, что происходит после t.

    Считаем признаки на полной истории и на её префиксе — общие строки
    обязаны совпасть.
    """
    cut = len(bars) - 500
    full = builder.build_features(bars, context)
    prefix_context = {tf: df[df.index <= bars.index[cut - 1]] for tf, df in context.items()}
    prefix = builder.build_features(bars.iloc[:cut], prefix_context)

    common = full.index.intersection(prefix.index)
    assert len(common) > 100
    # Последний бар префикса отбрасываем: у старших ТФ его бар ещё не закрыт.
    common = common[:-1]
    pd.testing.assert_frame_equal(
        full.loc[common], prefix.loc[common], check_exact=False, rtol=1e-9
    )


def test_scale_is_robust_to_outliers(features):
    params = builder.robust_scale_params(features)
    spoiled = features.copy()
    spoiled.iloc[0] = spoiled.iloc[0] * 1000
    scaled = builder.apply_scale(spoiled, params)
    # Выброс обрезается, а не растягивает шкалу остальных строк.
    assert np.abs(scaled).max() <= 5.0


def test_event_blocks_are_stable_and_typed(bars):
    blocks = events.build_event_blocks(bars)
    assert len(blocks) == len(bars)
    assert blocks["event_block_id"].str.startswith("event_block_").all()
    assert blocks["intensity"].isin(["sparse", "moderate", "dense"]).all()
    assert (blocks["family_count"] <= blocks["atom_count"]).all()

    # Один и тот же набор атомов всегда даёт один и тот же id.
    repeated = events.build_event_blocks(bars)
    assert repeated["event_block_id"].equals(blocks["event_block_id"])


def test_block_statistics_shares_sum_to_one(bars):
    blocks = events.build_event_blocks(bars)
    stats = events.block_statistics(blocks)
    assert abs(stats["row_share"].sum() - 1.0) < 1e-9
    assert stats["rarity"].isin(["rare", "uncommon", "common"]).all()


def test_context_atoms_survive_to_the_output(bars):
    """
    Регрессия: контекстные атомы считались и молча выбрасывались.

    До правки build_event_blocks срезал detect_atoms по SIGNATURE_ATOMS, и
    девять фоновых атомов не доходили ни до выдачи, ни до bar_events — мерить
    лифт по фону было не по чему.
    """
    blocks = events.build_event_blocks(bars)
    assert "context_atoms" in blocks.columns

    seen = {atom for row in blocks["context_atoms"] for atom in row}
    assert seen, "ни одного контекстного атома не сохранилось"
    assert seen <= events.CONTEXT_ATOMS

    # Сессии покрывают сутки целиком, поэтому на любой истории обязаны быть.
    assert seen & {"asia_session", "europe_session", "us_session"}

    # Обратное разделение: в atoms контекст протечь не должен.
    in_atoms = {atom for row in blocks["atoms"] for atom in row}
    assert not (in_atoms & events.CONTEXT_ATOMS)


def test_context_atoms_do_not_change_block_id(bars):
    """
    event_block_id — функция только от signature-атомов.

    Бары с одинаковым набором signature-атомов обязаны получить один и тот же
    блок, как бы ни различался их контекст. Иначе добавление любого фонового
    атома дробило бы историческую выборку.
    """
    blocks = events.build_event_blocks(bars)
    frame = pd.DataFrame({
        "signature": blocks["atoms"].map(tuple),
        "context": blocks["context_atoms"].map(tuple),
        "block": blocks["event_block_id"],
    })

    grouped = frame.groupby("signature")
    assert (grouped["block"].nunique() == 1).all()

    # Проверка, что тест не вырожден: контекст внутри групп реально различается.
    assert (grouped["context"].nunique() > 1).any()


def test_signature_bits_are_pinned(bars):
    """
    Номера битов существующих атомов зафиксированы навсегда.

    Вставка нового signature-атома в середину ATOM_FAMILY сдвинула бы биты, и
    все исторические event_block_id сменили бы смысл — молча, без единой
    ошибки. Новые атомы добавляются только в конец словаря.
    """
    assert events.SIGNATURE_ATOMS[:20] == [
        "breakout_1d_high",
        "breakdown_1d_low",
        "breakout_1w_high",
        "breakdown_1w_low",
        "wide_range_bar",
        "inside_bar",
        "vol_expansion",
        "vol_contraction",
        "atr_spike",
        "volume_spike",
        "volume_dry",
        "ema_cross_up",
        "ema_cross_down",
        "rsi_overbought",
        "rsi_oversold",
        "momentum_reversal_up",
        "momentum_reversal_down",
        "at_range_high",
        "at_range_low",
        "round_level_touch",
    ]
    assert [events.ATOM_BIT[a] for a in events.SIGNATURE_ATOMS[:20]] == list(range(20))
    # Ровно двадцать: всё, что добавлялось после, — контекстное. Смена этого
    # числа означает решение платить дроблением блоков и принимается отдельно
    # (фаза 4 задачи SMC), а не заезжает попутной правкой.
    assert len(events.SIGNATURE_ATOMS) == 20
    # Знаковый int64 в build_event_blocks: 62 бита — потолок без переполнения.
    assert len(events.SIGNATURE_ATOMS) <= 62
    # Ни один атом не может быть одновременно фоном и происшествием.
    assert not (set(events.SIGNATURE_ATOMS) & events.CONTEXT_ATOMS)
    assert len(events.SIGNATURE_ATOMS) + len(events.CONTEXT_ATOM_LIST) == len(events.ATOMS)


def _synthetic_walk(level: float, n: int = 20000, seed: int = 1):
    """Случайное блуждание вокруг заданного уровня цены."""
    import numpy as np
    import pandas as pd

    index = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(seed)
    close = pd.Series(level * np.exp(np.cumsum(rng.normal(0, 0.002, n))), index=index)
    return pd.DataFrame(
        {"open": close, "high": close * 1.002, "low": close * 0.998,
         "close": close, "volume": 1000.0},
        index=index,
    )


def test_round_level_touch_works_on_cheap_coins():
    """
    Атом обязан жить на монете любой цены.

    До 2026-08-11 порог был задан в долларах (`close % 1000 < 50`), и на SOL
    в диапазоне $100–200 условие не выполнялось никогда: бит 19 маски у
    монеты постоянно нулевой, то есть signature-атом, за который платят
    дроблением блоков, работал ровно на одной монете из трёх.
    """
    for level in (150.0, 3000.0, 60000.0):
        atoms = events.detect_atoms(_synthetic_walk(level))
        share = atoms["round_level_touch"].mean()
        assert 0.02 < share < 0.25, (
            f"на уровне цены {level} доля срабатываний {share:.1%} — "
            "детектор либо мёртв, либо срабатывает постоянно"
        )


def test_round_level_touch_is_scale_invariant():
    """
    Тот же рынок, умноженный на степень десяти, обязан дать ту же разметку.

    Это общее требование к детекторам (`docs/extending_features.md`, 2.1) и
    ровно тот тест, который валит любой порог, заданный в деньгах.
    """
    close = _synthetic_walk(30000.0, n=5000, seed=2)["close"]
    baseline = events.round_level_touch(close)
    for factor in (100.0, 0.001):
        assert (events.round_level_touch(close * factor) == baseline).all(), (
            f"разметка поехала при умножении цены на {factor}"
        )


def test_round_level_touch_survives_bad_prices():
    """Ноль в цене не должен ронять разметку целиком через log10(0)."""
    import numpy as np
    import pandas as pd

    close = pd.Series([100.0, 0.0, np.nan, 150.0])
    got = events.round_level_touch(close)
    assert got.iloc[1] is np.False_ or got.iloc[1] == False  # noqa: E712
    assert not got.isna().any()
