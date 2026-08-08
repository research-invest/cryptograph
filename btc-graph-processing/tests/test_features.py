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
