"""
Тесты гейта N и fgi_residual (btcproc/analysis/fgi_novelty.py).

Синтетика с ИЗВЕСТНОЙ связью: fgi_level = 0.6·f1 - 0.3·f2 + шум. OOS R² на
train/test обязан поймать эту связь, а остаток — растерять её (по построению
он то, что регрессия объяснить не смогла).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcproc.analysis.fgi_novelty import (
    base_feature_columns,
    compute_fgi_residual,
    novelty_r2,
)


def make_features(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"f1": rng.normal(size=n), "f2": rng.normal(size=n), "f3": rng.normal(size=n)},
        index=index,
    )


def test_novelty_r2_recovers_a_strong_linear_relationship():
    features = make_features()
    rng = np.random.default_rng(1)
    fgi = pd.Series(
        0.6 * features["f1"] - 0.3 * features["f2"] + rng.normal(scale=0.05, size=len(features)),
        index=features.index,
    )

    r2, model, top5, columns, index, cut = novelty_r2(fgi, features, columns=["f1", "f2", "f3"])
    assert r2 > 0.8
    assert top5[0][0] == "f1"  # самый сильный вклад


def test_novelty_r2_is_low_for_unrelated_series():
    features = make_features()
    rng = np.random.default_rng(2)
    fgi = pd.Series(rng.normal(size=len(features)), index=features.index)

    r2, *_ = novelty_r2(fgi, features, columns=["f1", "f2", "f3"])
    assert r2 < 0.1


def test_residual_removes_the_explained_part():
    """Остаток не должен коррелировать с тем же предиктором, что его объяснил."""
    features = make_features()
    rng = np.random.default_rng(3)
    fgi = pd.Series(
        0.7 * features["f1"] + rng.normal(scale=0.05, size=len(features)),
        index=features.index,
    )

    residual = compute_fgi_residual(fgi, features, columns=["f1", "f2", "f3"])
    assert residual.notna().all()
    corr = np.corrcoef(residual.to_numpy(), features["f1"].to_numpy())[0, 1]
    assert abs(corr) < 0.15


def test_residual_uses_only_train_fraction_coefficients():
    """
    Коэффициент, подобранный на train, применяется и к test — если бы
    коэффициенты считались на всей истории, test-часть остатка тоже
    объясняла бы связь, добавленную только в test-половине.
    """
    features = make_features(n=1000)
    n = len(features)
    fgi = pd.Series(0.0, index=features.index)
    # Связь есть только во второй половине (test-часть при train_frac=0.7
    # захватывает конец) — заведомо в самом хвосте, за train_frac.
    tail = slice(int(n * 0.9), n)
    fgi.iloc[tail] = 5.0 * features["f1"].iloc[tail]

    residual = compute_fgi_residual(fgi, features, train_frac=0.7, columns=["f1", "f2", "f3"])
    # Коэффициент на train (где связи нет) должен быть маленьким, поэтому
    # остаток в хвосте почти не отличается от самого fgi (сильная связь там).
    assert residual.iloc[tail].abs().mean() > 1.0


def test_base_feature_columns_excludes_registered_sources(monkeypatch):
    from btcproc import config
    from btcproc.features import fear_greed as fg

    monkeypatch.setattr(config, "fgi", config.FearGreedConfig(enabled=True, features_enabled=True))
    columns = ["ret_1h", "rv_rank", *fg.FEATURE_CANDIDATES]
    features = pd.DataFrame(columns=columns)
    base = base_feature_columns(features)
    assert set(base) == {"ret_1h", "rv_rank"}
