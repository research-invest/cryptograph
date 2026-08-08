from __future__ import annotations

import numpy as np
import pandas as pd

from btcproc import config
from btcproc.features import builder
from btcproc.states import assign, clustering, graph


def test_smoothing_removes_flicker():
    # Дребезг 1-2 бара гасится, устойчивая серия — принимается.
    labels = np.array([1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1], dtype=float)
    smoothed = assign.smooth_labels(labels, min_run=3)
    assert list(smoothed) == [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1] or \
           list(smoothed[:6]) == [1, 1, 1, 1, 1, 1]
    # Одиночный выброс не создаёт перехода.
    changes = (pd.Series(smoothed).diff() != 0).sum()
    assert changes <= 3


def test_assign_states_age_and_transitions():
    index = pd.date_range("2021-01-01", periods=40, freq="15min", tz="UTC")
    labels = np.array([1.0] * 20 + [2.0] * 20)
    states = assign.assign_states(index, labels, smoothing_bars=1, trajectory_window=8)

    assert states["is_transition"].sum() == 1
    transition = states[states["is_transition"]].iloc[0]
    assert transition["transition_id"] == "1->2"
    assert transition["age_minutes"] == 0
    assert transition["age_bucket"] == "age_lt_30"

    # Возраст растёт по 15 минут и переключает бакеты на 30/60/120.
    tail = states.iloc[20:]
    assert tail["age_minutes"].iloc[2] == 30
    assert tail["age_bucket"].iloc[2] == "age_30_60"
    assert tail["age_bucket"].iloc[-1] == "age_gt_120"
    assert states["entropy"].isin(["low", "medium", "high"]).all()


def test_clustering_finds_separated_clusters():
    """Три явно разделённых облака должны стать разными состояниями."""
    rng = np.random.default_rng(3)
    blobs = np.vstack([
        rng.normal(centre, 0.35, size=(2500, 4))
        for centre in ([-6, -6, 0, 0], [6, 6, 0, 0], [0, 0, 8, 8])
    ])
    cfg = config.StatesConfig(seed_clusters=3, min_group_size=300, max_depth=3)
    model, labels = clustering.fit_states(
        blobs, [f"f{i}" for i in range(4)],
        {"median": np.zeros(4), "iqr": np.ones(4)}, cfg,
    )

    assert 3 <= model.n_groups <= 8
    # Точки одного облака попали преимущественно в одну группу.
    for start in (0, 2500, 5000):
        chunk = labels[start:start + 2500]
        dominant = pd.Series(chunk).value_counts(normalize=True).iloc[0]
        assert dominant > 0.9


def test_clustering_does_not_shatter_homogeneous_cloud():
    """Однородный шар дробить не на что — gap-критерий должен это увидеть."""
    rng = np.random.default_rng(11)
    cloud = rng.normal(0, 1, size=(6000, 5))
    cfg = config.StatesConfig(seed_clusters=2, min_group_size=500, max_depth=3)
    model, _ = clustering.fit_states(
        cloud, [f"f{i}" for i in range(5)],
        {"median": np.zeros(5), "iqr": np.ones(5)}, cfg,
    )
    assert model.n_groups <= 4


def test_model_roundtrip_and_predict():
    rng = np.random.default_rng(5)
    x = np.vstack([rng.normal(-4, 0.3, (600, 3)), rng.normal(4, 0.3, (600, 3))])
    cfg = config.StatesConfig(seed_clusters=2, min_group_size=200, max_depth=1)
    model, labels = clustering.fit_states(
        x, ["a", "b", "c"], {"median": np.zeros(3), "iqr": np.ones(3)}, cfg
    )
    restored = clustering.StateModel.from_dict(model.to_dict())
    assert np.array_equal(model.predict(x), restored.predict(x))


def test_transition_rarity_covers_all_buckets(features):
    scale = builder.robust_scale_params(features)
    matrix = builder.apply_scale(features, scale)
    cfg = config.StatesConfig(seed_clusters=4, min_group_size=400, max_depth=2)
    model, labels = clustering.fit_states(matrix, list(features.columns), scale, cfg)
    states = assign.assign_states(features.index, labels)

    transitions = graph.transition_stats(states)
    assert not transitions.empty
    assert transitions["rarity"].isin(["rare", "uncommon", "common"]).all()
    assert abs(transitions["share"].sum() - 1.0) < 1e-9

    groups = graph.group_stats(states, features=features)
    assert len(groups) == states["group_id"].nunique()
    payload = graph.to_cytoscape(groups, transitions)
    assert payload["nodes"] and payload["edges"]
