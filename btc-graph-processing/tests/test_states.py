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


def test_split_gain_is_independent_of_dimensionality():
    """
    Критерий приёмки P0-2: на двух явно разделимых облаках gain положителен,
    на однородном — отрицателен, и в 32, и в 64 измерениях ОДИНАКОВО.

    Именно это свойство было сломано абсолютным порогом: разность силуэтов
    зависит от размерности, поэтому константа, откалиброванная на 32
    признаках, в 44 означала другую строгость — и двенадцать признаков любой
    природы, включая чистый шум, обрушивали граф. Порог в сигмах референса
    самонормируется.
    """
    cfg = config.StatesConfig()
    for dim in (32, 64):
        rng = np.random.default_rng(1)
        homogeneous = rng.normal(0, 1, size=(2000, dim))
        separable = np.vstack([
            rng.normal(-4, 0.5, (1000, dim)),
            rng.normal(+4, 0.5, (1000, dim)),
        ])
        assert clustering._split_gain(homogeneous, np.random.default_rng(42), cfg) < 0, \
            f"однородное облако в {dim} измерениях не должно дробиться"
        assert clustering._split_gain(separable, np.random.default_rng(42), cfg) > 0, \
            f"разделимые облака в {dim} измерениях обязаны дробиться"


def test_split_gain_threshold_is_inside_the_value():
    """
    Порог живёт внутри gain, поэтому решение — это `gain > 0`.

    Следствие, которое и проверяем: рост split_gain_sigma делает критерий
    строже монотонно. Если порог когда-нибудь снова вынесут наружу, тест
    упадёт и заставит поправить `_recursive_split` вместе с ним.
    """
    rng = np.random.default_rng(2)
    borderline = np.vstack([
        rng.normal(-1.0, 1.0, (1000, 12)),
        rng.normal(+1.0, 1.0, (1000, 12)),
    ])
    gains = [
        clustering._split_gain(
            borderline, np.random.default_rng(7),
            config.StatesConfig(split_gain_sigma=sigma),
        )
        for sigma in (0.0, 1.0, 3.0)
    ]
    assert gains[0] > gains[1] > gains[2]


def test_split_gain_averages_over_references():
    """
    Референсов несколько, и это видно по результату: с B = 1 сигму оценить
    не по чему, поэтому порог не вычитается вовсе и критерий мягче.

    Проверка на то, что параметр реально доходит до расчёта, — B = 1 был
    поведением до 2026-08-11, и оно не должно вернуться молча.
    """
    rng = np.random.default_rng(3)
    cloud = rng.normal(0, 1, size=(1500, 16))
    single = clustering._split_gain(
        cloud, np.random.default_rng(9), config.StatesConfig(split_reference_draws=1))
    many = clustering._split_gain(
        cloud, np.random.default_rng(9), config.StatesConfig(split_reference_draws=10))
    assert single > many


def test_model_params_record_the_calibration():
    """
    Параметры дробления попадают в params модели — без них прогон нельзя
    воспроизвести, а число состояний двух прогонов нельзя объяснить.
    """
    rng = np.random.default_rng(5)
    x = np.vstack([rng.normal(-4, 0.3, (600, 3)), rng.normal(4, 0.3, (600, 3))])
    cfg = config.StatesConfig(seed_clusters=2, min_group_size=200, max_depth=1)
    model, _ = clustering.fit_states(
        x, ["a", "b", "c"], {"median": np.zeros(3), "iqr": np.ones(3)}, cfg
    )
    assert model.params["split_gain_sigma"] == cfg.split_gain_sigma
    assert model.params["split_reference_draws"] == cfg.split_reference_draws
    # Абсолютный порог удалён — его присутствие означало бы откат калибровки.
    assert "split_gain" not in model.params


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
