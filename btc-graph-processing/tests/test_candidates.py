from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcproc import config
from btcproc.candidates import builder as cand
from btcproc.candidates.outcomes import compute_outcomes
from btcproc.features import builder as feat
from btcproc.features import events as ev
from btcproc.states import assign, clustering, graph


# На синтетике 125 дней переходов набирается немного, поэтому порог выборки
# ослаблен: тест проверяет корректность сборки кандидата, а не статистическую
# значимость (её обеспечивает боевой CAND_MIN_SAMPLE_SIZE).
TEST_CAND_CFG = config.CandidateConfig(min_sample_size=10)


@pytest.fixture(scope="module")
def pipeline_data(bars, context):
    """Прогон всей цепочки на синтетике — база для проверок кандидатов."""
    features = feat.build_features(bars, context)
    scale = feat.robust_scale_params(features)
    matrix = feat.apply_scale(features, scale)
    cfg = config.StatesConfig(seed_clusters=4, min_group_size=400, max_depth=2)
    model, labels = clustering.fit_states(matrix, list(features.columns), scale, cfg)

    states = assign.assign_states(features.index, labels)
    events = ev.build_event_blocks(bars).reindex(features.index).dropna(
        subset=["event_block_id"]
    )
    blocks = ev.block_statistics(events)
    outcomes = compute_outcomes(bars).reindex(features.index)
    transitions = graph.transition_stats(states, outcomes)

    snapshots = cand.build_snapshots(states, events, outcomes)
    return {
        "features": features, "states": states, "events": events, "blocks": blocks,
        "outcomes": outcomes, "transitions": transitions, "snapshots": snapshots,
    }


def test_outcomes_align_with_prices(bars):
    horizon = 96
    outcomes = compute_outcomes(bars, horizon)

    i = 500
    expected_ret = (bars["close"].iloc[i + horizon] / bars["close"].iloc[i] - 1) * 100
    assert outcomes["ret_pct"].iloc[i] == pytest.approx(expected_ret)

    expected_mfe = (bars["high"].iloc[i + 1:i + 1 + horizon].max()
                    / bars["close"].iloc[i] - 1) * 100
    assert outcomes["mfe_pct"].iloc[i] == pytest.approx(expected_mfe)

    # У последних баров горизонта нет — метка невалидна.
    assert not outcomes["valid"].iloc[-1]
    assert outcomes["valid"].iloc[:-horizon].all()


def test_outcomes_marks_gaps_invalid(bars):
    """Дыра в данных внутри горизонта делает метку невалидной."""
    holed = pd.concat([bars.iloc[:1000], bars.iloc[1100:]])
    outcomes = compute_outcomes(holed, 96)
    around_gap = outcomes["valid"].iloc[904:1000]
    assert not around_gap.any()


def test_snapshots_cover_all_age_buckets(pipeline_data):
    snapshots = pipeline_data["snapshots"]
    assert not snapshots.empty
    buckets = set(snapshots["age_bucket"])
    # Смещения 0/45/90/180 минут должны дать все четыре бакета схемы.
    assert buckets == {"age_lt_30", "age_30_60", "age_60_120", "age_gt_120"}
    assert snapshots["ts"].is_monotonic_increasing


def test_generated_candidates_match_btc_graph_schema(pipeline_data):
    """
    Главная проверка совместимости: каждый кандидат обязан пройти pydantic-модель
    btc-graph без единой правки. Если схема разъедется — тест упадёт здесь,
    а не в проде при отправке.
    """
    pytest.importorskip("pydantic")
    Candidate = _load_btc_graph_model()

    rarity = dict(zip(pipeline_data["transitions"]["transition_id"],
                      pipeline_data["transitions"]["rarity"]))
    blocks = pipeline_data["blocks"].set_index("event_block_id").to_dict("index")

    produced = list(cand.generate(
        pipeline_data["snapshots"], rarity, blocks, "BTCUSDT", cfg=TEST_CAND_CFG
    ))
    assert produced, "На синтетике не выпущено ни одного кандидата"

    for item in produced[:200]:
        parsed = Candidate(**cand.strip_meta(item))
        assert parsed.symbol == "BTCUSDT"
        assert 0.0 <= parsed.research_score <= 1.0
        assert parsed.valid_label_count + parsed.invalid_label_count == parsed.sample_size
        assert parsed.long_outcome_count + parsed.short_outcome_count == parsed.valid_label_count


def test_candidate_sample_uses_only_matured_past(pipeline_data):
    """
    Кандидат в момент t не должен видеть ни будущее, ни исходы, которые
    к моменту t ещё не закрылись.
    """
    snapshots = pipeline_data["snapshots"]
    rarity = dict(zip(pipeline_data["transitions"]["transition_id"],
                      pipeline_data["transitions"]["rarity"]))
    blocks = pipeline_data["blocks"].set_index("event_block_id").to_dict("index")

    horizon = pd.Timedelta(minutes=config.data.horizon_minutes)
    produced = list(cand.generate(snapshots, rarity, blocks, "BTCUSDT", cfg=TEST_CAND_CFG))

    for item in produced[:100]:
        ts = pd.Timestamp(item["_meta"]["ts"])
        key_scope = item["_meta"]["scope"]
        mask = snapshots["transition_id"] == item["transition_id"]
        if key_scope == "transition+event_block":
            mask &= snapshots["event_block_id"] == item["event_block_id"]
        # Столько случаев было доступно на момент t по правилам генератора.
        available = int((mask & (snapshots["ts"] + horizon <= ts)).sum())
        assert item["sample_size"] == available


def test_research_score_rewards_strong_samples():
    weak = cand.Accumulator()
    strong = cand.Accumulator()
    base = pd.Timestamp("2021-01-01", tz="UTC")

    for i in range(40):
        weak.add(base + pd.Timedelta(days=i % 3), 0.1, 1.0, -1.0, True)
    for i in range(1200):
        ts = base + pd.Timedelta(days=i % 540)
        strong.add(ts, 1.0 if i % 4 else -1.0, 2.0, -0.4, True)

    assert cand.research_score(strong, "rare", "rare") > cand.research_score(weak, "common", "common")
    assert 0.0 <= cand.research_score(weak, "common", "common") <= 1.0


def test_percentile_helper_matches_numpy():
    values = sorted(np.random.default_rng(1).normal(3, 1, 500).tolist())
    for q in (50.0, 70.0, 80.0, 95.0):
        assert cand._percentile_sorted(values, q) == pytest.approx(
            float(np.percentile(values, q)), rel=1e-9
        )


def _load_btc_graph_model():
    """Модель Candidate берём из самого btc-graph — дублировать её нельзя."""
    import sys

    path = str(config.sink.btc_graph_path)
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        from src.models.candidate import Candidate
    except ImportError:  # pragma: no cover — репозиторий рядом не лежит
        pytest.skip(f"btc-graph не найден по пути {path}")
    return Candidate
