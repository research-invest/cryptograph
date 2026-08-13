"""
Тесты фильтрации, дедупликации по family_key и детекции конфликтов.
"""
from __future__ import annotations

import pytest

from src.filters.candidate_filter import (
    FRESH_BONUS,
    detect_conflicts,
    filter_candidates,
    select_best_per_family,
)
from src.scorer.candidate_scorer import score_candidate


def _strong(make_candidate, **overrides):
    """Заведомо сильный кандидат (quality_score > 0.9)."""
    base = dict(
        valid_label_pct=0.95, sample_size=5000, monthly_concentration=0.03,
        repeatability_months=24, long_outcome_share=0.85,
        historical_outcome_skew=0.7, long_favorable_adverse_ratio_p70_p80=6.0,
        context_status="fresh", current_group_age_bucket="age_lt_30",
        trajectory_entropy="low", event_rarity_bucket="rare",
        transition_rarity="rare", research_score=0.95,
    )
    return make_candidate(**{**base, **overrides})


def _weak(make_candidate, **overrides):
    """Заведомо слабый кандидат (quality_score < 0.3)."""
    base = dict(
        valid_label_pct=0.60, sample_size=50, monthly_concentration=0.60,
        repeatability_months=1, long_outcome_share=0.51,
        historical_outcome_skew=0.05, long_favorable_adverse_ratio_p70_p80=1.1,
        context_status="stale", current_group_age_bucket="age_gt_120",
        trajectory_entropy="high", event_rarity_bucket="common",
        transition_rarity="common", research_score=0.30,
    )
    return make_candidate(**{**base, **overrides})


# ─── filter_candidates ────────────────────────────────────────────────────────

def test_filter_drops_below_threshold(make_candidate):
    strong = _strong(make_candidate, candidate_id="strong")
    weak = _weak(make_candidate, candidate_id="weak")

    result = filter_candidates([strong, weak], min_quality_score=0.60)

    assert [c.candidate_id for c, _ in result] == ["strong"]


def test_filter_sorts_by_score_descending(make_candidate):
    high = _strong(make_candidate, candidate_id="high")
    mid = _strong(make_candidate, candidate_id="mid", research_score=0.5,
                  event_rarity_bucket="common", transition_rarity="common")

    result = filter_candidates([mid, high], min_quality_score=0.0)

    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)
    assert result[0][0].candidate_id == "high"


def test_filter_threshold_is_inclusive(make_candidate):
    """Порог сравнивается через >=, кандидат ровно на пороге проходит."""
    c = _strong(make_candidate)
    exact = score_candidate(c).total
    assert len(filter_candidates([c], min_quality_score=exact)) == 1


def test_filter_empty_input():
    assert filter_candidates([], min_quality_score=0.6) == []


def test_filter_zero_threshold_keeps_everything(make_candidate):
    candidates = [_strong(make_candidate, candidate_id="a"),
                  _weak(make_candidate, candidate_id="b")]
    assert len(filter_candidates(candidates, min_quality_score=0.0)) == 2


# ─── select_best_per_family ───────────────────────────────────────────────────

def test_best_per_family_keeps_one_per_key(make_candidate):
    a1 = _strong(make_candidate, candidate_id="a1", candidate_family_key="A")
    a2 = _strong(make_candidate, candidate_id="a2", candidate_family_key="A",
                 research_score=0.5)
    b1 = _strong(make_candidate, candidate_id="b1", candidate_family_key="B")

    scored = filter_candidates([a1, a2, b1], min_quality_score=0.0)
    result = select_best_per_family(scored)

    families = {c.candidate_family_key for c, _ in result}
    assert families == {"A", "B"}
    assert len(result) == 2


def test_best_per_family_picks_higher_score_within_same_freshness(make_candidate):
    better = _strong(make_candidate, candidate_id="better", candidate_family_key="A")
    worse = _strong(make_candidate, candidate_id="worse", candidate_family_key="A",
                    research_score=0.4, event_rarity_bucket="common")

    result = select_best_per_family(filter_candidates([worse, better], 0.0))

    assert result[0][0].candidate_id == "better"


def test_best_per_family_prefers_fresh_when_scores_are_close(make_candidate):
    """При сопоставимых оценках свежесть решает — бонус FRESH_BONUS."""
    fresh = _strong(make_candidate, candidate_id="fresh", candidate_family_key="A",
                    context_status="fresh")
    stale = _strong(make_candidate, candidate_id="stale", candidate_family_key="A",
                    context_status="stale")

    result = select_best_per_family(filter_candidates([stale, fresh], 0.0))

    assert result[0][0].candidate_id == "fresh"


def test_best_per_family_strong_stale_beats_weak_fresh(make_candidate):
    """
    Регрессия на замечание №9: раньше свежесть была абсолютным приоритетом и
    слабый fresh вытеснял заметно более сильный stale.
    """
    fresh_weak = _strong(make_candidate, candidate_id="fresh", candidate_family_key="A",
                         context_status="fresh", research_score=0.2,
                         event_rarity_bucket="common", transition_rarity="common",
                         long_outcome_share=0.55, historical_outcome_skew=0.1,
                         long_favorable_adverse_ratio_p70_p80=1.2)
    stale_strong = _strong(make_candidate, candidate_id="stale", candidate_family_key="A",
                           context_status="stale")

    scored = dict(
        (c.candidate_id, s) for c, s in filter_candidates([fresh_weak, stale_strong], 0.0)
    )
    assert scored["stale"] - scored["fresh"] > FRESH_BONUS, "тестовые данные не различаются"

    result = select_best_per_family(filter_candidates([fresh_weak, stale_strong], 0.0))
    assert result[0][0].candidate_id == "stale"


def test_fresh_bonus_is_bounded(make_candidate):
    """Бонус за свежесть должен оставаться небольшой надбавкой, а не приоритетом."""
    assert 0.0 < FRESH_BONUS <= 0.15


def test_candidates_without_family_key_are_not_merged(make_candidate):
    """Без family_key ключом становится candidate_id — кандидаты не схлопываются."""
    a = _strong(make_candidate, candidate_id="a", candidate_family_key=None)
    b = _strong(make_candidate, candidate_id="b", candidate_family_key=None)

    result = select_best_per_family(filter_candidates([a, b], 0.0))

    assert len(result) == 2


def test_best_per_family_result_is_sorted(make_candidate):
    candidates = [
        _strong(make_candidate, candidate_id="a", candidate_family_key="A"),
        _strong(make_candidate, candidate_id="b", candidate_family_key="B",
                research_score=0.5, event_rarity_bucket="common"),
        _strong(make_candidate, candidate_id="c", candidate_family_key="C",
                research_score=0.75, transition_rarity="uncommon"),
    ]
    result = select_best_per_family(filter_candidates(candidates, 0.0))
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)


# ─── detect_conflicts ─────────────────────────────────────────────────────────

def test_no_conflicts_on_clean_candidates(make_candidate):
    clean = _strong(make_candidate, candidate_id="clean")
    assert detect_conflicts([clean]) == []


def test_contradictory_direction(make_candidate):
    long_c = _strong(make_candidate, candidate_id="l", transition_id="42->1",
                     research_side="long")
    short_c = _strong(make_candidate, candidate_id="s", transition_id="42->1",
                      research_side="short")

    conflicts = detect_conflicts([long_c, short_c])

    types = [c.conflict_type for c in conflicts]
    assert "contradictory_direction" in types
    conflict = next(c for c in conflicts if c.conflict_type == "contradictory_direction")
    assert set(conflict.candidate_ids) == {"l", "s"}


def test_same_direction_is_not_a_conflict(make_candidate):
    a = _strong(make_candidate, candidate_id="a", transition_id="42->1", research_side="long")
    b = _strong(make_candidate, candidate_id="b", transition_id="42->1", research_side="long")

    types = [c.conflict_type for c in detect_conflicts([a, b])]
    assert "contradictory_direction" not in types


def test_different_transitions_are_not_a_conflict(make_candidate):
    a = _strong(make_candidate, candidate_id="a", transition_id="42->1", research_side="long")
    b = _strong(make_candidate, candidate_id="b", transition_id="7->3", research_side="short")

    types = [c.conflict_type for c in detect_conflicts([a, b])]
    assert "contradictory_direction" not in types


def test_stale_and_old_context_conflict(make_candidate):
    c = _strong(make_candidate, candidate_id="x", context_status="stale",
                current_group_age_bucket="age_gt_120")

    conflicts = detect_conflicts([c])

    assert [k.conflict_type for k in conflicts] == ["stale_and_old_context"]
    assert conflicts[0].candidate_ids == ["x"]


def test_false_confidence_conflict(make_candidate):
    c = _strong(make_candidate, candidate_id="x", research_score=0.95, valid_label_pct=0.60)

    types = [k.conflict_type for k in detect_conflicts([c])]

    assert "false_confidence" in types


def test_seasonal_concentration_conflict(make_candidate):
    c = _strong(make_candidate, candidate_id="x", monthly_concentration=0.55)

    types = [k.conflict_type for k in detect_conflicts([c])]

    assert "seasonal_concentration" in types


def test_multiple_conflicts_on_one_candidate(make_candidate):
    """Один кандидат может нарушить сразу несколько правил."""
    c = _strong(
        make_candidate, candidate_id="x",
        context_status="stale", current_group_age_bucket="age_gt_120",
        research_score=0.95, valid_label_pct=0.60, monthly_concentration=0.55,
    )

    types = {k.conflict_type for k in detect_conflicts([c])}

    assert types == {"stale_and_old_context", "false_confidence", "seasonal_concentration"}


def test_detect_conflicts_on_empty_list():
    assert detect_conflicts([]) == []


@pytest.mark.parametrize("research_score,valid_pct,expected", [
    (0.86, 0.69, True),
    (0.85, 0.69, False),    # ровно на границе research_score
    (0.86, 0.70, False),    # ровно на границе valid_label_pct
])
def test_false_confidence_boundaries(make_candidate, research_score, valid_pct, expected):
    c = _strong(make_candidate, research_score=research_score, valid_label_pct=valid_pct)
    types = [k.conflict_type for k in detect_conflicts([c])]
    assert ("false_confidence" in types) is expected
