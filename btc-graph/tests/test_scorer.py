"""
Тесты скорера: каждая ступень каждой из 4 осей + веса + границы рейтингов.

Именно здесь жил дефект с недостижимой веткой age_60_120 — см.
docs/audit_findings.md, замечание №1.
"""
from __future__ import annotations

import pytest

from src.scorer.candidate_scorer import (
    RATING_MODERATE_MIN,
    RATING_STRONG_MIN,
    WEIGHTS,
    fa_ratio_for,
    get_rating,
    score_candidate,
    win_rate_for,
    _score_context,
    _score_directional,
    _score_rarity,
    _score_statistical,
)


# ─── Ось A: статистическая надёжность ─────────────────────────────────────────

@pytest.mark.parametrize("valid_label_pct,expected", [
    (0.90, 1.0),
    (0.86, 1.0),
    (0.85, 0.7),    # граница: строгое «>», ровно 0.85 попадает в следующую ступень
    (0.82, 0.7),
    (0.80, 0.4),
    (0.76, 0.4),
    (0.75, 0.0),
    (0.50, 0.0),
])
def test_statistical_valid_label_pct_steps(make_candidate, valid_label_pct, expected):
    """Каждая ступень valid_label_pct даёт свой вклад; остальные три — по 1.0."""
    c = make_candidate(
        valid_label_pct=valid_label_pct,
        sample_size=1500,
        monthly_concentration=0.05,
        repeatability_months=20,
    )
    assert _score_statistical(c) == pytest.approx((expected + 3.0) / 4.0)


@pytest.mark.parametrize("sample_size,expected", [
    (1500, 1.0),
    (1001, 1.0),
    (1000, 0.7),
    (501, 0.7),
    (500, 0.4),
    (201, 0.4),
    (200, 0.1),
    (10, 0.1),
])
def test_statistical_sample_size_steps(make_candidate, sample_size, expected):
    c = make_candidate(
        valid_label_pct=0.90,
        sample_size=sample_size,
        monthly_concentration=0.05,
        repeatability_months=20,
    )
    assert _score_statistical(c) == pytest.approx((expected + 3.0) / 4.0)


@pytest.mark.parametrize("monthly_concentration,expected", [
    (0.05, 1.0),
    (0.099, 1.0),
    (0.10, 0.7),
    (0.14, 0.7),
    (0.15, 0.3),
    (0.29, 0.3),
    (0.30, 0.0),
    (0.80, 0.0),
])
def test_statistical_monthly_concentration_steps(make_candidate, monthly_concentration, expected):
    """Сезонная концентрация: чем выше, тем хуже — оценка убывает."""
    c = make_candidate(
        valid_label_pct=0.90,
        sample_size=1500,
        monthly_concentration=monthly_concentration,
        repeatability_months=20,
    )
    assert _score_statistical(c) == pytest.approx((expected + 3.0) / 4.0)


@pytest.mark.parametrize("repeatability_months,expected", [
    (24, 1.0),
    (16, 1.0),
    (15, 0.7),
    (13, 0.7),
    (12, 0.4),
    (7, 0.4),
    (6, 0.1),
    (0, 0.1),
])
def test_statistical_repeatability_steps(make_candidate, repeatability_months, expected):
    c = make_candidate(
        valid_label_pct=0.90,
        sample_size=1500,
        monthly_concentration=0.05,
        repeatability_months=repeatability_months,
    )
    assert _score_statistical(c) == pytest.approx((expected + 3.0) / 4.0)


# ─── Ось B: directional asymmetry ─────────────────────────────────────────────

@pytest.mark.parametrize("long_outcome_share,expected", [
    (0.80, 1.0),
    (0.71, 1.0),
    (0.70, 0.7),
    (0.66, 0.7),
    (0.65, 0.4),
    (0.61, 0.4),
    (0.60, 0.1),
    (0.30, 0.1),
])
def test_directional_win_rate_steps_long(make_candidate, long_outcome_share, expected):
    c = make_candidate(
        research_side="long",
        long_outcome_share=long_outcome_share,
        historical_outcome_skew=0.5,
        long_favorable_adverse_ratio_p70_p80=5.0,
    )
    assert _score_directional(c) == pytest.approx((expected + 2.0) / 3.0)


@pytest.mark.parametrize("long_outcome_share,expected", [
    (0.20, 0.80),
    (0.30, 0.70),
    (0.40, 0.60),
    (0.74, 0.26),
])
def test_win_rate_inverted_for_short(make_candidate, long_outcome_share, expected):
    """Для short-кандидата win_rate = 1 − long_outcome_share."""
    short_c = make_candidate(research_side="short", long_outcome_share=long_outcome_share)
    long_c = make_candidate(research_side="long", long_outcome_share=long_outcome_share)

    assert win_rate_for(short_c) == pytest.approx(expected)
    assert win_rate_for(long_c) == pytest.approx(long_outcome_share)


@pytest.mark.parametrize("side,expected", [
    ("long", 4.333179476414578),
    ("short", None),
])
def test_fa_ratio_applies_only_to_long(make_candidate, side, expected):
    """
    p70/p80 описывают движение вверх, поэтому к short-кандидату F/A ratio
    неприменим — см. docs/audit_findings.md, замечание №8.
    """
    c = make_candidate(research_side=side)
    if expected is None:
        assert fa_ratio_for(c) is None
    else:
        assert fa_ratio_for(c) == pytest.approx(expected)


def test_directional_uses_two_criteria_for_short(make_candidate):
    """
    Для short ось directional усредняется по двум критериям (win rate + skew),
    а не по трём: F/A ratio исключён и не может тянуть оценку вверх.
    """
    short_c = make_candidate(
        research_side="short",
        long_outcome_share=0.20,                        # win_rate 0.80 → 1.0
        historical_outcome_skew=0.5,                    # → 1.0
        long_favorable_adverse_ratio_p70_p80=0.5,       # был бы 0.1, теперь игнорируется
    )
    assert _score_directional(short_c) == pytest.approx(1.0)


def test_short_score_independent_of_long_fa_ratio(make_candidate):
    """Изменение long-ratio не должно влиять на оценку short-кандидата."""
    scores = {
        ratio: _score_directional(
            make_candidate(research_side="short", long_favorable_adverse_ratio_p70_p80=ratio)
        )
        for ratio in (0.1, 2.5, 9.9)
    }
    assert len(set(scores.values())) == 1, f"ratio влияет на short: {scores}"


@pytest.mark.parametrize("skew,expected", [
    (0.50, 1.0),
    (-0.50, 1.0),   # знак не важен, важен модуль
    (0.41, 1.0),
    (0.40, 0.7),
    (0.31, 0.7),
    (0.30, 0.4),
    (0.21, 0.4),
    (0.20, 0.1),
    (0.0, 0.1),
])
def test_directional_skew_steps(make_candidate, skew, expected):
    c = make_candidate(
        research_side="long",
        long_outcome_share=0.80,
        historical_outcome_skew=skew,
        long_favorable_adverse_ratio_p70_p80=5.0,
    )
    assert _score_directional(c) == pytest.approx((expected + 2.0) / 3.0)


@pytest.mark.parametrize("ratio,expected", [
    (10.0, 1.0),
    (4.1, 1.0),
    (4.0, 0.7),
    (3.1, 0.7),
    (3.0, 0.4),
    (2.1, 0.4),
    (2.0, 0.1),
    (0.5, 0.1),
])
def test_directional_fa_ratio_steps(make_candidate, ratio, expected):
    c = make_candidate(
        research_side="long",
        long_outcome_share=0.80,
        historical_outcome_skew=0.5,
        long_favorable_adverse_ratio_p70_p80=ratio,
    )
    assert _score_directional(c) == pytest.approx((expected + 2.0) / 3.0)


# ─── Ось C: контекстуальная свежесть ──────────────────────────────────────────

@pytest.mark.parametrize("age_bucket,expected", [
    ("age_lt_30", 1.0),
    ("age_30_60", 0.75),
    ("age_60_120", 0.5),
    ("age_gt_120", 0.0),
])
def test_context_age_bucket_is_graduated(make_candidate, age_bucket, expected):
    """
    Регрессия на замечание №1: раньше `!= age_gt_120 → 1.0` делало ветку
    age_60_120 недостижимой, и три первых бакета давали одинаковый балл.
    """
    c = make_candidate(
        context_status="fresh",
        current_group_age_bucket=age_bucket,
        trajectory_entropy="low",
    )
    assert _score_context(c) == pytest.approx((2.0 + expected) / 3.0)


def test_context_age_buckets_are_all_distinct(make_candidate):
    """Все четыре бакета должны давать четыре разных оценки."""
    scores = {
        bucket: _score_context(
            make_candidate(
                context_status="fresh",
                current_group_age_bucket=bucket,
                trajectory_entropy="low",
            )
        )
        for bucket in ("age_lt_30", "age_30_60", "age_60_120", "age_gt_120")
    }
    assert len(set(scores.values())) == 4, f"бакеты слиплись: {scores}"


@pytest.mark.parametrize("context_status,expected", [
    ("fresh", 1.0),
    ("stale", 0.0),
])
def test_context_freshness_steps(make_candidate, context_status, expected):
    c = make_candidate(
        context_status=context_status,
        current_group_age_bucket="age_lt_30",
        trajectory_entropy="low",
    )
    assert _score_context(c) == pytest.approx((expected + 2.0) / 3.0)


@pytest.mark.parametrize("entropy,expected", [
    ("low", 1.0),
    ("medium", 0.5),
    ("high", 0.0),
])
def test_context_entropy_steps(make_candidate, entropy, expected):
    c = make_candidate(
        context_status="fresh",
        current_group_age_bucket="age_lt_30",
        trajectory_entropy=entropy,
    )
    assert _score_context(c) == pytest.approx((2.0 + expected) / 3.0)


# ─── Ось D: редкость и специфичность ──────────────────────────────────────────

@pytest.mark.parametrize("event_rarity,expected", [
    ("rare", 1.0),
    ("uncommon", 0.7),
    ("common", 0.2),
])
def test_rarity_event_bucket_steps(make_candidate, event_rarity, expected):
    c = make_candidate(
        event_rarity_bucket=event_rarity,
        transition_rarity="rare",
        research_score=0.9,
    )
    assert _score_rarity(c) == pytest.approx((expected + 2.0) / 3.0)


@pytest.mark.parametrize("transition_rarity,expected", [
    ("rare", 1.0),
    ("uncommon", 0.7),
    ("common", 0.2),
])
def test_rarity_transition_steps(make_candidate, transition_rarity, expected):
    c = make_candidate(
        event_rarity_bucket="rare",
        transition_rarity=transition_rarity,
        research_score=0.9,
    )
    assert _score_rarity(c) == pytest.approx((expected + 2.0) / 3.0)


@pytest.mark.parametrize("research_score,expected", [
    (0.95, 1.0),
    (0.86, 1.0),
    (0.85, 0.6),
    (0.71, 0.6),
    (0.70, 0.2),
    (0.10, 0.2),
])
def test_rarity_research_score_steps(make_candidate, research_score, expected):
    c = make_candidate(
        event_rarity_bucket="rare",
        transition_rarity="rare",
        research_score=research_score,
    )
    assert _score_rarity(c) == pytest.approx((expected + 2.0) / 3.0)


# ─── Агрегация ────────────────────────────────────────────────────────────────

def test_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_total_is_weighted_sum_of_axes(reference_candidate):
    b = score_candidate(reference_candidate)
    expected = (
        b.statistical * WEIGHTS["statistical"]
        + b.directional * WEIGHTS["directional"]
        + b.context * WEIGHTS["context"]
        + b.rarity * WEIGHTS["rarity"]
    )
    assert b.total == pytest.approx(expected, abs=1e-4)


def test_score_is_bounded(make_candidate):
    """Ни один вход не должен выводить quality_score за пределы [0..1]."""
    best = make_candidate(
        valid_label_pct=1.0, sample_size=10_000, monthly_concentration=0.0,
        repeatability_months=36, research_side="long", long_outcome_share=1.0,
        historical_outcome_skew=1.0, long_favorable_adverse_ratio_p70_p80=99.0,
        context_status="fresh", current_group_age_bucket="age_lt_30",
        trajectory_entropy="low", event_rarity_bucket="rare",
        transition_rarity="rare", research_score=1.0,
    )
    worst = make_candidate(
        valid_label_pct=0.0, sample_size=0, monthly_concentration=1.0,
        repeatability_months=0, research_side="long", long_outcome_share=0.0,
        historical_outcome_skew=0.0, long_favorable_adverse_ratio_p70_p80=0.0,
        context_status="stale", current_group_age_bucket="age_gt_120",
        trajectory_entropy="high", event_rarity_bucket="common",
        transition_rarity="common", research_score=0.0,
    )
    assert score_candidate(best).total == pytest.approx(1.0)
    assert 0.0 <= score_candidate(worst).total <= 1.0


def test_reference_candidate_is_strong(reference_candidate):
    """
    Эталонный кандидат из ТЗ: статистика и перекос идеальны, контекст провален
    (stale + age_gt_120 + medium entropy → только 0.5 балла из 3). Вывод — STRONG.

    Эти числа зафиксированы в README, раздел «Как читать результат оценки»:
    если тест упал после правки порогов — обнови и README.
    """
    b = score_candidate(reference_candidate)
    assert b.statistical == pytest.approx(1.0)
    assert b.directional == pytest.approx(1.0)
    assert b.context == pytest.approx(0.1667, abs=1e-4)
    assert b.rarity == pytest.approx(0.6333, abs=1e-4)
    assert b.total == pytest.approx(0.7783, abs=1e-4)
    assert get_rating(b.total) == "STRONG"


@pytest.mark.parametrize("score,rating", [
    (1.0, "STRONG"),
    (0.76, "STRONG"),
    (0.75, "STRONG"),       # граница включительно
    (0.7499, "MODERATE"),
    (0.55, "MODERATE"),     # граница включительно
    (0.5499, "WEAK"),
    (0.0, "WEAK"),
])
def test_get_rating_boundaries(score, rating):
    assert get_rating(score) == rating
