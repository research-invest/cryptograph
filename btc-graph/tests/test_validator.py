"""
Тесты валидатора: каждый warning_flag должен ставиться ровно тогда,
когда выполнено его условие, и не ставиться иначе.
"""
from __future__ import annotations

import pytest

from src.validator.candidate_validator import validate_candidate


def test_clean_candidate_has_no_flags(make_candidate):
    """Кандидат без единой аномалии не должен получать флагов."""
    c = make_candidate(
        symbol="BTCUSDT",
        context_status="fresh",
        current_group_age_bucket="age_lt_30",
        transition_rarity="rare",
        valid_label_pct=0.90,
        research_score=0.95,
        monthly_concentration=0.05,
        sample_size=1000,
        repeatability_months=12,
    )
    assert validate_candidate(c) == []


@pytest.mark.parametrize("symbol,expected", [
    ("BTCUSDT", False),
    ("ETHUSDT", True),
    ("btcusdt", True),      # регистр учитывается
])
def test_symbol_flag(make_candidate, symbol, expected):
    flags = validate_candidate(make_candidate(symbol=symbol))
    assert ("symbol_not_btcusdt" in flags) is expected


@pytest.mark.parametrize("status,expected", [
    ("stale", True),
    ("fresh", False),
])
def test_stale_flag(make_candidate, status, expected):
    flags = validate_candidate(make_candidate(context_status=status))
    assert ("context_status_stale" in flags) is expected


@pytest.mark.parametrize("bucket,expected", [
    ("age_gt_120", True),
    ("age_60_120", False),
    ("age_30_60", False),
    ("age_lt_30", False),
])
def test_age_flag(make_candidate, bucket, expected):
    flags = validate_candidate(make_candidate(current_group_age_bucket=bucket))
    assert ("age_gt_120" in flags) is expected


@pytest.mark.parametrize("status,bucket,expected", [
    ("stale", "age_gt_120", True),
    ("stale", "age_lt_30", False),
    ("fresh", "age_gt_120", False),
    ("fresh", "age_lt_30", False),
])
def test_combined_stale_and_aged_flag(make_candidate, status, bucket, expected):
    """Комбинированный флаг ставится только при одновременном совпадении."""
    flags = validate_candidate(
        make_candidate(context_status=status, current_group_age_bucket=bucket)
    )
    assert ("stale_and_aged_combined" in flags) is expected


@pytest.mark.parametrize("rarity,expected", [
    ("common", True),
    ("uncommon", False),
    ("rare", False),
])
def test_transition_rarity_flag(make_candidate, rarity, expected):
    flags = validate_candidate(make_candidate(transition_rarity=rarity))
    assert ("transition_rarity_common" in flags) is expected


@pytest.mark.parametrize("pct,expected", [
    (0.50, True),
    (0.69, True),
    (0.70, False),      # граница: строгое «<»
    (0.90, False),
])
def test_low_valid_label_pct_flag(make_candidate, pct, expected):
    flags = validate_candidate(make_candidate(valid_label_pct=pct))
    assert ("low_valid_label_pct" in flags) is expected


@pytest.mark.parametrize("research_score,valid_pct,expected", [
    (0.95, 0.50, True),     # высокая уверенность на дырявых данных
    (0.95, 0.90, False),    # уверенность оправдана
    (0.50, 0.50, False),    # данные плохие, но и уверенности нет
    (0.85, 0.50, False),    # ровно 0.85 — не «> 0.85»
])
def test_false_confidence_flag(make_candidate, research_score, valid_pct, expected):
    flags = validate_candidate(
        make_candidate(research_score=research_score, valid_label_pct=valid_pct)
    )
    assert ("false_confidence_high_score_low_validity" in flags) is expected


@pytest.mark.parametrize("concentration,expected", [
    (0.50, True),
    (0.31, True),
    (0.30, False),      # граница: строгое «>»
    (0.05, False),
])
def test_seasonal_concentration_flag(make_candidate, concentration, expected):
    flags = validate_candidate(make_candidate(monthly_concentration=concentration))
    assert ("high_monthly_concentration_seasonal_risk" in flags) is expected


@pytest.mark.parametrize("sample_size,expected", [
    (50, True),
    (99, True),
    (100, False),       # граница: строгое «<»
    (1339, False),
])
def test_small_sample_flag(make_candidate, sample_size, expected):
    flags = validate_candidate(make_candidate(sample_size=sample_size))
    assert ("very_small_sample_size" in flags) is expected


@pytest.mark.parametrize("months,expected", [
    (0, True),
    (2, True),
    (3, False),         # граница: строгое «<»
    (19, False),
])
def test_low_repeatability_flag(make_candidate, months, expected):
    flags = validate_candidate(make_candidate(repeatability_months=months))
    assert ("low_repeatability_months" in flags) is expected


def test_reference_candidate_flags(reference_candidate):
    """Эталонный кандидат из ТЗ: ровно четыре флага, все про контекст."""
    assert validate_candidate(reference_candidate) == [
        "context_status_stale",
        "age_gt_120",
        "stale_and_aged_combined",
        "transition_rarity_common",
    ]


def test_validator_never_raises(make_candidate):
    """Валидатор только помечает, но не бросает исключений даже на худших данных."""
    worst = make_candidate(
        symbol="DOGEUSDT", context_status="stale", current_group_age_bucket="age_gt_120",
        transition_rarity="common", valid_label_pct=0.0, research_score=1.0,
        monthly_concentration=1.0, sample_size=0, repeatability_months=0,
    )
    flags = validate_candidate(worst)
    assert len(flags) == 10                 # сработали все правила
    assert len(set(flags)) == len(flags)    # без дублей
