"""
Валидатор кандидатов: проверяет диапазоны значений и формирует warning_flags.
"""
from __future__ import annotations

from src.models.candidate import Candidate, ContextStatus, AgeBucket, TransitionRarity


def validate_candidate(candidate: Candidate) -> list[str]:
    """
    Проверяет кандидата на аномальные значения.
    Возвращает список warning_flags.
    """
    flags: list[str] = []

    if candidate.symbol != "BTCUSDT":
        flags.append("symbol_not_btcusdt")

    if candidate.context_status == ContextStatus.stale:
        flags.append("context_status_stale")

    if candidate.current_group_age_bucket == AgeBucket.age_gt_120:
        flags.append("age_gt_120")

    if candidate.context_status == ContextStatus.stale and candidate.current_group_age_bucket == AgeBucket.age_gt_120:
        flags.append("stale_and_aged_combined")

    if candidate.transition_rarity == TransitionRarity.common:
        flags.append("transition_rarity_common")

    if candidate.valid_label_pct < 0.70:
        flags.append("low_valid_label_pct")

    if candidate.research_score > 0.85 and candidate.valid_label_pct < 0.70:
        flags.append("false_confidence_high_score_low_validity")

    if candidate.monthly_concentration > 0.30:
        flags.append("high_monthly_concentration_seasonal_risk")

    if candidate.sample_size < 100:
        flags.append("very_small_sample_size")

    if candidate.repeatability_months < 3:
        flags.append("low_repeatability_months")

    return flags
