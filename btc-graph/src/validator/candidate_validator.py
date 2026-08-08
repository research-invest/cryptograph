"""
Валидатор кандидатов: проверяет диапазоны значений и формирует warning_flags.

Контракт: валидатор только помечает и никогда не бросает исключений. Кандидат
по монете без профиля обязан быть оценён — с флагом, но оценён.
"""
from __future__ import annotations

from src.config.profiles import ScoringProfile, get_profile, is_known_symbol
from src.models.candidate import Candidate, ContextStatus, AgeBucket, TransitionRarity


def validate_candidate(
    candidate: Candidate,
    profile: ScoringProfile | None = None,
) -> list[str]:
    """
    Проверяет кандидата на аномальные значения.
    Возвращает список warning_flags.

    Пороги берутся из профиля монеты: «выборка меньше 100 случаев» — это
    характеристика калибровки, а не универсальная истина.
    """
    profile = profile if profile is not None else get_profile(candidate.symbol)
    limits = profile.validator
    flags: list[str] = []

    # Поле symbol имеет значение по умолчанию, поэтому payload без него молча
    # становился биткоином. model_fields_set показывает, что реально пришло.
    if "symbol" not in candidate.model_fields_set:
        flags.append("symbol_defaulted")

    # Реестр профилей работает whitelist'ом монет: своего профиля нет —
    # значит монета считается чужой линейкой, и знать об этом надо.
    if not is_known_symbol(candidate.symbol):
        flags.append("unknown_symbol_profile")

    if candidate.context_status == ContextStatus.stale:
        flags.append("context_status_stale")

    if candidate.current_group_age_bucket == AgeBucket.age_gt_120:
        flags.append("age_gt_120")

    if candidate.context_status == ContextStatus.stale and candidate.current_group_age_bucket == AgeBucket.age_gt_120:
        flags.append("stale_and_aged_combined")

    if candidate.transition_rarity == TransitionRarity.common:
        flags.append("transition_rarity_common")

    if candidate.valid_label_pct < limits.low_valid_label_pct:
        flags.append("low_valid_label_pct")

    if (candidate.research_score > limits.false_confidence_research_score
            and candidate.valid_label_pct < limits.low_valid_label_pct):
        flags.append("false_confidence_high_score_low_validity")

    if candidate.monthly_concentration > limits.high_monthly_concentration:
        flags.append("high_monthly_concentration_seasonal_risk")

    if candidate.sample_size < limits.very_small_sample_size:
        flags.append("very_small_sample_size")

    if candidate.repeatability_months < limits.low_repeatability_months:
        flags.append("low_repeatability_months")

    return flags
