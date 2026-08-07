"""
Scorer: вычисляет quality_score [0..1] по 4 осям согласно ТЗ.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.models.candidate import (
    Candidate,
    ContextStatus,
    AgeBucket,
    TrajectoryEntropy,
    TransitionRarity,
    EventRarityBucket,
    ResearchSide,
)

# Границы итогового рейтинга. Единственное определение в проекте —
# все потребители обязаны звать get_rating(), а не сравнивать числа сами.
RATING_STRONG_MIN = 0.75
RATING_MODERATE_MIN = 0.55


@dataclass
class ScoreBreakdown:
    statistical: float   # A. Статистическая надёжность
    directional: float   # B. Directional asymmetry
    context: float       # C. Контекстуальная свежесть
    rarity: float        # D. Редкость и специфичность
    total: float


def _score_statistical(c: Candidate) -> float:
    """A. Статистическая надёжность выборки."""
    score = 0.0

    if c.valid_label_pct > 0.85:
        score += 1.0
    elif c.valid_label_pct > 0.80:
        score += 0.7
    elif c.valid_label_pct > 0.75:
        score += 0.4
    else:
        score += 0.0

    if c.sample_size > 1000:
        score += 1.0
    elif c.sample_size > 500:
        score += 0.7
    elif c.sample_size > 200:
        score += 0.4
    else:
        score += 0.1

    if c.monthly_concentration < 0.10:
        score += 1.0
    elif c.monthly_concentration < 0.15:
        score += 0.7
    elif c.monthly_concentration < 0.30:
        score += 0.3
    else:
        score += 0.0

    if c.repeatability_months > 15:
        score += 1.0
    elif c.repeatability_months > 12:
        score += 0.7
    elif c.repeatability_months > 6:
        score += 0.4
    else:
        score += 0.1

    return score / 4.0


def win_rate_for(c: Candidate) -> float:
    """
    Доля исходов в сторону research_side.

    Единственное место, где long_outcome_share переводится в win rate —
    используется и скорером, и сборкой CandidateEvaluation.
    """
    if c.research_side == ResearchSide.long:
        return c.long_outcome_share
    return 1.0 - c.long_outcome_share


def fa_ratio_for(c: Candidate) -> float | None:
    """
    Favorable/adverse ratio, применимый к research_side кандидата.

    p70_long_favorable_pct / p80_long_adverse_pct описывают движение ВВЕРХ:
    первое — потенциал long, второе — риск long. Для short-кандидата выгода и
    риск меняются местами, а симметричных short-полей источник данных не даёт.
    Поэтому для short метрика неприменима и возвращается None — вместо того
    чтобы начислять баллы за чужую сторону рынка.
    """
    if c.research_side == ResearchSide.long:
        return c.long_favorable_adverse_ratio_p70_p80
    return None


def _score_directional(c: Candidate) -> float:
    """
    B. Сила directional asymmetry.

    Для long усредняется по трём критериям, для short — по двум:
    F/A ratio к short-кандидату неприменим (см. fa_ratio_for).
    """
    score = 0.0
    criteria = 2

    win_rate = win_rate_for(c)

    if win_rate > 0.70:
        score += 1.0
    elif win_rate > 0.65:
        score += 0.7
    elif win_rate > 0.60:
        score += 0.4
    else:
        score += 0.1

    skew = abs(c.historical_outcome_skew)
    if skew > 0.40:
        score += 1.0
    elif skew > 0.30:
        score += 0.7
    elif skew > 0.20:
        score += 0.4
    else:
        score += 0.1

    ratio = fa_ratio_for(c)
    if ratio is not None:
        criteria = 3
        if ratio > 4.0:
            score += 1.0
        elif ratio > 3.0:
            score += 0.7
        elif ratio > 2.0:
            score += 0.4
        else:
            score += 0.1

    return score / criteria


# Чем дольше рынок сидит в текущем состоянии, тем ближе смена фазы —
# оценка убывает по мере старения состояния.
_AGE_SCORE = {
    AgeBucket.age_lt_30: 1.0,
    AgeBucket.age_30_60: 0.75,
    AgeBucket.age_60_120: 0.5,
    AgeBucket.age_gt_120: 0.0,
}


def _score_context(c: Candidate) -> float:
    """C. Контекстуальная свежесть."""
    score = 0.0

    if c.context_status == ContextStatus.fresh:
        score += 1.0
    else:
        score += 0.0

    score += _AGE_SCORE[c.current_group_age_bucket]

    if c.trajectory_entropy == TrajectoryEntropy.low:
        score += 1.0
    elif c.trajectory_entropy == TrajectoryEntropy.medium:
        score += 0.5
    else:
        score += 0.0

    return score / 3.0


def _score_rarity(c: Candidate) -> float:
    """D. Редкость и специфичность."""
    score = 0.0

    if c.event_rarity_bucket == EventRarityBucket.rare:
        score += 1.0
    elif c.event_rarity_bucket == EventRarityBucket.uncommon:
        score += 0.7
    else:
        score += 0.2

    if c.transition_rarity == TransitionRarity.rare:
        score += 1.0
    elif c.transition_rarity == TransitionRarity.uncommon:
        score += 0.7
    else:
        score += 0.2

    if c.research_score > 0.85:
        score += 1.0
    elif c.research_score > 0.70:
        score += 0.6
    else:
        score += 0.2

    return score / 3.0


WEIGHTS = {
    "statistical": 0.30,
    "directional": 0.35,
    "context": 0.20,
    "rarity": 0.15,
}


def score_candidate(candidate: Candidate) -> ScoreBreakdown:
    """Вычисляет quality_score и разбивку по осям."""
    stat = _score_statistical(candidate)
    direc = _score_directional(candidate)
    ctx = _score_context(candidate)
    rar = _score_rarity(candidate)

    total = (
        stat * WEIGHTS["statistical"]
        + direc * WEIGHTS["directional"]
        + ctx * WEIGHTS["context"]
        + rar * WEIGHTS["rarity"]
    )

    return ScoreBreakdown(
        statistical=round(stat, 4),
        directional=round(direc, 4),
        context=round(ctx, 4),
        rarity=round(rar, 4),
        total=round(total, 4),
    )


def get_rating(quality_score: float) -> str:
    """STRONG / MODERATE / WEAK по quality_score. Единственный источник порогов."""
    if quality_score >= RATING_STRONG_MIN:
        return "STRONG"
    elif quality_score >= RATING_MODERATE_MIN:
        return "MODERATE"
    else:
        return "WEAK"
