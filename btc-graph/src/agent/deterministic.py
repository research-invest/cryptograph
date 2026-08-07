"""
Детерминированная оценка кандидата — без обращения к LLM.

Вынесена в отдельный модуль, чтобы pipeline и llm_node могли пользоваться ею
без взаимного импорта, а окружение без пакета `anthropic` (тесты, лёгкие
воркеры) не тянуло SDK ради формул из ТЗ.
"""
from __future__ import annotations

from src.models.candidate import AgeBucket, Candidate, CandidateEvaluation, TransitionRarity
from src.scorer.candidate_scorer import ScoreBreakdown, fa_ratio_for, get_rating, win_rate_for


def deterministic_evaluation(
    candidate: Candidate,
    score: ScoreBreakdown,
    warning_flags: list[str],
) -> CandidateEvaluation:
    """Генерирует оценку без вызова LLM на основе правил ТЗ."""

    strengths = []
    risks = []

    if candidate.research_score > 0.85:
        strengths.append(f"Очень высокий research_score ({candidate.research_score:.3f})")
    elif candidate.research_score > 0.70:
        strengths.append(f"Хороший research_score ({candidate.research_score:.3f})")

    win_rate = win_rate_for(candidate)
    if win_rate > 0.70:
        strengths.append(f"{win_rate:.1%} win rate при выборке {candidate.valid_label_count}")
    elif win_rate > 0.60:
        strengths.append(f"Умеренный win rate {win_rate:.1%}")

    fa_ratio = fa_ratio_for(candidate)
    if fa_ratio is None:
        risks.append(
            "F/A ratio неприменим к short-кандидату: p70/p80 в данных описывают "
            "движение вверх — потенциал и риск не оценены"
        )
    elif fa_ratio > 4.0:
        strengths.append(f"F/A ratio {fa_ratio:.2f}x — отличное соотношение")
    elif fa_ratio > 2.0:
        strengths.append(f"F/A ratio {fa_ratio:.2f}x — приемлемое соотношение")

    if candidate.repeatability_months > 12:
        strengths.append(f"Повторяемость в {candidate.repeatability_months} месяцах — устойчивый паттерн")

    if candidate.monthly_concentration < 0.15:
        strengths.append(f"Низкая сезонная концентрация ({candidate.monthly_concentration:.1%})")

    if candidate.context_status.value == "stale":
        risks.append("Контекст устарел (stale) — состояние может быть неактуальным")

    if candidate.current_group_age_bucket == AgeBucket.age_gt_120:
        risks.append("Состояние группы > 120 мин — возможна смена рыночной фазы")

    if candidate.transition_rarity == TransitionRarity.common:
        risks.append(f"Переход {candidate.transition_id} часто встречается (common) — менее специфичный сигнал")

    if candidate.valid_label_pct < 0.80:
        risks.append(f"Доля валидных меток {candidate.valid_label_pct:.1%} < 80% — ограниченная надёжность")

    if not strengths:
        strengths = ["Нет выраженных сильных сторон"]
    if not risks:
        risks = ["Явных рисков не обнаружено"]

    rating = get_rating(score.total)
    fa_text = f"F/A ratio={fa_ratio:.2f}x." if fa_ratio is not None else "F/A ratio неприменим."
    summary = (
        f"{rating} {candidate.research_side.value.upper()} кандидат: "
        f"quality_score={score.total:.3f}, win_rate={win_rate:.1%}, {fa_text}"
    )

    return CandidateEvaluation(
        candidate_id=candidate.candidate_id,
        quality_score=score.total,
        rating=rating,
        direction=candidate.research_side.value,
        win_rate=round(win_rate, 4),
        favorable_adverse_ratio=round(fa_ratio, 2) if fa_ratio is not None else None,
        context_freshness=candidate.context_status.value,
        warning_flags=warning_flags,
        strengths=strengths,
        risks=risks,
        summary=summary,
        score_statistical=score.statistical,
        score_directional=score.directional,
        score_context=score.context,
        score_rarity=score.rarity,
    )
