"""
Задача 3: генерация текстового аналитического резюме по формату ТЗ.
"""
from __future__ import annotations

from src.models.candidate import Candidate, CandidateEvaluation


def format_report(candidate: Candidate, evaluation: CandidateEvaluation) -> str:
    win_rate_pct = evaluation.win_rate * 100

    strengths_lines = "\n".join(f"  + {s}" for s in evaluation.strengths)
    risks_lines = "\n".join(f"  - {r}" for r in evaluation.risks)

    return f"""КАНДИДАТ: {evaluation.candidate_id}
Горизонт: {candidate.horizon} | Направление: {evaluation.direction.upper()} | Win rate: {win_rate_pct:.1f}%

КЛЮЧЕВЫЕ МЕТРИКИ:
- Выборка: {candidate.valid_label_count} валидных из {candidate.sample_size} ({candidate.valid_label_pct:.0%})
- Win rate: {candidate.long_outcome_share:.1%} в сторону {candidate.research_side.value}
- Потенциал / риск: +{candidate.p70_long_favorable_pct:.2f}% / -{candidate.p80_long_adverse_pct:.2f}% (ratio: {candidate.long_favorable_adverse_ratio_p70_p80:.1f}x)
- Повторяемость: {candidate.repeatability_months} мес, {candidate.repeatability_days} дней

КОНТЕКСТ:
- Переход: {candidate.transition_id} ({candidate.transition_rarity.value})
- Состояние: group {candidate.current_group_id}, возраст {candidate.current_group_age_bucket.value}, статус {candidate.context_status.value}
- События: {candidate.event_block_id}, интенсивность {candidate.event_intensity_bucket.value}, семей {candidate.event_family_count}

ОЦЕНКА:
- quality_score: {evaluation.quality_score:.3f}
- statistical: {evaluation.score_statistical:.3f} | directional: {evaluation.score_directional:.3f} | context: {evaluation.score_context:.3f} | rarity: {evaluation.score_rarity:.3f}
- Флаги: {", ".join(evaluation.warning_flags) if evaluation.warning_flags else "нет"}

ВЫВОД: {evaluation.rating} {evaluation.direction.upper()} кандидат
Сильные стороны:
{strengths_lines}
Риски:
{risks_lines}

{evaluation.summary}
"""
