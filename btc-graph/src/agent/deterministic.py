"""
Детерминированная оценка кандидата — без обращения к LLM.

Вынесена в отдельный модуль, чтобы pipeline и llm_node могли пользоваться ею
без взаимного импорта, а окружение без пакета `anthropic` (тесты, лёгкие
воркеры) не тянуло SDK ради формул из ТЗ.

Пороги формулировок берутся из ступеней профиля, а не из литералов: «очень
высокий research_score» — это верхняя ступень оси rarity той монеты, о которой
идёт речь. Иначе текст про альткоин описывал бы линейку биткоина.
"""
from __future__ import annotations

from src.config.profiles import ScoringProfile, get_profile
from src.models.candidate import (
    AgeBucket,
    Candidate,
    CandidateEvaluation,
    SampleScope,
    TransitionRarity,
)
from src.scorer.candidate_scorer import ScoreBreakdown, fa_ratio_for, get_rating, win_rate_for


def deterministic_evaluation(
    candidate: Candidate,
    score: ScoreBreakdown,
    warning_flags: list[str],
    profile: ScoringProfile | None = None,
) -> CandidateEvaluation:
    """Генерирует оценку без вызова LLM на основе правил ТЗ."""
    profile = profile if profile is not None else get_profile(candidate.symbol)

    research_ladder = profile.rarity.research_score
    win_ladder = profile.directional.win_rate
    fa_ladder = profile.directional.fa_ratio
    repeat_ladder = profile.statistical.repeatability_months
    concentration_ladder = profile.statistical.monthly_concentration
    validity_ladder = profile.statistical.valid_label_pct

    strengths = []
    risks = []

    if candidate.research_score > research_ladder.step(0):
        strengths.append(f"Очень высокий research_score ({candidate.research_score:.3f})")
    elif candidate.research_score > research_ladder.step(-1):
        strengths.append(f"Хороший research_score ({candidate.research_score:.3f})")

    win_rate = win_rate_for(candidate)
    if win_rate > win_ladder.step(0):
        strengths.append(f"{win_rate:.1%} win rate при выборке {candidate.valid_label_count}")
    elif win_rate > win_ladder.step(-1):
        strengths.append(f"Умеренный win rate {win_rate:.1%}")

    fa_ratio = fa_ratio_for(candidate)
    if fa_ratio is None:
        risks.append(
            "F/A ratio неприменим к short-кандидату: p70/p80 в данных описывают "
            "движение вверх — потенциал и риск не оценены"
        )
    elif fa_ratio > fa_ladder.step(0):
        strengths.append(f"F/A ratio {fa_ratio:.2f}x — отличное соотношение")
    elif fa_ratio > fa_ladder.step(-1):
        strengths.append(f"F/A ratio {fa_ratio:.2f}x — приемлемое соотношение")

    if candidate.repeatability_months > repeat_ladder.step(1):
        strengths.append(f"Повторяемость в {candidate.repeatability_months} месяцах — устойчивый паттерн")

    if candidate.monthly_concentration < concentration_ladder.step(1):
        strengths.append(f"Низкая сезонная концентрация ({candidate.monthly_concentration:.1%})")

    if candidate.context_status.value == "stale":
        risks.append("Контекст устарел (stale) — состояние может быть неактуальным")

    if candidate.current_group_age_bucket == AgeBucket.age_gt_120:
        risks.append("Состояние группы > 120 мин — возможна смена рыночной фазы")

    if candidate.sample_scope == SampleScope.transition:
        risks.append(
            f"Выборка собрана по переходу целиком, а не по паре «переход + блок "
            f"{candidate.event_block_id}»: характеристики блока (редкость "
            f"{candidate.event_rarity_bucket.value}, доля строк) описывают текущий "
            f"бар, но к этой статистике не относятся"
        )

    if (candidate.effective_sample_size is not None
            and candidate.effective_sample_size < candidate.sample_size):
        risks.append(
            f"Независимых случаев {candidate.effective_sample_size} из "
            f"{candidate.sample_size} строк выборки: остальное — снимки той же "
            f"реализации перехода со смещением 45/90/180 минут"
        )

    if candidate.transition_rarity == TransitionRarity.common:
        risks.append(f"Переход {candidate.transition_id} часто встречается (common) — менее специфичный сигнал")

    validity_floor = validity_ladder.step(1)
    if candidate.valid_label_pct < validity_floor:
        risks.append(
            f"Доля валидных меток {candidate.valid_label_pct:.1%} < "
            f"{validity_floor:.0%} — ограниченная надёжность"
        )

    if "unknown_symbol_profile" in warning_flags:
        risks.append(
            f"У монеты {candidate.symbol} нет своего профиля оценки — кандидат "
            f"посчитан базовой калибровкой ({profile.name}), пороги могут быть чужими"
        )

    if not strengths:
        strengths = ["Нет выраженных сильных сторон"]
    if not risks:
        risks = ["Явных рисков не обнаружено"]

    rating = get_rating(score.total, profile)
    fa_text = f"F/A ratio={fa_ratio:.2f}x." if fa_ratio is not None else "F/A ratio неприменим."
    summary = (
        f"{rating} {candidate.research_side.value.upper()} кандидат "
        f"по {candidate.symbol}: quality_score={score.total:.3f}, "
        f"win_rate={win_rate:.1%}, {fa_text}"
    )

    return CandidateEvaluation(
        candidate_id=candidate.candidate_id,
        symbol=candidate.symbol,
        scoring_profile=score.profile,
        profile_fingerprint=score.fingerprint,
        quality_score=score.total,
        quality_score_baseline=score.baseline_total,
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
