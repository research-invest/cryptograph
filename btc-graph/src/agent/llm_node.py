"""
LLM Reasoning node: генерирует strengths, risks и summary через Claude API.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import anthropic

from src.config.profiles import ScoringProfile, get_profile
from src.models.candidate import Candidate, CandidateEvaluation
from src.scorer.candidate_scorer import ScoreBreakdown, fa_ratio_for, get_rating, win_rate_for

logger = logging.getLogger(__name__)

# Клиент сам ретраит сетевые сбои и 429/5xx; таймаут не даёт запросу висеть вечно.
LLM_TIMEOUT_SECONDS = 30.0
LLM_MAX_RETRIES = 2

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )
    return _client


_SYSTEM_TEMPLATE = """Ты аналитик торговых кандидатов по инструменту {symbol}.
Тебе дают структурированные данные кандидата — снимок рыночной конфигурации с историческими метриками.
Твоя задача: дать краткий профессиональный анализ.

ВАЖНО:
- research_score — это НЕ вероятность прибыли, а исследовательская метрика силы кандидата.
- Кандидат — это ИДЕЯ для исследования, а не торговый сигнал.
- Будь честен: называй риски и слабые стороны явно.
- quality_score посчитан профилем калибровки {scoring_profile}, откалиброванным
  под этот инструмент. Сравнивать его с оценками других монет нельзя.
{market_hint}"""

_USER_TEMPLATE = """Проанализируй кандидата:

candidate_id: {candidate_id}
symbol: {symbol} | профиль оценки: {scoring_profile}
research_side: {research_side} | horizon: {horizon}
research_score: {research_score}
transition: {transition_id} ({transition_rarity})
context_status: {context_status} | age: {current_group_age_bucket} | entropy: {trajectory_entropy}
sample_size: {sample_size} | valid: {valid_label_count} ({valid_label_pct:.1%})
{win_rate_line}
{fa_line}
repeatability: {repeatability_months} мес, {repeatability_days} дней | monthly_conc: {monthly_concentration:.1%}
event_block: {event_block_id} | intensity: {event_intensity_bucket} | rarity: {event_rarity_bucket}
quality_score: {quality_score:.3f} | rating: {rating}
warning_flags: {warning_flags}

Верни JSON с полями:
- "strengths": список из 3-5 строк (сильные стороны)
- "risks": список из 2-3 строк (основные риски)
- "summary": одно предложение — итоговая оценка кандидата
"""


def _build_system_prompt(candidate: Candidate, profile: ScoringProfile) -> str:
    """
    Системный промпт параметризуется монетой и подсказкой из её профиля.

    «Ты аналитик кандидатов BTC» на кандидате по ETH — не мелочь: модель
    начинает рассуждать о ликвидности и глубине истории не того рынка.
    """
    hint = profile.llm.market_hint.strip()
    return _SYSTEM_TEMPLATE.format(
        symbol=candidate.symbol,
        scoring_profile=profile.name,
        market_hint=f"- Контекст рынка: {hint}\n" if hint else "",
    )


def _build_prompt(
    candidate: Candidate,
    score: ScoreBreakdown,
    warning_flags: list[str],
    profile: ScoringProfile | None = None,
) -> str:
    profile = profile if profile is not None else get_profile(candidate.symbol)
    rating = get_rating(score.total, profile)
    win_rate = win_rate_for(candidate)
    fa_ratio = fa_ratio_for(candidate)

    win_rate_line = (
        f"win_rate ({candidate.research_side.value}): {win_rate:.1%} | "
        f"long_outcome_share: {candidate.long_outcome_share:.1%} | "
        f"outcome_skew: {candidate.historical_outcome_skew:+.3f}"
    )

    if fa_ratio is not None:
        fa_line = (
            f"favorable/adverse: +{candidate.p70_long_favorable_pct:.2f}% / "
            f"-{candidate.p80_long_adverse_pct:.2f}% (ratio: {fa_ratio:.2f}x)"
        )
    else:
        # Для short p70/p80 описывают движение вверх — не выдаём их за метрику
        # кандидата, иначе LLM опишет чужую сторону рынка как сильную сторону.
        fa_line = (
            "favorable/adverse: неприменимо к short-кандидату "
            "(в данных есть только long-перцентили p70/p80)"
        )

    return _USER_TEMPLATE.format(
        win_rate_line=win_rate_line,
        fa_line=fa_line,
        candidate_id=candidate.candidate_id,
        symbol=candidate.symbol,
        scoring_profile=profile.name,
        research_side=candidate.research_side.value,
        horizon=candidate.horizon,
        research_score=candidate.research_score,
        transition_id=candidate.transition_id,
        transition_rarity=candidate.transition_rarity.value,
        context_status=candidate.context_status.value,
        current_group_age_bucket=candidate.current_group_age_bucket.value,
        trajectory_entropy=candidate.trajectory_entropy.value,
        sample_size=candidate.sample_size,
        valid_label_count=candidate.valid_label_count,
        valid_label_pct=candidate.valid_label_pct,
        repeatability_months=candidate.repeatability_months,
        repeatability_days=candidate.repeatability_days,
        monthly_concentration=candidate.monthly_concentration,
        event_block_id=candidate.event_block_id,
        event_intensity_bucket=candidate.event_intensity_bucket.value,
        event_rarity_bucket=candidate.event_rarity_bucket.value,
        quality_score=score.total,
        rating=rating,
        warning_flags=", ".join(warning_flags) if warning_flags else "нет",
    )


def _extract_text(response) -> str:
    """
    Достаёт текст из ответа Claude.

    Берём первый блок с type == "text", а не content[0]: при extended thinking
    или tool use первым идёт нетекстовый блок, и обращение к .text падает.
    """
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text.strip()
    raise ValueError("В ответе Claude нет текстового блока")


def _fallback_evaluation(
    candidate: Candidate,
    score: ScoreBreakdown,
    warning_flags: list[str],
    reason: str,
    profile: ScoringProfile | None = None,
) -> CandidateEvaluation:
    """
    Детерминированная оценка вместо LLM-объяснения.

    Числа те же, что и при успешном вызове, — меняются только формулировки,
    и в risks добавляется явная пометка о причине.
    """
    from src.agent.pipeline import _deterministic_evaluation

    evaluation = _deterministic_evaluation(candidate, score, warning_flags, profile)
    evaluation.risks = [
        f"Аналитическое резюме сгенерировано без LLM ({reason}) — тексты шаблонные",
        *evaluation.risks,
    ]
    return evaluation


def evaluate_with_llm(
    candidate: Candidate,
    score: ScoreBreakdown,
    warning_flags: list[str],
    model: str = "claude-haiku-4-5-20251001",
    profile: ScoringProfile | None = None,
) -> CandidateEvaluation:
    """
    Вызывает Claude для генерации strengths/risks/summary.
    Собирает итоговый CandidateEvaluation.

    Все числа (quality_score, rating, win_rate) считает детерминированный скорер —
    LLM отвечает только за формулировки. Поэтому при недоступности API оценка не
    теряется: возвращается детерминированный вариант с пометкой в risks.
    """
    import json as _json

    profile = profile if profile is not None else get_profile(candidate.symbol)
    prompt = _build_prompt(candidate, score, warning_flags, profile)

    try:
        client = _get_client()
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=_build_system_prompt(candidate, profile),
            messages=[{"role": "user", "content": prompt}],
        )
        content = _extract_text(response)
    except anthropic.APIError as exc:
        logger.warning(
            "Claude API недоступен (%s) — оценка %s собрана без LLM",
            type(exc).__name__, candidate.candidate_id, exc_info=True,
        )
        return _fallback_evaluation(
            candidate, score, warning_flags, reason="сбой Claude API", profile=profile
        )
    except Exception:
        logger.exception(
            "Непредвиденная ошибка при вызове Claude для %s", candidate.candidate_id
        )
        return _fallback_evaluation(
            candidate, score, warning_flags, reason="ошибка вызова LLM", profile=profile
        )

    # Извлекаем JSON из ответа
    json_match = content
    if "```json" in content:
        json_match = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        json_match = content.split("```")[1].split("```")[0].strip()

    try:
        llm_data = _json.loads(json_match)
    except _json.JSONDecodeError:
        logger.warning(
            "Ответ Claude по кандидату %s не разобрался как JSON", candidate.candidate_id
        )
        llm_data = {
            "strengths": ["Не удалось распарсить ответ LLM"],
            "risks": ["Проверьте логи"],
            "summary": content[:200],
        }

    rating = get_rating(score.total, profile)
    win_rate = win_rate_for(candidate)
    fa_ratio = fa_ratio_for(candidate)

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
        strengths=llm_data.get("strengths", []),
        risks=llm_data.get("risks", []),
        summary=llm_data.get("summary", ""),
        score_statistical=score.statistical,
        score_directional=score.directional,
        score_context=score.context,
        score_rarity=score.rarity,
    )
