"""
Главный pipeline: Parser → Validator → Scorer → LLM → DB + Redis → Output.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Union

from src.config.profiles import ScoringProfile, get_profile
from src.models.candidate import Candidate, CandidateEvaluation
from src.parser.candidate_parser import parse_candidate, parse_candidates
from src.validator.candidate_validator import validate_candidate
from src.scorer.candidate_scorer import ScoreBreakdown, score_candidate
from src.agent.deterministic import deterministic_evaluation
from src.agent.report_formatter import format_report

logger = logging.getLogger(__name__)

# Обратная совместимость: раньше функция жила здесь.
_deterministic_evaluation = deterministic_evaluation

_USE_DB = os.environ.get("USE_DB", "true").lower() == "true"
_USE_REDIS = os.environ.get("USE_REDIS", "true").lower() == "true"
_USE_GRAPH = os.environ.get("USE_GRAPH", "true").lower() == "true"


def _evaluate(
    candidate: Candidate,
    use_llm: bool,
    llm_model: str,
    profile: ScoringProfile | None = None,
    score: ScoreBreakdown | None = None,
) -> CandidateEvaluation:
    """
    Валидация → скоринг → объяснение (LLM или детерминированное).

    Профиль резолвится ОДИН раз и прокидывается во все узлы. Иначе
    POST /config/reload посреди батча дал бы кандидата, у которого score
    посчитан одной версией калибровки, а rating и тексты — другой.

    `score` — уже посчитанная разбивка. Её передаёт батчевый путь: фильтр
    считает score всем кандидатам, и пересчитывать его выжившим значит делать
    двойную работу (у монеты со своим профилем скоринг ещё и двухпроходный —
    профильный плюс baseline). Передавать сюда разбивку, посчитанную ДРУГИМ
    профилем, нельзя: rating и тексты разъедутся со score.
    """
    profile = profile if profile is not None else get_profile(candidate.symbol)

    warning_flags = validate_candidate(candidate, profile)
    if score is None:
        score = score_candidate(candidate, profile)

    if use_llm:
        # Импорт ленивый: без use_llm пакет anthropic не нужен вовсе.
        from src.agent.llm_node import evaluate_with_llm
        return evaluate_with_llm(
            candidate, score, warning_flags, model=llm_model, profile=profile
        )
    return deterministic_evaluation(candidate, score, warning_flags, profile)


def run_pipeline(
    raw: Union[str, dict],
    use_llm: bool = True,
    llm_model: str = "claude-haiku-4-5-20251001",
    print_report: bool = False,
    save: bool = True,
    use_cache: bool = True,
) -> CandidateEvaluation:
    """
    Полный pipeline для одного кандидата.

    save=True:      сохраняет в PostgreSQL, обновляет граф Neo4j, кэширует в Redis.
    use_cache=True: переиспользует свежую оценку того же кандидата из Redis
                    (дедупликация по configuration_hash, TTL 30 мин).
    """
    candidate = parse_candidate(raw)

    cached = _cached_evaluation(candidate) if use_cache and _USE_REDIS else None
    if cached is not None:
        return cached

    evaluation = _evaluate(candidate, use_llm, llm_model)

    if save:
        _persist(candidate, evaluation)

    if print_report:
        print(format_report(candidate, evaluation))

    return evaluation


def run_batch_pipeline(
    raw: Union[str, list],
    use_llm: bool = True,
    llm_model: str = "claude-haiku-4-5-20251001",
    min_quality_score: float | None = None,
    save: bool = True,
) -> list[CandidateEvaluation]:
    """
    Pipeline для списка кандидатов с фильтрацией и дедупликацией по family_key.

    min_quality_score=None — порог берётся из профиля каждой монеты. Явное
    число применяется ко всему батчу и перекрывает профили: так удобно
    прогонять разовые эксперименты, но для регулярной работы это ровно та
    абсолютная линейка, от которой уводит шаг 4.

    Батч может содержать несколько монет. Профили резолвятся один раз здесь,
    до первой оценки: hot-reload конфига посреди батча не должен приводить
    к тому, что часть кандидатов посчитана старой калибровкой, часть новой.
    """
    from src.filters.candidate_filter import score_and_filter, select_best_per_family

    candidates = parse_candidates(raw)
    profiles = {c.symbol: get_profile(c.symbol) for c in candidates}

    scored = score_and_filter(
        candidates, min_quality_score=min_quality_score, profiles=profiles
    )
    # Разбивки, посчитанные фильтром, переиспользуются в _evaluate. Ключ —
    # candidate_id: он уникален в пределах монеты, symbol добавлен потому, что
    # батч бывает мультимонетным.
    breakdowns = {(c.symbol, c.candidate_id): b for c, b in scored}
    best = select_best_per_family(
        [(c, b.total) for c, b in scored], profiles=profiles
    )

    results = []
    for candidate, _ in best:
        evaluation = _evaluate(
            candidate, use_llm, llm_model,
            profile=profiles.get(candidate.symbol),
            score=breakdowns.get((candidate.symbol, candidate.candidate_id)),
        )
        if save:
            _persist(candidate, evaluation)
        results.append(evaluation)

    return results


def _cached_evaluation(candidate: Candidate) -> CandidateEvaluation | None:
    """
    Достаёт из Redis свежую оценку ЭТОГО ЖЕ кандидата по configuration_hash.

    configuration_hash описывает конфигурацию рынка, а не кандидата: под одним
    хэшем могут идти разные candidate_id. Раньше кэш отдавался без проверки, и
    клиент получал чужую оценку с чужим candidate_id. Теперь чужая запись
    игнорируется — кандидат считается заново.
    """
    if not candidate.configuration_hash:
        return None

    from src.cache import redis_cache

    cached_id = redis_cache.is_hash_cached(
        candidate.symbol, candidate.configuration_hash
    )
    if not cached_id:
        return None

    if cached_id != candidate.candidate_id:
        logger.info(
            "configuration_hash %s (%s) уже встречался у кандидата %s — "
            "для %s считаем заново",
            candidate.configuration_hash, candidate.symbol,
            cached_id, candidate.candidate_id,
        )
        return None

    cached_json = redis_cache.get_cached_evaluation(candidate.symbol, cached_id)
    if not cached_json:
        return None

    try:
        return CandidateEvaluation(**json.loads(cached_json))
    except (ValueError, TypeError):
        logger.warning(
            "Кэш оценки %s повреждён — считаем заново", cached_id, exc_info=True
        )
        return None


def _persist(candidate: Candidate, evaluation: CandidateEvaluation) -> dict[str, bool | None]:
    """
    Сохраняет результат в PostgreSQL, Neo4j и Redis.

    Недоступность хранилища не ломает pipeline, но обязательно попадает в лог.
    Возвращает статус по каждому хранилищу: True — записано, False — сбой,
    None — отключено через USE_DB / USE_GRAPH / USE_REDIS.
    """
    status: dict[str, bool | None] = {"db": None, "graph": None, "redis": None}

    if _USE_DB:
        try:
            from src.db.connection import get_session
            from src.db import candidate_repo
            with get_session() as session:
                candidate_repo.save_evaluation(session, candidate, evaluation)
                candidate_repo.log_event(session, candidate, evaluation)
            status["db"] = True
        except Exception:
            status["db"] = False
            logger.exception(
                "Не удалось сохранить оценку %s в PostgreSQL", candidate.candidate_id
            )

    if _USE_GRAPH:
        try:
            from src.db import graph_repo
            status["graph"] = graph_repo.upsert_from_candidate(candidate, evaluation)
        except Exception:
            status["graph"] = False
            logger.exception(
                "Не удалось обновить граф Neo4j для кандидата %s", candidate.candidate_id
            )

    if _USE_REDIS:
        try:
            from src.cache import redis_cache
            eval_json = evaluation.model_dump_json()
            ok = redis_cache.cache_evaluation(
                candidate.symbol, candidate.candidate_id, eval_json
            )
            if candidate.configuration_hash:
                ok = redis_cache.mark_hash_cached(
                    candidate.symbol, candidate.configuration_hash, candidate.candidate_id
                ) and ok
            if evaluation.rating == "STRONG":
                ok = redis_cache.publish_strong_candidate(evaluation.model_dump()) and ok
            status["redis"] = ok
        except Exception:
            status["redis"] = False
            logger.exception(
                "Не удалось записать оценку %s в Redis", candidate.candidate_id
            )

    if False in status.values():
        logger.warning(
            "Кандидат %s сохранён частично: %s", candidate.candidate_id, status
        )

    return status
