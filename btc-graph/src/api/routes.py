"""
FastAPI routes: приём кандидатов, оценка, фильтрация, граф, поиск похожих.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

from src.agent.pipeline import run_pipeline, run_batch_pipeline
from src.filters.candidate_filter import detect_conflicts
from src.models.candidate import CandidateEvaluation
from src.parser.candidate_parser import parse_candidate, parse_candidates
from src.validator.candidate_validator import validate_candidate
from src.scorer.candidate_scorer import get_rating, score_candidate

logger = logging.getLogger(__name__)

app = FastAPI(
    title="BTC Market Candidate Agent",
    description="Агент анализа рыночных кандидатов BTC — детерминирование фазы рынка",
    version="2.0.0",
)


def _fail(exc: Exception, context: str) -> HTTPException:
    """
    Переводит исключение в корректный HTTP-код.

    422 — только когда виноват вход (не прошла валидация Pydantic или парсер
    отверг данные). Всё остальное — 500: сбой БД, недоступность внешнего API,
    баг в коде. Раньше сюда падало всё подряд с кодом 422, и отладка по коду
    ответа была невозможна — см. docs/audit_findings.md, №12.
    """
    if isinstance(exc, (ValidationError, ValueError, TypeError)):
        return HTTPException(status_code=422, detail=str(exc))
    logger.exception("%s", context)
    return HTTPException(status_code=500, detail=f"{context}: {exc}")


# ─── Input schemas ────────────────────────────────────────────────────────────

class RawTextInput(BaseModel):
    raw: str
    use_llm: bool = True
    save: bool = True


class JsonCandidateInput(BaseModel):
    candidate: dict
    use_llm: bool = True
    save: bool = True


class BatchInput(BaseModel):
    candidates: list[dict]
    use_llm: bool = True
    save: bool = True
    min_quality_score: float = 0.60


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


# ─── Evaluate ─────────────────────────────────────────────────────────────────

@app.post("/evaluate/raw", response_model=CandidateEvaluation)
def evaluate_raw(body: RawTextInput):
    """Оценить кандидата из raw text."""
    try:
        return run_pipeline(body.raw, use_llm=body.use_llm, save=body.save)
    except Exception as e:
        raise _fail(e, "Ошибка оценки кандидата из raw text")


@app.post("/evaluate/json", response_model=CandidateEvaluation)
def evaluate_json(body: JsonCandidateInput):
    """Оценить кандидата из JSON объекта."""
    try:
        return run_pipeline(body.candidate, use_llm=body.use_llm, save=body.save)
    except Exception as e:
        raise _fail(e, "Ошибка оценки кандидата из JSON")


@app.post("/evaluate/batch", response_model=list[CandidateEvaluation])
def evaluate_batch(body: BatchInput):
    """Оценить список кандидатов с фильтрацией и дедупликацией по family_key."""
    try:
        return run_batch_pipeline(
            body.candidates,
            use_llm=body.use_llm,
            min_quality_score=body.min_quality_score,
            save=body.save,
        )
    except Exception as e:
        raise _fail(e, "Ошибка пакетной оценки кандидатов")


@app.post("/score/quick")
def quick_score(body: JsonCandidateInput):
    """Быстрый score без LLM и без сохранения."""
    try:
        candidate = parse_candidate(body.candidate)
        flags = validate_candidate(candidate)
        breakdown = score_candidate(candidate)
        rating = get_rating(breakdown.total)
        return {
            "candidate_id": candidate.candidate_id,
            "quality_score": breakdown.total,
            "rating": rating,
            "score_breakdown": {
                "statistical": breakdown.statistical,
                "directional": breakdown.directional,
                "context": breakdown.context,
                "rarity": breakdown.rarity,
            },
            "warning_flags": flags,
        }
    except Exception as e:
        raise _fail(e, "Ошибка быстрой оценки кандидата")


# ─── Conflicts ────────────────────────────────────────────────────────────────

@app.post("/conflicts")
def detect(body: BatchInput):
    """Найти конфликты среди списка кандидатов."""
    try:
        candidates = parse_candidates(body.candidates)
        conflicts = detect_conflicts(candidates)
        return {
            "total_candidates": len(candidates),
            "conflicts_found": len(conflicts),
            "conflicts": [
                {"type": c.conflict_type, "candidate_ids": c.candidate_ids, "description": c.description}
                for c in conflicts
            ],
        }
    except Exception as e:
        raise _fail(e, "Ошибка поиска конфликтов")


# ─── Database queries ─────────────────────────────────────────────────────────

@app.get("/candidates/{candidate_id}")
def get_candidate(candidate_id: str):
    """Получить сохранённую оценку из PostgreSQL."""
    try:
        from src.db.connection import get_session
        from src.db import candidate_repo
        with get_session() as session:
            record = candidate_repo.get_by_id(session, candidate_id)
            if record is None:
                raise HTTPException(status_code=404, detail="Кандидат не найден")
            return {
                "candidate_id": record.candidate_id,
                "quality_score": record.quality_score,
                "rating": record.rating,
                "direction": record.direction,
                "warning_flags": record.warning_flags,
                "evaluated_at": record.evaluated_at,
            }
    except HTTPException:
        raise
    except Exception as e:
        raise _fail(e, f"Ошибка чтения кандидата {candidate_id}")


@app.get("/candidates/strong/{direction}")
def get_strong(direction: str, limit: int = 20):
    """Получить последние STRONG кандидаты по направлению (long/short)."""
    try:
        from src.db.connection import get_session
        from src.db import candidate_repo
        with get_session() as session:
            records = candidate_repo.get_strong_candidates(session, direction=direction, limit=limit)
            return [
                {
                    "candidate_id": r.candidate_id,
                    "quality_score": r.quality_score,
                    "transition_id": r.transition_id,
                    "context_status": r.context_status,
                    "evaluated_at": r.evaluated_at,
                }
                for r in records
            ]
    except HTTPException:
        raise
    except Exception as e:
        raise _fail(e, f"Ошибка выборки STRONG-кандидатов ({direction})")


@app.post("/candidates/similar")
def find_similar(body: JsonCandidateInput, limit: int = 10):
    """Найти исторически похожие кандидаты через pgvector."""
    try:
        from src.db.connection import get_session
        from src.db import candidate_repo
        from src.db.embedding import build_embedding
        candidate = parse_candidate(body.candidate)
        embedding = build_embedding(candidate)
        with get_session() as session:
            records = candidate_repo.find_similar(session, embedding, limit=limit)
            return [
                {
                    "candidate_id": r.candidate_id,
                    "quality_score": r.quality_score,
                    "rating": r.rating,
                    "direction": r.direction,
                    "transition_id": r.transition_id,
                }
                for r in records
            ]
    except Exception as e:
        raise _fail(e, "Ошибка поиска похожих кандидатов")


# ─── Graph queries ─────────────────────────────────────────────────────────────

@app.get("/graph/group/{group_id}")
def get_group(group_id: float):
    """Получить информацию об узле графа MarketGroup."""
    try:
        from src.db import graph_repo
        info = graph_repo.get_group_info(group_id)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Группа {group_id} не найдена в графе")
        return info
    except HTTPException:
        raise
    except Exception as e:
        raise _fail(e, "Ошибка чтения узла графа")


@app.get("/graph/transitions/to/{group_id}")
def get_transitions_to(group_id: float, rarity: str = "uncommon,rare"):
    """
    Найти переходы, ведущие в группу group_id.
    rarity — через запятую: uncommon,rare или common
    """
    try:
        from src.db import graph_repo
        rarity_filter = [r.strip() for r in rarity.split(",")]
        transitions = graph_repo.find_transitions_to_group(group_id, rarity_filter)
        return {"group_id": group_id, "transitions": transitions}
    except Exception as e:
        raise _fail(e, "Ошибка чтения переходов графа")


# ─── Stats (TimescaleDB continuous aggregates) ────────────────────────────────

@app.get("/stats/hourly")
def stats_hourly(hours: int = 24):
    """Почасовая статистика кандидатов из continuous aggregate."""
    try:
        from src.db.connection import get_session
        from src.db import stats_repo
        with get_session() as session:
            return stats_repo.get_hourly_stats(session, hours=hours)
    except Exception as e:
        raise _fail(e, "Ошибка чтения почасовой статистики")


@app.get("/stats/groups")
def stats_groups(days: int = 7, group_id: float | None = None):
    """Дневная статистика по группам состояний."""
    try:
        from src.db.connection import get_session
        from src.db import stats_repo
        with get_session() as session:
            return stats_repo.get_daily_group_stats(session, days=days, group_id=group_id)
    except Exception as e:
        raise _fail(e, "Ошибка чтения статистики по группам")


@app.get("/stats/ratings")
def stats_ratings(days: int = 7):
    """Распределение рейтингов STRONG / MODERATE / WEAK."""
    try:
        from src.db.connection import get_session
        from src.db import stats_repo
        with get_session() as session:
            return stats_repo.get_rating_distribution(session, days=days)
    except Exception as e:
        raise _fail(e, "Ошибка чтения распределения рейтингов")


@app.get("/stats/events")
def stats_events(limit: int = 50):
    """Последние события оценки из hypertable candidate_events."""
    try:
        from src.db.connection import get_session
        from src.db import stats_repo
        with get_session() as session:
            return stats_repo.get_recent_events(session, limit=limit)
    except Exception as e:
        raise _fail(e, "Ошибка чтения последних событий")


# ─── Stream queue ─────────────────────────────────────────────────────────────

@app.post("/queue/enqueue")
def enqueue(body: JsonCandidateInput):
    """
    Положить кандидата в Redis Stream для асинхронной обработки.

    Кандидат разбирается сразу: невалидные данные отвергаются здесь (422), а не
    оседают в очереди, где ошибка всплывёт только в логах воркера.
    """
    try:
        from src.cache import redis_cache
        parse_candidate(body.candidate)
        msg_id = redis_cache.enqueue_candidate(body.candidate)
        if not msg_id:
            # enqueue_candidate возвращает "" при недоступном Redis — раньше
            # роут всё равно отвечал queued: true (docs/audit_findings.md, №13).
            raise HTTPException(
                status_code=503, detail="Redis недоступен — кандидат не поставлен в очередь"
            )
        return {"queued": True, "msg_id": msg_id}
    except HTTPException:
        raise
    except Exception as e:
        raise _fail(e, "Ошибка постановки кандидата в очередь")
