"""
CRUD операции для таблицы candidates.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from src.db.orm_models import CandidateRecord
from src.db.embedding import build_embedding
from src.models.candidate import Candidate, CandidateEvaluation


def save_evaluation(
    session: Session,
    candidate: Candidate,
    evaluation: CandidateEvaluation,
) -> CandidateRecord:
    """Сохраняет или обновляет результат оценки кандидата."""
    embedding = build_embedding(candidate)

    record = session.get(CandidateRecord, candidate.candidate_id)
    if record is None:
        record = CandidateRecord(candidate_id=candidate.candidate_id)
        session.add(record)

    record.symbol = candidate.symbol
    record.configuration_hash = candidate.configuration_hash
    record.family_key = candidate.candidate_family_key
    record.research_score = candidate.research_score
    record.transition_id = candidate.transition_id
    record.context_status = candidate.context_status.value
    record.trajectory_entropy = candidate.trajectory_entropy.value
    record.transition_rarity = candidate.transition_rarity.value
    record.current_group_id = candidate.current_group_id
    record.previous_group_id = candidate.previous_group_id
    record.current_group_age_bucket = candidate.current_group_age_bucket.value
    record.event_block_id = candidate.event_block_id
    record.event_rarity_bucket = candidate.event_rarity_bucket.value
    record.event_intensity_bucket = candidate.event_intensity_bucket.value
    record.horizon = candidate.horizon
    record.sample_size = candidate.sample_size
    record.valid_label_pct = candidate.valid_label_pct
    record.repeatability_months = candidate.repeatability_months
    record.monthly_concentration = candidate.monthly_concentration
    record.long_outcome_share = candidate.long_outcome_share
    record.outcome_skew = candidate.historical_outcome_skew
    record.fa_ratio = candidate.long_favorable_adverse_ratio_p70_p80
    record.quality_score = evaluation.quality_score
    record.rating = evaluation.rating
    record.direction = evaluation.direction
    record.warning_flags = evaluation.warning_flags
    record.raw_payload = candidate.model_dump()
    record.embedding = embedding

    return record


def get_by_id(session: Session, candidate_id: str) -> CandidateRecord | None:
    return session.get(CandidateRecord, candidate_id)


def is_hash_evaluated(session: Session, configuration_hash: str) -> str | None:
    """Возвращает candidate_id если хэш уже оценивался, иначе None."""
    record = (
        session.query(CandidateRecord)
        .filter(CandidateRecord.configuration_hash == configuration_hash)
        .first()
    )
    return record.candidate_id if record else None


def find_similar(
    session: Session,
    embedding: list[float],
    limit: int = 10,
) -> list[CandidateRecord]:
    """Поиск ближайших кандидатов по векторному расстоянию (cosine)."""
    return (
        session.query(CandidateRecord)
        .order_by(CandidateRecord.embedding.cosine_distance(embedding))
        .limit(limit)
        .all()
    )


def log_event(
    session: Session,
    candidate: Candidate,
    evaluation: CandidateEvaluation,
) -> None:
    """Записывает событие оценки в hypertable candidate_events."""
    from datetime import datetime, timezone
    from src.db.orm_models import CandidateEventRecord

    event = CandidateEventRecord(
        event_time=datetime.now(timezone.utc),
        candidate_id=candidate.candidate_id,
        symbol=candidate.symbol,
        transition_id=candidate.transition_id,
        current_group_id=candidate.current_group_id,
        quality_score=evaluation.quality_score,
        rating=evaluation.rating,
        direction=evaluation.direction,
        win_rate=evaluation.win_rate,
        fa_ratio=evaluation.favorable_adverse_ratio,
        context_status=candidate.context_status.value,
    )
    session.add(event)


def get_strong_candidates(session: Session, direction: str | None = None, limit: int = 20) -> list[CandidateRecord]:
    q = session.query(CandidateRecord).filter(CandidateRecord.rating == "STRONG")
    if direction:
        q = q.filter(CandidateRecord.direction == direction)
    return q.order_by(CandidateRecord.evaluated_at.desc()).limit(limit).all()
