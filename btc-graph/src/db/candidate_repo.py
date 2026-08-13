"""
CRUD операции для таблицы candidates.

Все выборки принимают symbol. Без него выдача перемешивает монеты: «последние
STRONG long» без указания инструмента — это не запрос, а случайный срез.
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

    record = session.get(CandidateRecord, (candidate.symbol, candidate.candidate_id))
    if record is None:
        record = CandidateRecord(
            symbol=candidate.symbol, candidate_id=candidate.candidate_id
        )
        session.add(record)

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
    record.quality_score_baseline = evaluation.quality_score_baseline
    record.scoring_profile = evaluation.scoring_profile
    record.profile_fingerprint = evaluation.profile_fingerprint
    record.rating = evaluation.rating
    record.direction = evaluation.direction
    record.warning_flags = evaluation.warning_flags
    record.raw_payload = candidate.model_dump()
    record.embedding = embedding

    return record


def get_by_id(session: Session, symbol: str, candidate_id: str) -> CandidateRecord | None:
    return session.get(CandidateRecord, (symbol, candidate_id))


def is_hash_evaluated(
    session: Session, symbol: str, configuration_hash: str
) -> str | None:
    """Возвращает candidate_id если хэш этой монеты уже оценивался, иначе None."""
    record = (
        session.query(CandidateRecord)
        .filter(
            CandidateRecord.symbol == symbol,
            CandidateRecord.configuration_hash == configuration_hash,
        )
        .first()
    )
    return record.candidate_id if record else None


def find_similar(
    session: Session,
    embedding: list[float],
    symbol: str | None = None,
    limit: int = 10,
) -> list[CandidateRecord]:
    """
    Поиск ближайших кандидатов по векторному расстоянию (cosine).

    Вектор намеренно остаётся symbol-agnostic: это позволяет спросить
    «а у BTC такая конфигурация уже была?» по кандидату ETH. Изоляция —
    фильтр в SQL, а не свойство вектора; symbol=None ищет по всем монетам.
    """
    query = session.query(CandidateRecord)
    if symbol is not None:
        query = query.filter(CandidateRecord.symbol == symbol)
    return (
        query
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
        symbol=candidate.symbol,
        candidate_id=candidate.candidate_id,
        transition_id=candidate.transition_id,
        current_group_id=candidate.current_group_id,
        quality_score=evaluation.quality_score,
        quality_score_baseline=evaluation.quality_score_baseline,
        scoring_profile=evaluation.scoring_profile,
        rating=evaluation.rating,
        direction=evaluation.direction,
        win_rate=evaluation.win_rate,
        fa_ratio=evaluation.favorable_adverse_ratio,
        context_status=candidate.context_status.value,
    )
    session.add(event)


def get_strong_candidates(
    session: Session,
    symbol: str | None = None,
    direction: str | None = None,
    limit: int = 20,
) -> list[CandidateRecord]:
    """
    Последние STRONG-кандидаты. symbol=None — по всем монетам сразу.

    Смешанная выдача осмысленна только вместе с quality_score_baseline:
    профильные quality_score разных монет между собой не сравнимы.
    """
    q = session.query(CandidateRecord).filter(CandidateRecord.rating == "STRONG")
    if symbol:
        q = q.filter(CandidateRecord.symbol == symbol)
    if direction:
        q = q.filter(CandidateRecord.direction == direction)
    return q.order_by(CandidateRecord.evaluated_at.desc()).limit(limit).all()


def list_symbols(session: Session) -> list[str]:
    """Монеты, по которым есть сохранённые оценки."""
    rows = (
        session.query(CandidateRecord.symbol)
        .distinct()
        .order_by(CandidateRecord.symbol)
        .all()
    )
    return [row[0] for row in rows if row[0]]
