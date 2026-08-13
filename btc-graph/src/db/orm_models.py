"""
SQLAlchemy ORM модели: candidates и market_events.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from pgvector.sqlalchemy import Vector

from src.db.connection import Base
from src.db.embedding import VECTOR_DIM


class CandidateRecord(Base):
    """
    Оценённые кандидаты. Ключ составной: (symbol, candidate_id).

    Генератор строит candidate_id как хэш от (symbol, ts, переход, блок,
    смещение), то есть глобально уникальным он уже является. Символ в ключе
    нужен не от коллизий, а ради локальности индексов — все выборки по
    кандидатам идут «в пределах монеты» — и как страховка от смены схемы
    идентификатора на стороне генератора.
    """
    __tablename__ = "candidates"

    symbol = Column(String, primary_key=True, nullable=False)
    candidate_id = Column(String, primary_key=True)
    configuration_hash = Column(String, index=True)
    family_key = Column(String, index=True)

    research_score = Column(Float)
    transition_id = Column(String, index=True)
    context_status = Column(String)
    trajectory_entropy = Column(String)
    transition_rarity = Column(String)
    current_group_id = Column(Float)
    previous_group_id = Column(Float)
    current_group_age_bucket = Column(String)
    event_block_id = Column(String)
    event_rarity_bucket = Column(String)
    event_intensity_bucket = Column(String)

    horizon = Column(String)
    sample_size = Column(Integer)
    valid_label_pct = Column(Float)
    repeatability_months = Column(Integer)
    monthly_concentration = Column(Float)

    long_outcome_share = Column(Float)
    outcome_skew = Column(Float)
    fa_ratio = Column(Float)

    quality_score = Column(Float, index=True)
    # Тот же кандидат по базовому профилю: профильный quality_score сравним
    # только внутри монеты, baseline — между монетами.
    quality_score_baseline = Column(Float)
    rating = Column(String, index=True)
    direction = Column(String, index=True)

    # Метка калибровки, которой посчитана строка («ETHUSDT@3» + отпечаток
    # содержимого). Без неё оценки до и после перекалибровки неотличимы.
    scoring_profile = Column(String, index=True)
    profile_fingerprint = Column(String)

    warning_flags = Column(ARRAY(Text))
    raw_payload = Column(JSONB)
    embedding = Column(Vector(VECTOR_DIM))

    evaluated_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class MarketEventRecord(Base):
    """TimescaleDB hypertable — партиционируется по ts."""
    __tablename__ = "market_events"

    ts = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    symbol = Column(String, primary_key=True, nullable=False, default="BTCUSDT")
    event_family = Column(String)
    event_block_id = Column(String, index=True)
    group_id = Column(Float)
    payload = Column(JSONB)


class CandidateEventRecord(Base):
    """
    Лог событий оценки кандидатов — TimescaleDB hypertable.
    Служит источником для continuous aggregates.
    Каждая оценка (в том числе повторная) пишется как новая строка.
    """
    __tablename__ = "candidate_events"

    event_time = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    symbol = Column(String, primary_key=True, nullable=False)
    candidate_id = Column(String, primary_key=True, nullable=False)
    transition_id = Column(String)
    current_group_id = Column(Float)
    quality_score = Column(Float)
    quality_score_baseline = Column(Float)
    rating = Column(String)
    direction = Column(String)
    win_rate = Column(Float)
    fa_ratio = Column(Float)
    context_status = Column(String)
    # Входит в GROUP BY continuous aggregate'ов: средние не должны склеивать
    # оценки, посчитанные разными калибровками.
    scoring_profile = Column(String)
