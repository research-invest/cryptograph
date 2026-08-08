from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class AgeBucket(str, Enum):
    age_lt_30 = "age_lt_30"
    age_30_60 = "age_30_60"
    age_60_120 = "age_60_120"
    age_gt_120 = "age_gt_120"


class ContextStatus(str, Enum):
    fresh = "fresh"
    stale = "stale"


class TrajectoryEntropy(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class TransitionRarity(str, Enum):
    rare = "rare"
    uncommon = "uncommon"
    common = "common"


class EventIntensityBucket(str, Enum):
    sparse = "sparse"
    moderate = "moderate"
    dense = "dense"


class EventRarityBucket(str, Enum):
    common = "common"
    uncommon = "uncommon"
    rare = "rare"


class HistoricalBiasContext(str, Enum):
    long_skew = "long_skew"
    short_skew = "short_skew"
    neutral = "neutral"


class ResearchSide(str, Enum):
    long = "long"
    short = "short"


class Candidate(BaseModel):
    # Identity
    candidate_id: str
    symbol: str = "BTCUSDT"
    configuration_hash: Optional[str] = None
    candidate_family_key: Optional[str] = None
    research_score: float = Field(ge=0.0, le=1.0)

    # State / Trajectory Context
    previous_group_id: Optional[float] = None
    current_group_id: float
    transition_id: str
    current_group_age_bucket: AgeBucket
    context_status: ContextStatus
    trajectory_entropy: TrajectoryEntropy
    transition_rarity: TransitionRarity

    # Event Context
    event_block_id: str
    primary_event_family: Optional[str] = None
    event_intensity_bucket: EventIntensityBucket
    event_rarity_bucket: EventRarityBucket
    signature_atom_count: int = Field(ge=0)
    event_family_count: int = Field(ge=0)
    event_block_total_rows: int = Field(ge=0)
    event_block_row_share: float = Field(ge=0.0, le=1.0)

    # Historical Sample
    horizon: str
    sample_size: int = Field(ge=0)
    valid_label_count: int = Field(ge=0)
    invalid_label_count: int = Field(ge=0)
    valid_label_pct: float = Field(ge=0.0, le=1.0)
    repeatability_days: int = Field(ge=0)
    repeatability_months: int = Field(ge=0)
    monthly_concentration: float = Field(ge=0.0, le=1.0)

    # Outcome Profile
    historical_bias_context: HistoricalBiasContext
    research_side: ResearchSide
    long_outcome_count: int = Field(ge=0)
    short_outcome_count: int = Field(ge=0)
    long_outcome_share: float = Field(ge=0.0, le=1.0)
    historical_outcome_skew: float = Field(ge=-1.0, le=1.0)

    # Favorable / Adverse Distributions
    p70_long_favorable_pct: float
    p80_long_adverse_pct: float
    long_favorable_adverse_ratio_p70_p80: float


class CandidateEvaluation(BaseModel):
    candidate_id: str
    symbol: str = "BTCUSDT"

    # Каким профилем посчитана оценка. Без этого сохранённые quality_score
    # разных калибровок неотличимы и молча смешиваются в средних; fingerprint
    # ловит правку порогов без бампа version.
    scoring_profile: str = ""
    profile_fingerprint: str = ""

    quality_score: float = Field(ge=0.0, le=1.0)
    # Тот же кандидат по базовому профилю. Профильный quality_score сравним
    # только ВНУТРИ монеты; baseline существует для вопроса «какая монета
    # сегодня интереснее» и в rating не участвует.
    quality_score_baseline: float = Field(default=0.0, ge=0.0, le=1.0)

    rating: str  # STRONG / MODERATE / WEAK
    direction: str  # long / short
    win_rate: float
    # None для short-кандидатов: p70/p80 описывают движение вверх и к short
    # неприменимы, симметричных полей источник данных не даёт.
    favorable_adverse_ratio: Optional[float] = None
    context_freshness: str
    warning_flags: list[str]
    strengths: list[str]
    risks: list[str]
    summary: str

    # Score breakdown по осям
    score_statistical: Optional[float] = None
    score_directional: Optional[float] = None
    score_context: Optional[float] = None
    score_rarity: Optional[float] = None
