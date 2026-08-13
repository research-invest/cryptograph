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


class SampleScope(str, Enum):
    """
    На что обусловлена историческая выборка кандидата.

    `transition_event_block` — статистика собрана по паре (переход × блок
    событий), то есть ровно по той конфигурации, которую кандидат описывает.

    `transition` — генератор откатился на выборку по переходу целиком: пары
    не набралось до его порога. Поля блока событий у такого кандидата
    заполнены (они описывают текущий бар), но выборка о блоке не знает
    ничего, и начислять баллы за «редкий блок» по ней нельзя.
    По выгрузкам генератора это большинство кандидатов: 68–81% на BTC/ETH/SOL
    и 100% на HYPEUSDT.
    """

    transition_event_block = "transition+event_block"
    transition = "transition"


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
    # Число НЕЗАВИСИМЫХ реализаций перехода в выборке.
    #
    # `sample_size` считает строки снимков: генератор берёт снимок конфигурации
    # в момент перехода и спустя 45/90/180 минут, то есть до четырёх строк на
    # один случай, и окна их исходов при горизонте 24h совпадают на 87.5%.
    # «Выборка 1000» на деле означала около 250 случаев.
    #
    # Optional с дефолтом None — старые выгрузки поля не несут. Ступень
    # statistical.sample_size переводить на него можно только вместе с
    # перекалибровкой профилей: пороги вида 2342 откалиброваны на строках.
    effective_sample_size: Optional[int] = Field(default=None, ge=0)
    sample_scope: Optional[SampleScope] = None
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
    # Зеркало long-версии для случаев падения. Optional, потому что кандидаты,
    # выпущенные до 2026-08-13, поля не несут; None означает «не знаем», и ось
    # directional тогда считается по двум критериям, как и раньше. Пустым оно
    # приходит и у конфигураций, где падений в выборке не было вовсе.
    short_favorable_adverse_ratio_p70_p80: Optional[float] = None


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
