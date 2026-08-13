"""
Генерация feature-вектора из полей кандидата для pgvector.

Вектор табличный, а не текстовый: 18 нормализованных признаков кандидата.
Размерность 32 — 18 значимых позиций плюс запас, чтобы добавление новых
признаков не требовало миграции. Раньше было 384 (366 нулей) — размерность
из мира sentence-transformers, здесь она только раздувала HNSW-индекс.

ВАЖНО: порядок признаков фиксирован. Менять его или VECTOR_DIM можно только
вместе с миграцией, пересчитывающей embedding у всех существующих строк.
"""
from __future__ import annotations

import numpy as np

from src.models.candidate import (
    Candidate,
    ContextStatus,
    TrajectoryEntropy,
    TransitionRarity,
    EventRarityBucket,
    EventIntensityBucket,
    AgeBucket,
    ResearchSide,
)

VECTOR_DIM = 32


def _rarity_to_float(rarity: TransitionRarity | EventRarityBucket) -> float:
    return {"rare": 0.0, "uncommon": 0.5, "common": 1.0}.get(rarity.value, 0.5)


def _intensity_to_float(intensity: EventIntensityBucket) -> float:
    return {"sparse": 0.0, "moderate": 0.5, "dense": 1.0}.get(intensity.value, 0.5)


def _age_to_float(bucket: AgeBucket) -> float:
    return {"age_lt_30": 0.0, "age_30_60": 0.33, "age_60_120": 0.66, "age_gt_120": 1.0}.get(bucket.value, 0.5)


def _entropy_to_float(entropy: TrajectoryEntropy) -> float:
    return {"low": 0.0, "medium": 0.5, "high": 1.0}.get(entropy.value, 0.5)


def build_embedding(candidate: Candidate) -> list[float]:
    """
    Возвращает список из VECTOR_DIM (=32) float для хранения в pgvector.
    Порядок признаков фиксирован — нельзя менять без пересчёта всех embeddings.
    Размерность когда-то была 384; её ужала миграция
    alembic/versions/003_shrink_embedding_dim.py.
    """
    features = [
        candidate.research_score,
        candidate.valid_label_pct,
        candidate.long_outcome_share,
        (candidate.historical_outcome_skew + 1.0) / 2.0,  # [-1,1] → [0,1]
        min(candidate.long_favorable_adverse_ratio_p70_p80 / 10.0, 1.0),
        candidate.monthly_concentration,
        min(candidate.repeatability_months / 24.0, 1.0),
        candidate.event_block_row_share,
        min(candidate.p70_long_favorable_pct / 10.0, 1.0),
        min(candidate.p80_long_adverse_pct / 10.0, 1.0),
        1.0 if candidate.context_status == ContextStatus.fresh else 0.0,
        _entropy_to_float(candidate.trajectory_entropy),
        _rarity_to_float(candidate.transition_rarity),
        _rarity_to_float(candidate.event_rarity_bucket),
        _intensity_to_float(candidate.event_intensity_bucket),
        _age_to_float(candidate.current_group_age_bucket),
        1.0 if candidate.research_side == ResearchSide.long else 0.0,
        {"long_skew": 1.0, "neutral": 0.5, "short_skew": 0.0}.get(candidate.historical_bias_context.value, 0.5),
    ]

    vec = np.zeros(VECTOR_DIM, dtype=np.float32)
    vec[: len(features)] = features
    return vec.tolist()
