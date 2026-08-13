"""Initial schema: candidates, market_events, extensions, indexes.

Revision ID: 001
Revises:
Create Date: 2026-05-30
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Расширения
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")

    # Таблица candidates
    op.create_table(
        "candidates",
        sa.Column("candidate_id", sa.String, primary_key=True),
        sa.Column("symbol", sa.String, nullable=False, server_default="BTCUSDT"),
        sa.Column("configuration_hash", sa.String),
        sa.Column("family_key", sa.String),
        sa.Column("research_score", sa.Float),
        sa.Column("transition_id", sa.String),
        sa.Column("context_status", sa.String),
        sa.Column("trajectory_entropy", sa.String),
        sa.Column("transition_rarity", sa.String),
        sa.Column("current_group_id", sa.Float),
        sa.Column("previous_group_id", sa.Float),
        sa.Column("current_group_age_bucket", sa.String),
        sa.Column("event_block_id", sa.String),
        sa.Column("event_rarity_bucket", sa.String),
        sa.Column("event_intensity_bucket", sa.String),
        sa.Column("horizon", sa.String),
        sa.Column("sample_size", sa.Integer),
        sa.Column("valid_label_pct", sa.Float),
        sa.Column("repeatability_months", sa.Integer),
        sa.Column("monthly_concentration", sa.Float),
        sa.Column("long_outcome_share", sa.Float),
        sa.Column("outcome_skew", sa.Float),
        sa.Column("fa_ratio", sa.Float),
        sa.Column("quality_score", sa.Float),
        sa.Column("rating", sa.String),
        sa.Column("direction", sa.String),
        sa.Column("warning_flags", postgresql.ARRAY(sa.Text)),
        sa.Column("raw_payload", postgresql.JSONB),
        sa.Column("embedding", sa.Text),  # pgvector — добавляем как text, меняем ниже
        sa.Column(
            "evaluated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
        ),
    )

    # Меняем тип embedding на vector(384)
    op.execute("ALTER TABLE candidates ALTER COLUMN embedding TYPE vector(384) USING NULL")

    # Индексы candidates
    op.create_index("ix_candidates_family_key", "candidates", ["family_key"])
    op.create_index("ix_candidates_transition_id", "candidates", ["transition_id"])
    op.create_index("ix_candidates_rating_direction", "candidates", ["rating", "direction"])
    op.create_index("ix_candidates_evaluated_at", "candidates", ["evaluated_at"])
    op.create_index("ix_candidates_configuration_hash", "candidates", ["configuration_hash"])

    # HNSW индекс для pgvector (быстрый ANN поиск)
    op.execute(
        "CREATE INDEX ix_candidates_embedding ON candidates "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    # Таблица market_events (TimescaleDB hypertable)
    op.create_table(
        "market_events",
        sa.Column("ts", sa.DateTime(timezone=True), primary_key=True, nullable=False),
        sa.Column("symbol", sa.String, primary_key=True, nullable=False),
        sa.Column("event_family", sa.String),
        sa.Column("event_block_id", sa.String),
        sa.Column("group_id", sa.Float),
        sa.Column("payload", postgresql.JSONB),
    )

    op.create_index("ix_market_events_symbol_ts", "market_events", ["symbol", sa.text("ts DESC")])
    op.create_index("ix_market_events_event_block_id", "market_events", ["event_block_id"])

    # Конвертируем в TimescaleDB hypertable
    op.execute("SELECT create_hypertable('market_events', 'ts', if_not_exists => TRUE)")


def downgrade() -> None:
    op.drop_table("market_events")
    op.drop_table("candidates")
