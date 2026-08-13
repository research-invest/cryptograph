"""candidate_events hypertable + TimescaleDB continuous aggregates.

Revision ID: 002
Revises: 001
Create Date: 2026-05-30
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Hypertable: лог событий оценки ──────────────────────────────────────
    op.create_table(
        "candidate_events",
        sa.Column("event_time", sa.DateTime(timezone=True), primary_key=True, nullable=False),
        sa.Column("candidate_id", sa.String, primary_key=True, nullable=False),
        sa.Column("symbol", sa.String, server_default="BTCUSDT"),
        sa.Column("transition_id", sa.String),
        sa.Column("current_group_id", sa.Float),
        sa.Column("quality_score", sa.Float),
        sa.Column("rating", sa.String),
        sa.Column("direction", sa.String),
        sa.Column("win_rate", sa.Float),
        sa.Column("fa_ratio", sa.Float),
        sa.Column("context_status", sa.String),
    )

    op.execute("SELECT create_hypertable('candidate_events', 'event_time', if_not_exists => TRUE)")

    op.create_index("ix_candidate_events_candidate_id", "candidate_events", ["candidate_id"])
    op.create_index("ix_candidate_events_rating", "candidate_events", ["rating"])
    op.create_index("ix_candidate_events_group_id", "candidate_events", ["current_group_id"])

    # ── Continuous aggregate 1: почасовая статистика ─────────────────────────
    # Агрегирует: количество кандидатов, avg quality_score, avg win_rate по direction + rating
    op.execute("""
        CREATE MATERIALIZED VIEW hourly_candidate_stats
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 hour', event_time)   AS bucket,
            direction,
            rating,
            COUNT(*)                             AS candidate_count,
            AVG(quality_score)                   AS avg_quality_score,
            AVG(win_rate)                        AS avg_win_rate,
            AVG(fa_ratio)                        AS avg_fa_ratio,
            MIN(quality_score)                   AS min_quality_score,
            MAX(quality_score)                   AS max_quality_score
        FROM candidate_events
        GROUP BY bucket, direction, rating
        WITH NO DATA
    """)

    # Политика авто-обновления: обновлять каждый час, покрывать последние 3 часа
    op.execute("""
        SELECT add_continuous_aggregate_policy(
            'hourly_candidate_stats',
            start_offset  => INTERVAL '3 hours',
            end_offset    => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour'
        )
    """)

    # ── Continuous aggregate 2: дневная статистика по группам ────────────────
    # Агрегирует: кол-во событий + avg quality_score по group_id за каждый день
    op.execute("""
        CREATE MATERIALIZED VIEW daily_group_stats
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 day', event_time)    AS bucket,
            current_group_id,
            direction,
            COUNT(*)                             AS event_count,
            AVG(quality_score)                   AS avg_quality_score,
            AVG(win_rate)                        AS avg_win_rate,
            SUM(CASE WHEN rating = 'STRONG' THEN 1 ELSE 0 END) AS strong_count,
            SUM(CASE WHEN rating = 'WEAK'   THEN 1 ELSE 0 END) AS weak_count
        FROM candidate_events
        GROUP BY bucket, current_group_id, direction
        WITH NO DATA
    """)

    op.execute("""
        SELECT add_continuous_aggregate_policy(
            'daily_group_stats',
            start_offset  => INTERVAL '7 days',
            end_offset    => INTERVAL '1 day',
            schedule_interval => INTERVAL '1 day'
        )
    """)

    # ── Политика retention: хранить сырые события 90 дней ───────────────────
    op.execute("""
        SELECT add_retention_policy('candidate_events', INTERVAL '90 days')
    """)


def downgrade() -> None:
    op.execute("DROP MATERIALIZED VIEW IF EXISTS daily_group_stats CASCADE")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS hourly_candidate_stats CASCADE")
    op.drop_table("candidate_events")
