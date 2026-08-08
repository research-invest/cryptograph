"""Мультимонетность: составные ключи, профили калибровки, symbol в агрегатах.

Revision ID: 004
Revises: 003
Create Date: 2026-08-08

⚠️  ЧИТАТЬ ДО ЗАПУСКА.

Миграция ПЕРЕСОЗДАЁТ оба continuous aggregate — ALTER для них TimescaleDB не
поддерживает. Пересборка идёт из `candidate_events`, а у него retention
90 дней, поэтому **агрегаты за период старше 90 дней будут потеряны
безвозвратно**. Если история важна, сними дамп ДО миграции:

    CREATE TABLE hourly_candidate_stats_backup_003 AS
        SELECT * FROM hourly_candidate_stats;
    CREATE TABLE daily_group_stats_backup_003 AS
        SELECT * FROM daily_group_stats;

Что делает миграция:
  1. новые колонки: scoring_profile, profile_fingerprint, quality_score_baseline;
  2. бэкфилл существующих строк (symbol='BTCUSDT', scoring_profile='legacy@0');
  3. составные первичные ключи (symbol, candidate_id) и
     (event_time, symbol, candidate_id);
  4. составные индексы под запросы «в пределах монеты»;
  5. пересоздание CAGG с symbol и scoring_profile в GROUP BY.

Смысл п. 5: без symbol в группировке средние склеивают разные рынки, без
scoring_profile — до- и после-калибровочные оценки. И то и другое портит
статистику молча.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None

# Метка для строк, посчитанных до появления профилей. Не «BTCUSDT@1»:
# та калибровка формально совпадает, но проверить это по данным нельзя,
# а смешать старые оценки с новыми под одной меткой — потерять границу.
LEGACY_PROFILE = "legacy@0"


def upgrade() -> None:
    # ── 1. Новые колонки ────────────────────────────────────────────────────
    op.add_column("candidates", sa.Column("quality_score_baseline", sa.Float))
    op.add_column("candidates", sa.Column("scoring_profile", sa.String))
    op.add_column("candidates", sa.Column("profile_fingerprint", sa.String))

    op.add_column("candidate_events", sa.Column("quality_score_baseline", sa.Float))
    op.add_column("candidate_events", sa.Column("scoring_profile", sa.String))

    # ── 2. Бэкфилл ──────────────────────────────────────────────────────────
    # До шага 4 приёмник принимал только BTCUSDT — валидатор помечал остальные
    # флагом symbol_not_btcusdt, а генератор их не выпускал.
    op.execute("UPDATE candidates SET symbol = 'BTCUSDT' WHERE symbol IS NULL")
    op.execute("UPDATE candidate_events SET symbol = 'BTCUSDT' WHERE symbol IS NULL")
    op.execute(
        f"UPDATE candidates SET scoring_profile = '{LEGACY_PROFILE}' "
        "WHERE scoring_profile IS NULL"
    )
    op.execute(
        f"UPDATE candidate_events SET scoring_profile = '{LEGACY_PROFILE}' "
        "WHERE scoring_profile IS NULL"
    )
    # baseline у старых строк равен самой оценке: считали базовой калибровкой.
    op.execute(
        "UPDATE candidates SET quality_score_baseline = quality_score "
        "WHERE quality_score_baseline IS NULL"
    )
    op.execute(
        "UPDATE candidate_events SET quality_score_baseline = quality_score "
        "WHERE quality_score_baseline IS NULL"
    )

    # ── 3. symbol становится обязательным и частью ключа ────────────────────
    # server_default снимается: молчаливый дефолт превращал потерянное поле
    # в биткоин. Теперь отсутствие символа — ошибка вставки, а не тихая правка.
    op.alter_column("candidates", "symbol", nullable=False, server_default=None)
    op.alter_column("candidate_events", "symbol", nullable=False, server_default=None)

    op.execute("ALTER TABLE candidates DROP CONSTRAINT IF EXISTS candidates_pkey")
    op.create_primary_key("candidates_pkey", "candidates", ["symbol", "candidate_id"])

    # У hypertable ключ обязан содержать колонку партиционирования (event_time).
    op.execute(
        "ALTER TABLE candidate_events DROP CONSTRAINT IF EXISTS candidate_events_pkey"
    )
    op.create_primary_key(
        "candidate_events_pkey", "candidate_events",
        ["event_time", "symbol", "candidate_id"],
    )

    # ── 4. Индексы под запросы «в пределах монеты» ──────────────────────────
    op.create_index(
        "ix_candidates_symbol_quality", "candidates",
        ["symbol", sa.text("quality_score DESC")],
    )
    op.create_index(
        "ix_candidates_symbol_rating_direction", "candidates",
        ["symbol", "rating", "direction"],
    )
    op.create_index(
        "ix_candidates_symbol_confhash", "candidates", ["symbol", "configuration_hash"]
    )
    op.create_index(
        "ix_candidates_symbol_family", "candidates", ["symbol", "family_key"]
    )
    op.create_index("ix_candidates_scoring_profile", "candidates", ["scoring_profile"])
    op.create_index(
        "ix_candidate_events_symbol_time", "candidate_events",
        ["symbol", sa.text("event_time DESC")],
    )

    # Одиночные индексы, ставшие ведущими префиксами составных, больше не нужны.
    op.execute("DROP INDEX IF EXISTS ix_candidates_configuration_hash")
    op.execute("DROP INDEX IF EXISTS ix_candidates_family_key")
    op.execute("DROP INDEX IF EXISTS ix_candidates_rating_direction")

    # ── 5. Пересоздание continuous aggregates ───────────────────────────────
    _recreate_aggregates()


def _recreate_aggregates() -> None:
    op.execute("DROP MATERIALIZED VIEW IF EXISTS daily_group_stats CASCADE")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS hourly_candidate_stats CASCADE")

    op.execute("""
        CREATE MATERIALIZED VIEW hourly_candidate_stats
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 hour', event_time)   AS bucket,
            symbol,
            scoring_profile,
            direction,
            rating,
            COUNT(*)                             AS candidate_count,
            AVG(quality_score)                   AS avg_quality_score,
            AVG(quality_score_baseline)          AS avg_quality_score_baseline,
            AVG(win_rate)                        AS avg_win_rate,
            AVG(fa_ratio)                        AS avg_fa_ratio,
            MIN(quality_score)                   AS min_quality_score,
            MAX(quality_score)                   AS max_quality_score
        FROM candidate_events
        GROUP BY bucket, symbol, scoring_profile, direction, rating
        WITH NO DATA
    """)

    op.execute("""
        SELECT add_continuous_aggregate_policy(
            'hourly_candidate_stats',
            start_offset  => INTERVAL '3 hours',
            end_offset    => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour'
        )
    """)

    op.execute("""
        CREATE MATERIALIZED VIEW daily_group_stats
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 day', event_time)    AS bucket,
            symbol,
            scoring_profile,
            current_group_id,
            direction,
            COUNT(*)                             AS event_count,
            AVG(quality_score)                   AS avg_quality_score,
            AVG(quality_score_baseline)          AS avg_quality_score_baseline,
            AVG(win_rate)                        AS avg_win_rate,
            SUM(CASE WHEN rating = 'STRONG' THEN 1 ELSE 0 END) AS strong_count,
            SUM(CASE WHEN rating = 'WEAK'   THEN 1 ELSE 0 END) AS weak_count
        FROM candidate_events
        GROUP BY bucket, symbol, scoring_profile, current_group_id, direction
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

    # Первичное наполнение из уцелевших сырых событий (retention 90 дней).
    op.execute("COMMIT")
    op.execute("CALL refresh_continuous_aggregate('hourly_candidate_stats', NULL, NULL)")
    op.execute("CALL refresh_continuous_aggregate('daily_group_stats', NULL, NULL)")


def downgrade() -> None:
    op.execute("DROP MATERIALIZED VIEW IF EXISTS daily_group_stats CASCADE")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS hourly_candidate_stats CASCADE")

    op.execute("DROP INDEX IF EXISTS ix_candidate_events_symbol_time")
    op.execute("DROP INDEX IF EXISTS ix_candidates_scoring_profile")
    op.execute("DROP INDEX IF EXISTS ix_candidates_symbol_family")
    op.execute("DROP INDEX IF EXISTS ix_candidates_symbol_confhash")
    op.execute("DROP INDEX IF EXISTS ix_candidates_symbol_rating_direction")
    op.execute("DROP INDEX IF EXISTS ix_candidates_symbol_quality")

    op.create_index("ix_candidates_configuration_hash", "candidates", ["configuration_hash"])
    op.create_index("ix_candidates_family_key", "candidates", ["family_key"])
    op.create_index("ix_candidates_rating_direction", "candidates", ["rating", "direction"])

    # Откат ключей возможен только если в таблице одна монета: иначе
    # candidate_id перестанет быть уникальным. Падение здесь — правильное
    # поведение, а не помеха: молча удалять чужие строки нельзя.
    op.execute("ALTER TABLE candidates DROP CONSTRAINT IF EXISTS candidates_pkey")
    op.create_primary_key("candidates_pkey", "candidates", ["candidate_id"])
    op.execute(
        "ALTER TABLE candidate_events DROP CONSTRAINT IF EXISTS candidate_events_pkey"
    )
    op.create_primary_key(
        "candidate_events_pkey", "candidate_events", ["event_time", "candidate_id"]
    )

    op.alter_column("candidates", "symbol", server_default="BTCUSDT")
    op.alter_column("candidate_events", "symbol", server_default="BTCUSDT")

    op.drop_column("candidate_events", "scoring_profile")
    op.drop_column("candidate_events", "quality_score_baseline")
    op.drop_column("candidates", "profile_fingerprint")
    op.drop_column("candidates", "scoring_profile")
    op.drop_column("candidates", "quality_score_baseline")

    # Восстанавливаем агрегаты в форме ревизии 002.
    op.execute("""
        CREATE MATERIALIZED VIEW hourly_candidate_stats
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 hour', event_time)   AS bucket,
            direction, rating,
            COUNT(*) AS candidate_count,
            AVG(quality_score) AS avg_quality_score,
            AVG(win_rate) AS avg_win_rate,
            AVG(fa_ratio) AS avg_fa_ratio,
            MIN(quality_score) AS min_quality_score,
            MAX(quality_score) AS max_quality_score
        FROM candidate_events
        GROUP BY bucket, direction, rating
        WITH NO DATA
    """)
    op.execute("""
        CREATE MATERIALIZED VIEW daily_group_stats
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 day', event_time) AS bucket,
            current_group_id, direction,
            COUNT(*) AS event_count,
            AVG(quality_score) AS avg_quality_score,
            AVG(win_rate) AS avg_win_rate,
            SUM(CASE WHEN rating = 'STRONG' THEN 1 ELSE 0 END) AS strong_count,
            SUM(CASE WHEN rating = 'WEAK'   THEN 1 ELSE 0 END) AS weak_count
        FROM candidate_events
        GROUP BY bucket, current_group_id, direction
        WITH NO DATA
    """)
