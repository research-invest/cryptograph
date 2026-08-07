"""
Запросы к TimescaleDB continuous aggregates.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import sqlalchemy as sa
from sqlalchemy.orm import Session


def get_hourly_stats(session: Session, hours: int = 24) -> list[dict]:
    """
    Почасовая статистика кандидатов за последние N часов.
    Читает из continuous aggregate hourly_candidate_stats.
    """
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows = session.execute(
        sa.text("""
            SELECT
                bucket,
                direction,
                rating,
                candidate_count,
                ROUND(avg_quality_score::numeric, 4) AS avg_quality_score,
                ROUND(avg_win_rate::numeric, 4)      AS avg_win_rate,
                ROUND(avg_fa_ratio::numeric, 4)      AS avg_fa_ratio
            FROM hourly_candidate_stats
            WHERE bucket >= :since
            ORDER BY bucket DESC, direction, rating
        """),
        {"since": since},
    ).fetchall()

    return [dict(r._mapping) for r in rows]


def get_daily_group_stats(
    session: Session,
    days: int = 7,
    group_id: float | None = None,
) -> list[dict]:
    """
    Дневная статистика по группам состояний за последние N дней.
    Читает из continuous aggregate daily_group_stats.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)
    query = """
        SELECT
            bucket,
            current_group_id,
            direction,
            event_count,
            ROUND(avg_quality_score::numeric, 4) AS avg_quality_score,
            ROUND(avg_win_rate::numeric, 4)      AS avg_win_rate,
            strong_count,
            weak_count
        FROM daily_group_stats
        WHERE bucket >= :since
    """
    params: dict = {"since": since}
    if group_id is not None:
        query += " AND current_group_id = :gid"
        params["gid"] = group_id

    query += " ORDER BY bucket DESC, event_count DESC"

    rows = session.execute(sa.text(query), params).fetchall()
    return [dict(r._mapping) for r in rows]


def get_rating_distribution(session: Session, days: int = 7) -> dict:
    """
    Распределение рейтингов STRONG / MODERATE / WEAK за последние N дней.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = session.execute(
        sa.text("""
            SELECT
                rating,
                SUM(candidate_count) AS total
            FROM hourly_candidate_stats
            WHERE bucket >= :since
            GROUP BY rating
            ORDER BY total DESC
        """),
        {"since": since},
    ).fetchall()

    total = sum(r.total for r in rows) or 1
    return {
        r.rating: {"count": r.total, "pct": round(r.total / total, 4)}
        for r in rows
    }


def get_recent_events(session: Session, limit: int = 50) -> list[dict]:
    """Последние N событий из hypertable candidate_events (сырые данные)."""
    rows = session.execute(
        sa.text("""
            SELECT event_time, candidate_id, direction, rating,
                   quality_score, win_rate, fa_ratio, transition_id, context_status
            FROM candidate_events
            ORDER BY event_time DESC
            LIMIT :limit
        """),
        {"limit": limit},
    ).fetchall()
    return [dict(r._mapping) for r in rows]
