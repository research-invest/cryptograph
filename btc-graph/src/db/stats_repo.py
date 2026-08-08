"""
Запросы к TimescaleDB continuous aggregates.

symbol и scoring_profile входят в GROUP BY агрегатов (миграция 004), поэтому
любая сводка возвращает их явно: среднее «по всем монетам и всем калибровкам»
— это число, которое ничего не значит, и оно не должно получаться случайно.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import sqlalchemy as sa
from sqlalchemy.orm import Session


def get_hourly_stats(
    session: Session, hours: int = 24, symbol: str | None = None
) -> list[dict]:
    """
    Почасовая статистика кандидатов за последние N часов.
    Читает из continuous aggregate hourly_candidate_stats.
    """
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    query = """
        SELECT
            bucket,
            symbol,
            scoring_profile,
            direction,
            rating,
            candidate_count,
            ROUND(avg_quality_score::numeric, 4)          AS avg_quality_score,
            ROUND(avg_quality_score_baseline::numeric, 4) AS avg_quality_score_baseline,
            ROUND(avg_win_rate::numeric, 4)               AS avg_win_rate,
            ROUND(avg_fa_ratio::numeric, 4)               AS avg_fa_ratio
        FROM hourly_candidate_stats
        WHERE bucket >= :since
    """
    params: dict = {"since": since}
    if symbol:
        query += " AND symbol = :symbol"
        params["symbol"] = symbol

    query += " ORDER BY bucket DESC, symbol, direction, rating"
    rows = session.execute(sa.text(query), params).fetchall()
    return [dict(r._mapping) for r in rows]


def get_daily_group_stats(
    session: Session,
    days: int = 7,
    symbol: str | None = None,
    group_id: float | None = None,
) -> list[dict]:
    """
    Дневная статистика по группам состояний за последние N дней.
    Читает из continuous aggregate daily_group_stats.

    group_id без symbol почти наверняка ошибка: номера групп у монет свои,
    и «группа 7» смешает разные рынки. Поэтому фильтр по группе требует монеты.
    """
    if group_id is not None and not symbol:
        raise ValueError(
            "Фильтр по group_id требует symbol: номера групп у монет свои "
            "и между инструментами несопоставимы."
        )

    since = datetime.now(timezone.utc) - timedelta(days=days)
    query = """
        SELECT
            bucket,
            symbol,
            scoring_profile,
            current_group_id,
            direction,
            event_count,
            ROUND(avg_quality_score::numeric, 4)          AS avg_quality_score,
            ROUND(avg_quality_score_baseline::numeric, 4) AS avg_quality_score_baseline,
            ROUND(avg_win_rate::numeric, 4)               AS avg_win_rate,
            strong_count,
            weak_count
        FROM daily_group_stats
        WHERE bucket >= :since
    """
    params: dict = {"since": since}
    if symbol:
        query += " AND symbol = :symbol"
        params["symbol"] = symbol
    if group_id is not None:
        query += " AND current_group_id = :gid"
        params["gid"] = group_id

    query += " ORDER BY bucket DESC, event_count DESC"

    rows = session.execute(sa.text(query), params).fetchall()
    return [dict(r._mapping) for r in rows]


def get_rating_distribution(
    session: Session, days: int = 7, symbol: str | None = None
) -> dict:
    """
    Распределение рейтингов STRONG / MODERATE / WEAK за последние N дней.

    Без symbol доли считаются по всем монетам сразу — это осмысленно как
    «сколько сильного вообще», но не как характеристика инструмента.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)
    query = """
        SELECT rating, SUM(candidate_count) AS total
        FROM hourly_candidate_stats
        WHERE bucket >= :since
    """
    params: dict = {"since": since}
    if symbol:
        query += " AND symbol = :symbol"
        params["symbol"] = symbol
    query += " GROUP BY rating ORDER BY total DESC"

    rows = session.execute(sa.text(query), params).fetchall()

    total = sum(r.total for r in rows) or 1
    return {
        "symbol": symbol or "all",
        "distribution": {
            r.rating: {"count": r.total, "pct": round(r.total / total, 4)}
            for r in rows
        },
    }


def get_recent_events(
    session: Session, limit: int = 50, symbol: str | None = None
) -> list[dict]:
    """Последние N событий из hypertable candidate_events (сырые данные)."""
    query = """
        SELECT event_time, symbol, candidate_id, direction, rating,
               quality_score, quality_score_baseline, scoring_profile,
               win_rate, fa_ratio, transition_id, context_status
        FROM candidate_events
    """
    params: dict = {"limit": limit}
    if symbol:
        query += " WHERE symbol = :symbol"
        params["symbol"] = symbol
    query += " ORDER BY event_time DESC LIMIT :limit"

    rows = session.execute(sa.text(query), params).fetchall()
    return [dict(r._mapping) for r in rows]


def list_symbols(session: Session, days: int = 30) -> list[str]:
    """Монеты, по которым были события оценки за период."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = session.execute(
        sa.text(
            "SELECT DISTINCT symbol FROM candidate_events "
            "WHERE event_time >= :since ORDER BY symbol"
        ),
        {"since": since},
    ).fetchall()
    return [r[0] for r in rows if r[0]]
