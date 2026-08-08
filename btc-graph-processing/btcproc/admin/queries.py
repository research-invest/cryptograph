"""
Запросы для админки. Здесь только чтение — ничего не считает и не пишет.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from btcproc import config
from btcproc.db import runs as runs_repo
from btcproc.db.session import fetch_all, fetch_one


def overview() -> dict:
    """Сводка для дашборда."""
    coverage = fetch_all(
        "SELECT tf, count(*) AS bars, min(ts) AS first_ts, max(ts) AS last_ts "
        "FROM ohlcv WHERE symbol = %s GROUP BY tf ORDER BY tf",
        (config.data.symbol,),
    )
    last_train = runs_repo.latest_completed_run("train")
    run_id = last_train["run_id"] if last_train else None

    totals = fetch_one(
        "SELECT count(*) AS candidates, "
        "count(*) FILTER (WHERE emitted_at IS NOT NULL) AS emitted, "
        "count(*) FILTER (WHERE rating = 'STRONG') AS strong, "
        "count(*) FILTER (WHERE rating = 'MODERATE') AS moderate, "
        "count(*) FILTER (WHERE rating = 'WEAK') AS weak, "
        "avg(quality_score) AS avg_quality "
        "FROM candidates" + (" WHERE run_id = %s" if run_id else ""),
        (run_id,) if run_id else None,
    ) or {}

    graph_size = fetch_one(
        "SELECT (SELECT count(*) FROM market_groups WHERE run_id = %s) AS groups, "
        "(SELECT count(*) FROM transitions WHERE run_id = %s) AS transitions",
        (run_id, run_id),
    ) if run_id else {}

    return {
        "symbol": config.data.symbol,
        "base_tf": config.data.base_tf,
        "horizon": config.data.horizon,
        "coverage": coverage,
        "last_train": last_train,
        "totals": totals,
        "graph": graph_size or {},
        "active_run": runs_repo.active_run(),
    }


def rating_distribution(run_id: int | None = None) -> list[dict]:
    sql = (
        "SELECT rating, direction, count(*) AS n, avg(quality_score) AS avg_quality "
        "FROM candidates WHERE rating IS NOT NULL"
    )
    params: list[Any] = []
    if run_id:
        sql += " AND run_id = %s"
        params.append(run_id)
    sql += " GROUP BY rating, direction ORDER BY rating, direction"
    return fetch_all(sql, params)


def graph_payload(run_id: int, min_count: int = 1, rarity: str | None = None) -> dict:
    """Узлы и рёбра графа состояний в формате Cytoscape."""
    groups = fetch_all(
        "SELECT group_id, size AS count, share, dominant_bias, up_share, avg_ret_pct, "
        "top_features FROM market_groups WHERE run_id = %s ORDER BY group_id",
        (run_id,),
    )
    sql = "SELECT * FROM transitions WHERE run_id = %s AND count >= %s"
    params: list[Any] = [run_id, min_count]
    if rarity:
        sql += " AND rarity = ANY(%s)"
        params.append([r.strip() for r in rarity.split(",") if r.strip()])
    sql += " ORDER BY count DESC"
    transitions = fetch_all(sql, params)

    if not groups:
        return {"nodes": [], "edges": []}

    from btcproc.states.graph import to_cytoscape

    return to_cytoscape(pd.DataFrame(groups), pd.DataFrame(transitions))


def chart_data(run_id: int, start: str | None = None, end: str | None = None,
               limit: int = 1500) -> dict:
    """
    Свечи с раскраской по состоянию + маркеры кандидатов.

    Раскраска — главный смысл этой страницы: видно, где именно граф считает,
    что рынок сменил состояние, и совпадает ли это с тем, что видит глаз.
    """
    sql = (
        "SELECT o.ts, o.open, o.high, o.low, o.close, s.group_id, s.is_transition, "
        "s.transition_id, s.age_bucket, s.entropy "
        "FROM ohlcv o LEFT JOIN bar_states s "
        "  ON s.symbol = o.symbol AND s.ts = o.ts AND s.run_id = %s "
        "WHERE o.symbol = %s AND o.tf = %s"
    )
    params: list[Any] = [run_id, config.data.symbol, config.data.base_tf]
    if start:
        sql += " AND o.ts >= %s"
        params.append(start)
    if end:
        sql += " AND o.ts <= %s"
        params.append(end)
    sql += " ORDER BY o.ts DESC LIMIT %s"
    params.append(limit)

    bars = list(reversed(fetch_all(sql, params)))
    if not bars:
        return {"bars": [], "markers": [], "groups": []}

    first_ts, last_ts = bars[0]["ts"], bars[-1]["ts"]
    candidates = fetch_all(
        "SELECT candidate_id, ts, research_side, rating, quality_score, transition_id "
        "FROM candidates WHERE run_id = %s AND ts BETWEEN %s AND %s "
        "ORDER BY ts LIMIT 500",
        (run_id, first_ts, last_ts),
    )

    group_ids = sorted({b["group_id"] for b in bars if b["group_id"] is not None})
    palette = {gid: _color(i, len(group_ids)) for i, gid in enumerate(group_ids)}

    out_bars = []
    for b in bars:
        color = palette.get(b["group_id"], "#8892a0")
        out_bars.append({
            "time": int(b["ts"].timestamp()),
            "open": b["open"], "high": b["high"], "low": b["low"], "close": b["close"],
            "color": color, "borderColor": color, "wickColor": color,
            "group_id": b["group_id"],
            "transition_id": b["transition_id"],
        })

    markers = []
    for c in candidates:
        quality = "" if c["quality_score"] is None else f" {c['quality_score']:.2f}"
        markers.append({
            "time": int(c["ts"].timestamp()),
            "position": "belowBar" if c["research_side"] == "long" else "aboveBar",
            "color": _rating_color(c["rating"]),
            "shape": "arrowUp" if c["research_side"] == "long" else "arrowDown",
            "text": f"{c['rating'] or '—'} {c['research_side']}{quality}",
            "id": c["candidate_id"],
        })

    return {
        "bars": out_bars,
        "markers": markers,
        "groups": [{"group_id": gid, "color": palette[gid]} for gid in group_ids],
    }


def _color(index: int, total: int) -> str:
    """Равномерно разнесённые оттенки — соседние состояния не сливаются."""
    hue = int(360 * index / max(total, 1))
    return f"hsl({hue}, 62%, 55%)"


def _rating_color(rating: str | None) -> str:
    return {"STRONG": "#16a34a", "MODERATE": "#d97706", "WEAK": "#94a3b8"}.get(
        rating or "", "#64748b"
    )


def candidates_page(
    run_id: int | None = None,
    rating: str | None = None,
    direction: str | None = None,
    min_quality: float | None = None,
    transition: str | None = None,
    emitted: str | None = None,
    page: int = 1,
    per_page: int = 50,
) -> dict:
    where, params = [], []
    if run_id:
        where.append("run_id = %s")
        params.append(run_id)
    if rating:
        where.append("rating = %s")
        params.append(rating)
    if direction:
        where.append("research_side = %s")
        params.append(direction)
    if min_quality is not None:
        where.append("quality_score >= %s")
        params.append(min_quality)
    if transition:
        where.append("transition_id = %s")
        params.append(transition)
    if emitted == "yes":
        where.append("emitted_at IS NOT NULL")
    elif emitted == "no":
        where.append("emitted_at IS NULL")

    clause = f" WHERE {' AND '.join(where)}" if where else ""
    total = (fetch_one(f"SELECT count(*) AS n FROM candidates{clause}", params) or {}).get("n", 0)

    offset = max(page - 1, 0) * per_page
    rows = fetch_all(
        "SELECT candidate_id, ts, transition_id, event_block_id, research_side, "
        "research_score, sample_size, quality_score, rating, warning_flags, "
        "emitted_at, emit_error FROM candidates"
        + clause
        + " ORDER BY ts DESC, quality_score DESC NULLS LAST LIMIT %s OFFSET %s",
        params + [per_page, offset],
    )
    return {
        "rows": rows,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": max(1, (total + per_page - 1) // per_page),
    }


def candidate_detail(candidate_id: str) -> dict | None:
    return fetch_one("SELECT * FROM candidates WHERE candidate_id = %s", (candidate_id,))


def transitions_table(run_id: int, limit: int = 200) -> list[dict]:
    return fetch_all(
        "SELECT * FROM transitions WHERE run_id = %s ORDER BY count DESC LIMIT %s",
        (run_id, limit),
    )


def group_detail(run_id: int, group_id: float) -> dict | None:
    node = fetch_one(
        "SELECT * FROM market_groups WHERE run_id = %s AND group_id = %s",
        (run_id, group_id),
    )
    if not node:
        return None
    node["incoming"] = fetch_all(
        "SELECT * FROM transitions WHERE run_id = %s AND cur_group_id = %s "
        "ORDER BY count DESC LIMIT 20",
        (run_id, group_id),
    )
    node["outgoing"] = fetch_all(
        "SELECT * FROM transitions WHERE run_id = %s AND prev_group_id = %s "
        "ORDER BY count DESC LIMIT 20",
        (run_id, group_id),
    )
    return node
