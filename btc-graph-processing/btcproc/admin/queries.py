"""
Запросы для админки. Здесь только чтение — ничего не считает и не пишет.
"""
from __future__ import annotations

import colorsys
from typing import Any

import pandas as pd

from btcproc import config
from btcproc.db import runs as runs_repo
from btcproc.db.session import fetch_all, fetch_one

# Потолок маркеров на графике: больше пятисот стрелок всё равно сливаются в
# кашу и перекрывают раскраску состояний. Факт обрезки отдаётся наружу
# (`markers_truncated`) — молча терять кандидатов нельзя.
MARKER_LIMIT = 500


def overview(symbol: str | None = None) -> dict:
    """
    Сводка для дашборда по выбранной монете.

    symbol прокидывается аргументом, а не берётся из config: селектор в шапке
    должен переключать ВСЕ цифры страницы, а не только заголовок.
    """
    symbol = symbol or config.data.symbol
    coverage = fetch_all(
        "SELECT tf, count(*) AS bars, min(ts) AS first_ts, max(ts) AS last_ts "
        "FROM ohlcv WHERE symbol = %s GROUP BY tf ORDER BY tf",
        (symbol,),
    )
    last_train = runs_repo.latest_completed_run("train", symbol)
    run_id = last_train["run_id"] if last_train else None

    # Фильтр по монете нужен даже при фильтре по run_id: run_id уже задаёт
    # монету, но без symbol запрос без прогона (run_id=None) посчитал бы
    # кандидатов всех монет сразу.
    totals = fetch_one(
        "SELECT count(*) AS candidates, "
        "count(*) FILTER (WHERE emitted_at IS NOT NULL) AS emitted, "
        "count(*) FILTER (WHERE rating = 'STRONG') AS strong, "
        "count(*) FILTER (WHERE rating = 'MODERATE') AS moderate, "
        "count(*) FILTER (WHERE rating = 'WEAK') AS weak, "
        "avg(quality_score) AS avg_quality "
        "FROM candidates WHERE symbol = %s" + (" AND run_id = %s" if run_id else ""),
        (symbol, run_id) if run_id else (symbol,),
    ) or {}

    graph_size = fetch_one(
        "SELECT (SELECT count(*) FROM market_groups WHERE run_id = %s) AS groups, "
        "(SELECT count(*) FROM transitions WHERE run_id = %s) AS transitions",
        (run_id, run_id),
    ) if run_id else {}

    return {
        "symbol": symbol,
        "base_tf": config.data.base_tf,
        "horizon": config.data.horizon,
        "coverage": coverage,
        "last_train": last_train,
        "totals": totals,
        "graph": graph_size or {},
        "active_run": runs_repo.active_run(symbol=symbol),
        "active_runs": runs_repo.active_runs(),
    }


def rating_distribution(run_id: int | None = None, symbol: str | None = None) -> list[dict]:
    sql = (
        "SELECT rating, direction, count(*) AS n, avg(quality_score) AS avg_quality "
        "FROM candidates WHERE rating IS NOT NULL"
    )
    params: list[Any] = []
    if symbol:
        sql += " AND symbol = %s"
        params.append(symbol)
    if run_id:
        sql += " AND run_id = %s"
        params.append(run_id)
    sql += " GROUP BY rating, direction ORDER BY rating, direction"
    return fetch_all(sql, params)


def graph_payload(run_id: int, min_count: int = 1, rarity: str | None = None) -> dict:
    """Узлы и рёбра графа состояний в формате Cytoscape."""
    groups = fetch_all(
        "SELECT group_id, size AS count, share, dominant_bias, up_share, avg_ret_pct, "
        "top_features, name FROM market_groups WHERE run_id = %s ORDER BY group_id",
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


class SymbolRunMismatch(ValueError):
    """
    run_id принадлежит другой монете.

    Отдельное исключение, потому что симптом обманчив: LEFT JOIN по
    (symbol, ts, run_id) просто не совпадёт, и график BTC с прогоном ETH
    отдаст свечи вообще без раскраски. Выглядит как «граф ничего не нашёл»,
    а на самом деле выбраны несогласованные монета и прогон.
    """


def run_symbol(run_id: int) -> str | None:
    row = fetch_one("SELECT symbol FROM runs WHERE run_id = %s", (run_id,))
    return row["symbol"] if row else None


def chart_data(run_id: int, symbol: str | None = None, start: str | None = None,
               end: str | None = None, limit: int = 1500,
               rating: str | None = None) -> dict:
    """
    Свечи с раскраской по состоянию + маркеры кандидатов.

    Раскраска — главный смысл этой страницы: видно, где именно граф считает,
    что рынок сменил состояние, и совпадает ли это с тем, что видит глаз.

    symbol и run_id обязаны быть согласованы — иначе раскраски не будет, и
    понять почему по пустому графику невозможно. Проверяем явно.
    """
    symbol = symbol or config.data.symbol
    owner = run_symbol(run_id)
    if owner and owner != symbol:
        raise SymbolRunMismatch(
            f"Прогон #{run_id} посчитан по {owner}, а график запрошен для {symbol}. "
            f"Выбери прогон этой монеты — состояния между монетами несопоставимы."
        )

    sql = (
        "SELECT o.ts, o.open, o.high, o.low, o.close, s.group_id, s.is_transition, "
        "s.transition_id, s.age_bucket, s.entropy "
        "FROM ohlcv o LEFT JOIN bar_states s "
        "  ON s.symbol = o.symbol AND s.ts = o.ts AND s.run_id = %s "
        "WHERE o.symbol = %s AND o.tf = %s"
    )
    params: list[Any] = [run_id, symbol, config.data.base_tf]
    if start:
        sql += " AND o.ts >= %s"
        params.append(start)
    if end:
        sql += " AND o.ts <= %s"
        params.append(end)
    # Лимит всегда режет диапазон с одного конца — важно, с какого. Если задана
    # нижняя граница, окно отсчитывается от неё вперёд: при сортировке DESC
    # оставались последние N баров диапазона, и начало заданного периода молча
    # пропадало (запрос с 1 июля отдавал бары с 15-го). Без start смысл обратный
    # — нужны свежие бары, поэтому берём с конца.
    sql += " ORDER BY o.ts ASC LIMIT %s" if start else " ORDER BY o.ts DESC LIMIT %s"
    params.append(limit)

    rows = fetch_all(sql, params)
    bars = rows if start else list(reversed(rows))
    if not bars:
        return {"bars": [], "markers": [], "groups": [], "markers_truncated": False}

    first_ts, last_ts = bars[0]["ts"], bars[-1]["ts"]
    # rating="none" — маркеры не нужны вовсе: на длинном окне их сотни,
    # и они перекрывают саму раскраску состояний.
    truncated = False
    if rating == "none":
        candidates = []
    else:
        sql_c = (
            "SELECT candidate_id, ts, research_side, rating, quality_score, transition_id "
            "FROM candidates WHERE run_id = %s AND ts BETWEEN %s AND %s"
        )
        params_c: list[Any] = [run_id, first_ts, last_ts]
        if rating:
            sql_c += " AND rating = ANY(%s)"
            params_c.append([r.strip() for r in rating.split(",") if r.strip()])
        # DESC, а не ASC: при сортировке по возрастанию лимит отрезал самые
        # свежие маркеры — на окне в 5000 баров свечи шли до сегодня, а стрелки
        # обрывались двумя неделями раньше, и выглядело это как «кандидатов нет».
        # Берём последние MARKER_LIMIT, лишнюю строку — чтобы знать про обрезку.
        sql_c += " ORDER BY ts DESC LIMIT %s"
        params_c.append(MARKER_LIMIT + 1)
        rows_c = fetch_all(sql_c, params_c)
        truncated = len(rows_c) > MARKER_LIMIT
        # lightweight-charts требует маркеры по возрастанию времени.
        candidates = list(reversed(rows_c[:MARKER_LIMIT]))

    group_ids = sorted({b["group_id"] for b in bars if b["group_id"] is not None})
    palette = {gid: _color(i, len(group_ids)) for i, gid in enumerate(group_ids)}
    # Имена состояний для легенды: без них «состояние 7» ничего не говорит,
    # а сверяться с таблицей ради каждого цвета никто не будет.
    names = {
        float(r["group_id"]): r["name"] or ""
        for r in fetch_all(
            "SELECT group_id, name FROM market_groups WHERE run_id = %s", (run_id,)
        )
    }

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
        "groups": [
            {"group_id": gid, "color": palette[gid], "name": names.get(gid, "")}
            for gid in group_ids
        ],
        "markers_truncated": truncated,
    }


def _color(index: int, total: int) -> str:
    """
    Равномерно разнесённые оттенки — соседние состояния не сливаются.

    Возвращаем именно hex: lightweight-charts парсит цвета сам и на строке
    вида `hsl(32, 62%, 55%)` падает с «Cannot parse color».
    """
    hue = 360.0 * index / max(total, 1)
    red, green, blue = colorsys.hls_to_rgb(hue / 360.0, 0.55, 0.62)
    return "#{:02x}{:02x}{:02x}".format(
        round(red * 255), round(green * 255), round(blue * 255)
    )


def _rating_color(rating: str | None) -> str:
    return {"STRONG": "#16a34a", "MODERATE": "#d97706", "WEAK": "#94a3b8"}.get(
        rating or "", "#64748b"
    )


def candidates_page(
    run_id: int | None = None,
    symbol: str | None = None,
    rating: str | None = None,
    direction: str | None = None,
    min_quality: float | None = None,
    transition: str | None = None,
    emitted: str | None = None,
    page: int = 1,
    per_page: int = 50,
) -> dict:
    where, params = [], []
    if symbol:
        where.append("symbol = %s")
        params.append(symbol)
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
        "SELECT candidate_id, symbol, ts, transition_id, event_block_id, research_side, "
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


def top_groups(run_id: int, limit: int = 10) -> list[dict]:
    """Крупнейшие состояния — где рынок проводит больше всего времени."""
    return fetch_all(
        "SELECT group_id, size, share, dominant_bias, up_share, avg_ret_pct "
        "FROM market_groups WHERE run_id = %s ORDER BY size DESC LIMIT %s",
        (run_id, limit),
    )


def transitions_table(run_id: int, limit: int = 200) -> list[dict]:
    return fetch_all(
        "SELECT * FROM transitions WHERE run_id = %s ORDER BY count DESC LIMIT %s",
        (run_id, limit),
    )


def group_detail(run_id: int, group_id: float) -> dict | None:
    """
    Узел графа. Монета не аргумент: run_id уже однозначно её задаёт
    (один прогон = одна монета). Но подписать её в UI обязательно —
    «group_id 7» без монеты бессмысленен.
    """
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
