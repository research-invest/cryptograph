"""
Запись и чтение результатов расчёта: признаки, события, состояния, граф,
исходы, кандидаты.

Кандидаты дублируются здесь намеренно: в btc-graph уходит только то, что
прошло его фильтр, а нам для админки и для разбора нужен полный список —
включая тех, кого отсеяли.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import psycopg2.extras

from btcproc import config
from btcproc.db.session import bulk_upsert, connect, fetch_all, fetch_one

logger = logging.getLogger(__name__)


# ─── Признаки ───────────────────────────────────────────────────────────────
def save_feature_set(version: str, names: Sequence[str], params: dict | None = None) -> None:
    bulk_upsert(
        "feature_sets",
        ["version", "names", "params"],
        [(version, list(names), psycopg2.extras.Json(params or {}))],
        conflict_columns=["version"],
    )


def save_features(features: pd.DataFrame, version: str, symbol: str | None = None) -> int:
    symbol = symbol or config.data.symbol
    values = features.to_numpy(dtype=float)
    rows = [
        (symbol, ts.to_pydatetime(), version, row.tolist())
        for ts, row in zip(features.index, values)
    ]
    return bulk_upsert(
        "features",
        ["symbol", "ts", "version", "values"],
        rows,
        conflict_columns=["symbol", "ts", "version"],
    )


def last_feature_ts(symbol: str, version: str):
    """
    Последний бар, для которого посчитаны признаки НАБОРА `version`.

    Точка продолжения для `live`: он дописывает хвост признаков, а не всю
    историю (её пишет `train`). None — набора в таблице нет вовсе.
    """
    row = fetch_one(
        "SELECT max(ts) AS ts FROM features WHERE symbol = %s AND version = %s",
        (symbol, version),
    )
    return row["ts"] if row and row["ts"] else None


def model_feature_version(run_id: int) -> str | None:
    """Метка набора признаков, которым обучена модель прогона."""
    row = fetch_one("SELECT feature_ver FROM state_models WHERE run_id = %s", (run_id,))
    return row["feature_ver"] if row else None


def load_features(version: str, symbol: str | None = None) -> pd.DataFrame:
    meta = fetch_one("SELECT names FROM feature_sets WHERE version = %s", (version,))
    if not meta:
        return pd.DataFrame()
    rows = fetch_all(
        "SELECT ts, values FROM features WHERE symbol = %s AND version = %s ORDER BY ts",
        (symbol or config.data.symbol, version),
    )
    if not rows:
        return pd.DataFrame()
    index = pd.DatetimeIndex([r["ts"] for r in rows], name="ts")
    return pd.DataFrame([r["values"] for r in rows], index=index, columns=meta["names"])


# ─── События ────────────────────────────────────────────────────────────────
def save_events(events: pd.DataFrame, symbol: str | None = None,
                version: str | None = None) -> int:
    """
    Разметка баров атомами.

    `version` — метка НАБОРА атомов (`events.event_version()`), а не формата
    таблицы. Без неё строки прогонов с разным составом детекторов внешне
    неотличимы, а смысл `event_block_id` у них разный. Дефолт берётся из
    текущей конфигурации: вызывающему не надо помнить про параметр, но
    подменить его (бэкфилл, сверка) можно.
    """
    from btcproc.features import events as ev

    symbol = symbol or config.data.symbol
    version = version or ev.event_version()
    # itertuples, а не iterrows: последний собирает Series на КАЖДУЮ строку,
    # и на трёхстах тысячах баров train это секунды чистого CPU на одну
    # таблицу — при том, что дальше всё равно доминирует вставка
    # (аудит 2026-08-15, O5).
    rows = [
        (
            symbol, r.Index.to_pydatetime(), r.event_block_id, list(r.atoms), list(r.families),
            int(r.atom_count), int(r.family_count), r.intensity, r.primary_family,
            list(r.context_atoms), version,
        )
        for r in events.itertuples()
    ]
    return bulk_upsert(
        "bar_events",
        ["symbol", "ts", "event_block_id", "atoms", "families", "atom_count",
         "family_count", "intensity", "primary_family", "context_atoms", "version"],
        rows,
        conflict_columns=["symbol", "ts"],
    )


def save_event_blocks(run_id: int, blocks: pd.DataFrame,
                      version: str | None = None) -> int:
    from btcproc.features import events as ev

    version = version or ev.event_version()
    rows = [
        (
            run_id, r.event_block_id, int(r.total_rows), float(r.row_share), r.rarity,
            r.intensity, int(r.atom_count), int(r.family_count), r.primary_family,
            list(r.families) if isinstance(r.families, (list, tuple)) else None,
            version,
        )
        for _, r in blocks.iterrows()
    ]
    return bulk_upsert(
        "event_blocks",
        ["run_id", "event_block_id", "total_rows", "row_share", "rarity", "intensity",
         "atom_count", "family_count", "primary_family", "families", "version"],
        rows,
        conflict_columns=["run_id", "event_block_id"],
    )


# ─── Состояния и граф ───────────────────────────────────────────────────────
def save_state_model(run_id: int, model, feature_version: str) -> None:
    payload = json.dumps(model.to_dict(), ensure_ascii=False).encode("utf-8")
    bulk_upsert(
        "state_models",
        ["run_id", "version", "n_groups", "feature_ver", "params", "artifact"],
        [(
            run_id, "v1", model.n_groups, feature_version,
            psycopg2.extras.Json(model.params), psycopg2.Binary(payload),
        )],
        conflict_columns=["run_id"],
    )


def load_state_model(run_id: int):
    """
    Модель по run_id. Монета не спрашивается: run_id уже однозначно её задаёт
    (один прогон = одна монета), а лишний аргумент только позволил бы им
    разъехаться.
    """
    from btcproc.states.clustering import StateModel

    row = fetch_one("SELECT artifact FROM state_models WHERE run_id = %s", (run_id,))
    if not row or row["artifact"] is None:
        return None
    return StateModel.from_dict(json.loads(bytes(row["artifact"]).decode("utf-8")))


def latest_model_run_id(symbol: str) -> int | None:
    """run_id последнего успешного train'а монеты, у которого есть модель."""
    row = fetch_one(
        "SELECT r.run_id FROM runs r JOIN state_models m ON m.run_id = r.run_id "
        "WHERE r.kind = 'train' AND r.status = 'done' AND r.symbol = %s "
        "ORDER BY r.finished_at DESC LIMIT 1",
        (symbol,),
    )
    return int(row["run_id"]) if row else None


def save_bar_states(run_id: int, states: pd.DataFrame, symbol: str | None = None) -> int:
    symbol = symbol or config.data.symbol
    rows = [
        (
            symbol, r.Index.to_pydatetime(), run_id, float(r.group_id),
            None if pd.isna(r.prev_group_id) else float(r.prev_group_id),
            int(r.state_seq), int(r.age_minutes), r.age_bucket, r.entropy,
            bool(r.is_transition), r.transition_id,
        )
        for r in states.itertuples()
    ]
    return bulk_upsert(
        "bar_states",
        ["symbol", "ts", "run_id", "group_id", "prev_group_id", "state_seq",
         "age_minutes", "age_bucket", "entropy", "is_transition", "transition_id"],
        rows,
        conflict_columns=["symbol", "ts", "run_id"],
    )


def save_groups(run_id: int, groups: pd.DataFrame, centroids: dict | None = None) -> int:
    rows = []
    for _, r in groups.iterrows():
        gid = float(r.group_id)
        rows.append((
            run_id, gid, int(r["count"]), float(r.share), r.get("dominant_bias"),
            _nan_to_none(r.get("up_share")), _nan_to_none(r.get("avg_ret_pct")),
            _nan_to_none(r.get("avg_vol_pct")),
            (centroids or {}).get(gid),
            psycopg2.extras.Json(r.get("top_features") or {}),
            r.get("name"),
        ))
    return bulk_upsert(
        "market_groups",
        ["run_id", "group_id", "size", "share", "dominant_bias", "up_share",
         "avg_ret_pct", "avg_vol_pct", "centroid", "top_features", "name"],
        rows,
        conflict_columns=["run_id", "group_id"],
    )


# ─── Фон состояний ──────────────────────────────────────────────────────────
#
# Насколько часто контекстный атом встречается внутри состояния по сравнению
# со всей размеченной историей прогона. Считается ОДИН РАЗ, в конце train, и
# ложится в `state_context` — до 2026-08-16 админка гоняла этот агрегат на
# каждое открытие узла графа (журнал 43).
#
# Считается целиком на сервере: разворот массивов атомов по трёмстам тысячам
# баров в питон тянуть незачем, наружу выходят полторы тысячи строк.
#
# Пороги отбора (доля, лифт, верхушка) здесь НЕ применяются — они живут в
# админке и меняются без пересчёта агрегата.

#: Размеченные бары прогона вместе с их фоном — во временную таблицу.
#:
#: Форма запроса выбрана так, чтобы НЕ ЗАВИСЕТЬ ОТ ОЦЕНОК ПЛАНИРОВЩИКА, и это
#: не перестраховка. У свежей монеты статистики по `symbol` ещё нет (ANALYZE
#: не прошёл), и PostgreSQL оценивает выборку в ОДНУ строку — после чего
#: берёт nested loop, в котором внутренний скан идёт по `bar_events` без
#: условия на `ts`: 80 тысяч строк × 80 тысяч строк с проверкой в join-фильтре.
#: На боевом контуре это и наблюдалось: у пяти монет со статистикой агрегат
#: считался за 3–4 секунды, у только что заведённой TAOUSDT тот же запрос шёл
#: больше десяти минут (журнал 43.7).
#:
#: Оба входа обёрнуты в MATERIALIZED CTE: у материализованного набора индекса
#: нет, поэтому вложенный цикл по нему невыгоден при любой оценке, и
#: планировщику остаются hash/merge join. Цена — временный набор на время
#: запроса; выигрыш — устойчивость к незнанию статистики.
_LABELED_SQL = """
CREATE TEMP TABLE _state_context_labeled ON COMMIT DROP AS
WITH ev AS MATERIALIZED (
    SELECT ts, context_atoms FROM bar_events
    WHERE symbol = %(symbol)s AND context_atoms IS NOT NULL
),
st AS MATERIALIZED (
    SELECT ts, group_id FROM bar_states
    WHERE symbol = %(symbol)s AND run_id = %(run_id)s
)
SELECT st.group_id, ev.context_atoms FROM st JOIN ev ON ev.ts = st.ts
"""

#: Сам агрегат — уже по временной таблице, то есть join сделан один раз.
#: Общая частота атома считается из `atom_group`, а не вторым `unnest`ом по
#: всей истории: тот удваивал время запроса на ровном месте.
_STATE_CONTEXT_SQL = """
INSERT INTO state_context (run_id, group_id, atom, share, lift)
WITH per_group AS (
    SELECT group_id, count(*) AS bars FROM _state_context_labeled GROUP BY 1
),
atom_group AS (
    SELECT l.group_id, a.atom, count(*) AS n
    FROM _state_context_labeled l, unnest(l.context_atoms) AS a(atom)
    GROUP BY 1, 2
),
atom_total AS (SELECT atom, sum(n) AS n FROM atom_group GROUP BY 1),
totals AS (SELECT sum(bars) AS bars FROM per_group)
SELECT %(run_id)s,
       ag.group_id,
       ag.atom,
       ag.n::float / pg.bars,
       (ag.n::float / pg.bars) / NULLIF(at.n::float / t.bars, 0)
FROM atom_group ag
JOIN per_group pg ON pg.group_id = ag.group_id
JOIN atom_total at ON at.atom = ag.atom
CROSS JOIN totals t
ON CONFLICT (run_id, group_id, atom) DO UPDATE
   SET share = EXCLUDED.share, lift = EXCLUDED.lift
"""


def save_state_context(run_id: int, symbol: str | None = None,
                       timeout_ms: int | None = None) -> dict:
    """
    Считает и сохраняет фон состояний прогона. Идемпотентно.

    Возвращает {"bars": ..., "rows": ...} — сколько баров участвовало и
    сколько пар «состояние × атом» получилось. Ноль баров — законный исход
    (история размечена до появления `bar_events.context_atoms`), и отметка
    в `state_context_runs` ставится всё равно: без неё «посчитано и пусто»
    неотличимо от «не считалось», и читатель гонял бы агрегат заново.
    """
    symbol = symbol or config.data.symbol
    params = {"symbol": symbol, "run_id": run_id}
    with connect(timeout_ms=timeout_ms) as conn, conn.cursor() as cur:
        # ON COMMIT DROP: соединение уходит обратно в пул, и оставленная на
        # нём временная таблица досталась бы следующему запросу.
        cur.execute(_LABELED_SQL, params)
        cur.execute("SELECT count(*) FROM _state_context_labeled")
        bars = cur.fetchone()[0]
        # Пересчёт, а не досчёт: состав атомов мог измениться бэкфиллом, и
        # строка, переставшая проходить, обязана исчезнуть.
        cur.execute("DELETE FROM state_context WHERE run_id = %s", (run_id,))
        cur.execute(_STATE_CONTEXT_SQL, params)
        rows = cur.rowcount
        cur.execute(
            "INSERT INTO state_context_runs (run_id, bars, atom_rows, computed_at) "
            "VALUES (%s, %s, %s, NOW()) "
            "ON CONFLICT (run_id) DO UPDATE SET bars = EXCLUDED.bars, "
            "  atom_rows = EXCLUDED.atom_rows, computed_at = EXCLUDED.computed_at",
            (run_id, bars, rows),
        )
    logger.info("Фон состояний прогона %s: %s баров, %s строк", run_id, bars, rows)
    return {"bars": bars, "rows": rows}


#: Таблицы, чья статистика для планировщика устаревает за один прогон
#: настолько, что это меняет планы: в них прогон пишет от десятков тысяч до
#: сотен тысяч строк.
_ANALYZE_TABLES = ("ohlcv", "features", "bar_events", "bar_states", "candidates")


def refresh_planner_stats(tables: Sequence[str] = _ANALYZE_TABLES) -> None:
    """
    Собирает статистику для планировщика по таблицам, которые только что
    пополнил прогон.

    Нужно не «для скорости вообще», а из-за НОВОЙ МОНЕТЫ. Пока ANALYZE по ней
    не прошёл, планировщик не знает про её `symbol` вовсе и оценивает любую
    выборку в одну строку — после чего выбирает планы, рассчитанные на одну
    строку, для сотен тысяч. Автовакуум это исправляет, но своим чередом, а
    оператор открывает граф сразу после обучения (журнал 43.7).

    Здесь же собирается расширенная статистика `bar_states (symbol, run_id)`:
    сама по себе `CREATE STATISTICS` только объявляет её, данные появляются
    при ANALYZE.

    Три-четыре секунды на прогон в полчаса — цена, которую не жалко.
    """
    with connect(autocommit=True) as conn, conn.cursor() as cur:
        for table in tables:
            cur.execute(f"ANALYZE {table}")
    logger.info("Статистика планировщика обновлена: %s", ", ".join(tables))


def state_context_ready(run_id: int) -> bool:
    """Посчитан ли фон этого прогона. Пустой результат — тоже посчитанный."""
    return fetch_one(
        "SELECT 1 FROM state_context_runs WHERE run_id = %s", (run_id,)
    ) is not None


def load_state_context(run_id: int) -> list[dict]:
    """Фон состояний прогона, от сильнейшего лифта внутри состояния."""
    return fetch_all(
        "SELECT group_id, atom, share, lift FROM state_context "
        "WHERE run_id = %s ORDER BY group_id, lift DESC NULLS LAST",
        (run_id,),
    )


def save_transitions(run_id: int, transitions: pd.DataFrame) -> int:
    rows = [
        (
            run_id, r.transition_id, _nan_to_none(r.prev_group_id), float(r.cur_group_id),
            int(r["count"]), float(r.share), r.rarity,
            _nan_to_none(r.get("avg_horizon_return")), _nan_to_none(r.get("up_share")),
        )
        for _, r in transitions.iterrows()
    ]
    return bulk_upsert(
        "transitions",
        ["run_id", "transition_id", "prev_group_id", "cur_group_id", "count",
         "share", "rarity", "avg_horizon_return", "up_share"],
        rows,
        conflict_columns=["run_id", "transition_id"],
    )


# ─── Исходы ─────────────────────────────────────────────────────────────────
def save_outcomes(outcomes: pd.DataFrame, symbol: str | None = None,
                  horizon: str | None = None) -> int:
    symbol = symbol or config.data.symbol
    horizon = horizon or config.data.horizon
    rows = [
        (
            symbol, r.Index.to_pydatetime(), horizon,
            _nan_to_none(r.ret_pct), _nan_to_none(r.mfe_pct), _nan_to_none(r.mae_pct),
            None if r.is_up is None or pd.isna(r.ret_pct) else bool(r.is_up),
            bool(r.valid),
        )
        for r in outcomes.itertuples()
    ]
    return bulk_upsert(
        "outcomes",
        ["symbol", "ts", "horizon", "ret_pct", "mfe_pct", "mae_pct", "is_up", "valid"],
        rows,
        conflict_columns=["symbol", "ts", "horizon"],
    )


# ─── Кандидаты ──────────────────────────────────────────────────────────────
def save_candidates(run_id: int, candidates: Iterable[dict]) -> int:
    """
    Пишет кандидатов прогона. Повтор безопасен: `candidate_id` детерминирован,
    запись идёт upsert'ом.

    `run_id` — прогон, ВПЕРВЫЕ выпустивший кандидата, и при повторной встрече
    он не переписывается. Окна live намеренно перекрываются, поэтому одного и
    того же кандидата видят несколько прогонов подряд; пока `run_id` был в
    `update_columns`, каждый такой прогон «перевозил» его к себе. Выборка
    «кандидаты прогона N» от этого худела со временем сама по себе, а
    счётчики `prune_runs.models()` плыли. Ни то, ни другое не выглядело
    ошибкой — просто числа медленно становились неправильными.

    Кандидатов одной МОДЕЛИ (а не одного запуска) выбирают через
    `runs.model_run_scope`.
    """
    rows = []
    for c in candidates:
        meta = c.get("_meta", {})
        payload = {k: v for k, v in c.items() if not k.startswith("_")}
        rows.append((
            c["candidate_id"], run_id, c["symbol"], meta.get("ts"),
            c["transition_id"], c["event_block_id"], c.get("candidate_family_key"),
            c["research_side"], float(c["research_score"]), int(c["sample_size"]),
            psycopg2.extras.Json(payload),
        ))
    return bulk_upsert(
        "candidates",
        ["candidate_id", "run_id", "symbol", "ts", "transition_id", "event_block_id",
         "family_key", "research_side", "research_score", "sample_size", "payload"],
        rows,
        conflict_columns=["candidate_id"],
        # run_id намеренно НЕ обновляется — см. docstring.
        update_columns=["payload", "research_score", "sample_size"],
    )


def last_candidate_ts(symbol: str) -> pd.Timestamp | None:
    """
    Время самого свежего выпущенного кандидата — точка, с которой продолжает
    live-прогон. Берётся по символу, а не по run_id: каждый live создаёт свой
    прогон, и привязка к нему потеряла бы всю предысторию.

    Аргумент обязателен. Молчаливый дефолт из .env здесь означал бы, что live
    по ETH продолжает с точки BTC — и выпускает либо дыру, либо десятки тысяч
    кандидатов разом.
    """
    row = fetch_one(
        "SELECT max(ts) AS ts FROM candidates WHERE symbol = %s", (symbol,)
    )
    return pd.Timestamp(row["ts"]) if row and row["ts"] else None


def save_evaluations(evaluations: Iterable[dict]) -> int:
    """Проставляет кандидатам оценку, вернувшуюся из btc-graph."""
    rows = [
        (
            e["candidate_id"], _nan_to_none(e.get("quality_score")), e.get("rating"),
            e.get("direction"), list(e.get("warning_flags") or []),
            psycopg2.extras.Json(e),
        )
        for e in evaluations
    ]
    if not rows:
        return 0
    sql = """
        UPDATE candidates AS c SET
            quality_score = v.quality_score,
            rating        = v.rating,
            direction     = v.direction,
            warning_flags = v.warning_flags,
            evaluation    = v.evaluation,
            emitted_at    = NOW(),
            emit_error    = NULL
        FROM (VALUES %s) AS v(candidate_id, quality_score, rating, direction,
                              warning_flags, evaluation)
        WHERE c.candidate_id = v.candidate_id
    """
    with connect() as conn, conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur, sql, rows,
            template="(%s, %s::double precision, %s, %s, %s::text[], %s::jsonb)",
        )
        return cur.rowcount


def mark_emit_error(
    candidate_ids: Sequence[str], error: str, mark_emitted: bool = False
) -> None:
    """
    mark_emitted=True — кандидат обработан, но оценки не получил (например,
    отсеян фильтром btc-graph). Без этого он бесконечно возвращался бы
    в очередь отправки.
    """
    if not candidate_ids:
        return
    emitted = ", emitted_at = NOW()" if mark_emitted else ""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE candidates SET emit_error = %s{emitted} WHERE candidate_id = ANY(%s)",
            (error[:2000], list(candidate_ids)),
        )


def _nan_to_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
