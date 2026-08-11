"""
Полный прогон истории — «обучение графа».

Последовательность:

    бары → признаки → события → исходы → состояния → граф → кандидаты → btc-graph

Каждая стадия пишет результат в БД под общим run_id, поэтому прогон можно
разобрать постфактум: какая модель состояний использовалась, какие переходы
получились, какие кандидаты из них выросли и как их оценил btc-graph.

Прогресс и лог пишутся в processing.runs — админка читает их оттуда в
реальном времени.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

import pandas as pd

from btcproc import config, notify, symbols
from btcproc.candidates import builder as cand
from btcproc.candidates.outcomes import compute_outcomes
from btcproc.db import repo, runs
from btcproc.db.runs import dumps as runs_dumps
from btcproc.features import builder as feat
from btcproc.features import events as ev
from btcproc.ingest import binance
from btcproc.sink import graph_sink
from btcproc.states import assign, clustering, graph

logger = logging.getLogger(__name__)

# Доля прогресса, приходящаяся на каждую стадию (сумма = 1.0).
STAGE_WEIGHTS = {
    "ingest": 0.20,
    "features": 0.10,
    "events": 0.08,
    "outcomes": 0.05,
    "states": 0.22,
    "graph": 0.05,
    "candidates": 0.15,
    "emit": 0.15,
}


class Progress:
    """Считает общий прогресс по стадиям и пишет его в runs."""

    def __init__(self, run_id: int):
        self.run_id = run_id
        self.done = 0.0
        self.stage = ""

    def start(self, stage: str, message: str = "") -> None:
        self.stage = stage
        runs.log(self.run_id, f"[{stage}] {message or 'старт'}",
                 stage=stage, progress=self.done)

    def within(self, fraction: float, message: str = "") -> None:
        """fraction — доля выполненного внутри текущей стадии [0..1]."""
        weight = STAGE_WEIGHTS.get(self.stage, 0.0)
        runs.update_run(
            self.run_id,
            progress=min(0.999, self.done + weight * max(0.0, min(1.0, fraction))),
            log_line=f"[{self.stage}] {message}" if message else None,
        )

    def finish(self, message: str = "") -> None:
        self.done = min(1.0, self.done + STAGE_WEIGHTS.get(self.stage, 0.0))
        runs.log(self.run_id, f"[{self.stage}] готово {message}", progress=self.done)


def run_train(
    *,
    run_id: int | None = None,
    symbol: str | None = None,
    start: str | None = None,
    end: str | None = None,
    do_ingest: bool = True,
    do_emit: bool = True,
    emit_min_quality: float | None = None,
    on_progress: Callable[[str, float], None] | None = None,
) -> dict:
    """
    Полный прогон. Возвращает сводку; она же ложится в runs.stats.

    do_ingest=False — работать по тому, что уже в БД (быстрый пересчёт модели
    без повторной закачки истории).
    """
    spec = symbols.resolve(symbol)
    symbol = spec.ticker
    # Дата листинга монеты перекрывает общий HISTORY_START: качать ETH с 2017,
    # а SOL с 2020 — это сотни лишних 404 на каждый ingest.
    start = start or spec.start_date()

    # Пороги кластеризации приводятся к монете здесь, до запуска: точечные
    # переопределения из реестра + масштабирование min_group_size по длине
    # истории внутри fit_states.
    states_cfg = spec.states_config()

    own_run = run_id is None
    if own_run:
        run_id = runs.start_run(
            "train",
            {"symbol": symbol, "start": start, "end": end,
             "do_ingest": do_ingest, "do_emit": do_emit,
             "states_overrides": spec.states_overrides or None},
            symbol=symbol,
        )

    progress = Progress(run_id)
    stats: dict = {"run_id": run_id, "symbol": symbol}
    started = time.time()

    try:
        # ── 1. История ──────────────────────────────────────────────────────
        progress.start("ingest", "загрузка баров Binance")
        if do_ingest:
            result = binance.sync_history(
                symbol, config.data.base_tf, start,
                progress=lambda i, n, msg: progress.within(i / max(n, 1), msg),
            )
            stats["ingest"] = result
            runs.log(run_id, f"Загружено баров: {result['rows']}")
            ctx_rows = binance.rebuild_context_timeframes(symbol)
            stats["context_timeframes"] = ctx_rows
        base = binance.load_ohlcv(symbol, config.data.base_tf, start, end)
        if base.empty:
            raise RuntimeError(
                "В БД нет баров. Запусти прогон с загрузкой истории (do_ingest=True)."
            )
        stats["bars"] = len(base)
        progress.finish(f"{len(base)} баров {base.index[0].date()}…{base.index[-1].date()}")

        # ── 2. Признаки ─────────────────────────────────────────────────────
        progress.start("features", "расчёт признаков")
        context = {
            tf: binance.load_ohlcv(symbol, tf, start, end)
            for tf in config.data.context_tfs
        }
        features = feat.build_features(base, context)
        if features.empty:
            raise RuntimeError("Признаки не посчитались — слишком короткая история.")
        repo.save_feature_set(
            feat.feature_version(), list(features.columns),
            {"base_tf": config.data.base_tf, "context": config.data.context_tfs},
        )
        repo.save_features(features, feat.feature_version(), symbol)
        stats["features"] = {"rows": len(features), "columns": features.shape[1]}
        progress.finish(f"{features.shape[1]} признаков × {len(features)} строк")

        # ── 3. События ──────────────────────────────────────────────────────
        progress.start("events", "детекторы событий")
        events = ev.build_event_blocks(base).reindex(features.index).dropna(
            subset=["event_block_id"]
        )
        blocks = ev.block_statistics(events)
        repo.save_events(events, symbol)
        repo.save_event_blocks(run_id, blocks)
        stats["events"] = {"blocks": len(blocks)}
        progress.finish(f"{len(blocks)} уникальных блоков")

        # ── 4. Исходы ───────────────────────────────────────────────────────
        progress.start("outcomes", f"разметка исходов на {config.data.horizon}")
        outcomes = compute_outcomes(base).reindex(features.index)
        repo.save_outcomes(outcomes, symbol)
        stats["outcomes"] = {
            "valid": int(outcomes["valid"].sum()),
            "total": int(len(outcomes)),
        }
        progress.finish(f"{stats['outcomes']['valid']} валидных меток")

        # ── 5. Состояния ────────────────────────────────────────────────────
        progress.start("states", "адаптивная кластеризация")
        scale = feat.robust_scale_params(features)
        matrix = feat.apply_scale(features, scale)
        model, raw_labels = clustering.fit_states(
            matrix, list(features.columns), scale,
            cfg=states_cfg,
            progress=lambda msg, frac: progress.within(frac, msg),
        )
        states = assign.assign_states(features.index, raw_labels)
        repo.save_state_model(run_id, model, feat.feature_version())
        repo.save_bar_states(run_id, states, symbol)
        stats["states"] = {
            "groups": model.n_groups,
            "transitions": int(states["is_transition"].sum()),
        }
        progress.finish(f"{model.n_groups} состояний, "
                        f"{stats['states']['transitions']} переходов")

        # ── 5а. Сброс графа монеты ──────────────────────────────────────────
        # Новая модель записана — значит нумерация group_id только что
        # поехала, и накопленный в Neo4j граф ей больше не соответствует.
        # Ключ узла там `(symbol, group_id)`, измерения модели в нём нет,
        # поэтому счётчики рёбер продолжили бы расти поверх узлов прежнего
        # поколения, а отличить одно от другого в графе невозможно: те же
        # свойства, правдоподобные значения.
        #
        # Место выбрано именно здесь, а не перед отправкой, потому что
        # штатный недельный train идёт с `--no-emit`: кандидатов он не шлёт,
        # но модель подменяет, и ближайший `live` начнёт писать в граф уже
        # новой нумерацией. Привязка очистки к отправке оставила бы этот
        # путь открытым.
        #
        # Ошибка не глотается: прогон обязан упасть с диагнозом. Молчаливое
        # смешивание поколений — ровно то, что очистка закрывает, и «не
        # получилось почистить, но модель сохранили» его воспроизводит.
        if config.sink.mode != "none":
            deleted = graph_sink.reset_graph(symbol)
            stats["graph_reset"] = {"nodes_deleted": deleted}
            runs.log(run_id, f"Граф {symbol} очищен под новую модель: "
                             f"узлов удалено {deleted}")

        # ── 6. Граф ─────────────────────────────────────────────────────────
        progress.start("graph", "статистика графа")
        transitions = graph.transition_stats(states, outcomes)
        groups = graph.group_stats(states, outcomes, features)
        centroids = {
            float(gid): model.centroids[i].tolist()
            for i, gid in enumerate(model.group_ids)
        }
        repo.save_transitions(run_id, transitions)
        repo.save_groups(run_id, groups, centroids)
        stats["graph"] = {"nodes": len(groups), "edges": len(transitions)}
        progress.finish(f"{len(groups)} узлов, {len(transitions)} рёбер")

        # ── 7. Кандидаты ────────────────────────────────────────────────────
        progress.start("candidates", "сборка кандидатов")
        snapshots = cand.build_snapshots(states, events, outcomes)
        rarity_map = dict(zip(transitions["transition_id"], transitions["rarity"]))
        block_map = blocks.set_index("event_block_id").to_dict("index")

        produced, buffer = 0, []
        scopes: dict[str, int] = {}
        for candidate in cand.generate(
            snapshots, rarity_map, block_map, symbol,
            progress=lambda i, n: progress.within(i / max(n, 1), ""),
        ):
            buffer.append(candidate)
            produced += 1
            scope = candidate["_meta"]["scope"]
            scopes[scope] = scopes.get(scope, 0) + 1
            if len(buffer) >= 5000:
                repo.save_candidates(run_id, buffer)
                buffer.clear()
        if buffer:
            repo.save_candidates(run_id, buffer)
        # Доля scope=transition+event_block показывает, насколько специфичны
        # блоки событий: если она близка к нулю, блоки слишком дробные и
        # выборка по ним никогда не набирается.
        stats["candidates"] = {
            "produced": produced, "snapshots": len(snapshots), "scopes": scopes,
        }
        progress.finish(f"{produced} кандидатов из {len(snapshots)} снимков")

        # ── 8. Отправка в btc-graph ─────────────────────────────────────────
        progress.start("emit", f"режим {config.sink.mode}")
        if do_emit and config.sink.mode != "none" and produced:
            stats["emit"] = emit_pending(
                run_id,
                min_quality=emit_min_quality,
                on_batch=lambda done, total: progress.within(done / max(total, 1),
                                                             f"{done}/{total}"),
            )
        else:
            stats["emit"] = {"sent": 0, "skipped": True}
        progress.finish(str(stats["emit"]))

        stats["seconds"] = round(time.time() - started, 1)
        if own_run:
            runs.finish_run(run_id, stats)
        if on_progress:
            on_progress("done", 1.0)
        return stats

    except Exception as exc:  # noqa: BLE001 — прогон обязан завершиться с диагнозом
        logger.exception("Прогон %s упал", run_id)
        if own_run:
            runs.fail_run(run_id, f"{type(exc).__name__}: {exc}")
        raise


def emit_pending(
    run_id: int,
    batch_size: int | None = None,
    min_quality: float | None = None,
    limit: int | None = None,
    on_batch: Callable[[int, int], None] | None = None,
    retry_failed: bool = True,
) -> dict:
    """
    Отправляет в btc-graph кандидатов прогона, которые ещё не отправлялись.

    Оценок возвращается меньше, чем отправлено: btc-graph отбрасывает слабых
    и схлопывает дубликаты по family_key. Отправленным считается весь батч —
    повторно слать отсеянных смысла нет.

    `min_quality` — ЛОКАЛЬНЫЙ порог поверх чужого фильтра, он влияет только на
    статистику прогона. Оценки сохраняются до него: кандидат, которого
    btc-graph принял и сохранил у себя, обязан иметь свою оценку и в
    `processing.candidates`, иначе базы расходятся. Причина в
    `mark_emit_error` у таких кандидатов своя — путать её с «отсеян фильтром
    btc-graph» нельзя, это разные события.

    on_batch(готово, всего) зовётся после каждой пачки. Для исторического
    бэкфилла это не украшательство: очередь в десятки тысяч разбирается часами,
    и без обратной связи команда неотличима от зависшей.

    `retry_failed` — добирать кандидатов ТОЙ ЖЕ модели, у которых прошлая
    отправка сорвалась (`emit_error` есть, `emitted_at` нет). Раньше их
    подхватывало само собой: перекрывающиеся окна live «перевозили» кандидата
    в свежий `run_id`, и следующий прогон видел его как своего. С тех пор как
    `run_id` перестал переписываться (кандидат остаётся за прогоном, который
    его впервые выпустил), этот путь исчез — сорвавшаяся отправка залипала бы
    навсегда, а заметить это можно было бы только глазами в админке.

    Добираются именно СОРВАВШИЕСЯ, а не все неотправленные: `train --no-emit`
    накапливает десятки тысяч кандидатов без `emit_error`, и первый же live
    после него принялся бы разгребать этот бэкфилл без спроса, часами.
    """
    from btcproc.db.session import fetch_all
    from btcproc.db.runs import model_root, model_run_scope

    batch_size = batch_size or config.sink.batch_size
    params: list = [run_id]
    where = "run_id = %s"
    if retry_failed:
        scope_sql, scope_params = model_run_scope(model_root(run_id))
        where = f"({where} OR (emit_error IS NOT NULL AND {scope_sql}))"
        params.extend(scope_params)
    sql = (
        f"SELECT payload FROM candidates "
        f"WHERE {where} AND emitted_at IS NULL ORDER BY ts"
    )
    if limit:
        sql += " LIMIT %s"
        params.append(limit)

    # Выборка идёт целиком в память: на очереди в 34 тысячи это десятки
    # мегабайт JSONB и несколько секунд тишины перед первой пачкой. Сообщаем
    # о начале ДО запроса — иначе непонятно, началось ли что-нибудь вообще.
    if on_batch:
        on_batch(0, 0)

    pending = [row["payload"] for row in fetch_all(sql, params)]
    total = len(pending)
    sent, evaluated, errors = 0, 0, 0
    notified: dict[str, int] = {}

    for start in range(0, total, batch_size):
        batch = pending[start:start + batch_size]
        ids = [c["candidate_id"] for c in batch]
        try:
            results = graph_sink.emit_batch(batch)
            # Сохраняем оценки ВСЕХ вернувшихся, до локального порога. Каждая
            # такая запись у btc-graph уже есть; отбросив её здесь, мы бы
            # развели базы и потеряли готовую оценку без возможности
            # восстановить (повторно кандидат не отправляется).
            repo.save_evaluations(results)
            evaluated_ids = {r["candidate_id"] for r in results}

            # Две разные судьбы с разными причинами. Смешивать их нельзя:
            # «отсеян фильтром btc-graph» на кандидате, которого btc-graph
            # как раз принял и сохранил, — ложный диагноз.
            rejected = [cid for cid in ids if cid not in evaluated_ids]
            if rejected:
                repo.mark_emit_error(
                    rejected, "отсеян фильтром btc-graph", mark_emitted=True
                )

            accepted = results
            if min_quality is not None:
                weak_ids = [
                    r["candidate_id"] for r in results
                    if (r.get("quality_score") or 0) < min_quality
                ]
                if weak_ids:
                    repo.mark_emit_error(
                        weak_ids,
                        f"оценён btc-graph, но ниже локального "
                        f"emit_min_quality={min_quality}",
                        mark_emitted=True,
                    )
                weak_set = set(weak_ids)
                accepted = [r for r in results if r["candidate_id"] not in weak_set]

            sent += len(batch)
            # Успешно отправленными считаются те, что прошли и чужой фильтр,
            # и локальный порог, — по ним и считается статистика прогона.
            evaluated += len(accepted)
        except Exception as exc:  # noqa: BLE001
            errors += len(batch)
            repo.mark_emit_error(ids, f"{type(exc).__name__}: {exc}")
            runs.log(run_id, f"Ошибка отправки батча: {exc}")

        # Уведомления шлём здесь, а не в конце: оценка уже записана, и
        # правилам есть по чему фильтровать (rating и quality_score считает
        # btc-graph, у нас их нет). Отправка не блокирующая — задания уходят
        # в очередь, прогон идёт дальше. Кандидаты старше окна свежести
        # отсекаются внутри dispatch, поэтому бэкфилл на всей истории не
        # превращается в десятки тысяч вебхуков.
        _merge_counters(notified, notify.dispatch(ids, reason=f"emit run={run_id}"))

        if on_batch:
            on_batch(min(start + batch_size, total), total)

    # Очередь разбирается фоновыми потоками, а крон-процесс завершится сразу
    # после прогона и убьёт их вместе с неотправленным. Ждём с дедлайном.
    if notified.get("queued"):
        runs.log(run_id, f"Уведомления: {runs_dumps(notified)}")
    notify.flush()

    return {"sent": sent, "evaluated": evaluated, "errors": errors, "pending": total,
            "notify": notified}


def _merge_counters(target: dict[str, int], summary: dict) -> None:
    """Складывает сводки рассылки по батчам в одну. Строковые поля переносит."""
    for key, value in summary.items():
        if isinstance(value, int):
            target[key] = target.get(key, 0) + value
        else:
            target[key] = value
