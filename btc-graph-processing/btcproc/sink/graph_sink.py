"""
Доставка кандидатов в btc-graph.

Режим `direct` — рабочий: мы импортируем пакет btc-graph и вызываем его
`run_batch_pipeline(save=True)`. Запись идёт прямо в PostgreSQL, Neo4j и Redis,
без HTTP-прослойки, но проходит через его валидатор, скорер, фильтр по
family_key и дедупликацию — то есть смысл btc-graph как фильтра сохраняется,
уходит только транспорт.

Почему это вообще возможно без конфликта имён: пакет btc-graph называется
`src`, наш — `btcproc`. Их каталог добавляется в sys.path один раз, и их
ленивые импорты (`from src.db...` внутри `_persist`) резолвятся корректно.

Режим `http` оставлен как запасной путь: если btc-graph поднят в контейнере,
а мы работаем снаружи и не хотим тянуть его зависимости.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, Iterable, Sequence

import httpx

from btcproc import config
from btcproc.candidates.builder import strip_meta

logger = logging.getLogger(__name__)

_pipeline = None  # кэш импортированного модуля btc-graph


def _load_btc_graph():
    """
    Импортирует pipeline btc-graph, предварительно выставив ему окружение.

    USE_DB / USE_REDIS / USE_GRAPH читаются на импорте модуля, поэтому
    выставлять их после первого импорта бесполезно — отсюда порядок действий.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    path = str(config.sink.btc_graph_path)
    if not os.path.isdir(os.path.join(path, "src")):
        raise RuntimeError(
            f"BTC_GRAPH_PATH={path} не похож на репозиторий btc-graph "
            "(не найден каталог src). Поправь .env или переключи SINK_MODE=http."
        )
    if path not in sys.path:
        sys.path.insert(0, path)

    os.environ.setdefault("DATABASE_URL", config.db.url)
    os.environ.setdefault("REDIS_URL", config.sink.btc_graph_redis_url)
    os.environ.setdefault("USE_DB", "true")
    os.environ.setdefault("USE_REDIS", "true")
    os.environ.setdefault("USE_GRAPH", "true")

    from src.agent import pipeline  # noqa: E402  (пакет btc-graph)

    _pipeline = pipeline
    logger.info("btc-graph подключён из %s", path)
    return _pipeline


def _to_result(evaluation: Any) -> dict:
    """CandidateEvaluation (pydantic) или dict из HTTP → одинаковый dict."""
    if hasattr(evaluation, "model_dump"):
        return evaluation.model_dump()
    return dict(evaluation)


def emit_direct(batch: Sequence[dict], use_llm: bool | None = None) -> list[dict]:
    pipeline = _load_btc_graph()
    payload = [strip_meta(c) for c in batch]
    evaluations = pipeline.run_batch_pipeline(
        payload,
        use_llm=config.sink.use_llm if use_llm is None else use_llm,
        min_quality_score=config.sink.min_quality_score,
        save=True,
    )
    return [_to_result(e) for e in evaluations]


def emit_http(batch: Sequence[dict], use_llm: bool | None = None) -> list[dict]:
    payload = {
        "candidates": [strip_meta(c) for c in batch],
        "use_llm": config.sink.use_llm if use_llm is None else use_llm,
        "save": True,
        "min_quality_score": config.sink.min_quality_score,
    }
    url = config.sink.btc_graph_url.rstrip("/") + "/evaluate/batch"
    with httpx.Client(timeout=300.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    # Формат ответа /evaluate/batch — список оценок либо {"results": [...]}.
    if isinstance(data, dict):
        data = data.get("results", data.get("evaluations", []))
    return [_to_result(item) for item in data]


def emit_batch(batch: Sequence[dict], mode: str | None = None, use_llm: bool | None = None) -> list[dict]:
    """
    Отправляет пачку кандидатов и возвращает оценки.

    Оценок может прийти меньше, чем кандидатов: btc-graph отбрасывает слабых
    по min_quality_score и оставляет по одному на candidate_family_key.
    """
    mode = mode or config.sink.mode
    if not batch or mode == "none":
        return []
    if mode == "direct":
        return emit_direct(batch, use_llm)
    if mode == "http":
        return emit_http(batch, use_llm)
    raise ValueError(f"Неизвестный SINK_MODE: {mode}")


def chunked(items: Iterable[dict], size: int) -> Iterable[list[dict]]:
    chunk: list[dict] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def sink_status() -> dict:
    """Диагностика для админки: доступен ли приёмник кандидатов."""
    mode = config.sink.mode
    status: dict[str, Any] = {"mode": mode, "ok": False, "detail": ""}
    if mode == "none":
        status.update(ok=True, detail="отправка отключена")
        return status
    if mode == "direct":
        try:
            _load_btc_graph()
            status.update(ok=True, detail=f"пакет btc-graph из {config.sink.btc_graph_path}")
        except Exception as exc:  # noqa: BLE001 — показываем причину в админке
            status["detail"] = str(exc)
        return status
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(config.sink.btc_graph_url.rstrip("/") + "/health")
        status.update(ok=resp.status_code == 200, detail=f"HTTP {resp.status_code}")
    except Exception as exc:  # noqa: BLE001
        status["detail"] = str(exc)
    return status
