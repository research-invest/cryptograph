"""
Диагностика окружения для SINK_MODE=direct.

Проблема, ради которой это написано: pgvector и neo4j импортируются не нами,
а btc-graph, причём лениво — уже внутри сохранения. Если запустить из venv,
где их нет, кандидат оценивается, «отправляется», и только в логе видно
ModuleNotFoundError из чужого пакета. Проверка должна называть проблему
раньше и своими словами.
"""
from __future__ import annotations

import importlib.util
import sys

from btcproc.sink import graph_sink


def test_all_dependencies_present_by_default():
    assert graph_sink.missing_direct_dependencies() == {}


def test_missing_dependency_is_detected(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "pgvector":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    missing = graph_sink.missing_direct_dependencies()
    assert "pgvector" in missing
    assert "neo4j" not in missing


def test_hint_names_the_interpreter():
    """
    Подсказка обязана указывать конкретный python: вся суть ошибки в том,
    что пакеты стоят в другом интерпретаторе.
    """
    hint = graph_sink._dependency_hint({"pgvector": "запись оценок в PostgreSQL"})

    assert sys.executable in hint
    assert "pip install -r requirements.txt" in hint
    assert "pgvector" in hint


# ── Порог качества при отправке (B1) ────────────────────────────────────────
#
# SINK_MIN_QUALITY по умолчанию не задан, и это должно означать «порог берёт
# btc-graph из профиля каждой монеты». Раньше дефолтом было 0.0 — явное число,
# которое btc-graph трактует как «перекрыть профили на весь батч», и все
# калиброванные batch.min_quality_score не применялись никогда.


def _set_threshold(monkeypatch, value):
    """SinkConfig — frozen dataclass, поэтому подменяем его целиком."""
    import dataclasses

    monkeypatch.setattr(
        graph_sink.config,
        "sink",
        dataclasses.replace(graph_sink.config.sink, min_quality_score=value),
    )


def test_unset_threshold_reaches_btc_graph_as_none(monkeypatch):
    captured = {}

    class FakePipeline:
        @staticmethod
        def run_batch_pipeline(payload, **kwargs):
            captured.update(kwargs)
            return []

    monkeypatch.setattr(graph_sink, "_load_btc_graph", lambda: FakePipeline)
    _set_threshold(monkeypatch, None)

    graph_sink.emit_direct([{"candidate_id": "x"}])

    assert captured["min_quality_score"] is None


def test_explicit_threshold_reaches_btc_graph_as_number(monkeypatch):
    captured = {}

    class FakePipeline:
        @staticmethod
        def run_batch_pipeline(payload, **kwargs):
            captured.update(kwargs)
            return []

    monkeypatch.setattr(graph_sink, "_load_btc_graph", lambda: FakePipeline)
    _set_threshold(monkeypatch, 0.7)

    graph_sink.emit_direct([{"candidate_id": "x"}])

    assert captured["min_quality_score"] == 0.7


def test_http_payload_omits_unset_threshold(monkeypatch):
    """
    В HTTP-режиме «не задано» выражается отсутствием ключа: BatchInput
    в btc-graph объявляет min_quality_score как float | None с дефолтом None.
    """
    sent = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return []

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json):
            sent.update(json)
            return FakeResponse()

    monkeypatch.setattr(graph_sink.httpx, "Client", FakeClient)
    _set_threshold(monkeypatch, None)

    graph_sink.emit_http([{"candidate_id": "x"}])
    assert "min_quality_score" not in sent

    _set_threshold(monkeypatch, 0.5)
    graph_sink.emit_http([{"candidate_id": "x"}])
    assert sent["min_quality_score"] == 0.5


def test_env_parses_empty_as_none_and_number_as_number():
    from btcproc.config import _env_float_optional

    import os

    os.environ["_TEST_MIN_Q"] = ""
    assert _env_float_optional("_TEST_MIN_Q") is None
    os.environ["_TEST_MIN_Q"] = "0.55"
    assert _env_float_optional("_TEST_MIN_Q") == 0.55
    del os.environ["_TEST_MIN_Q"]
