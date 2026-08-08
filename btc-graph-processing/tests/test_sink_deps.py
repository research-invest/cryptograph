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
