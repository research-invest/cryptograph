"""
Общие фикстуры тестов.

Тесты не требуют ни PostgreSQL, ни Redis, ни Neo4j, ни ключа Anthropic:
проверяется чистая логика парсинга, валидации, скоринга и фильтрации.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# src.db.connection создаёт engine прямо на импорте. Уводим его на in-memory
# sqlite: тесты не нуждаются в psycopg2 и гарантированно не ходят в боевую БД.
# Значение должно быть выставлено ДО первого импорта src.db.*.
os.environ["DATABASE_URL"] = "sqlite://"

# Хранилища выключены по умолчанию — тесты, которым нужен _persist,
# включают их точечно через monkeypatch.
os.environ.setdefault("USE_DB", "false")
os.environ.setdefault("USE_REDIS", "false")
os.environ.setdefault("USE_GRAPH", "false")

def _stub_module(name: str, **attrs) -> None:
    """
    Регистрирует заглушку модуля, если настоящий пакет не установлен.

    Тесты проверяют чистую логику, а SDK внешних сервисов импортируются на
    верхнем уровне (`src.agent.pipeline` → `llm_node` → `anthropic` и т.д.).
    Там, где реальный пакет есть, заглушка не подставляется.
    """
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    sys.modules[name] = module

    # Для составных имён вида «pgvector.sqlalchemy» регистрируем и родителя.
    if "." in name:
        parent_name, _, child = name.rpartition(".")
        parent = sys.modules.get(parent_name) or types.ModuleType(parent_name)
        setattr(parent, child, module)
        sys.modules[parent_name] = parent


class _StubError(Exception):
    """Базовое исключение для заглушек внешних SDK."""


_stub_module("anthropic", Anthropic=object, APIError=_StubError)
_stub_module("pgvector.sqlalchemy", Vector=lambda dim: None)
_stub_module(
    "neo4j",
    GraphDatabase=object,
    Driver=object,
    exceptions=types.SimpleNamespace(ServiceUnavailable=_StubError),
)
_stub_module("neo4j.exceptions", ServiceUnavailable=_StubError)
_stub_module("redis", Redis=object, RedisError=_StubError, from_url=lambda *a, **kw: None)

from src.models.candidate import Candidate  # noqa: E402


# Эталонный кандидат из ТЗ (README_agent_spec.md, Reference Case).
REFERENCE_PAYLOAD: dict = {
    "candidate_id": "245be5fb0908d59f6e89",
    "symbol": "BTCUSDT",
    "configuration_hash": "0f8928cb2fc1547b",
    "candidate_family_key": "1.0|42->1|event_block_098200|long_skew",
    "research_score": 0.9571800918456002,
    "previous_group_id": 42.0,
    "current_group_id": 1.0,
    "transition_id": "42->1",
    "current_group_age_bucket": "age_gt_120",
    "context_status": "stale",
    "trajectory_entropy": "medium",
    "transition_rarity": "common",
    "event_block_id": "event_block_098200",
    "primary_event_family": "zone_context_events",
    "event_intensity_bucket": "dense",
    "event_rarity_bucket": "uncommon",
    "signature_atom_count": 6,
    "event_family_count": 2,
    "event_block_total_rows": 23444,
    "event_block_row_share": 0.0067561570250315,
    "horizon": "24h",
    "sample_size": 1339,
    "valid_label_count": 1151,
    "invalid_label_count": 188,
    "valid_label_pct": 0.859596713965646,
    "repeatability_days": 21,
    "repeatability_months": 19,
    "monthly_concentration": 0.0999131190269331,
    "historical_bias_context": "long_skew",
    "research_side": "long",
    "long_outcome_count": 857,
    "short_outcome_count": 294,
    "long_outcome_share": 0.7445699391833188,
    "historical_outcome_skew": 0.4891398783666377,
    "p70_long_favorable_pct": 3.175384334258424,
    "p80_long_adverse_pct": 0.732807017851166,
    "long_favorable_adverse_ratio_p70_p80": 4.333179476414578,
}


@pytest.fixture
def reference_payload() -> dict:
    """Копия эталонного payload — правки в тесте не текут в соседние."""
    return dict(REFERENCE_PAYLOAD)


@pytest.fixture
def reference_candidate() -> Candidate:
    return Candidate(**REFERENCE_PAYLOAD)


@pytest.fixture
def make_candidate():
    """
    Фабрика кандидатов: берёт эталон и переопределяет нужные поля.

        c = make_candidate(context_status="fresh", sample_size=50)
    """
    def _make(**overrides) -> Candidate:
        return Candidate(**{**REFERENCE_PAYLOAD, **overrides})

    return _make
