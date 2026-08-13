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


# ─── Профили в тестах ─────────────────────────────────────────────────────────
#
# По умолчанию вся сюита считает по ЗАМОРОЖЕННЫМ профилям из
# tests/fixtures/profiles: там `_default` — копия боевого, а BTCUSDT — пустое
# наследование от него.
#
# Развязка нужна вот зачем. Скорер, вызванный без явного профиля, резолвит его
# по символу кандидата, а эталонный кандидат из ТЗ — BTCUSDT. Пока боевой
# BTCUSDT был пустым наследованием, тесты ступеней `_default` проходили по
# совпадению. Первая же реальная калибровка BTC уронила 66 случаев, не имеющих
# к калибровке отношения: ступени, pipeline, формулировки, фильтры.
#
# Калибровка по правилу 6 — штатное повторяющееся событие. Тесты механики
# обязаны его переживать. Тем немногим тестам, которым нужны именно боевые
# профили (снапшот базовой линейки, sanity-якоря, реестр), их отдаёт фикстура
# `shipped_profiles`.

FROZEN_PROFILES = Path(__file__).parent / "fixtures" / "profiles"


@pytest.fixture(autouse=True, scope="session")
def _frozen_profiles() -> None:
    from src.config import profiles

    os.environ["PROFILES_DIR"] = str(FROZEN_PROFILES)
    profiles.reload_profiles()
    yield
    os.environ.pop("PROFILES_DIR", None)
    profiles.reload_profiles()


@pytest.fixture
def shipped_profiles():
    """
    Боевые профили из config/symbols на время одного теста.

    Нужна там, где проверяется именно то, что поставляется: снапшот базовой
    линейки, sanity-якоря монет, содержимое реестра.
    """
    from src.config import profiles

    os.environ.pop("PROFILES_DIR", None)
    profiles.reload_profiles()
    yield profiles
    os.environ["PROFILES_DIR"] = str(FROZEN_PROFILES)
    profiles.reload_profiles()


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


@pytest.fixture(scope="session")
def eth_profile():
    """
    Профиль монеты с короткой историей и меньшей ликвидностью.

    Собирается в памяти, а не читается из config/symbols: тесты не должны
    зависеть от того, заведена ли ETH в боевом реестре. Ступени сдвинуты
    ровно настолько, чтобы отличие от _default было видно на глаз.
    """
    from src.config.profiles import ScoringProfile, _deep_merge, default_profile

    base = default_profile().model_dump(mode="json")
    override = {
        "symbol": "ETHUSDT",
        "version": 2,
        "description": "Тестовый профиль альткоина",
        "statistical": {
            "sample_size": {"mode": "higher_better",
                            "ladder": [[400, 1.0], [200, 0.7], [80, 0.4]], "floor": 0.1},
            "repeatability_months": {"mode": "higher_better",
                                     "ladder": [[8, 1.0], [5, 0.7], [3, 0.4]], "floor": 0.1},
        },
        "weights": {"statistical": 0.35, "directional": 0.30,
                    "context": 0.20, "rarity": 0.15},
        "rating": {"strong_min": 0.70, "moderate_min": 0.50},
        "batch": {"min_quality_score": 0.50, "fresh_bonus": 0.05},
        "validator": {"very_small_sample_size": 60, "low_repeatability_months": 2},
        "llm": {"market_hint": "ETH — история короче, выборки меньше"},
    }
    return ScoringProfile(**_deep_merge(base, override))


# ─── Снапшот скорера ──────────────────────────────────────────────────────────
# Набор кандидатов, на котором зафиксированы значения ScoreBreakdown до перехода
# скорера на профили (шаг 4.2). Генерация обязана быть детерминированной и
# неизменной: любая правка этой функции обесценивает снапшот.

SNAPSHOT_SIZE = 200


def make_snapshot_candidates(count: int = SNAPSHOT_SIZE) -> list[Candidate]:
    """
    Псевдослучайная выборка кандидатов, покрывающая все ступени всех осей.

    Значения берутся из фиксированного генератора: между запусками и между
    машинами набор один и тот же, поэтому снапшот сравним побайтово.
    """
    import random

    rng = random.Random(20260808)

    age_buckets = ["age_lt_30", "age_30_60", "age_60_120", "age_gt_120"]
    entropies = ["low", "medium", "high"]
    rarities = ["rare", "uncommon", "common"]
    intensities = ["sparse", "moderate", "dense"]
    biases = ["long_skew", "short_skew", "neutral"]

    candidates = []
    for i in range(count):
        valid_pct = round(rng.uniform(0.50, 1.0), 6)
        sample = rng.choice([10, 50, 99, 100, 200, 201, 500, 501, 1000, 1001, 5000])
        valid_count = int(sample * valid_pct)
        long_count = int(valid_count * rng.uniform(0.0, 1.0))
        long_share = long_count / valid_count if valid_count else 0.0
        favorable = round(rng.uniform(0.1, 8.0), 6)
        adverse = round(rng.uniform(0.1, 4.0), 6)

        candidates.append(Candidate(
            candidate_id=f"snapshot_{i:03d}",
            symbol="BTCUSDT",
            configuration_hash=f"hash_{i:03d}",
            candidate_family_key=f"family_{i % 17}",
            research_score=round(rng.uniform(0.0, 1.0), 6),
            previous_group_id=float(rng.randint(1, 40)),
            current_group_id=float(rng.randint(1, 40)),
            transition_id=f"{rng.randint(1, 40)}->{rng.randint(1, 40)}",
            current_group_age_bucket=rng.choice(age_buckets),
            context_status=rng.choice(["fresh", "stale"]),
            trajectory_entropy=rng.choice(entropies),
            transition_rarity=rng.choice(rarities),
            event_block_id=f"event_block_{i % 23:06d}",
            primary_event_family=rng.choice([None, "zone_context_events", "breakout_events"]),
            event_intensity_bucket=rng.choice(intensities),
            event_rarity_bucket=rng.choice(rarities),
            signature_atom_count=rng.randint(0, 12),
            event_family_count=rng.randint(0, 7),
            event_block_total_rows=rng.randint(10, 50_000),
            event_block_row_share=round(rng.uniform(0.0, 1.0), 6),
            horizon="24h",
            sample_size=sample,
            valid_label_count=valid_count,
            invalid_label_count=sample - valid_count,
            valid_label_pct=valid_pct,
            repeatability_days=rng.randint(0, 400),
            repeatability_months=rng.choice([0, 3, 6, 7, 12, 13, 15, 16, 24, 36]),
            monthly_concentration=round(rng.uniform(0.0, 1.0), 6),
            historical_bias_context=rng.choice(biases),
            research_side=rng.choice(["long", "short"]),
            long_outcome_count=long_count,
            short_outcome_count=valid_count - long_count,
            long_outcome_share=long_share,
            historical_outcome_skew=round(2 * long_share - 1, 6),
            p70_long_favorable_pct=favorable,
            p80_long_adverse_pct=adverse,
            long_favorable_adverse_ratio_p70_p80=round(favorable / adverse, 6),
        ))

    return candidates


@pytest.fixture(scope="session")
def snapshot_candidates() -> list[Candidate]:
    return make_snapshot_candidates()
