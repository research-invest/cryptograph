"""
Публичное API чтения: ключ и форма выдачи.

В базу тесты не ходят — `src.db` подменяется заглушками через monkeypatch на
модуль. Проверяется ровно то, что можно проверить без стека: кого пускают,
кого нет, и во что превращается строка таблицы `candidates`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api import auth, public
from src.api.routes import app
from tests.conftest import REFERENCE_PAYLOAD

KEY = "b" * 40


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv(auth.ENV_VAR, f"reader:{KEY}")
    return TestClient(app)


def _record(**overrides):
    """Строка таблицы candidates в том виде, в каком её видит роутер."""
    base = dict(
        candidate_id=REFERENCE_PAYLOAD["candidate_id"],
        symbol="BTCUSDT",
        evaluated_at=datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc),
        quality_score=0.81,
        quality_score_baseline=0.79,
        rating="STRONG",
        direction="long",
        scoring_profile="BTCUSDT@2",
        profile_fingerprint="abc123",
        horizon="24h",
        sample_size=1339,
        valid_label_pct=0.8596,
        repeatability_months=19,
        monthly_concentration=0.0999,
        long_outcome_share=0.7446,
        outcome_skew=0.4891,
        context_status="stale",
        transition_id="42->1",
        transition_rarity="common",
        current_group_id=1.0,
        previous_group_id=42.0,
        event_block_id="event_block_098200",
        family_key="1.0|42->1|event_block_098200|long_skew",
        configuration_hash="0f8928cb2fc1547b",
        warning_flags=["stale_context"],
        raw_payload=dict(REFERENCE_PAYLOAD),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def stub_db(monkeypatch):
    """Подменяет репозиторий и сессию: тестам не нужен PostgreSQL."""
    import contextlib
    import sys
    import types

    rows = [_record()]
    calls = {}

    repo = types.ModuleType("src.db.candidate_repo")

    def list_candidates(session, **kwargs):
        calls.update(kwargs)
        return rows, len(rows)

    def get_by_id(session, symbol, candidate_id):
        for row in rows:
            if (row.symbol, row.candidate_id) == (symbol, candidate_id):
                return row
        return None

    repo.list_candidates = list_candidates
    repo.get_by_id = get_by_id
    repo.list_symbols = lambda session: ["BTCUSDT"]

    connection = types.ModuleType("src.db.connection")
    connection.get_session = lambda: contextlib.nullcontext(object())

    monkeypatch.setitem(sys.modules, "src.db.candidate_repo", repo)
    monkeypatch.setitem(sys.modules, "src.db.connection", connection)
    db_pkg = sys.modules.get("src.db")
    if db_pkg is not None:
        monkeypatch.setattr(db_pkg, "candidate_repo", repo, raising=False)
        monkeypatch.setattr(db_pkg, "connection", connection, raising=False)
    return SimpleNamespace(rows=rows, calls=calls)


# ─── Ключ ─────────────────────────────────────────────────────────────────────

def test_without_key_is_401(client):
    assert client.get("/api/v1/ping").status_code == 401


def test_wrong_key_is_401(client):
    assert client.get("/api/v1/ping", headers={"X-API-Key": "wrong-key"}).status_code == 401


def test_header_key_passes(client):
    response = client.get("/api/v1/ping", headers={"X-API-Key": KEY})
    assert response.status_code == 200
    assert response.json()["key"] == "reader"


def test_bearer_key_passes(client):
    response = client.get("/api/v1/ping", headers={"Authorization": f"Bearer {KEY}"})
    assert response.status_code == 200


def test_empty_api_keys_closes_the_api_instead_of_opening_it(monkeypatch):
    """
    Не настроенный API отвечает 503, а не пускает всех.

    Самая дорогая из возможных ошибок конфигурации: `API_KEYS=` в .env
    выглядит безобидно и молча открыл бы наружу всю выдачу.
    """
    monkeypatch.setenv(auth.ENV_VAR, "")
    response = TestClient(app).get("/api/v1/ping", headers={"X-API-Key": KEY})
    assert response.status_code == 503


def test_health_stays_open(client):
    """`/health` ключом не закрыт — иначе healthcheck контейнера упрётся в 401."""
    assert client.get("/health").status_code == 200


def test_keys_are_parsed_with_and_without_labels():
    keys = auth.parse_keys("reader:aaa, bbb ,, ccc:")
    assert keys["aaa"] == "reader"
    assert keys["bbb"] == "key2"
    # «ccc:» без секрета — не ключ, а мусор; пустой строкой пускать нельзя.
    assert "" not in keys


# ─── Выдача ───────────────────────────────────────────────────────────────────

def test_list_returns_scores_and_derived_metrics(client, stub_db):
    response = client.get(
        "/api/v1/candidates", params={"symbol": "btcusdt"}, headers={"X-API-Key": KEY}
    )
    assert response.status_code == 200
    body = response.json()

    assert body["symbol"] == "BTCUSDT"
    assert body["total"] == 1
    item = body["items"][0]
    assert item["quality_score"] == 0.81
    assert item["rating"] == "STRONG"
    # win_rate и F/A ratio колонками не хранятся — считаются из raw_payload
    # единственными функциями перевода.
    assert item["win_rate"] == pytest.approx(REFERENCE_PAYLOAD["long_outcome_share"])
    assert item["favorable_adverse_ratio"] == pytest.approx(
        REFERENCE_PAYLOAD["long_favorable_adverse_ratio_p70_p80"]
    )
    assert stub_db.calls["symbol"] == "BTCUSDT"


def test_symbol_all_is_explicit_and_sorts_by_baseline(client, stub_db):
    response = client.get(
        "/api/v1/candidates",
        params={"symbol": "all", "order": "score"},
        headers={"X-API-Key": KEY},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "all"
    assert stub_db.calls["symbol"] is None
    # Предупреждение о несравнимости профильных score обязано ехать в теле:
    # клиент читает JSON, а не документацию.
    assert any("baseline" in note for note in body["notes"])


def test_filters_reach_the_repository(client, stub_db):
    response = client.get(
        "/api/v1/candidates",
        params={
            "symbol": "BTCUSDT",
            "rating": "strong,moderate",
            "direction": "long",
            "min_score": 0.5,
            "hours": 24,
            "limit": 10,
            "offset": 5,
        },
        headers={"X-API-Key": KEY},
    )
    assert response.status_code == 200
    assert stub_db.calls["ratings"] == ["STRONG", "MODERATE"]
    assert stub_db.calls["direction"] == "long"
    assert stub_db.calls["min_quality_score"] == 0.5
    assert stub_db.calls["limit"] == 10
    assert stub_db.calls["offset"] == 5
    assert stub_db.calls["since"] is not None


def test_unknown_rating_is_422(client, stub_db):
    response = client.get(
        "/api/v1/candidates",
        params={"symbol": "BTCUSDT", "rating": "AWESOME"},
        headers={"X-API-Key": KEY},
    )
    assert response.status_code == 422


def test_limit_is_capped(client, stub_db):
    response = client.get(
        "/api/v1/candidates",
        params={"symbol": "BTCUSDT", "limit": public.MAX_LIMIT + 1},
        headers={"X-API-Key": KEY},
    )
    assert response.status_code == 422


def test_detail_hides_raw_payload_by_default(client, stub_db):
    url = f"/api/v1/candidates/BTCUSDT/{REFERENCE_PAYLOAD['candidate_id']}"
    plain = client.get(url, headers={"X-API-Key": KEY}).json()
    assert plain["raw_payload"] is None

    full = client.get(url, params={"include_raw": True}, headers={"X-API-Key": KEY}).json()
    assert full["raw_payload"]["sample_size"] == REFERENCE_PAYLOAD["sample_size"]


def test_detail_404(client, stub_db):
    response = client.get("/api/v1/candidates/BTCUSDT/no-such-id", headers={"X-API-Key": KEY})
    assert response.status_code == 404


def test_range_block_is_null_when_generator_said_nothing(client, stub_db):
    """Отсутствие полей размаха — норма, а не аномалия: null, а не ошибка."""
    item = client.get(
        "/api/v1/candidates", params={"symbol": "BTCUSDT"}, headers={"X-API-Key": KEY}
    ).json()["items"][0]
    assert item["range"]["lift"] is None


def test_range_block_is_passed_through(client, stub_db):
    payload = dict(REFERENCE_PAYLOAD)
    payload.update(range_lift=1.31, range_regime="wide",
                   expected_range_ratio_p50=1.2, expected_range_ratio_p90=2.4)
    stub_db.rows[:] = [_record(raw_payload=payload)]

    item = client.get(
        "/api/v1/candidates", params={"symbol": "BTCUSDT"}, headers={"X-API-Key": KEY}
    ).json()["items"][0]
    assert item["range"]["lift"] == 1.31
    assert item["range"]["regime"] == "wide"


def test_symbols_marks_coins_without_profile(client, stub_db):
    body = client.get("/api/v1/symbols", headers={"X-API-Key": KEY}).json()
    assert body["symbols"][0]["symbol"] == "BTCUSDT"
    assert "has_profile" in body["symbols"][0]
