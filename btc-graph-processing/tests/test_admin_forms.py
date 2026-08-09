"""
Формы запуска прогонов в админке.

Классическая ловушка HTML-форм: снятый чекбокс браузер не отправляет вовсе.
Пока дефолт параметра был `Form(True)`, отсутствие поля означало «включено» —
то есть выключить «качать историю» и «отправлять кандидатов» из админки было
невозможно, галки работали только в одну сторону. Проверяем оба направления.

Здесь нужен настоящий HTTP: смысл бага ровно в том, как FastAPI подставляет
дефолты отсутствующим полям формы, а прямой вызов функции это обходит.
"""
from __future__ import annotations

import pytest

from btcproc import config
from btcproc.admin import auth

fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        config, "admin",
        config.AdminConfig(user="operator", password="very-long-password-42",
                           secret_key="k" * 64, ip_allowlist=[]),
    )
    from btcproc.admin import app as admin_app

    # Авторизация и схема БД к предмету теста отношения не имеют.
    monkeypatch.setattr(auth, "current_user", lambda request: "operator")
    monkeypatch.setattr(admin_app, "init_schema", lambda: None, raising=False)
    monkeypatch.setattr(admin_app, "_guard_capacity", lambda tickers: None)

    with fastapi_testclient.TestClient(admin_app.app) as test_client:
        yield test_client


@pytest.fixture
def train_calls(monkeypatch):
    calls = []
    from btcproc.pipeline import train as train_module

    monkeypatch.setattr(train_module, "run_train", lambda **kw: calls.append(kw))
    return calls


@pytest.fixture
def live_calls(monkeypatch):
    calls = []
    from btcproc.pipeline import live as live_module

    monkeypatch.setattr(live_module, "run_live", lambda **kw: calls.append(kw))
    return calls


def test_unchecked_boxes_turn_features_off(client, train_calls):
    """Снятые галки → поля в запросе нет → прогон обязан их выключить."""
    client.post("/runs/train", data={"symbol": "BTCUSDT"}, follow_redirects=False)

    assert train_calls, "прогон должен быть поставлен в очередь"
    assert train_calls[0]["do_ingest"] is False
    assert train_calls[0]["do_emit"] is False


def test_checked_boxes_turn_features_on(client, train_calls):
    client.post(
        "/runs/train",
        data={"symbol": "BTCUSDT", "ingest": "true", "emit": "true"},
        follow_redirects=False,
    )

    assert train_calls[0]["do_ingest"] is True
    assert train_calls[0]["do_emit"] is True


def test_live_emit_checkbox_works_both_ways(client, live_calls):
    client.post("/runs/live", data={"symbol": "BTCUSDT"}, follow_redirects=False)
    assert live_calls[0]["do_emit"] is False

    client.post(
        "/runs/live", data={"symbol": "BTCUSDT", "emit": "true"}, follow_redirects=False
    )
    assert live_calls[1]["do_emit"] is True


def test_empty_lookback_means_auto(client, live_calls):
    """
    Пустое поле «свежесть» = авто-режим. Раньше поле было int с дефолтом 240 и
    пустым быть не могло: админка всегда задавала окно явно, а короткое окно
    после долгой паузы оставляет дыру в кандидатах навсегда.
    """
    client.post(
        "/runs/live", data={"symbol": "BTCUSDT", "lookback": ""}, follow_redirects=False
    )

    assert live_calls[0]["lookback_minutes"] is None


def test_explicit_lookback_is_passed_through(client, live_calls):
    client.post(
        "/runs/live", data={"symbol": "BTCUSDT", "lookback": "60"}, follow_redirects=False
    )

    assert live_calls[0]["lookback_minutes"] == 60
