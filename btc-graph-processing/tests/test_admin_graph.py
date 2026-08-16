"""
Селектор прогона на странице графа.

Регрессия: `/graph` брал `runs_repo.list_runs(20, symbol)` без фильтра по
`kind` и полагался на то, что шаблон сам отсеет всё, кроме train. Train пишет
market_groups/transitions раз в неделю, live — каждые 30 минут по крону:
через ~10 часов после недельного retrain (20 live-прогонов) train вытесняется
из топ-20 ЛЮБЫХ прогонов монеты, и селектор показывает пустой список до
следующего train — хотя граф на месте и прекрасно открывается по прямой
ссылке `?run=`. Заметили на бою вскоре после планового retrain.

Фикс — фильтровать по `kind="train"` в самом запросе к БД, а не только в
Jinja-цикле.
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

    monkeypatch.setattr(auth, "current_user", lambda request: "operator")
    monkeypatch.setattr(admin_app, "init_schema", lambda: None, raising=False)

    with fastapi_testclient.TestClient(admin_app.app) as test_client:
        yield test_client


def test_graph_page_asks_for_train_runs_only(client, monkeypatch):
    """
    Даже если бы Jinja-фильтр в шаблоне пропал, запрос к БД уже должен нести
    `kind="train"` — иначе список за пределами свежего окна после retrain
    неизбежно опустеет вслед за приливом live-прогонов.
    """
    from btcproc.admin import app as admin_app

    calls = []

    def fake_list_runs(limit, symbol, kind=None):
        calls.append({"limit": limit, "symbol": symbol, "kind": kind})
        return []

    monkeypatch.setattr(admin_app.runs_repo, "list_runs", fake_list_runs)
    monkeypatch.setattr(admin_app, "_latest_train_id", lambda symbol: None)

    resp = client.get("/graph", params={"symbol": "BTCUSDT"})

    assert resp.status_code == 200
    assert calls, "list_runs должен был быть вызван"
    assert calls[0]["kind"] == "train"


def test_graph_run_selector_survives_a_flood_of_live_runs(client, monkeypatch):
    """
    Сквозной сценарий: у монеты один train и полсотни live-прогонов ПОСЛЕ
    него (типичная картина через день-два после недельного retrain). Опция
    train должна остаться в разметке страницы, а не потеряться в топ-20.
    """
    from btcproc.admin import app as admin_app
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 8, 16, 3, 20, tzinfo=timezone.utc)
    train_run = {
        "run_id": 1237, "kind": "train", "status": "done", "started_at": base,
    }
    live_runs = [
        {
            "run_id": 1300 + i, "kind": "live", "status": "done",
            "started_at": base + timedelta(minutes=30 * (i + 1)),
        }
        for i in range(50)  # 25 часов live-прогонов каждые полчаса
    ]
    all_runs = [train_run, *live_runs]

    def fake_list_runs(limit, symbol, kind=None):
        rows = [r for r in all_runs if kind is None or r["kind"] == kind]
        rows.sort(key=lambda r: r["started_at"], reverse=True)
        return rows[:limit]

    monkeypatch.setattr(admin_app.runs_repo, "list_runs", fake_list_runs)
    monkeypatch.setattr(admin_app, "_latest_train_id", lambda symbol: 1237)

    resp = client.get("/graph", params={"symbol": "BTCUSDT"})

    assert resp.status_code == 200
    assert "#1237" in resp.text, (
        "train-прогон должен остаться в селекторе, даже когда live-прогонов "
        "после него набралось больше 20"
    )
