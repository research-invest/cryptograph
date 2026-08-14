"""
Тесты загрузчика внешних дневных рядов (btcproc/ingest/external.py).

Ни в БД, ни в сеть не ходят — как и весь остальной набор: HTTP подменяется
`httpx.MockTransport`, запись в БД — monkeypatch на `connect`/`fetch_all`.
"""
from __future__ import annotations

import json
from contextlib import contextmanager

import httpx
import pandas as pd
import pytest

from btcproc.ingest import external


def _transport(payload: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


def _fng_payload(rows: list[tuple[str, str, int]]) -> dict:
    """rows: (value, classification, unix_timestamp)."""
    return {
        "name": "Fear and Greed Index",
        "data": [
            {
                "value": value,
                "value_classification": classification,
                "timestamp": str(ts),
                "time_until_update": "3600",
            }
            for value, classification, ts in rows
        ],
        "metadata": {"error": None},
    }


def test_fetch_parses_fields_and_sorts_chronologically():
    # Поставщик отдаёт свежее первым — проверяем, что мы разворачиваем.
    payload = _fng_payload([
        ("50", "Neutral", 1621382400),   # 2021-05-19 00:00 UTC
        ("29", "Fear", 1621296000),      # 2021-05-18 00:00 UTC
    ])
    client = httpx.Client(transport=_transport(payload))
    frame = external.fetch_fear_greed(client)

    assert list(frame["day"]) == [pd.Timestamp("2021-05-18").date(),
                                   pd.Timestamp("2021-05-19").date()]
    assert list(frame["value"]) == [29.0, 50.0]
    assert frame.iloc[0]["meta"]["value_classification"] == "Fear"
    # Служебные поля (сама дата, значение) в meta не дублируются.
    assert "timestamp" not in frame.iloc[0]["meta"]
    assert "value" not in frame.iloc[0]["meta"]


def test_fetch_handles_empty_history():
    client = httpx.Client(transport=_transport({"data": [], "metadata": {}}))
    frame = external.fetch_fear_greed(client)
    assert frame.empty


# ── Ревизии истории (гейт D, docs/task_fear_greed.md §3.3) ──────────────────
@pytest.fixture
def captured_upsert(monkeypatch):
    """Подменяет запись в БД и возвращает список вставленных строк."""
    captured: dict = {}

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class FakeConn:
        def cursor(self):
            return FakeCursor()

    @contextmanager
    def fake_connect():
        yield FakeConn()

    def fake_execute_values(cur, sql, rows):
        captured["rows"] = list(rows)

    monkeypatch.setattr(external, "connect", fake_connect)
    monkeypatch.setattr(external, "execute_values", fake_execute_values)
    return captured


def _frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame({
        "day": [pd.Timestamp(d).date() for d, _ in rows],
        "value": [v for _, v in rows],
        "meta": [{} for _ in rows],
    })


def test_store_detects_no_revision_when_values_match(monkeypatch, captured_upsert):
    monkeypatch.setattr(
        external, "fetch_all",
        lambda sql, params: [{"day": pd.Timestamp("2021-05-18").date(), "value": 29.0}],
    )
    result = external.store_external_daily(_frame([("2021-05-18", 29.0)]), "fear_greed")

    assert result["revised_days"] == 0
    assert result["rows"] == 1
    assert captured_upsert["rows"][0][2] == 29.0


def test_store_flags_revised_days_but_still_writes_the_fresh_value(monkeypatch, captured_upsert, caplog):
    """
    Поставщик задним числом пересчитал историю: значение сохранённого дня
    поменялось. Расхождение обязано попасть в лог с числом дней (не
    молчаливый upsert), но запись всё равно идёт — свежий снимок авторитетнее.
    """
    monkeypatch.setattr(
        external, "fetch_all",
        lambda sql, params: [{"day": pd.Timestamp("2021-05-18").date(), "value": 20.0}],
    )
    with caplog.at_level("WARNING"):
        result = external.store_external_daily(_frame([("2021-05-18", 29.0)]), "fear_greed")

    assert result["revised_days"] == 1
    assert result["revised"][0] == {
        "day": pd.Timestamp("2021-05-18").date(), "old": 20.0, "new": 29.0,
    }
    assert any("разошлись" in message for message in caplog.messages)
    # Значение всё равно записано свежим, а не отброшено.
    assert captured_upsert["rows"][0][2] == 29.0


def test_store_empty_frame_is_a_noop(captured_upsert):
    result = external.store_external_daily(pd.DataFrame(columns=["day", "value", "meta"]), "fear_greed")
    assert result == {"rows": 0, "revised_days": 0, "revised": []}
    assert "rows" not in captured_upsert


def test_sync_drops_days_before_history_start(monkeypatch, captured_upsert):
    """История индекса начинается 2018-02-01 — более ранние сутки не пишутся."""
    payload = _fng_payload([
        ("50", "Neutral", 1514764800),   # 2018-01-01 — до начала истории
        ("29", "Fear", 1517443200),      # 2018-02-01 — начало истории
    ])
    real_client_cls = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client_cls(transport=_transport(payload)))
    monkeypatch.setattr(external, "fetch_all", lambda sql, params: [])

    result = external.sync_fear_greed()

    assert result["rows"] == 1
    assert captured_upsert["rows"][0][1] == pd.Timestamp("2018-02-01").date()
