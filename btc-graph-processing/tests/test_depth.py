"""
Снимки стакана: единственные данные проекта, которые нельзя докачать задним
числом.

Отсюда и то, что здесь проверяется. Не «правильно ли посчитан дисбаланс» —
никаких величин из стакана пока не считается вовсе, — а ровно те три способа
порвать ряд, после которых через год выяснится, что копилось не то:

1. **неполный ответ не должен попадать в базу.** Пустая сторона стакана у
   торгуемой пары означает сломанный ответ, а не нулевой объём, и через год
   отличить одно от другого будет нечем;
2. **одна недоступная монета не отменяет остальные.** Ценность ряда ровно в
   непрерывности, и терять пять снимков из-за шестого нельзя;
3. **ноль снимков — ненулевой код возврата.** Иначе крон рапортует об успехе
   ровно тогда, когда ряд рвётся.

Сеть не нужна: httpx.Client подменяется.
"""
from __future__ import annotations

import datetime as dt
import json

import pytest

from btcproc.ingest import depth


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeClient:
    """Клиент, отвечающий по URL. Всё, что не описано, считается сбоем сети."""

    def __init__(self, by_url: dict):
        self.by_url = by_url
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None):
        self.calls.append((url, params or {}))
        payload = self.by_url.get(url)
        if isinstance(payload, Exception):
            raise payload
        if payload is None:
            raise RuntimeError(f"нет ответа для {url}")
        return FakeResponse(payload)


BINANCE_OK = {"bids": [["100.5", "2"], ["100.0", "3"]],
              "asks": [["101.0", "1"], ["101.5", "4"]]}
BYBIT_OK = {"retCode": 0, "result": {"b": [["10.5", "7"]], "a": [["10.6", "9"]]}}


def test_snapshot_reads_both_venues():
    """Две биржи отдают разную форму ответа — наружу обе дают одну строку."""
    now = dt.datetime(2026, 8, 24, 12, 0, 30, 123456, tzinfo=dt.timezone.utc)

    client = FakeClient({depth.BINANCE_URL: BINANCE_OK})
    row = depth.snapshot("BTCUSDT", "binance_spot", client, now=now)
    symbol, ts, venue, bid, ask, bid_vol, ask_vol, levels = row
    assert (symbol, venue, bid, ask) == ("BTCUSDT", "binance_spot", 100.5, 101.0)
    assert (bid_vol, ask_vol) == (5.0, 5.0)
    assert ts.microsecond == 0, "время усекается до секунды"
    assert json.loads(levels)["b"][0] == [100.5, 2.0]

    client = FakeClient({depth.BYBIT_URL: BYBIT_OK})
    row = depth.snapshot("HYPEUSDT", "bybit_spot", client, now=now)
    assert row[3] == 10.5 and row[4] == 10.6


def test_bybit_error_arrives_with_http_200():
    """
    У Bybit ошибка приезжает кодом 200 с ненулевым retCode.

    Без проверки `result` оказался бы пустым, и снимок лёг бы в базу нулевым
    вместо того, чтобы отсутствовать. Через год такую строку не отличить от
    настоящего пустого стакана.
    """
    client = FakeClient({depth.BYBIT_URL: {"retCode": 10001, "retMsg": "params error"}})
    with pytest.raises(ValueError, match="retCode"):
        depth.snapshot("HYPEUSDT", "bybit_spot", client)


def test_empty_side_is_skipped_not_stored():
    """Пустая сторона — сломанный ответ, а не нулевой объём."""
    client = FakeClient({depth.BINANCE_URL: {"bids": [], "asks": [["1", "1"]]}})
    assert depth.snapshot("BTCUSDT", "binance_spot", client) is None


def test_one_broken_symbol_does_not_lose_the_others(monkeypatch):
    """Ценность ряда в непрерывности: одна недоступная пара не отменяет пять."""
    saved: list = []

    class Spec:
        def __init__(self, ticker, venue):
            self.ticker, self.venue = ticker, venue

    monkeypatch.setattr("btcproc.symbols.resolve_many",
                        lambda names, all_symbols: [Spec("BTCUSDT", "binance_spot"),
                                                    Spec("ETHUSDT", "binance_spot")])
    monkeypatch.setattr(depth, "bulk_upsert",
                        lambda table, columns, rows, conflict: saved.extend(rows) or len(rows))

    calls = {"n": 0}

    def flaky(symbol, venue, client, now=None):
        calls["n"] += 1
        if symbol == "BTCUSDT":
            raise RuntimeError("сеть недоступна")
        return (symbol, dt.datetime.now(dt.timezone.utc), venue, 1.0, 2.0, 3.0, 4.0, "{}")

    monkeypatch.setattr(depth, "snapshot", flaky)
    monkeypatch.setattr(depth.httpx, "Client", lambda **kw: _NullClient())

    assert depth.collect(None, True) == 1
    assert calls["n"] == 2, "вторая монета обязана быть опрошена после сбоя первой"
    assert saved[0][0] == "ETHUSDT"


class _NullClient:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_unknown_venue_fails_loudly():
    """Опечатка в площадке — ошибка, а не молчаливый пропуск монеты."""
    with pytest.raises(ValueError, match="не реализован"):
        depth.snapshot("XXX", "kraken_spot", FakeClient({}))
