"""
Тесты источника хвоста tv-quotes-api.

Он заведён ради обхода геоблокировки: REST Bybit отвечает 403 с боевого
адреса, и без замены хвост HYPEUSDT отставал до суток. Отсюда же характер
проверок — все ошибки здесь тихие:

* незакрытый бар, попавший в `ohlcv`, выглядит как обычный, но его `close`
  снят на середине интервала;
* отказ источника, принятый за «баров нет», молча отменяет заход в REST —
  монета остаётся без хвоста там, где он был доступен;
* включённый ключ в окружении превращает офлайновый тест в поход в сеть.

В сеть и в БД, как и остальные тесты проекта, не ходят.
"""
from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from btcproc import config
from btcproc.ingest import bybit, tvquotes
from btcproc.ingest.bars import OHLCV_COLUMNS


@pytest.fixture
def configured(monkeypatch):
    """Источник настроен. `config.data` заморожен — подменяется копией целиком."""
    monkeypatch.setattr(
        config, "data",
        dataclasses.replace(config.data, tvq_url="https://tvq.test",
                            tvq_api_key="test-key", tvq_timeout=5.0),
    )


def _response(candles, status=200, meta=None):
    class FakeResponse:
        status_code = status
        text = "тело"

        def json(self):
            if status != 200:
                return {"error": {"code": "bad_request", "message": "плохой символ"}}
            return {"candles": candles,
                    "meta": meta or {"cached": False, "stale": False, "age_sec": 0}}

    return FakeResponse()


def _candle(ts: pd.Timestamp, price: float = 100.0, volume: float = 5.0) -> dict:
    return {"time": int(ts.timestamp()), "datetime": ts.isoformat(),
            "open": price, "high": price + 1, "low": price - 1,
            "close": price + 0.5, "volume": volume}


# ─── Разбор ответа ────────────────────────────────────────────────────────────

def test_unclosed_bar_is_dropped(configured, monkeypatch):
    """
    Источник отдаёт текущий бар наравне с закрытыми — у него это штатно.

    Для нас это единственный бар, чей `close` ещё изменится. Попав в `ohlcv`,
    он неотличим от закрытого, и признаки до следующего прогона считаются по
    цене, которой на закрытии не было.
    """
    now = pd.Timestamp.now(tz="UTC").floor("15min")
    candles = [_candle(now - pd.Timedelta(minutes=30)),
               _candle(now - pd.Timedelta(minutes=15)),
               _candle(now)]  # текущий, ещё идёт
    monkeypatch.setattr(tvquotes.httpx, "get", lambda *a, **kw: _response(candles))

    frame = tvquotes.fetch_bars("HYPEUSDT", "15m", "bybit_spot", 3)

    assert len(frame) == 2, "незакрытый бар обязан быть отброшен"
    assert frame["ts"].max() == now - pd.Timedelta(minutes=15)


def test_frame_matches_the_ohlcv_schema(configured, monkeypatch):
    """
    Колонки — ровно схема `ohlcv`, иначе `store_bars` упадёт при вставке.

    `trades` и `taker_buy_base` пустые сознательно: сторона агрессора есть
    только в тиковых архивах. `quote_volume` пуст по той же причине —
    оборот в котируемой валюте источник не отдаёт, а посчитать его как
    `volume × цена` значило бы положить оценку в колонку с измерениями.
    """
    ts = pd.Timestamp.now(tz="UTC").floor("15min") - pd.Timedelta(hours=1)
    monkeypatch.setattr(tvquotes.httpx, "get", lambda *a, **kw: _response([_candle(ts)]))

    frame = tvquotes.fetch_bars("HYPEUSDT", "15m", "bybit_spot", 1)

    assert list(frame.columns) == OHLCV_COLUMNS
    assert frame["symbol"].iloc[0] == "HYPEUSDT" and frame["tf"].iloc[0] == "15m"
    assert pd.isna(frame["quote_volume"].iloc[0])
    assert pd.isna(frame["taker_buy_base"].iloc[0])


def test_symbol_carries_the_venue_prefix():
    """Символ TradingView — `БИРЖА:ТИКЕР`; без префикса источник ответит 404."""
    assert tvquotes.tv_symbol("HYPEUSDT", "bybit_spot") == "BYBIT:HYPEUSDT"
    assert tvquotes.tv_symbol("btcusdt", "binance_spot") == "BINANCE:BTCUSDT"
    with pytest.raises(tvquotes.TvQuotesError):
        tvquotes.tv_symbol("HYPEUSDT", "kraken_spot")


def test_unknown_timeframe_does_not_go_to_the_network(configured, monkeypatch):
    """
    Наших сеток источник знает не все, и подменять 6h соседней нельзя молча:
    в `ohlcv` легли бы бары чужого размера под нашей меткой `tf`.
    """
    monkeypatch.setattr(
        tvquotes.httpx, "get",
        lambda *a, **kw: pytest.fail("до сети дойти было не должно"),
    )
    with pytest.raises(tvquotes.TvQuotesError):
        tvquotes.fetch_bars("HYPEUSDT", "6h", "bybit_spot", 10)


def test_error_code_becomes_a_readable_failure(configured, monkeypatch):
    """Отказ источника — исключение с его текстом, а не пустой ответ."""
    monkeypatch.setattr(tvquotes.httpx, "get", lambda *a, **kw: _response([], status=400))
    with pytest.raises(tvquotes.TvQuotesError, match="400"):
        tvquotes.fetch_bars("HYPEUSDT", "15m", "bybit_spot", 10)


def test_disabled_source_is_not_a_silent_empty(monkeypatch):
    """
    Без ключа источник выключен, и это должно быть слышно вызывающему:
    пустой ответ он принял бы за «баров нет» и не пошёл бы в REST.
    """
    monkeypatch.setattr(
        config, "data", dataclasses.replace(config.data, tvq_api_key=""),
    )
    assert not tvquotes.enabled()
    with pytest.raises(tvquotes.TvQuotesError):
        tvquotes.fetch_bars("HYPEUSDT", "15m", "bybit_spot", 10)


# ─── Встройка в загрузчик ─────────────────────────────────────────────────────

def test_tail_comes_from_tvq_instead_of_rest(configured, monkeypatch):
    """
    Когда источник настроен, REST Bybit не тревожится вовсе: с боевого адреса
    он всё равно отвечает 403, и лишний поход туда — только задержка прогона.
    """
    ts = pd.Timestamp.now(tz="UTC").floor("15min") - pd.Timedelta(minutes=30)
    monkeypatch.setattr(bybit, "_sync_daily_tail", lambda *a, **kw: 3)
    monkeypatch.setattr(bybit.bars, "last_ts", lambda *a, **kw: ts - pd.Timedelta(minutes=15))
    monkeypatch.setattr(bybit.bars, "store_bars", lambda frame: len(frame))
    monkeypatch.setattr(tvquotes.httpx, "get", lambda *a, **kw: _response([_candle(ts)]))

    # Клиент нужен и архивам, поэтому проверяется не создание клиента,
    # а факт запроса: их у REST-хвоста быть не должно ни одного.
    calls = {"rest": 0}

    class CountingClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

        def get(self, *a, **kw):
            calls["rest"] += 1
            raise AssertionError("REST не должен вызываться при живом tv-quotes-api")

    monkeypatch.setattr(bybit.httpx, "Client", CountingClient)

    assert bybit.sync_recent("HYPEUSDT", "15m") == 3 + 1
    assert calls["rest"] == 0


def test_tvq_failure_falls_back_to_rest(configured, monkeypatch):
    """
    Отказ источника не окончателен: следом пробуется REST. Установка, где он
    доступен, не должна терять хвост из-за чужого сбоя.
    """
    calls = {"rest": 0}

    class FakeResponse:
        status_code = 403

        def raise_for_status(self):  # pragma: no cover
            raise AssertionError("403 обрабатывается мягко")

    class FakeClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

        def get(self, *a, **kw):
            calls["rest"] += 1
            return FakeResponse()

    def boom(*a, **kw):
        raise tvquotes.TvQuotesError("источник лежит")

    monkeypatch.setattr(bybit, "_sync_daily_tail", lambda *a, **kw: 2)
    monkeypatch.setattr(bybit.bars, "last_ts", lambda *a, **kw: pd.Timestamp("2026-08-01", tz="UTC"))
    monkeypatch.setattr(tvquotes, "fetch_bars", boom)
    monkeypatch.setattr(bybit.httpx, "Client", FakeClient)

    assert bybit.sync_recent("HYPEUSDT", "15m") == 2
    assert calls["rest"] == 1, "после отказа tv-quotes-api полагается заход в REST"


def test_deep_lag_is_left_to_the_archives(configured, monkeypatch):
    """
    Дыра шире потолка источника закрывается архивами, а не пачками запросов:
    архив вдобавок принесёт `trades` и `taker_buy_base`, которых здесь нет.
    """
    monkeypatch.setattr(bybit, "_sync_daily_tail", lambda *a, **kw: 0)
    monkeypatch.setattr(bybit.bars, "last_ts", lambda *a, **kw: pd.Timestamp("2020-01-01", tz="UTC"))
    monkeypatch.setattr(
        tvquotes, "fetch_bars",
        lambda *a, **kw: pytest.fail("запрос глубже MAX_BARS отправлять незачем"),
    )
    assert bybit._sync_tv_tail("HYPEUSDT", "15m", int(pd.Timestamp("2020-01-01").timestamp() * 1000)) is None
