"""
Тесты загрузчика Bybit и выбора площадки.

Bybit заведён ради монет, которых на споте Binance нет вовсе. Ценность этих
тестов в том, что все проверяемые здесь ошибки — тихие: они не роняют прогон,
а портят бары, и увидеть это по данным почти невозможно.

Самый важный — test_monthly_header_does_not_shift_columns. В месячных архивах
Bybit заголовок перечисляет пять имён при шести колонках данных, и pandas
на заголовке молча уводит `id` в индекс: цена уезжает в поле времени, объём —
в цену. История при этом загружается «успешно».

В сеть и в БД, как и остальные тесты проекта, не ходят.
"""
from __future__ import annotations

import gzip

import pandas as pd
import pytest

from btcproc import symbols
from btcproc.ingest import bybit, sources
from btcproc.ingest.bars import OHLCV_COLUMNS

# Заголовок из пяти имён при шести колонках данных — ровно как в месячных
# архивах Bybit.
MONTHLY_CSV = (
    "id,timestamp,price,volume,side\n"
    "1,1752224400000,46.00,1.0,buy,0\n"
    "2,1752224460000,47.00,2.0,sell,0\n"
    "3,1752225300000,50.00,4.0,buy,0\n"
)

# Дневные архивы объявляют все шесть.
DAILY_CSV = (
    "id,timestamp,price,volume,side,rpi\n"
    "1,1752224400000,46.00,1.0,buy,0\n"
    "2,1752224460000,47.00,2.0,sell,0\n"
)


def _ticks(csv: str) -> pd.DataFrame:
    return bybit.parse_ticks(gzip.compress(csv.encode()))


# ─── Разбор тиковых архивов ───────────────────────────────────────────────────

def test_monthly_header_does_not_shift_columns():
    """
    Заголовок месячного архива короче строки данных на одну колонку.
    Довериться ему — значит сдвинуть все поля: время окажется в `id`,
    цена во времени, объём в цене. Бары после такого выглядят правдоподобно.
    """
    ticks = _ticks(MONTHLY_CSV)

    assert list(ticks.columns)[:5] == ["id", "timestamp", "price", "volume", "side"]
    assert ticks["timestamp"].iloc[0] == 1752224400000
    assert ticks["price"].iloc[0] == 46.00
    assert ticks["volume"].iloc[0] == 1.0
    assert ticks["side"].iloc[0] == "buy"


def test_daily_header_with_all_columns_parses_too():
    ticks = _ticks(DAILY_CSV)

    assert ticks["price"].tolist() == [46.0, 47.0]
    assert ticks["side"].tolist() == ["buy", "sell"]


def test_headerless_csv_parses():
    """Заголовок в архивах появился не сразу — файлы без него читаемы."""
    ticks = _ticks("1,1752224400000,46.00,1.0,buy,0\n")

    assert ticks["price"].iloc[0] == 46.0


# ─── Сборка баров из сделок ───────────────────────────────────────────────────

def test_bars_carry_the_full_schema():
    """
    Схема `ohlcv` общая на все площадки. Загрузчик, отдающий бары без
    `trades` и `taker_buy_base`, молча лишает монету признака taker_bias
    и обоих атомов доминирования тейкеров.
    """
    frame = bybit.ticks_to_bars(_ticks(MONTHLY_CSV), "HYPEUSDT", "15m")

    assert list(frame.columns) == OHLCV_COLUMNS
    assert frame["trades"].sum() == 3
    assert frame["taker_buy_base"].notna().all()


def test_taker_side_is_summed_from_trades():
    """Объём покупок тейкеров — сумма сделок со стороной buy, а не половина."""
    frame = bybit.ticks_to_bars(_ticks(MONTHLY_CSV), "HYPEUSDT", "15m")

    first = frame.iloc[0]
    assert first["volume"] == pytest.approx(3.0)     # 1.0 buy + 2.0 sell
    assert first["taker_buy_base"] == pytest.approx(1.0)
    assert first["quote_volume"] == pytest.approx(46.0 * 1.0 + 47.0 * 2.0)


def test_bar_open_inherits_previous_close():
    """
    Конвенция Bybit: открытие бара равно закрытию предыдущего, а не цене
    первой сделки внутри бара (проверено сверкой с их kline — 199 баров
    из 199). Без этого бары не сходятся с публичными данными биржи.
    """
    frame = bybit.ticks_to_bars(_ticks(MONTHLY_CSV), "HYPEUSDT", "15m")

    assert len(frame) == 2
    assert frame["open"].iloc[1] == pytest.approx(frame["close"].iloc[0])


def test_open_is_inside_the_bar_range():
    """
    Открытие входит в диапазон бара. Иначе на разрыве получается бар
    с `high` ниже открытия — величина, которой не бывает.
    """
    frame = bybit.ticks_to_bars(_ticks(MONTHLY_CSV), "HYPEUSDT", "15m")

    assert (frame["high"] >= frame["open"]).all()
    assert (frame["low"] <= frame["open"]).all()
    assert (frame["high"] >= frame["close"]).all()


def test_previous_close_from_db_closes_the_seam_between_archives():
    """
    Первый бар архива не должен открываться первой сделкой: архивы месячные,
    и такой «гэп» появлялся бы раз в месяц ровно на стыке файлов.
    """
    frame = bybit.ticks_to_bars(_ticks(MONTHLY_CSV), "HYPEUSDT", "15m", prev_close=44.0)

    assert frame["open"].iloc[0] == pytest.approx(44.0)
    assert frame["low"].iloc[0] <= 44.0


def test_trades_in_one_millisecond_close_in_fill_order():
    """
    Порядок строк архива внутри одной миллисекунды не совпадает с порядком
    исполнения, а бар закрывается по последней исполненной сделке. Агрессивная
    заявка выедает стакан от лучшей цены к худшей: покупка снизу вверх.

    Взято с живого расхождения: три последние сделки бара 2025-08-01 01:30
    записаны как 41.43, 41.43, 41.42, биржа закрывает бар по 41.43. По порядку
    файла таких баров 26 из 1000, по восстановленному — ноль.
    """
    csv = (
        "id,timestamp,price,volume,side,rpi\n"
        "1,1752224400000,41.30,1.0,buy,0\n"
        "2,1752224460000,41.43,9.655,buy,0\n"
        "3,1752224460000,41.43,4.308,buy,0\n"
        "4,1752224460000,41.42,12.072,buy,0\n"
    )
    frame = bybit.ticks_to_bars(_ticks(csv), "HYPEUSDT", "15m")

    assert frame["close"].iloc[0] == pytest.approx(41.43)


def test_sell_side_fills_from_the_top():
    """Зеркальный случай: продажа выедает стакан сверху вниз."""
    csv = (
        "id,timestamp,price,volume,side,rpi\n"
        "1,1752224400000,41.30,1.0,sell,0\n"
        "2,1752224460000,41.20,1.0,sell,0\n"
        "3,1752224460000,41.25,1.0,sell,0\n"
    )
    frame = bybit.ticks_to_bars(_ticks(csv), "HYPEUSDT", "15m")

    assert frame["close"].iloc[0] == pytest.approx(41.20)


def test_bars_without_trades_are_flat_not_missing():
    """
    Биржа публикует бар и тогда, когда сделок не было: цена стоит,
    объём ноль. Дыра в сетке базового ТФ ломала бы скользящие окна признаков.
    """
    csv = (
        "id,timestamp,price,volume,side,rpi\n"
        "1,1752224400000,46.00,1.0,buy,0\n"
        "2,1752226200000,48.00,2.0,buy,0\n"   # +30 минут: между ними пустой бар
    )
    frame = bybit.ticks_to_bars(_ticks(csv), "HYPEUSDT", "15m")

    assert len(frame) == 3
    idle = frame.iloc[1]
    assert idle["volume"] == 0
    assert idle["trades"] == 0
    assert idle["open"] == idle["high"] == idle["low"] == idle["close"] == pytest.approx(46.0)


def test_timestamps_in_seconds_do_not_land_in_1970():
    """
    Метку времени Bybit отдаёт в миллисекундах, но не везде. Ошибка в
    единицах не роняет прогон — она уносит историю в 1970-й.
    """
    csv = "id,timestamp,price,volume,side,rpi\n1,1752224400,46.00,1.0,buy,0\n"
    frame = bybit.ticks_to_bars(_ticks(csv), "HYPEUSDT", "15m")

    assert frame["ts"].iloc[0].year == 2025


def test_empty_archive_gives_empty_frame():
    assert bybit.ticks_to_bars(pd.DataFrame(), "HYPEUSDT", "15m").empty


# ─── Гео-блокировка REST ──────────────────────────────────────────────────────

def test_geoblocked_rest_does_not_break_the_run(monkeypatch):
    """
    С американского хоста REST Bybit отвечает 403 — вместе со всеми зеркалами
    (api.bytick.com, api.bybit.nl: та же заглушка CloudFront). Боевой VPS
    именно такой.

    Ронять на этом прогон нельзя: тиковые архивы качаются, данные идут, а
    `live` монеты падал бы каждые полчаса. Отставание в таком режиме — до
    суток, до выхода дневного архива.
    """
    calls = {"daily": 0, "rest": 0}

    def fake_daily(symbol, tf, client, progress=None):
        calls["daily"] += 1
        return 7

    class FakeResponse:
        status_code = 403

        def raise_for_status(self):  # pragma: no cover — не должен вызываться
            raise AssertionError("на 403 полагается мягкая деградация, а не исключение")

    class FakeClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

        def get(self, *a, **kw):
            calls["rest"] += 1
            return FakeResponse()

    monkeypatch.setattr(bybit, "_sync_daily_tail", fake_daily)
    monkeypatch.setattr(bybit.bars, "last_ts", lambda *a, **kw: pd.Timestamp("2026-08-01", tz="UTC"))
    monkeypatch.setattr(bybit.httpx, "Client", FakeClient)

    rows = bybit.sync_recent("HYPEUSDT", "15m")

    assert rows == 7, "бары из дневных архивов обязаны сохраниться"
    assert calls["rest"] == 1, "после 403 повторять запросы незачем"


# ─── Выбор площадки ───────────────────────────────────────────────────────────

def test_loader_is_chosen_by_venue():
    assert sources.loader_for("binance_spot") is __import__(
        "btcproc.ingest.binance", fromlist=["binance"]
    )
    assert sources.loader_for("bybit_spot") is bybit


def test_unknown_venue_fails_with_a_readable_error():
    """
    Опечатка в площадке иначе всплыла бы как «нет данных по монете»:
    качать её было бы некому, а прогон продолжился бы.
    """
    with pytest.raises(symbols.UnknownSymbolError) as exc:
        sources.loader_for("kraken_spot")

    assert "kraken_spot" in str(exc.value)
    assert "binance_spot" in str(exc.value)


def test_every_symbol_declares_a_known_venue():
    for spec in symbols.SYMBOLS:
        assert spec.venue in symbols.VENUES, spec.ticker


def test_loaders_share_the_interface():
    """
    Диспетчер зовёт у загрузчика три функции и о различиях площадок не знает.
    Разъехавшаяся сигнатура проявилась бы только в прогоне живой монеты.
    """
    import inspect

    from btcproc.ingest import binance

    for name in ("sync_history", "sync_recent", "has_month"):
        left = inspect.signature(getattr(binance, name))
        right = inspect.signature(getattr(bybit, name))
        assert list(left.parameters) == list(right.parameters), name


def test_default_venue_keeps_existing_symbols_on_binance():
    """
    Молчаливый переезд заведённой монеты на другую площадку — худшее из
    возможных изменений: цены разъедутся на доли процента, а модель
    состояний переучится на другой рынок без единой ошибки.
    """
    for ticker in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        assert symbols.get(ticker).venue == "binance_spot"
