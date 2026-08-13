"""
Бары в базе: запись, чтение, агрегация старших таймфреймов.

Отделено от загрузчиков намеренно. Загрузчик знает про площадку (адреса
архивов, формат CSV, конвенции биржи), а всё, что ниже по течению, — уже нет:
в `ohlcv` лежат одинаковые строки независимо от того, откуда они приехали.
Пока модуль назывался `binance`, чтение баров бай­битовой монеты выглядело бы
как обращение к Binance — и это ровно тот вид путаницы, который живёт годами.

Схема таблицы одна на все площадки, поэтому загрузчик обязан отдать полный
набор полей, включая `trades` и `taker_buy_base`: без них молча пропадают
признак `taker_bias` и атомы доминирования тейкеров.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from btcproc import config
from btcproc.db.session import bulk_upsert, fetch_all, fetch_one

OHLCV_COLUMNS = [
    "symbol", "tf", "ts", "open", "high", "low", "close",
    "volume", "quote_volume", "trades", "taker_buy_base",
]

# Правило агрегации базовых баров в старший ТФ.
_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "quote_volume": "sum",
    "trades": "sum",
    "taker_buy_base": "sum",
}


def store_bars(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    df = df.copy()
    # trades — INTEGER в схеме, а после агрегации старших ТФ приходит float.
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype(int)
    rows = list(df.itertuples(index=False, name=None))
    return bulk_upsert(
        "ohlcv",
        OHLCV_COLUMNS,
        rows,
        conflict_columns=["symbol", "tf", "ts"],
    )


def rebuild_context_timeframes(
    symbol: str | None = None,
    tfs: list[str] | None = None,
    full: bool = False,
) -> dict:
    """
    Пересобирает старшие ТФ из базового и складывает в ту же таблицу.

    По умолчанию инкрементально: базовые бары читаются не с 2017 года, а от
    последнего уже сохранённого бара старшего ТФ. Этот последний бар обязан
    попасть в пересчёт заново — на момент прошлого запуска он мог быть
    недособран (метка бара — время открытия, а окно ещё не закрылось).

    Полный пересчёт (`full=True`) нужен, только если базовые бары правились
    задним числом: обычная догрузка хвоста этого не делает.

    Раньше `live` каждые полчаса читал всю историю базового ТФ и перезаписывал
    upsert'ом ВСЕ старшие — сотни тысяч строк ради нескольких свежих.
    """
    symbol = symbol or config.data.symbol
    tfs = tfs or config.data.context_tfs

    # Граница пересчёта — самый ранний хвост среди всех ТФ: базовые бары
    # читаются один раз на всех, а недельный ТФ отступает назад дальше
    # четырёхчасового.
    starts: dict[str, pd.Timestamp | None] = {}
    for tf in tfs:
        if full:
            starts[tf] = None
            continue
        row = fetch_one(
            "SELECT max(ts) AS t FROM ohlcv WHERE symbol = %s AND tf = %s",
            (symbol, tf),
        )
        starts[tf] = pd.Timestamp(row["t"]) if row and row["t"] is not None else None

    earliest = None if any(v is None for v in starts.values()) else min(starts.values())
    base = load_ohlcv(symbol, config.data.base_tf, start=earliest)

    out = {}
    for tf in tfs:
        # Срез строго от границы своего ТФ: она совпадает с открытием бара
        # (label=left), поэтому агрегация от неё даёт тот же результат, что и
        # агрегация всей истории — просто без лишней работы.
        window = base if starts[tf] is None else base[base.index >= starts[tf]]
        if window.empty:
            out[tf] = 0
            continue
        agg = resample(window, tf)
        agg = agg.reset_index().rename(columns={"index": "ts"})
        agg["symbol"], agg["tf"] = symbol, tf
        out[tf] = store_bars(agg[OHLCV_COLUMNS])
    return out


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Агрегация баров в старший ТФ.

    df — с DatetimeIndex по ts. label/closed=left: метка бара — время его
    открытия, как у Binance.
    """
    rule = {"1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h", "1d": "1D"}[tf]
    out = df.resample(rule, label="left", closed="left").agg(_AGG)
    return out.dropna(subset=["open", "close"])


def _as_utc(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    """
    Границу диапазона принимаем и наивной, и с зоной.

    `pd.Timestamp(x, tz="UTC")` на уже локализованном значении бросает
    ValueError — а границы приходят из обоих источников: наивные строки от
    оператора и tz-aware метки из БД (`max(ts)`).
    """
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def load_ohlcv(
    symbol: str | None = None,
    tf: str | None = None,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
) -> pd.DataFrame:
    """Бары из БД в DataFrame с DatetimeIndex (UTC)."""
    symbol = symbol or config.data.symbol
    tf = tf or config.data.base_tf
    sql = (
        "SELECT ts, open, high, low, close, volume, quote_volume, trades, taker_buy_base "
        "FROM ohlcv WHERE symbol = %s AND tf = %s"
    )
    params: list = [symbol, tf]
    if start is not None:
        sql += " AND ts >= %s"
        params.append(_as_utc(start))
    if end is not None:
        sql += " AND ts <= %s"
        params.append(_as_utc(end))
    sql += " ORDER BY ts"

    rows = fetch_all(sql, params)
    if not rows:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "quote_volume",
                     "trades", "taker_buy_base"],
            index=pd.DatetimeIndex([], tz="UTC", name="ts"),
        )
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index("ts").astype(float)


def last_ts(symbol: str | None = None, tf: str | None = None) -> datetime | None:
    row = fetch_one(
        "SELECT max(ts) AS ts FROM ohlcv WHERE symbol = %s AND tf = %s",
        (symbol or config.data.symbol, tf or config.data.base_tf),
    )
    return row["ts"] if row and row["ts"] else None


def last_close(symbol: str, tf: str, before: pd.Timestamp) -> float | None:
    """
    Цена закрытия последнего бара строго раньше `before`.

    Нужна загрузчикам, которые воспроизводят биржевую конвенцию «открытие
    бара равно закрытию предыдущего» (Bybit). Без неё первый бар каждого
    архива получал бы открытие по первой сделке — раз в месяц, на стыке
    файлов, и выглядело бы это как настоящий гэп рынка.
    """
    row = fetch_one(
        "SELECT close FROM ohlcv WHERE symbol = %s AND tf = %s AND ts < %s "
        "ORDER BY ts DESC LIMIT 1",
        (symbol, tf, _as_utc(before)),
    )
    return float(row["close"]) if row and row["close"] is not None else None


def coverage(symbol: str | None = None) -> list[dict]:
    """Что реально лежит в базе: период и число баров по каждому ТФ."""
    return fetch_all(
        "SELECT tf, count(*) AS bars, min(ts) AS first_ts, max(ts) AS last_ts "
        "FROM ohlcv WHERE symbol = %s GROUP BY tf ORDER BY tf",
        (symbol or config.data.symbol,),
    )


def coverage_all(symbols_filter: list[str] | None = None, tf: str | None = None) -> list[dict]:
    """
    Сводка покрытия по всем монетам сразу — одним запросом, а не N.

    tf=None — по всем таймфреймам; иначе только базовый, что и нужно
    помонетной таблице `status`.
    """
    sql = (
        "SELECT symbol, tf, count(*) AS bars, min(ts) AS first_ts, max(ts) AS last_ts "
        "FROM ohlcv"
    )
    conditions, params = [], []
    if symbols_filter:
        conditions.append("symbol = ANY(%s)")
        params.append(list(symbols_filter))
    if tf:
        conditions.append("tf = %s")
        params.append(tf)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    return fetch_all(sql + " GROUP BY symbol, tf ORDER BY symbol, tf", params)
