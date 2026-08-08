"""
Загрузка истории торговых пар с Binance.

Монет может быть несколько: `sync_many()` проходит их подряд, беря дату
начала истории у каждой из реестра `btcproc/symbols.py`. Остальные функции
работают по одной монете и принимают её символом.

Основной источник — публичные дампы data.binance.vision: месячные zip-архивы
klines. Это на порядок быстрее REST (вся история 15m с 2017 — несколько минут
против часов постраничного опроса) и не упирается в rate-limit.

Свежий хвост (текущий месяц и последние часы) добирается через REST
api.binance.com, потому что месячный дамп появляется только после закрытия
месяца.

Старшие таймфреймы не качаются отдельно — они агрегируются из базового.
Так исключены расхождения между ТФ на границах баров.
"""
from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Sequence

import httpx
import pandas as pd

from btcproc import config
from btcproc.db.session import bulk_upsert, fetch_all, fetch_one

logger = logging.getLogger(__name__)

VISION_URL = "https://data.binance.vision/data/spot/monthly/klines/{sym}/{tf}/{sym}-{tf}-{ym}.zip"
DAILY_URL = "https://data.binance.vision/data/spot/daily/klines/{sym}/{tf}/{sym}-{tf}-{ymd}.zip"
REST_URL = "https://api.binance.com/api/v3/klines"

# Порядок колонок в CSV Binance.
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]

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


def _to_utc(series: pd.Series) -> pd.Series:
    """
    Приводит время Binance к UTC.

    В дампах до 2025 года open_time в миллисекундах, начиная с 2025-01 —
    в микросекундах. Различаем по порядку величины, иначе половина истории
    уезжает в 1970-й.
    """
    unit = "us" if series.iloc[0] > 1e14 else "ms"
    return pd.to_datetime(series, unit=unit, utc=True)


def _parse_csv(raw: bytes) -> pd.DataFrame:
    # У части дампов появилась строка заголовка — определяем по первому байту.
    header = 0 if raw[:1].isalpha() else None
    df = pd.read_csv(
        io.BytesIO(raw),
        header=header,
        names=KLINE_COLUMNS if header is None else None,
    )
    if header == 0:
        df.columns = KLINE_COLUMNS[: len(df.columns)]
    return df


def _months(start: datetime, end: datetime) -> Iterator[str]:
    cur = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    while cur <= end:
        yield cur.strftime("%Y-%m")
        cur = (cur + timedelta(days=32)).replace(day=1)


def _cache_path(symbol: str, tf: str, key: str) -> Path:
    path = config.data.data_dir / "binance" / symbol / tf
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{key}.zip"


def _fetch_zip(
    url: str, cache: Path, client: httpx.Client, attempts: int = 4
) -> bytes | None:
    """
    Скачивает архив с диска-кэша или из сети. None — архива нет (404/403).

    data.binance.vision при плотной череде запросов отдаёт 503 и 429 —
    это не «файла нет», а просьба подождать, поэтому такие ответы
    повторяются с растущей паузой. Иначе одна случайная 503 роняла бы
    закачку всей истории на середине.
    """
    if cache.exists() and cache.stat().st_size > 0:
        return cache.read_bytes()

    delay = 2.0
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = client.get(url)
            if resp.status_code in (403, 404):
                return None
            if resp.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(
                    f"{resp.status_code} от {url}", request=resp.request, response=resp
                )
            resp.raise_for_status()
            cache.write_bytes(resp.content)
            return resp.content
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            logger.warning("Повтор %s через %.0f с (%s)", url, delay, exc)
            time.sleep(delay)
            delay *= 2

    logger.error("Не удалось скачать %s: %s", url, last_error)
    return None


def _unzip(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = zf.namelist()[0]
        return _parse_csv(zf.read(name))


def _normalize(df: pd.DataFrame, symbol: str, tf: str) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = _to_utc(df["open_time"])
    df["symbol"] = symbol
    df["tf"] = tf
    numeric = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df[OHLCV_COLUMNS]


def _store(df: pd.DataFrame) -> int:
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


def sync_history(
    symbol: str | None = None,
    tf: str | None = None,
    start: str | None = None,
    end: datetime | None = None,
    progress=None,
) -> dict:
    """
    Качает месячные дампы за весь период и кладёт в ohlcv.

    Уже скачанные архивы берутся с диска, уже загруженные бары перезаписываются
    upsert'ом — команду можно гонять повторно без вреда.
    """
    symbol = symbol or config.data.symbol
    tf = tf or config.data.base_tf
    start_dt = pd.Timestamp(start or config.data.history_start, tz="UTC").to_pydatetime()
    end_dt = end or datetime.now(timezone.utc)

    months = list(_months(start_dt, end_dt))
    total_rows, missing = 0, []

    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        for i, ym in enumerate(months):
            url = VISION_URL.format(sym=symbol, tf=tf, ym=ym)
            payload = _fetch_zip(url, _cache_path(symbol, tf, ym), client)
            if payload is None:
                missing.append(ym)
                if progress:
                    progress(i + 1, len(months), f"{ym}: дампа нет")
                continue
            df = _normalize(_unzip(payload), symbol, tf)
            total_rows += _store(df)
            if progress:
                progress(i + 1, len(months), f"{ym}: {len(df)} баров")

        # Текущий месяц закрывается дневными архивами, остаток — через REST.
        total_rows += _sync_daily_tail(symbol, tf, client, progress)

    total_rows += sync_recent(symbol, tf)
    return {"months": len(months), "rows": total_rows, "missing_months": missing}


def sync_many(
    tickers: Sequence[str] | None = None,
    tf: str | None = None,
    start: str | None = None,
    context: bool = True,
    progress=None,
    on_symbol=None,
) -> dict:
    """
    Загрузка истории по нескольким монетам подряд.

    Цикл живёт здесь, а не в CLI, потому что он нужен трём вызывающим —
    команде `ingest --all`, форме админки и обкатке новой монеты — и у всех
    трёх одинаковые требования к поведению.

    Как устроено и почему:

      * **Дата начала берётся из реестра, у каждой монеты своя.** Общий
        HISTORY_START на альткоине означал бы полсотни запросов за месяцы
        до листинга: каждый вернёт 404 и попадёт в missing_months, замусорив
        отчёт настолько, что настоящий пропуск в нём не разглядеть.
        Явный `start` перекрывает реестр — для точечных догрузок.
      * **Падение одной монеты не останавливает остальные.** Сеть отвалилась
        на третьей из семи — остальные четыре всё равно должны загрузиться,
        иначе ночной прогон превращается в лотерею. Диагноз копится
        и возвращается вызывающему.
      * **Последовательно, а не параллельно.** Упирается всё в сеть и в
        запись пачками в одну таблицу; параллельная закачка дала бы выигрыш
        только до первого 429 от Binance.

    Возвращает сводку: что загрузилось, что упало и сколько всего баров.
    Печатает — вызывающий, а не эта функция.

    tickers=None — все активные монеты реестра.
    """
    from btcproc import symbols

    specs = (
        [symbols.get(t) for t in tickers] if tickers else symbols.enabled()
    )
    tf = tf or config.data.base_tf

    result: dict = {"symbols": {}, "failed": {}, "rows_total": 0}

    for index, spec in enumerate(specs, start=1):
        if on_symbol:
            on_symbol(index, len(specs), spec.ticker)
        try:
            report = sync_history(
                spec.ticker, tf, start or spec.start_date(),
                progress=(
                    (lambda i, n, msg, _t=spec.ticker: progress(i, n, f"{_t}: {msg}"))
                    if progress else None
                ),
            )
            if context:
                report["context"] = rebuild_context_timeframes(spec.ticker)
            result["symbols"][spec.ticker] = report
            result["rows_total"] += report["rows"]
        except Exception as exc:  # noqa: BLE001 — одна монета не рушит пакет
            logger.exception("Загрузка %s упала", spec.ticker)
            result["failed"][spec.ticker] = f"{type(exc).__name__}: {exc}"

    result["ok"] = len(result["symbols"])
    result["total"] = len(specs)
    return result


def _sync_daily_tail(symbol: str, tf: str, client: httpx.Client, progress=None) -> int:
    """Дневные дампы за текущий месяц: месячный появится только после его конца."""
    today = datetime.now(timezone.utc).date()
    first_of_month = today.replace(day=1)
    rows = 0
    day = first_of_month
    while day < today:
        ymd = day.strftime("%Y-%m-%d")
        url = DAILY_URL.format(sym=symbol, tf=tf, ymd=ymd)
        payload = _fetch_zip(url, _cache_path(symbol, tf, ymd), client)
        if payload is not None:
            rows += _store(_normalize(_unzip(payload), symbol, tf))
        day += timedelta(days=1)
    if progress:
        progress(1, 1, f"дневной хвост: {rows} баров")
    return rows


def sync_recent(symbol: str | None = None, tf: str | None = None, limit_batches: int = 20) -> int:
    """
    Добирает свежие бары через REST, начиная с последнего сохранённого.

    Используется и в live-режиме: вызов дешёвый, ходит максимум
    limit_batches × 1000 баров.
    """
    from btcproc import symbols

    symbol = symbol or config.data.symbol
    tf = tf or config.data.base_tf
    last = last_ts(symbol, tf)
    if last:
        start_ms = int(last.timestamp() * 1000) + 1
    else:
        # Баров ещё нет вовсе. Точка старта — дата листинга монеты из реестра,
        # а не общий HISTORY_START: для поздно listed-монеты общий дефолт
        # означал бы запрос за годы до появления пары.
        try:
            fallback = symbols.get(symbol).start_date()
        except symbols.UnknownSymbolError:
            fallback = config.data.history_start
        start_ms = int(pd.Timestamp(fallback, tz="UTC").timestamp() * 1000)

    rows = 0
    with httpx.Client(timeout=30.0) as client:
        for _ in range(limit_batches):
            resp = client.get(
                REST_URL,
                params={"symbol": symbol, "interval": tf, "startTime": start_ms, "limit": 1000},
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            df = pd.DataFrame(batch, columns=KLINE_COLUMNS)
            # Последний бар ещё не закрыт — в историю его брать нельзя.
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            df = df[pd.to_numeric(df["close_time"]) < now_ms]
            if df.empty:
                break
            rows += _store(_normalize(df, symbol, tf))
            start_ms = int(df["open_time"].iloc[-1]) + 1
            if len(batch) < 1000:
                break
    return rows


def rebuild_context_timeframes(symbol: str | None = None, tfs: list[str] | None = None) -> dict:
    """Пересобирает старшие ТФ из базового и складывает в ту же таблицу."""
    symbol = symbol or config.data.symbol
    tfs = tfs or config.data.context_tfs
    base = load_ohlcv(symbol, config.data.base_tf)
    out = {}
    for tf in tfs:
        agg = resample(base, tf)
        agg = agg.reset_index().rename(columns={"index": "ts"})
        agg["symbol"], agg["tf"] = symbol, tf
        out[tf] = _store(agg[OHLCV_COLUMNS])
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
        params.append(pd.Timestamp(start, tz="UTC"))
    if end is not None:
        sql += " AND ts <= %s"
        params.append(pd.Timestamp(end, tz="UTC"))
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
