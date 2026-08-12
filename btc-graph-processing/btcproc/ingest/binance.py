"""
Загрузка истории торговых пар со спота Binance.

Модуль знает ровно одну площадку. Цикл по монетам живёт в
`btcproc/ingest/sources.py`: монеты бывают с разных бирж, и выбирать
загрузчик — его работа, а не наша. Хранение и чтение баров — `bars.py`.

Основной источник — публичные дампы data.binance.vision: месячные zip-архивы
klines. Это на порядок быстрее REST (вся история 15m с 2017 — несколько минут
против часов постраничного опроса) и не упирается в rate-limit.

Свежий хвост (текущий месяц и последние часы) добирается через REST, потому
что месячный дамп появляется только после закрытия месяца. Адрес REST задаётся
переменной BINANCE_REST_URL: api.binance.com отвечает 451 из ряда юрисдикций,
и там его меняют на data-api.binance.vision.

Старшие таймфреймы не качаются отдельно — они агрегируются из базового
(`bars.rebuild_context_timeframes`). Так исключены расхождения между ТФ
на границах баров.
"""
from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import httpx
import pandas as pd

from btcproc import config
from btcproc.ingest import bars
from btcproc.ingest.bars import OHLCV_COLUMNS

logger = logging.getLogger(__name__)

VISION_URL = "https://data.binance.vision/data/spot/monthly/klines/{sym}/{tf}/{sym}-{tf}-{ym}.zip"
DAILY_URL = "https://data.binance.vision/data/spot/daily/klines/{sym}/{tf}/{sym}-{tf}-{ymd}.zip"
# Адрес настраиваемый: из США api.binance.com отвечает 451, и хвост баров
# не добирается вовсе. Подробности и альтернатива — в DataConfig.
REST_URL = config.data.binance_rest_url

# Порядок колонок в CSV Binance.
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


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


def has_month(symbol: str, ym: str, client: httpx.Client) -> bool:
    """
    Есть ли месячный дамп за `ym` (формат YYYY-MM).

    Нужна `symbols.resolve_history_start`: дату листинга проверяют фактом,
    а «факт» у каждой площадки свой. HEAD вместо GET — файлы бывают
    в сотни мегабайт, а ответ нужен один бит.
    """
    url = VISION_URL.format(sym=symbol.upper(), tf=config.data.base_tf, ym=ym)
    return client.head(url).status_code == 200


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
            total_rows += bars.store_bars(df)
            if progress:
                progress(i + 1, len(months), f"{ym}: {len(df)} баров")

        # Текущий месяц закрывается дневными архивами, остаток — через REST.
        total_rows += _sync_daily_tail(symbol, tf, client, progress)

    total_rows += sync_recent(symbol, tf)
    return {"months": len(months), "rows": total_rows, "missing_months": missing}


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
            rows += bars.store_bars(_normalize(_unzip(payload), symbol, tf))
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
    last = bars.last_ts(symbol, tf)
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
            rows += bars.store_bars(_normalize(df, symbol, tf))
            start_ms = int(df["open_time"].iloc[-1]) + 1
            if len(batch) < 1000:
                break
    return rows
