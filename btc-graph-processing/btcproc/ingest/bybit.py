"""
Загрузка истории торговых пар со спота Bybit.

Появился ради монет, которых на споте Binance нет вовсе (HYPEUSDT — первая
такая). Интерфейс совпадает с `binance.py` — `sync_history`, `sync_recent`,
`has_month`, — потому что диспетчер `sources.py` выбирает между ними по полю
`venue` монеты и о различиях знать не должен.

**Источник — тиковые архивы, а не готовые klines.** У Bybit есть и то и
другое, но kline-эндпоинт отдаёт лишь OHLCV с оборотом: ни числа сделок, ни
объёма покупок тейкеров. Схема `ohlcv` у нас общая на все площадки, и монета
без `taker_buy_base` молча теряет признак `taker_bias` и оба атома
доминирования тейкеров — вектор признаков становится другим, а метка набора
остаётся прежней. В тиках сторона агрессора есть, поэтому бары собираются
из сделок и набор полей выходит полным.

Цена решения — объём качаемого: месяц тиков HYPEUSDT это ~6 МБ gz против
~0.3 МБ у klines. Архивы кэшируются на диск, как и у Binance.

**Конвенция открытия бара воспроизводит биржевую.** У Bybit открытие каждого
бара равно закрытию предыдущего (проверено сверкой с их kline: 199 из 199
баров), а не цене первой сделки внутри бара. Разница вылезает на ~7% баров и
до 0.2% по цене; без неё наши бары не сходятся с публичными данными биржи,
и `high` иногда оказывается ниже открытия.
"""
from __future__ import annotations

import gzip
import io
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import httpx
import numpy as np
import pandas as pd

from btcproc import config
from btcproc.ingest import bars
from btcproc.ingest.bars import OHLCV_COLUMNS

logger = logging.getLogger(__name__)

# Месячные архивы — с дефисом, дневные — с подчёркиванием. Это не опечатка,
# а формат Bybit.
MONTHLY_URL = "https://public.bybit.com/spot/{sym}/{sym}-{ym}.csv.gz"
DAILY_URL = "https://public.bybit.com/spot/{sym}/{sym}_{ymd}.csv.gz"
# Настраиваемый по той же причине, что и у Binance, — см. DataConfig.
REST_URL = config.data.bybit_rest_url

# Порядок колонок в тиковых CSV. Заголовок в файлах есть, но доверять ему
# нельзя: в месячных архивах он перечисляет пять имён при шести колонках
# данных (`rpi` не объявлен), и pandas молча уводит `id` в индекс, сдвигая
# все поля на одно. Поэтому колонки задаются позиционно, по их числу.
TICK_COLUMNS = ["id", "timestamp", "price", "volume", "side", "rpi"]

# Наш базовый ТФ → значение параметра `interval` REST-эндпоинта Bybit
# (минуты числом, без суффикса).
_REST_INTERVAL = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720", "1d": "D",
}


def _cache_path(symbol: str, key: str) -> Path:
    path = config.data.data_dir / "bybit" / symbol / "ticks"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{key}.csv.gz"


def _months(start: datetime, end: datetime) -> Iterator[str]:
    cur = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    while cur <= end:
        yield cur.strftime("%Y-%m")
        cur = (cur + timedelta(days=32)).replace(day=1)


def _fetch(url: str, cache: Path, client: httpx.Client, attempts: int = 4) -> bytes | None:
    """
    Архив с диска-кэша или из сети. None — архива нет (404/403).

    Повторы с растущей паузой по той же причине, что у Binance: при плотной
    череде запросов отдача упирается в rate-limit, и одна случайная 503
    иначе роняла бы закачку всей истории на середине.
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


def parse_ticks(payload: bytes) -> pd.DataFrame:
    """
    Тиковый CSV (gzip) → DataFrame со сделками.

    Колонки задаются позиционно по их фактическому числу: заголовку в этих
    файлах доверять нельзя (см. TICK_COLUMNS).
    """
    raw = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
    lines = raw.split(b"\n", 2)
    if not lines or not lines[0].strip():
        return pd.DataFrame(columns=TICK_COLUMNS[:5])

    skip = 1 if lines[0][:1].isalpha() else 0
    sample = lines[skip] if len(lines) > skip and lines[skip].strip() else lines[0]
    ncols = len(sample.split(b","))
    if ncols > len(TICK_COLUMNS):
        # Bybit дописал ещё колонку — берём известные, лишние отбрасываем.
        names = TICK_COLUMNS + [f"extra_{i}" for i in range(ncols - len(TICK_COLUMNS))]
    else:
        names = TICK_COLUMNS[:ncols]

    return pd.read_csv(io.BytesIO(raw), header=None, skiprows=skip, names=names)


def _tick_timestamps(series: pd.Series) -> pd.Series:
    """
    Время сделок к UTC.

    Bybit отдаёт метку в миллисекундах на споте и в секундах с дробной частью
    в части архивов деривативов. Различаем по порядку величины, иначе
    история уезжает в 1970-й — ровно та ловушка, что уже случалась
    с микросекундами в дампах Binance.
    """
    values = pd.to_numeric(series, errors="coerce")
    first = values.dropna().iloc[0] if values.notna().any() else 0
    if first > 1e17:
        unit = "ns"
    elif first > 1e14:
        unit = "us"
    elif first > 1e11:
        unit = "ms"
    else:
        unit = "s"
    return pd.to_datetime(values, unit=unit, utc=True)


def ticks_to_bars(
    ticks: pd.DataFrame,
    symbol: str,
    tf: str,
    prev_close: float | None = None,
) -> pd.DataFrame:
    """
    Сделки → бары базового ТФ, в схеме `ohlcv`.

    `prev_close` — закрытие бара, предшествующего первому в этой пачке.
    Открытие бара у Bybit равно закрытию предыдущего, поэтому без него
    первый бар каждого архива получал бы открытие по первой сделке —
    раз в месяц, на стыке файлов, и выглядело бы это как гэп рынка.

    Бары без единой сделки биржа всё равно публикует (O=H=L=C=закрытие
    предыдущего, объём ноль) — повторяем это же: дыры в сетке базового ТФ
    ломают скользящие окна признаков сильнее, чем плоский бар.
    """
    empty = pd.DataFrame(columns=OHLCV_COLUMNS)
    if ticks.empty:
        return empty

    df = pd.DataFrame({
        "price": pd.to_numeric(ticks["price"], errors="coerce"),
        "volume": pd.to_numeric(ticks["volume"], errors="coerce"),
        "side": ticks["side"].astype(str).str.lower(),
    })
    df.index = _tick_timestamps(ticks["timestamp"])
    df = df.dropna(subset=["price", "volume"])

    # Внутри одной миллисекунды порядок строк в архиве НЕ совпадает с порядком
    # исполнения, а закрытие бара — это цена последней исполненной сделки.
    # Одна агрессивная заявка выедает стакан от лучшей цены к худшей: покупка
    # снизу вверх, продажа сверху вниз. Восстанавливаем этот порядок ключом
    # по цене со знаком стороны — на августе 2025 он даёт ровно ноль
    # расхождений закрытия с klines биржи против 26 баров из 1000 при порядке
    # строк файла. Обе сортировки устойчивые, иначе восстановленный порядок
    # тут же перемешается обратно.
    df["_fill_order"] = np.where(df["side"] == "buy", df["price"], -df["price"])
    df = df.sort_values("_fill_order", kind="stable").sort_index(kind="stable")
    if df.empty:
        return empty

    df["quote"] = df["price"] * df["volume"]
    df["buy"] = np.where(df["side"] == "buy", df["volume"], 0.0)

    grouped = df.resample(f"{_minutes(tf)}min", label="left", closed="left")
    out = pd.DataFrame({
        "high": grouped["price"].max(),
        "low": grouped["price"].min(),
        "close": grouped["price"].last(),
        "volume": grouped["volume"].sum(),
        "quote_volume": grouped["quote"].sum(),
        "trades": grouped["price"].count(),
        "taker_buy_base": grouped["buy"].sum(),
    })

    # Пустые бары: цена держится на последнем закрытии, объёмы нулевые.
    out["close"] = out["close"].ffill()
    if prev_close is not None:
        out["close"] = out["close"].fillna(prev_close)
    out = out.dropna(subset=["close"])
    if out.empty:
        return empty

    # Биржевая конвенция: открытие = закрытие предыдущего бара. Для самого
    # первого бара выборки предыдущего нет — берём его собственное закрытие,
    # если не передали хвост из базы.
    first_open = prev_close if prev_close is not None else float(out["close"].iloc[0])
    out["open"] = out["close"].shift(1).fillna(first_open)
    # Открытие входит в диапазон бара: у биржи high/low считаются с ним
    # заодно, и без этого high оказывается ниже открытия на разрывах.
    out["high"] = out[["high", "open"]].max(axis=1)
    out["low"] = out[["low", "open"]].min(axis=1)

    out[["volume", "quote_volume", "trades", "taker_buy_base"]] = (
        out[["volume", "quote_volume", "trades", "taker_buy_base"]].fillna(0)
    )
    out["symbol"] = symbol
    out["tf"] = tf
    return out.reset_index(names="ts")[OHLCV_COLUMNS]


def _minutes(tf: str) -> int:
    return config.TIMEFRAME_MINUTES[tf]


def has_month(symbol: str, ym: str, client: httpx.Client) -> bool:
    """Есть ли месячный тиковый архив за `ym` (формат YYYY-MM)."""
    url = MONTHLY_URL.format(sym=symbol.upper(), ym=ym)
    return client.head(url).status_code == 200


def _store_ticks(payload: bytes, symbol: str, tf: str, first_ts: pd.Timestamp | None = None) -> int:
    ticks = parse_ticks(payload)
    if ticks.empty:
        return 0
    stamps = _tick_timestamps(ticks["timestamp"])
    prev = bars.last_close(symbol, tf, first_ts if first_ts is not None else stamps.min())
    frame = ticks_to_bars(ticks, symbol, tf, prev_close=prev)
    return bars.store_bars(frame)


def sync_history(
    symbol: str | None = None,
    tf: str | None = None,
    start: str | None = None,
    end: datetime | None = None,
    progress=None,
) -> dict:
    """
    Качает месячные тиковые архивы за весь период и кладёт бары в ohlcv.

    Уже скачанные архивы берутся с диска, уже загруженные бары перезаписываются
    upsert'ом — команду можно гонять повторно без вреда.
    """
    symbol = (symbol or config.data.symbol).upper()
    tf = tf or config.data.base_tf
    start_dt = pd.Timestamp(start or config.data.history_start, tz="UTC").to_pydatetime()
    end_dt = end or datetime.now(timezone.utc)

    months = list(_months(start_dt, end_dt))
    total_rows, missing = 0, []

    with httpx.Client(timeout=180.0, follow_redirects=True) as client:
        for i, ym in enumerate(months):
            url = MONTHLY_URL.format(sym=symbol, ym=ym)
            payload = _fetch(url, _cache_path(symbol, ym), client)
            if payload is None:
                missing.append(ym)
                if progress:
                    progress(i + 1, len(months), f"{ym}: архива нет")
                continue
            rows = _store_ticks(payload, symbol, tf, pd.Timestamp(f"{ym}-01", tz="UTC"))
            total_rows += rows
            if progress:
                progress(i + 1, len(months), f"{ym}: {rows} баров")

        # Текущий месяц закрывается дневными архивами, остаток — через REST.
        total_rows += _sync_daily_tail(symbol, tf, client, progress)

    total_rows += sync_recent(symbol, tf)
    return {"months": len(months), "rows": total_rows, "missing_months": missing}


def _sync_daily_tail(symbol: str, tf: str, client: httpx.Client, progress=None) -> int:
    """
    Дневные архивы за текущий месяц: месячный появится только после его конца.

    Захватывает и последние дни прошлого месяца: месячный архив выкладывается
    не в полночь первого числа, и без этого нахлёста в истории оставалась бы
    дыра шириной в несколько суток ровно на границе месяцев.
    """
    today = datetime.now(timezone.utc).date()
    day = (today.replace(day=1) - timedelta(days=3))
    rows = 0
    while day < today:
        ymd = day.strftime("%Y-%m-%d")
        payload = _fetch(
            DAILY_URL.format(sym=symbol, ymd=ymd), _cache_path(symbol, ymd), client
        )
        if payload is not None:
            rows += _store_ticks(payload, symbol, tf, pd.Timestamp(ymd, tz="UTC"))
        day += timedelta(days=1)
    if progress:
        progress(1, 1, f"дневной хвост: {rows} баров")
    return rows


def sync_recent(symbol: str | None = None, tf: str | None = None, limit_batches: int = 20) -> int:
    """
    Добирает свежие бары, начиная с последнего сохранённого.

    Два источника, и порядок между ними важен:

    1. **Дневные тиковые архивы** за уже закрытые сутки — полный набор полей.
    2. **REST kline** за незакрытые сутки, которых в архивах ещё нет. Здесь
       `trades` и `taker_buy_base` неизвестны, и это единственное место, где
       они остаются пустыми.

    Пустоту закрывает следующий же запуск: архив за эти сутки выйдет и
    перезапишет бары точными значениями (upsert по `(symbol, tf, ts)`).
    Поэтому REST берётся строго после архивов и только за то, что позже них, —
    иначе приблизительные бары затирали бы точные.

    Признаки такую дыру переживают: `taker_bias` усредняется скользящим окном
    в сутки и NaN в нём пропускаются, атомы доминирования на NaN дают False.
    """
    symbol = (symbol or config.data.symbol).upper()
    tf = tf or config.data.base_tf

    rows = 0
    with httpx.Client(timeout=180.0, follow_redirects=True) as client:
        rows += _sync_daily_tail(symbol, tf, client)

    last = bars.last_ts(symbol, tf)
    if last is None:
        from btcproc import symbols

        try:
            fallback = symbols.get(symbol).start_date()
        except symbols.UnknownSymbolError:
            fallback = config.data.history_start
        start_ms = int(pd.Timestamp(fallback, tz="UTC").timestamp() * 1000)
    else:
        start_ms = int(last.timestamp() * 1000) + 1

    interval = _REST_INTERVAL.get(tf)
    if interval is None:
        raise ValueError(f"Bybit не отдаёт klines для таймфрейма {tf}")

    bar_ms = _minutes(tf) * 60_000
    with httpx.Client(timeout=30.0) as client:
        for _ in range(limit_batches):
            resp = client.get(
                REST_URL,
                params={
                    "category": "spot", "symbol": symbol, "interval": interval,
                    "start": start_ms, "limit": 1000,
                },
            )
            if resp.status_code in (403, 451):
                # Гео-блокировка, а не сбой: с американского хоста REST Bybit
                # закрыт целиком, вместе со всеми зеркалами. Ронять прогон
                # нельзя — тогда `live` монеты падал бы каждые полчаса,
                # хотя тиковые архивы качаются и данные идут. Отставание в
                # этом режиме — до суток, ровно до выхода дневного архива.
                logger.warning(
                    "REST Bybit недоступен (%s), хвост берётся только из "
                    "дневных архивов: отставание до суток. Адрес — %s",
                    resp.status_code, REST_URL,
                )
                break
            resp.raise_for_status()
            body = resp.json()
            if body.get("retCode") != 0:
                raise RuntimeError(f"Bybit: {body.get('retMsg')} (retCode {body.get('retCode')})")
            batch = body.get("result", {}).get("list") or []
            if not batch:
                break
            # Bybit отдаёт от новых к старым — нам нужен хронологический порядок.
            df = pd.DataFrame(
                batch, columns=["ts", "open", "high", "low", "close", "volume", "turnover"],
            ).astype(float).sort_values("ts")
            # Последний бар ещё не закрыт — в историю его брать нельзя.
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            df = df[df["ts"] + bar_ms <= now_ms]
            if df.empty:
                break

            frame = pd.DataFrame({
                "symbol": symbol,
                "tf": tf,
                "ts": pd.to_datetime(df["ts"], unit="ms", utc=True),
                "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
                "volume": df["volume"], "quote_volume": df["turnover"],
                "trades": 0,
                "taker_buy_base": np.nan,
            })
            rows += bars.store_bars(frame[OHLCV_COLUMNS])
            start_ms = int(df["ts"].iloc[-1]) + 1
            if len(batch) < 1000:
                break
    return rows
