"""
Свежий хвост баров через tv-quotes-api — прокси к котировкам TradingView.

Появился ради одной задачи: REST Bybit отвечает 403 с боевого VPS (блокировка
по адресу, зеркала закрыты тем же), и хвост HYPEUSDT добирался только дневными
тиковыми архивами — то есть отставал до суток. Прокси стоит на машине, до
которой TradingView доступен, и отдаёт те же бары.

**Это источник хвоста, а не истории.** Историю по-прежнему качают тиковые
архивы `public.bybit.com`: только в них есть сторона агрессора, из которой
считаются `trades` и `taker_buy_base`. Здесь их нет — ровно то же ограничение,
что было у REST Bybit, и закрывается оно тем же способом: вышедший архив
перезаписывает бар точными значениями (upsert по `(symbol, tf, ts)`).

`quote_volume` остаётся пустым, а не считается как `volume × цена`.
Приближение легло бы в ту же колонку, что и точные обороты из архивов, и
`cross_section` сравнивал бы оценку с измерением, не отличая одно от другого.
Пустое значение — то же самое знание, но видимое.

Сверка перед включением (2026-09-03, HYPEUSDT, 15m, 313 общих баров):
расхождение с нашими барами 0.0000% по OHLC и объёму. Конвенция открытия бара
у TradingView совпадает с биржевой — той самой, которую воспроизводит
`bybit._store_ticks`.

Спецификация источника — `tv-quotes-api/docs/API.md`.
"""
from __future__ import annotations

import logging

import httpx
import numpy as np
import pandas as pd

from btcproc import config
from btcproc.ingest.bars import OHLCV_COLUMNS

logger = logging.getLogger(__name__)

# Площадка монеты → префикс символа TradingView.
VENUE_PREFIX = {"binance_spot": "BINANCE", "bybit_spot": "BYBIT"}

# Наши таймфреймы → таймфреймы источника. Совпадают везде, кроме суток:
# у TradingView это `1D` с большой буквы, и `1d` он отвергает как `400`.
# Часть наших сеток (3m, 6h, 12h) источник не знает вовсе, и молча подменять
# их соседними нельзя — отсутствие в словаре означает «этот ТФ не отсюда».
TIMEFRAMES = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "1d": "1D",
}

# Потолок источника (`PROVIDER__MAX_BARS`). Больше просить бессмысленно:
# запрос уйдёт в `400`, а не отдаст сколько сможет.
MAX_BARS = 5000


class TvQuotesError(RuntimeError):
    """Источник не отдал данные. Вызывающий решает, фатально это или нет."""


def enabled() -> bool:
    """Настроен ли источник. Без ключа он не источник, а `401` на каждый запрос."""
    return bool(config.data.tvq_api_key and config.data.tvq_url)


def tv_symbol(symbol: str, venue: str) -> str:
    """`HYPEUSDT` + `bybit_spot` → `BYBIT:HYPEUSDT`."""
    prefix = VENUE_PREFIX.get(venue)
    if prefix is None:
        raise TvQuotesError(f"Нет префикса TradingView для площадки {venue!r}")
    return f"{prefix}:{symbol.upper()}"


def fetch_bars(symbol: str, tf: str, venue: str, count: int) -> pd.DataFrame:
    """
    Последние `count` баров в формате `OHLCV_COLUMNS`.

    Запрашиваются именно последние N: у источника нет параметра «с момента»,
    он умеет только хвост заданной длины. Сколько баров нужно, считает
    вызывающий — здесь известна лишь длина.

    Незакрытый бар отбрасывается: источник отдаёт текущий бар как обычный,
    и без отсева в `ohlcv` попал бы `close` на середине интервала. Следующий
    прогон его бы перезаписал, но между прогонами признаки считались бы по
    цене, которой на закрытии не было.
    """
    tv_tf = TIMEFRAMES.get(tf)
    if tv_tf is None:
        raise TvQuotesError(f"tv-quotes-api не отдаёт таймфрейм {tf!r}")
    if not enabled():
        raise TvQuotesError("TVQ_API_KEY не задан — источник выключен")

    count = max(1, min(int(count), MAX_BARS))
    url = config.data.tvq_url.rstrip("/") + "/v1/history"
    try:
        resp = httpx.get(
            url,
            params={"symbol": tv_symbol(symbol, venue), "timeframe": tv_tf,
                    "bars": count},
            headers={"X-API-Key": config.data.tvq_api_key},
            timeout=config.data.tvq_timeout,
        )
    except httpx.HTTPError as exc:
        raise TvQuotesError(f"tv-quotes-api недоступен: {exc}") from exc

    if resp.status_code != 200:
        # Тело ошибки у источника всегда одной формы; если разобрать не
        # вышло — в диагностике важнее код, чем сырой HTML прокси.
        try:
            message = resp.json()["error"]["message"]
        except Exception:  # noqa: BLE001
            message = resp.text[:200]
        raise TvQuotesError(f"tv-quotes-api {resp.status_code}: {message}")

    body = resp.json()
    candles = body.get("candles") or []
    if body.get("meta", {}).get("stale"):
        # Просроченный кеш — законный ответ источника (stale-if-error), но для
        # добора хвоста он бесполезен: свежих баров в нём по определению нет.
        logger.warning(
            "tv-quotes-api отдал просроченный кеш (%s с) — свежих баров нет",
            body.get("meta", {}).get("age_sec"),
        )
    if not candles:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    df = pd.DataFrame(candles)
    ts = pd.to_datetime(df["time"], unit="s", utc=True)
    bar = pd.Timedelta(minutes=config.TIMEFRAME_MINUTES[tf])
    closed = ts + bar <= pd.Timestamp.now(tz="UTC")

    frame = pd.DataFrame({
        "symbol": symbol.upper(),
        "tf": tf,
        "ts": ts,
        "open": df["open"], "high": df["high"], "low": df["low"],
        "close": df["close"],
        "volume": df["volume"],
        # См. шапку модуля: оба поля есть только в тиковых архивах, а оборот
        # в котируемой валюте источник не отдаёт вовсе.
        "quote_volume": np.nan,
        "trades": 0,
        "taker_buy_base": np.nan,
    })[closed.values]
    return frame[OHLCV_COLUMNS].dropna(subset=["open", "high", "low", "close"])
