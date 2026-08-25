"""
Снимки стакана — единственный модуль проекта, который копит данные раньше,
чем известно, зачем они нужны.

ТЗ — `crypto-graph/docs/tz_wave_a_24-08-26.md`, задача E. Обоснование в одном
абзаце: у стакана **нет истории и не будет**. Свечи, метрики деривативов и
индексы настроения можно докачать за любой прошлый год; глубину рынка —
нельзя ни за какие деньги, её отдают только «сейчас». Значит вопрос «полезен
ли стакан» решается не раньше, чем через год накопления, а вопрос «копить ли»
— только сегодня, и каждый день промедления вычитается из будущей выборки
безвозвратно.

Отсюда правила, которые действуют, пока не появится отдельное ТЗ:

* **никаких признаков, атомов и гейтов** из этих данных не заводится;
* `train` и `live` таблицу `depth_snapshots` не читают вовсе;
* команда стоит в кроне отдельно, как `fetch-external` и `ingest-metrics`:
  сетевой поход в чужой API не должен ронять регулярный расчёт.

## Что снимается и почему именно так

Снимок — верхние `DEPTH_LEVELS` уровней каждой стороны, целиком, как их отдал
поставщик. **Сырьё, а не агрегаты**: посчитать дисбаланс из уровней можно
всегда, восстановить уровни из дисбаланса — никогда, а какие именно агрегаты
понадобятся через год, сейчас не знает никто.

Момент снимка усекается до секунды и берётся по **часам этой машины**, а не
из ответа биржи: Binance отдаёт `lastUpdateId` без времени, Bybit — свой `ts`,
и складывать в одну колонку две разные шкалы хуже, чем иметь одну свою с
известной погрешностью. Погрешность — задержка сети, единицы сотен
миллисекунд, и для минутной сетки она несущественна.

Одна монета не роняет остальные: сеть — самая ненадёжная часть контура, и
недоступность одной пары не повод потерять снимок по пяти другим.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import httpx

from btcproc import config
from btcproc.db.session import bulk_upsert

logger = logging.getLogger(__name__)

#: Адрес берётся из того же `BINANCE_REST_URL`, что и klines, а не пишется
#: здесь: `api.binance.com` отвечает 451 из ряда юрисдикций (боевой VPS —
#: как раз такая), и на нём переменная переопределена на публичное зеркало
#: `data-api.binance.vision`. Хардкод означал бы пустую таблицу стакана без
#: единой ошибки в расчёте — крон пишет в лог только сбой, а копится ряд,
#: который потом невозможно докупить.
BINANCE_URL = config.data.binance_rest_url.rsplit("/", 1)[0] + "/depth"
BYBIT_URL = "https://api.bybit.com/v5/market/orderbook"

#: Сколько уровней с каждой стороны. Двадцать — потолок дешёвого лимита
#: Binance (вес запроса 1 до 100 уровней) и одновременно та глубина, за
#: которой у ликвидных пар начинается шум от алгоритмов, переставляющих
#: заявки. Меняя число, помни: старые строки пересняты не будут.
LEVELS = 20

#: Таймаут одного запроса. Короткий намеренно: снимок, приехавший через
#: полминуты, относится уже к другому рынку, и лучше его не иметь вовсе, чем
#: положить в базу с чужим временем.
TIMEOUT = 5.0

COLUMNS = ("symbol", "ts", "venue", "best_bid", "best_ask",
           "bid_volume", "ask_volume", "levels")


def _pairs(raw) -> list[tuple[float, float]]:
    """[[«цена», «объём»], …] → [(цена, объём), …]. Обе биржи отдают строки."""
    return [(float(price), float(size)) for price, size in raw]


def fetch_binance(symbol: str, client: httpx.Client) -> tuple[list, list]:
    resp = client.get(BINANCE_URL, params={"symbol": symbol, "limit": LEVELS})
    resp.raise_for_status()
    body = resp.json()
    return _pairs(body["bids"]), _pairs(body["asks"])


def fetch_bybit(symbol: str, client: httpx.Client) -> tuple[list, list]:
    resp = client.get(BYBIT_URL,
                      params={"category": "spot", "symbol": symbol, "limit": LEVELS})
    resp.raise_for_status()
    body = resp.json()
    # У Bybit ошибка приезжает кодом 200 с ненулевым retCode — проверять
    # обязательно, иначе `result` окажется пустым и снимок будет «нулевым»,
    # а не отсутствующим.
    if body.get("retCode") not in (0, None):
        raise ValueError(f"Bybit: retCode={body.get('retCode')} {body.get('retMsg')}")
    result = body["result"]
    return _pairs(result["b"]), _pairs(result["a"])


FETCHERS = {"binance_spot": fetch_binance, "bybit_spot": fetch_bybit}


def snapshot(symbol: str, venue: str, client: httpx.Client,
             now: datetime | None = None) -> tuple | None:
    """
    Один снимок одной монеты. None, если стакан пуст с любой стороны.

    Пустая сторона — это не «объёма ноль», это сломанный или неполный ответ:
    у торгуемой пары обеих сторон не бывает пустыми. Класть такую строку
    нельзя — через год её нельзя будет отличить от настоящего нуля.
    """
    fetch = FETCHERS.get(venue)
    if fetch is None:
        raise ValueError(f"Для площадки {venue!r} снимок стакана не реализован")
    bids, asks = fetch(symbol, client)
    if not bids or not asks:
        logger.warning("%s: пустая сторона стакана, снимок пропущен", symbol)
        return None

    ts = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    return (
        symbol, ts, venue,
        bids[0][0], asks[0][0],
        sum(size for _, size in bids), sum(size for _, size in asks),
        json.dumps({"b": bids, "a": asks}),
    )


def collect(symbols_list: list[str] | None = None, all_symbols: bool = False) -> int:
    """
    Снимок по монетам реестра, одной пачкой в базу.

    Возвращает число сохранённых строк. Монета, по которой запрос сорвался,
    логируется и пропускается: потерять пять снимков из шести из-за одной
    недоступной пары — худший из возможных исходов для задачи, у которой
    ценность ровно в непрерывности ряда.
    """
    from btcproc import symbols as registry

    specs = registry.resolve_many(symbols_list, all_symbols)
    rows = []
    with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
        for spec in specs:
            try:
                row = snapshot(spec.ticker, spec.venue, client)
            except Exception as exc:  # noqa: BLE001 — причина неважна, важен пропуск
                logger.warning("%s: снимок стакана не снят (%s)", spec.ticker, exc)
                continue
            if row:
                rows.append(row)

    if not rows:
        return 0
    return bulk_upsert("depth_snapshots", COLUMNS, rows, ("symbol", "ts"))


def coverage() -> list[dict]:
    """Что уже накоплено — для `status` и для будущего разбора."""
    from btcproc.db.session import fetch_all

    return fetch_all(
        "SELECT symbol, count(*) AS n, min(ts) AS first_ts, max(ts) AS last_ts "
        "FROM depth_snapshots GROUP BY symbol ORDER BY symbol"
    )
