"""
Выбор загрузчика по площадке монеты.

Монеты живут на разных биржах: HYPEUSDT на споте Binance не существует
вовсе, он есть на Bybit. Площадка объявлена в реестре (`SymbolSpec.venue`)
и дальше не всплывает нигде: прогоны, признаки и кандидаты видят одинаковые
строки в `ohlcv` независимо от происхождения.

Модули-загрузчики обязаны совпадать по интерфейсу — `sync_history`,
`sync_recent`, `has_month`. Диспетчер намеренно тонкий: как только он начнёт
знать про особенности конкретной биржи, различия расползутся по коду,
и «добавить площадку» перестанет быть работой на один модуль.
"""
from __future__ import annotations

import logging
from types import ModuleType
from typing import Sequence

from btcproc import config
from btcproc.ingest import bars

logger = logging.getLogger(__name__)


def loader_for(venue: str) -> ModuleType:
    """Модуль-загрузчик площадки. Импорт ленивый — каждый тянет pandas и httpx."""
    if venue == "binance_spot":
        from btcproc.ingest import binance

        return binance
    if venue == "bybit_spot":
        from btcproc.ingest import bybit

        return bybit

    from btcproc.symbols import VENUES, UnknownSymbolError

    raise UnknownSymbolError(
        f"Неизвестная площадка {venue!r}. Известные: {', '.join(VENUES)}."
    )


def loader_for_symbol(symbol: str) -> ModuleType:
    """Загрузчик монеты по её месту в реестре."""
    from btcproc import symbols

    return loader_for(symbols.get(symbol).venue)


def sync_history(
    symbol: str | None = None,
    tf: str | None = None,
    start: str | None = None,
    end=None,
    progress=None,
) -> dict:
    """История монеты с её площадки."""
    symbol = symbol or config.data.symbol
    return loader_for_symbol(symbol).sync_history(symbol, tf, start, end, progress)


def sync_recent(symbol: str | None = None, tf: str | None = None, **kwargs) -> int:
    """Свежий хвост баров монеты с её площадки."""
    symbol = symbol or config.data.symbol
    return loader_for_symbol(symbol).sync_recent(symbol, tf, **kwargs)


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
      * **Площадка тоже берётся из реестра**, у каждой монеты своя: с одной
        биржи качается вся история, с другой нет и половины пары.
      * **Падение одной монеты не останавливает остальные.** Сеть отвалилась
        на третьей из семи — остальные четыре всё равно должны загрузиться,
        иначе ночной прогон превращается в лотерею. Диагноз копится
        и возвращается вызывающему.
      * **Последовательно, а не параллельно.** Упирается всё в сеть и в
        запись пачками в одну таблицу; параллельная закачка дала бы выигрыш
        только до первого 429 от биржи.

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
            report = loader_for(spec.venue).sync_history(
                spec.ticker, tf, start or spec.start_date(),
                progress=(
                    (lambda i, n, msg, _t=spec.ticker: progress(i, n, f"{_t}: {msg}"))
                    if progress else None
                ),
            )
            report["venue"] = spec.venue
            if context:
                report["context"] = bars.rebuild_context_timeframes(spec.ticker)
            result["symbols"][spec.ticker] = report
            result["rows_total"] += report["rows"]
        except Exception as exc:  # noqa: BLE001 — одна монета не рушит пакет
            logger.exception("Загрузка %s упала", spec.ticker)
            result["failed"][spec.ticker] = f"{type(exc).__name__}: {exc}"

    result["ok"] = len(result["symbols"])
    result["total"] = len(specs)
    return result
