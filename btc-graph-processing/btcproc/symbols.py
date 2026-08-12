"""
Реестр торгуемых пар.

Список живёт в коде, а не в окружении: дата листинга, заметки и точечные
переопределения порогов кластеризации — это знание о рынке, оно должно
версионироваться вместе с кодом и проходить ревью в диффе. `.env` остаётся
для того, что зависит от окружения; `SYMBOL` в нём становится «монетой
по умолчанию» для одиночных команд.

Ключевое следствие решения «своя модель состояний на каждую монету»:
**один прогон = одна монета**. Поэтому таблицы, привязанные к `run_id`
(`state_models`, `market_groups`, `transitions`, `event_blocks`), помонетны
автоматически, без колонки `symbol`. И поэтому же `group_id` осмыслен только
в паре `(symbol, run_id)` — номера состояний разных монет несопоставимы.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from btcproc import config


#: Площадки, с которых умеем брать бары. Значение — модуль-загрузчик в
#: `btcproc/ingest/`; диспетчер живёт в `btcproc/ingest/sources.py`.
#:
#: Площадка — свойство монеты, а не окружения: HYPEUSDT на споте Binance
#: не существует вовсе, и это не настройка, которую захочется поменять.
VENUES: tuple[str, ...] = ("binance_spot", "bybit_spot")


@dataclass(frozen=True)
class SymbolSpec:
    ticker: str

    # Раньше этой даты дампов на площадке просто нет.
    # Занижать безопасно (лишние месяцы дадут 404 и попадут в missing_months),
    # но каждый лишний месяц — это лишний HTTP-запрос при каждом ingest.
    # Проверять фактом, а не по памяти: см. resolve_history_start ниже.
    history_start: str

    enabled: bool = True

    # Откуда берутся бары. Дефолт — Binance: на нём заведены все монеты,
    # с которых начиналась система, и молчаливая смена площадки у них была бы
    # худшим из возможных изменений — цены разъехались бы на доли процента,
    # а модель состояний переучилась бы на другой рынок без единой ошибки.
    venue: str = "binance_spot"

    # Точечные переопределения StatesConfig для монет, которым общая формула
    # масштабирования min_group_size не подошла (раздел 7 задачи).
    # Применяются через dataclasses.replace — ключи обязаны быть полями
    # StatesConfig, иначе get() бросит понятную ошибку.
    states_overrides: dict[str, Any] = field(default_factory=dict)

    note: str = ""

    def states_config(self, base: config.StatesConfig | None = None) -> config.StatesConfig:
        """StatesConfig монеты: базовый конфиг с её переопределениями."""
        base = base if base is not None else config.states
        if not self.states_overrides:
            return base
        return replace(base, **self.states_overrides)

    def start_date(self) -> str:
        """
        Дата начала истории. Спека монеты перекрывает HISTORY_START из .env:
        общая дата для всех монет означала бы сотни лишних 404 на альткоинах.
        """
        return self.history_start or config.data.history_start


# Даты начала истории на Binance (спот, пары к USDT).
#
# Значения здесь ЗАНИЖЕНЫ до начала месяца и проверены только по BTCUSDT,
# на котором система работает. Для остальных монет перед первым ingest
# уточни фактом: `symbols.resolve_history_start("SOLUSDT")` перебирает месяцы
# и возвращает первый, за который дамп существует. Занижение безопасно —
# лишние месяцы дадут 404 и попадут в missing_months, — но каждый лишний
# месяц это лишний HTTP-запрос при каждом ingest.
SYMBOLS: tuple[SymbolSpec, ...] = (
    SymbolSpec(
        "BTCUSDT", "2017-08-01",
        note="Эталон: на нём откалиброваны пороги кластеризации и профиль оценки в btc-graph",
    ),
    SymbolSpec(
        "ETHUSDT", "2017-08-01",
        note="История сопоставима с BTC, ликвидность ниже",
    ),
    SymbolSpec(
        "SOLUSDT", "2020-08-01",
        note=(
            "Листинг 2020-08 (проверено resolve_history_start). Истории вдвое "
            "меньше, чем у BTC и ETH — это первая монета, на которой работает "
            "относительный порог дробления min_group_share"
        ),
    ),
    SymbolSpec(
        "HYPEUSDT", "2025-07-01",
        venue="bybit_spot",
        note=(
            "Первая монета не с Binance: пары HYPEUSDT на его споте нет вовсе "
            "(есть только бессрочный фьючерс), на Bybit она с 2025-07-11. "
            "Истории всего год с небольшим — вчетверо короче, чем у SOLUSDT, "
            "и профиль оценки в btc-graph ей нужен по пресету young. "
            "Кандидаты выходят ТОЛЬКО со scope=transition: на 8.4 тыс. снимков "
            "приходится 4.5 тыс. пар (переход × блок событий), и до "
            "min_sample_size=30 аналогов не добирается ни одна. Это следствие "
            "короткой истории, а не поломки — понижать порог ради появления "
            "scope=transition+event_block нельзя, выборка и так на пределе"
        ),
    ),
)

_BY_TICKER = {spec.ticker.upper(): spec for spec in SYMBOLS}


class UnknownSymbolError(ValueError):
    """
    Тикера нет в реестре.

    Отдельный класс, а не KeyError: опечатка в CLI не должна выглядеть
    как «нет данных по монете».
    """


def get(ticker: str) -> SymbolSpec:
    """Спека монеты. Неизвестный тикер → понятная ошибка со списком известных."""
    key = (ticker or "").strip().upper()
    spec = _BY_TICKER.get(key)
    if spec is None:
        raise UnknownSymbolError(
            f"Монета {ticker!r} не заведена в btcproc/symbols.py. "
            f"Известные: {', '.join(tickers())}. "
            "Новая монета добавляется туда же — SymbolSpec с датой листинга."
        )
    _validate_overrides(spec)
    _validate_venue(spec)
    return spec


def _validate_overrides(spec: SymbolSpec) -> None:
    """
    Опечатка в states_overrides иначе всплыла бы TypeError'ом в середине
    прогона — после закачки истории и получаса кластеризации.
    """
    if not spec.states_overrides:
        return
    known = {f.name for f in config.StatesConfig.__dataclass_fields__.values()}
    unknown = set(spec.states_overrides) - known
    if unknown:
        raise UnknownSymbolError(
            f"{spec.ticker}: states_overrides содержит неизвестные параметры "
            f"{sorted(unknown)}. Поля StatesConfig: {', '.join(sorted(known))}."
        )


def _validate_venue(spec: SymbolSpec) -> None:
    """
    Опечатка в площадке обязана падать здесь, а не там, где её последствия
    выглядят как отсутствие данных: диспетчер `ingest/sources.py` выбирает
    загрузчик по этой строке, и незнакомое значение означало бы монету,
    которую некому качать.
    """
    if spec.venue not in VENUES:
        raise UnknownSymbolError(
            f"{spec.ticker}: неизвестная площадка {spec.venue!r}. "
            f"Известные: {', '.join(VENUES)}."
        )


def enabled() -> list[SymbolSpec]:
    """Только активные монеты, в порядке объявления."""
    return [spec for spec in SYMBOLS if spec.enabled]


def tickers(only_enabled: bool = False) -> list[str]:
    source = enabled() if only_enabled else SYMBOLS
    return [spec.ticker for spec in source]


def default() -> SymbolSpec:
    """
    Монета по умолчанию (`SYMBOL` из .env) для одиночных команд.

    Если её нет в реестре — это ошибка конфигурации, и обнаружиться она должна
    на старте, а не на середине прогона.
    """
    return get(config.data.symbol)


def resolve(ticker: str | None) -> SymbolSpec:
    """Спека переданной монеты либо монеты по умолчанию."""
    return get(ticker) if ticker else default()


def resolve_many(
    tickers_arg: list[str] | tuple[str, ...] | None,
    all_symbols: bool = False,
) -> list[SymbolSpec]:
    """
    Разбор аргументов CLI `--symbol` (можно несколько) и `--all`.

    Без флагов — монета из .env: существующие скрипты и `make train`
    не должны сломаться от появления мультимонетности.
    """
    if all_symbols:
        if tickers_arg:
            raise UnknownSymbolError(
                "--all и --symbol вместе не имеют смысла: либо все активные, "
                "либо перечисленные явно."
            )
        active = enabled()
        if not active:
            raise UnknownSymbolError(
                "В btcproc/symbols.py нет ни одной активной монеты (enabled=True)."
            )
        return active

    if not tickers_arg:
        return [default()]

    # dict.fromkeys — дедупликация с сохранением порядка: «--symbol ETHUSDT
    # --symbol ETHUSDT» не должно означать два прогона одной монеты.
    return [get(t) for t in dict.fromkeys(tickers_arg)]


def validate_default() -> None:
    """
    Проверка на старте: монета по умолчанию есть в реестре.

    Зовётся рядом с проверкой плейсхолдеров паролей — по той же причине:
    дефолт, указывающий в никуда, обнаруживается иначе слишком поздно.
    """
    default()


def resolve_history_start(
    ticker: str,
    probe_from: str = "2017-07-01",
    venue: str = "binance_spot",
) -> str:
    """
    Ищет фактическую дату начала истории перебором месяцев.

    Хелпер для заведения новой монеты: ходит в сеть, поэтому в прогонах
    не используется. Настоящее начало истории — первый месяц, за который
    дамп существует.

        python3 -c "from btcproc import symbols; print(symbols.resolve_history_start('SOLUSDT'))"
        python3 -c "from btcproc import symbols; print(symbols.resolve_history_start('HYPEUSDT', venue='bybit_spot'))"

    Площадку приходится передавать явно: монеты в реестре ещё нет — она
    заводится по результату этого вызова.
    """
    from datetime import datetime, timezone

    import httpx
    import pandas as pd

    if venue not in VENUES:
        raise UnknownSymbolError(
            f"Неизвестная площадка {venue!r}. Известные: {', '.join(VENUES)}."
        )

    from btcproc.ingest import sources

    loader = sources.loader_for(venue)
    month = pd.Timestamp(probe_from, tz="UTC")
    now = pd.Timestamp(datetime.now(timezone.utc))

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        while month < now:
            if loader.has_month(ticker.upper(), month.strftime("%Y-%m"), client):
                return month.strftime("%Y-%m-01")
            month += pd.DateOffset(months=1)

    raise UnknownSymbolError(
        f"Дампов {ticker} на площадке {venue} не найдено с {probe_from}. "
        "Проверь написание тикера."
    )
