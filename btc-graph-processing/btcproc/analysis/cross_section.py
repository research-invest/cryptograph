"""
Поперечное сечение: шесть монет как один объект — ТЗ
`crypto-graph/docs/tz_cross_section_20-08-26.md`.

## На какой вопрос отвечает

Не на «пойдёт ли цена вверх» — эта задача закрыта дважды (26.3, 26.4, 31), и
воскрешать её нельзя. Здесь другая: **обгонит ли монета корзину.** Целевая
величина — доходность за вычетом средней по корзине, то есть величина, из
которой общий рыночный фактор вычтен по построению. Другая цель, другая
нулёвка (IC = 0, а не «всегда long»), другой источник эффекта.

Отдельный модуль, а не расширение `samples.py`: тот загружает кандидатов ОДНОЙ
монеты по ОДНОЙ модели, здесь же нужен срез по всем монетам на общей сетке
времени, и модель состояний не участвует вовсе. `group_id` в этом модуле не
упоминается ни разу — намеренно (разметка нестабильна, 21.17).

## Две ловушки, названные до первой строки кода

**Выживаемость корзины.** Шесть монет отобраны владельцем задним числом, в
2026 году, то есть в корзину попали те, про кого УЖЕ известно, что они дожили
и остались ликвидными. Состав корзины здесь сделан зависящим от времени
(`membership`) — это чинит половину проблемы, «монета ещё не существовала».
Вторая половина наличными данными не чинится вовсе, пока вселенная не задана
механическим правилом. Поэтому любой положительный результат этого модуля —
**верхняя граница** эффекта, и в конвейер он не идёт без расширения вселенной.
Отрицательный результат от смещения не страдает: оно работает в плюс.

**Общерыночная величина, показанная шесть раз.** Урок FGI: частоты атомов у
BTC и ETH совпали до десятой доли процента, потому что ряд был один на все
монеты. Величины здесь делятся по этой границе жёстко: `basket_dispersion` и
`btc_share_chg_1d` — ОДИН ряд, сколько бы монет ни было, и устойчивость у них
проверяется эпохами, а не монетами. Признак класса — максимальная попарная
корреляция ряда между монетами (`cross_symbol_correlation`), и печатать её
обязан любой отчёт по этому модулю.

## Правила, встроенные в код, а не в инструкцию

* бар, на котором в корзине меньше `MIN_BASKET` монет, **выбрасывается
  целиком**, а не достраивается последним известным значением: протяжка — это
  заглядывание в прошлое, которое выглядит как настоящее (на дневном ряде FGI
  проект на этом уже попался);
* корзина **равновесная**. Взвешенная по обороту превратила бы её в BTC с
  добавками, то есть в общерыночный ряд из абзаца выше;
* доходность перед ранжированием нормируется на СОБСТВЕННУЮ волатильность
  монеты, посчитанную по прошлому. Без этого ранг мерил бы разницу
  волатильностей, а не разницу движений;
* всё считается строго внутри одного `ts` и только из прошлого каждой монеты.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from btcproc import config, symbols
from btcproc.features import indicators as ind

#: Минимальный размер корзины. Ранг внутри корзины из двух монет — булев флаг
#: «кто из двоих выше», из трёх — три уровня; поперечная статистика на таком
#: сечении не имеет смысла. Объявлено в §2.2.2 ТЗ и не подбирается.
MIN_BASKET = 4

#: Прогрев монеты перед входом в корзину — 4 недели, как у `build_features`.
#: Нормировки монеты должны быть посчитаны на её собственной истории, а не на
#: первых днях листинга.
WARMUP = pd.Timedelta(weeks=4)

#: Окна величин в барах базового ТФ: час, сутки, месяц.
WINDOW_1H, WINDOW_1D, WINDOW_1M = 4, 96, 2880

#: Окно собственной волатильности монеты для нормировки доходности.
RV_WINDOW = WINDOW_1D

#: Потолок суммирования автокорреляций ряда IC, в БАРАХ. Короче, чем у
#: волатильности (`range_model.AUTOCORR_MAX_LAG_BARS` = 5000): месячная память
#: у поперечной корреляции была бы находкой, а не нормой.
AUTOCORR_MAX_LAG_BARS = 500

FIELDS = ("open", "high", "low", "close", "quote_volume")


@dataclass
class Basket:
    """
    Панель `ts × symbol` по каждому полю бара плюс состав корзины по времени.

    `membership` — булева панель того же размера: True, если монета на этом
    баре в корзине (прогрев прошёл И данные есть). Все величины модуля
    считаются только по True-ячейкам, а бары с составом меньше `MIN_BASKET`
    из выдачи исчезают.
    """

    frames: dict[str, pd.DataFrame]
    membership: pd.DataFrame

    @property
    def close(self) -> pd.DataFrame:
        return self.frames["close"]

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.membership.index

    @property
    def tickers(self) -> list[str]:
        return list(self.membership.columns)

    def size(self) -> pd.Series:
        """Число монет в корзине по барам. Печатается всегда: читать ранг, не
        зная N, нельзя."""
        return self.membership.sum(axis=1)

    def masked(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Обнулить в NaN всё, что вне корзины на своём баре."""
        return frame.where(self.membership)


# ─── Загрузка ───────────────────────────────────────────────────────────────
def load_basket(tf: str | None = None, start: str | None = None,
                end: str | None = None, tickers: list[str] | None = None,
                min_basket: int = MIN_BASKET) -> Basket:
    """
    Панель из `processing.ohlcv` по всем активным монетам.

    Индекс — объединение меток времени, а не пересечение: пересечение молча
    выбросило бы бары, где одна монета отстала на минуту, и сделало бы дыры
    невидимыми. Отсев по размеру корзины идёт ПОСЛЕ, явным правилом.
    """
    from btcproc.ingest import bars

    tf = tf or config.data.base_tf
    specs = [spec for spec in symbols.enabled()
             if tickers is None or spec.ticker in tickers]
    loaded: dict[str, pd.DataFrame] = {}
    for spec in specs:
        frame = bars.load_ohlcv(spec.ticker, tf, start or spec.start_date(), end)
        if not frame.empty:
            loaded[spec.ticker] = frame
    if not loaded:
        raise ValueError("в базе нет баров ни по одной монете корзины")

    index = pd.DatetimeIndex(sorted(set().union(*(f.index for f in loaded.values()))))
    frames = {
        field: pd.DataFrame(
            {ticker: frame[field].reindex(index) for ticker, frame in loaded.items()},
            index=index,
        )
        for field in FIELDS
    }
    membership = basket_membership(index, list(loaded), frames["close"])
    keep = membership.sum(axis=1) >= min_basket
    return Basket(
        frames={field: frame.loc[keep] for field, frame in frames.items()},
        membership=membership.loc[keep],
    )


def basket_membership(index: pd.DatetimeIndex, tickers: list[str],
                      close: pd.DataFrame) -> pd.DataFrame:
    """
    Кто в корзине на баре `t`: прогрев монеты прошёл И бар у неё есть.

    Первое условие — против заглядывания в состав корзины (§0.2 ТЗ): на баре
    2022 года TAOUSDT не должна влиять ни на один ранг, потому что её тогда не
    существовало. Второе — против протяжки: дыра у монеты означает, что на
    этом баре корзина мельче, а не что цена не изменилась.
    """
    columns = {}
    for ticker in tickers:
        listed = pd.Timestamp(symbols.get(ticker).start_date(), tz="UTC") + WARMUP
        columns[ticker] = (index >= listed) & close[ticker].notna().to_numpy()
    return pd.DataFrame(columns, index=index)


# ─── Величины ───────────────────────────────────────────────────────────────
def normalised_return(basket: Basket, window: int) -> pd.DataFrame:
    """
    Доходность за `window` баров, делённая на собственную волатильность монеты.

    Знаменатель — `realized_vol` по суткам, умноженная на `√window`: доходность
    за k баров имеет масштаб `rv · √k`, и без корня величина мерила бы длину
    окна. Волатильность считается по ПРОШЛОМУ и на баре `t` известна.
    """
    close = basket.close
    ret = np.log(close / close.shift(window))
    scale = pd.DataFrame(
        {ticker: ind.realized_vol(close[ticker], RV_WINDOW) for ticker in close},
        index=close.index,
    ) * np.sqrt(window)
    return basket.masked(ret / scale.where(scale > 0))


def cross_rank(frame: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """
    Ранг значения внутри корзины на своём баре, в [0, 1].

    Вырожденное сечение (все монеты дали одно и то же) даёт всем средний ранг,
    а не NaN: «никто никого не обогнал» — это измерение, а не его отсутствие.
    Строка, где значений меньше `MIN_BASKET`, уходит целиком в NaN: ранг из
    двух монет — булев флаг, и подмешивать его к рангам из шести нельзя.
    """
    masked = frame.where(membership)
    ranks = masked.rank(axis=1, pct=True, method="average")
    enough = masked.notna().sum(axis=1) >= MIN_BASKET
    return ranks.where(enough, other=np.nan)


def basket_return(basket: Basket, window: int) -> pd.Series:
    """Равновесная доходность корзины за `window` баров (лог, по участникам бара)."""
    close = basket.close
    ret = basket.masked(np.log(close / close.shift(window)))
    return ret.mean(axis=1, skipna=True)


def beta_to_basket(basket: Basket, window: int = WINDOW_1M) -> pd.DataFrame:
    """
    Скользящая бета монеты к равновесной корзине по барным доходностям.

    Ковариация и дисперсия — по прошлому окну, поэтому величина на баре `t`
    известна на баре `t`. Монета вне корзины беты не получает: считать её
    относительно корзины, в которую монета не входит, бессмысленно.
    """
    close = basket.close
    ret = basket.masked(np.log(close / close.shift(1)))
    market = ret.mean(axis=1, skipna=True)
    variance = market.rolling(window, min_periods=window // 2).var()
    out = {}
    for ticker in ret:
        covariance = ret[ticker].rolling(window, min_periods=window // 2).cov(market)
        out[ticker] = covariance / variance.where(variance > 0)
    return basket.masked(pd.DataFrame(out, index=ret.index))


def idiosyncratic_return(basket: Basket, window: int = WINDOW_1D) -> pd.DataFrame:
    """
    Доходность монеты за окно за вычетом её беты, умноженной на доходность
    корзины: то, что осталось от движения после общего фактора.

    Связана с `beta_to_basket` по построению — считается через неё, — поэтому
    в поправку на множественность обе входят как ОДНА гипотеза (правило
    зеркальных пар).
    """
    close = basket.close
    ret = basket.masked(np.log(close / close.shift(window)))
    market = basket_return(basket, window)
    beta = beta_to_basket(basket)
    return ret - beta.mul(market, axis=0)


def realised_vol_rank(basket: Basket) -> pd.DataFrame:
    """
    Ранг собственной волатильности монеты внутри корзины.

    **Считается по закрытиям**, а не по размаху, и это осознанно: у HYPEUSDT
    `high`/`low` расширены синтетическим `open` (`ingest/bybit.py`), и
    range-величина сравнивала бы её с остальными не по одному правилу.
    """
    close = basket.close
    rv = pd.DataFrame(
        {ticker: ind.realized_vol(close[ticker], RV_WINDOW) for ticker in close},
        index=close.index,
    )
    return cross_rank(basket.masked(rv), basket.membership)


def dispersion(basket: Basket, window: int = WINDOW_1H) -> pd.Series:
    """
    Поперечное стандартное отклонение нормированных доходностей на баре.

    **Общерыночная величина**: ряд один на все монеты. «Знак совпал на трёх
    монетах» для неё не подтверждение, а один замер, показанный трижды;
    устойчивость проверяется эпохами.
    """
    return normalised_return(basket, window).std(axis=1, skipna=True)


def btc_share_change(basket: Basket, window: int = WINDOW_1D) -> pd.Series:
    """
    Изменение доли BTC в суточном обороте корзины — тоже ОБЩЕРЫНОЧНАЯ величина.

    Оборот берётся в котируемой валюте (`quote_volume`): базовый объём шести
    монет несопоставим между собой по определению.
    """
    volume = basket.masked(basket.frames["quote_volume"])
    rolled = volume.rolling(window, min_periods=window // 2).sum()
    total = rolled.sum(axis=1, skipna=True)
    if "BTCUSDT" not in rolled:
        return pd.Series(np.nan, index=basket.index)
    share = rolled["BTCUSDT"] / total.where(total > 0)
    return share.diff(window)


def measures(basket: Basket) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """
    Все величины ТЗ разом, разложенные по классу.

    Первый словарь — помонетные (H1–H3), второй — общерыночные (H4–H5). Класс
    определяет способ проверки устойчивости, поэтому он часть возвращаемого
    значения, а не комментарий рядом.
    """
    per_symbol = {
        "xs_rank_ret_1h": cross_rank(normalised_return(basket, WINDOW_1H),
                                     basket.membership),
        "xs_rank_ret_1d": cross_rank(normalised_return(basket, WINDOW_1D),
                                     basket.membership),
        "beta_basket_1m": beta_to_basket(basket),
        "idio_ret_1d": idiosyncratic_return(basket),
        "xs_rank_rv": realised_vol_rank(basket),
    }
    market_wide = {
        "basket_dispersion": dispersion(basket),
        "btc_share_chg_1d": btc_share_change(basket),
    }
    return per_symbol, market_wide


# ─── Цель и измеритель ──────────────────────────────────────────────────────
def cross_forward_return(basket: Basket, horizon_bars: int,
                         log: bool = True) -> pd.DataFrame:
    """
    `xs_fwd_ret(i, t, H)` — доходность монеты вперёд минус средняя по корзине
    на баре `t`.

    **`log=False` — не косметика, а обязательная контрольная проверка.**
    Логарифмическая доходность систематически ниже простой на величину σ²/2
    (неравенство Йенсена), и разница растёт с волатильностью монеты и с
    горизонтом. Значит ЛЮБОЙ предиктор, коррелированный с волатильностью —
    ранг `rv`, бета к корзине, — получает на логарифмической цели
    отрицательный поперечный IC, который к рынку отношения не имеет вовсе:
    это арифметика, а не поведение цен. Отличить одно от другого можно
    единственным способом — пересчитать ту же ячейку на простой доходности.

    Величина с нулевым средним по сечению по построению: общий рыночный фактор
    в ней отсутствует. Это и есть формальная причина, по которой замер не
    воскрешает закрытую задачу про направление.

    Единственная функция модуля, которая смотрит вперёд, — и поэтому она здесь
    одна и названа явно.
    """
    close = basket.close
    ratio = close.shift(-horizon_bars) / close
    forward = basket.masked(np.log(ratio) if log else ratio - 1.0)
    return forward.sub(forward.mean(axis=1, skipna=True), axis=0)


def _row_ranks(frame: pd.DataFrame, valid: np.ndarray) -> np.ndarray:
    """
    Ранги ПО СТРОКЕ, считанные только по валидным ячейкам, связки — средним.

    Векторно и через pandas, а не питоновским циклом по барам: панель в двести
    тысяч строк, и цикл здесь стоил десятки минут на ячейку замера. Связки
    обязаны получать средний ранг: иначе две одинаковые доходности
    различались бы порядком колонок, то есть алфавитом тикеров.
    """
    masked = frame.where(pd.DataFrame(valid, index=frame.index,
                                      columns=frame.columns))
    return masked.rank(axis=1, method="average").to_numpy(dtype=float)


def _centered(values: np.ndarray) -> np.ndarray:
    """Отклонения от среднего строки; пустая строка даёт нули, а не NaN."""
    mask = np.isfinite(values)
    counts = mask.sum(axis=1, keepdims=True)
    total = np.where(mask, values, 0.0).sum(axis=1, keepdims=True)
    mean = np.divide(total, counts, out=np.zeros_like(total),
                     where=counts > 0)
    return np.where(mask, values - mean, 0.0)


def _row_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Пирсон по строке на рангах — то есть Спирмен, построчно и векторно."""
    # Среднее по строке считается вручную, а не `nanmean`: строка без единой
    # валидной пары (корзина мельче порога) — штатная ситуация, а не повод
    # сыпать RuntimeWarning на каждый такой бар.
    cx = _centered(x)
    cy = _centered(y)
    numerator = np.einsum("ij,ij->i", cx, cy)
    denominator = np.sqrt(np.einsum("ij,ij->i", cx, cx)
                          * np.einsum("ij,ij->i", cy, cy))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def _pair(predictor: pd.DataFrame, target: pd.DataFrame,
          min_basket: int) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Общие бары, маска валидных пар и ранги обеих панелей."""
    common = predictor.index.intersection(target.index)
    x = predictor.loc[common]
    y = target.loc[common].reindex(columns=predictor.columns)
    valid = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    enough = valid.sum(axis=1) >= min_basket
    valid = valid & enough[:, None]
    return common, valid, _row_ranks(x, valid), _row_ranks(y, valid)


def information_coefficient(predictor: pd.DataFrame, target: pd.DataFrame,
                            min_basket: int = MIN_BASKET) -> pd.Series:
    """
    IC(t) — ранговая корреляция Спирмена МЕЖДУ МОНЕТАМИ на одном баре.

    Считается только по монетам, у которых на этом баре есть и предиктор, и
    цель; строки с меньше чем `min_basket` парами уходят в NaN. Ряд `IC(t)` —
    обычный временной ряд, и вся дисциплина проекта (блочный бутстрап, длина
    блока по перекрытию окон) применяется к нему без изменений.

    Вырожденная строка — все значения предиктора равны — даёт NaN, а не ноль:
    корреляция с константой не определена, и подменять её нулём значит
    занижать оценку разброса IC.
    """
    common, _valid, rank_x, rank_y = _pair(predictor, target, min_basket)
    return pd.Series(_row_correlation(rank_x, rank_y), index=common)


def within_symbol(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Убрать из панели средний уровень КАЖДОЙ МОНЕТЫ, оставив только её
    собственную динамику.

    Зачем это обязательная процедура, а не диагностика. Поперечный IC на баре
    складывается из двух совершенно разных вещей: «какие монеты в среднем
    волатильнее и в среднем отстают» (between) и «когда монета волатильнее
    обычного, она отстаёт» (within). Первое — это шесть наблюдений, сколько бы
    баров ни было, и блочный бутстрап по времени про него ничего не знает:
    он видит двести тысяч строк и выдаёт `p = 0.001` там, где независимых
    объектов пять. Второе — настоящая поперечная закономерность.

    Замер 2026-08-21: `xs_rank_rv` даёт IC −0.037, а после вычитания средних
    по монете +0.003; `beta_basket_1m` — −0.040 и +0.002. То есть весь
    наблюдённый эффект был первого рода, а корзина из шести монет отобрана
    задним числом (§0.2 ТЗ). Поэтому величина, не пережившая эту процедуру, в
    выводы не идёт независимо от p-value.
    """
    return frame - frame.mean()


def surrogate_ic(predictor: pd.DataFrame, target: pd.DataFrame,
                 rng: np.random.Generator, min_basket: int = MIN_BASKET,
                 draws: int = 1, block: int = 1) -> np.ndarray:
    """
    Средний IC при перестановке предиктора МЕЖДУ МОНЕТАМИ. Возвращает массив
    из `draws` реплик — распределение среднего IC под нулевой гипотезой.

    Вторая, независимая нулёвка (§4.2 ТЗ). Она сохраняет поперечное
    распределение значений и разрушает ровно проверяемую связь — в отличие от
    блочного бутстрапа, который проверяет другую гипотезу (о среднем ряда IC).

    **`block` не украшение, и по умолчанию единица здесь опасна.** Перестановка,
    сделанная в каждом баре независимо, разрушает не только связь, но и
    временну́ю зависимость ряда `IC(t)`: реплики становятся независимыми во
    времени, дисперсия их среднего падает в разы, и p-value выходит
    анти-консервативным. Это видно на данных: `xs_rank_ret_1d` на 4h давала
    `p_сурр = 0.005` при `p_блок = 0.117` — расхождение не про рынок, а про
    измеритель. Поэтому перестановка делается ОДНА НА БЛОК подряд идущих
    баров, длиной как у блочного бутстрапа: внутри блока состав корзины
    практически неизменен, а память ряда сохраняется.

    Переставляются РАНГИ исходного предиктора, а не свежий шум: у ранга бывают
    связки (две монеты с одинаковым значением), и шум их не воспроизводит.
    """
    _common, valid, rank_x, rank_y = _pair(predictor, target, min_basket)
    rows, columns = valid.shape
    # Валидные позиции строки в исходном порядке — в начале, невалидные в хвосте.
    destination = np.argsort(~valid, axis=1, kind="stable")
    block = max(1, int(block))
    n_blocks = int(np.ceil(rows / block))
    out = np.empty(draws)
    for draw in range(draws):
        # Ключ постоянен внутри блока: порядок перестановки монет держится
        # столько же баров, сколько живёт память ряда.
        key = np.repeat(rng.random((n_blocks, columns)), block, axis=0)[:rows]
        key = np.where(valid, key, np.inf)
        source = np.argsort(key, axis=1, kind="stable")
        shuffled = np.full_like(rank_x, np.nan)
        np.put_along_axis(shuffled, destination,
                          np.take_along_axis(rank_x, source, axis=1), axis=1)
        correlations = _row_correlation(shuffled, rank_y)
        out[draw] = float(np.nanmean(correlations))
    return out


def integrated_autocorr_time(values: pd.Series, max_lag: int = 200) -> float:
    """
    Интегральное время автокорреляции ряда: `τ = 1 + 2·Σ ρ_k` по начальным
    положительным лагам.

    Именно эта величина, а не «первый лаг, где корреляция упала ниже 0.2»,
    показывает, во сколько раз дисперсия среднего ряда больше, чем у
    независимых наблюдений. Разница не теоретическая: у ряда `IC(t)` с
    автокорреляциями 0.45 / 0.21 / 0.08 порог 0.2 даёт лаг 3, а `τ = 2.5`, и
    блок длиной 3–4 делает тест анти-консервативным (замер: 8.7 ложных
    срабатываний на 100 при номинале 5).

    Суммирование обрывается на первом неположительном лаге — стандартный
    приём (initial positive sequence): дальше идёт шум оценки, и добавлять его
    в сумму значит удлинять блок случайным образом.
    """
    series = values.dropna()
    total = 0.0
    for lag in range(1, max_lag + 1):
        rho = series.autocorr(lag)
        if rho is None or not np.isfinite(rho) or rho <= 0:
            break
        total += float(rho)
    return 1.0 + 2.0 * total


def ic_block_length(ic: pd.Series, ts: pd.Series, horizon_minutes: int,
                    factor: int = 4) -> int:
    """
    Длина блока бутстрапа для ряда `IC(t)`: максимум из перекрытия окон
    горизонта и `factor · τ`.

    Оба входа обязательны по разным причинам, и это то же правило, что у
    `range_model.bootstrap_block`, но с другой мерой памяти. Перекрытие окон
    есть всегда: соседние бары делят почти весь горизонт. Собственная память
    ряда бывает длиннее: предиктор считается на скользящем окне, и `IC(t)`
    её наследует.

    Множитель 4 — не украшение и не подобран под результат: блочный бутстрап
    воспроизводит дисперсию среднего, только когда блок заметно длиннее
    времени корреляции. Уровень процедуры замерен на негативном контроле
    (`test_negative_control_of_independent_walks_is_not_significant`):
    при блоке по горизонту — 8.7% ложных при номинале 5%, при `4·τ` — 6.7%.
    """
    from btcproc.analysis.lift import block_length_rows

    by_horizon = block_length_rows(ts, horizon_minutes)
    by_memory = int(np.ceil(factor * integrated_autocorr_time(
        ic, max_lag=AUTOCORR_MAX_LAG_BARS)))
    return max(by_horizon, by_memory)


def cross_symbol_correlation(frame: pd.DataFrame) -> float:
    """
    Максимальная попарная корреляция ряда величины МЕЖДУ МОНЕТАМИ — признак
    класса из §0.3 ТЗ.

    Значение близкое к 1.0 означает, что величина общерыночная, как бы она ни
    называлась и в какой бы таблице ТЗ ни стояла. Печатать обязательно для
    КАЖДОЙ величины, включая заявленные как помонетные: величина, случайно
    оказавшаяся общей, должна ловиться здесь, а не в выводах.
    """
    correlation = frame.corr(min_periods=100)
    # copy=True обязателен: на pandas 3 `to_numpy` отдаёт read-only массив, и
    # `fill_diagonal` падает. Ловушка записана в CLAUDE.md проекта.
    values = correlation.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(values, np.nan)
    return float(np.nanmax(values)) if np.isfinite(values).any() else float("nan")


def minimum_detectable_ic(basket_sizes: pd.Series, block_length: int,
                          z: float = 2.80) -> float:
    """
    MDE среднего IC — считается ДО прогона и печатается в каждой ячейке.

    Под нулевой гипотезой дисперсия Спирмена на выборке из N объектов
    приближённо `1/(N−1)`, отсюда `SE ≈ 1/√((N−1)·n_эфф)`. Множитель 2.80 —
    двусторонняя α = 0.05 при мощности 80%.

    Ячейка, где наблюдённый эффект меньше MDE, интерпретации не подлежит
    независимо от p-value. Именно эта арифметика заставила перенести горизонты
    задачи C на 1h/2h/4h: на 24h MDE выше порога практической величины 0.02,
    то есть ячейка была бы непроходимой в принципе.
    """
    sizes = basket_sizes[basket_sizes >= MIN_BASKET]
    if sizes.empty or block_length <= 0:
        return float("nan")
    n_eff = len(sizes) / block_length
    degrees = float((sizes - 1).mean())
    if degrees <= 0 or n_eff <= 0:
        return float("nan")
    return z / np.sqrt(degrees * n_eff)
