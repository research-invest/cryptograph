"""
Стенд поперечного сечения — задача A ТЗ
`crypto-graph/docs/tz_cross_section_20-08-26.md`, §2.5.

Проверяется стенд, а не рынок: в базу и в сеть тесты не ходят, панель строится
руками. Два теста здесь несущие, и без них остальные не имеют смысла:

* **позитивный контроль** — на панели, где ранг ЗАДАННО предсказывает
  относительную доходность, измеритель обязан это увидеть. Ноль без него
  неотличим от сломанного скрипта (§3.4 `extending_features.md`);
* **состав корзины зависит от времени** — монета с поздним листингом не влияет
  ни на один ранг до своей даты. Это единственная защита от заглядывания в
  СОСТАВ корзины, и тесты `test_features_do_not_look_ahead` его не ловят: они
  проверяют, что величина бара `t` не зависит от баров после `t`, а здесь
  зависимость идёт от решения, принятого в 2026 году.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcproc.analysis import cross_section as xs

TICKERS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "AAVEUSDT", "TAOUSDT")


def make_panel(n: int = 600, tickers: tuple[str, ...] = TICKERS,
               seed: int = 1, start: str = "2025-01-01") -> xs.Basket:
    """
    Панель независимых случайных блужданий на общей сетке.

    Даты листинга не подмешиваются: `membership` здесь True везде, чтобы тест
    проверял величину, а не состав. Тесты состава строят панель сами.
    """
    index = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(seed)
    frames: dict[str, pd.DataFrame] = {}
    close = pd.DataFrame(
        {t: 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n))) for t in tickers},
        index=index,
    )
    frames["close"] = close
    frames["open"] = close.shift(1).fillna(close.iloc[0])
    frames["high"] = close * 1.001
    frames["low"] = close * 0.999
    frames["quote_volume"] = pd.DataFrame(
        {t: rng.lognormal(10, 0.3, n) for t in tickers}, index=index
    )
    membership = pd.DataFrame(True, index=index, columns=list(tickers))
    return xs.Basket(frames=frames, membership=membership)


# ─── Позитивный и негативный контроль измерителя ────────────────────────────
def test_positive_control_recovers_a_planted_cross_sectional_signal():
    """
    Панель, где ранг предиктора ЗАДАННО связан с относительной доходностью:
    измеритель обязан вернуть IC, близкий к заданному.

    Связь вносится в цель напрямую (`target = predictor + шум`), потому что
    проверяется измеритель, а не способность рынка её породить.
    """
    index = pd.date_range("2025-01-01", periods=2000, freq="15min", tz="UTC")
    rng = np.random.default_rng(7)
    columns = list(TICKERS)
    predictor = pd.DataFrame(
        rng.normal(size=(len(index), len(columns))), index=index, columns=columns
    )
    target = predictor + rng.normal(0, 0.5, size=predictor.shape)
    target = target.sub(target.mean(axis=1), axis=0)

    ic = xs.information_coefficient(predictor, target)
    assert ic.notna().all()
    assert float(ic.mean()) > 0.5, f"средний IC {ic.mean():.3f} — сигнал не найден"


def test_negative_control_of_independent_walks_is_not_significant():
    """
    Сто панелей из независимых блужданий: доля ложных срабатываний ≤ 0.07 при
    номинале 0.05.

    Значимость считается блочным бутстрапом по ряду `IC(t)` — общим кодом
    (инвариант 11), потому что `IC(t)` это обычный временной ряд.

    Длина блока берётся у `ic_block_length`, и это НЕ формальность: при длине
    только по горизонту (4 бара) тест даёт 8.7 ложных срабатываний на 100 при
    номинале 5, потому что предиктор считается на скользящем окне и IC
    наследует его память. Правило `4·τ` возвращает уровень к 6.7%.
    """
    from btcproc.analysis.range_model import block_mean_p

    trials, rejected = 100, 0
    rng = np.random.default_rng(2026)
    for trial in range(trials):
        # 1200 баров, а не 400: на трёх сотнях точек ряда IC блочный бутстрап
        # меряет собственную дискретность, и уровень уплывает вверх (8/100 при
        # номинале 5). Это свойство короткой выборки, а не процедуры.
        basket = make_panel(1200, seed=500 + trial)
        predictor = xs.cross_rank(xs.normalised_return(basket, xs.WINDOW_1H),
                                  basket.membership)
        target = xs.cross_forward_return(basket, 4)
        ic = xs.information_coefficient(predictor, target).dropna()
        if len(ic) < 50:
            continue
        values = ic.to_numpy()
        block = xs.ic_block_length(ic, basket.index.to_series(), 60)
        # Двусторонность — через модуль: знак заранее не заявлен.
        p_up = block_mean_p(values, block, 300, rng)
        p_down = block_mean_p(-values, block, 300, rng)
        rejected += int(min(p_up, p_down) * 2 <= 0.05)
    assert rejected / trials <= 0.07, f"ложных срабатываний {rejected}/{trials}"


def test_surrogate_null_is_not_anti_conservative_in_time():
    """
    Суррогат, переставляющий монеты в каждом баре независимо, недооценивает
    дисперсию среднего IC: реплики становятся независимыми во времени, хотя
    настоящий ряд `IC(t)` автокоррелирован.

    Проверяется прямо: разброс средних по репликам при `block=1` заметно
    меньше, чем при блочной перестановке. Это и есть причина, по которой
    `measure_cross_section.py` передаёт сюда длину блока, а не оставляет
    единицу — на боевых данных расхождение выглядело как значимый эффект
    (`p_сурр = 0.005` против `p_блок = 0.117`).
    """
    basket = make_panel(3000, seed=13)
    predictor = xs.cross_rank(xs.normalised_return(basket, xs.WINDOW_1H),
                              basket.membership)
    target = xs.cross_forward_return(basket, 4)

    per_bar = xs.surrogate_ic(predictor, target, np.random.default_rng(14),
                              draws=40, block=1)
    blocked = xs.surrogate_ic(predictor, target, np.random.default_rng(14),
                              draws=40, block=32)
    # На синтетике фактор около 1.3 (0.0096 против 0.0125): побарная
    # перестановка занижает стандартную ошибку среднего примерно на треть,
    # то есть завышает z во столько же раз. Порог 1.2 — с запасом на шум
    # сорока реплик, а не подгонка под наблюдённое.
    assert blocked.std() > per_bar.std() * 1.2, (
        f"побарный суррогат {per_bar.std():.5f}, блочный {blocked.std():.5f} — "
        f"разницы нет, значит перестановка не сохраняет память ряда"
    )


def test_surrogate_null_destroys_the_planted_signal():
    """
    Суррогатная нулёвка (перестановка предиктора ВНУТРИ бара) обнуляет
    найденный эффект, сохраняя и временну́ю структуру, и распределение.

    Если бы перестановка ломала что-то ещё, средний IC суррогата отличался бы
    от нуля — и это был бы результат про измеритель, а не про рынок.
    """
    index = pd.date_range("2025-01-01", periods=1500, freq="15min", tz="UTC")
    rng = np.random.default_rng(11)
    columns = list(TICKERS)
    predictor = pd.DataFrame(
        rng.normal(size=(len(index), len(columns))), index=index, columns=columns
    )
    target = predictor + rng.normal(0, 0.5, size=predictor.shape)

    observed = float(xs.information_coefficient(predictor, target).mean())
    draws = xs.surrogate_ic(predictor, target, np.random.default_rng(12), draws=5)
    assert observed > 0.5
    assert len(draws) == 5
    assert abs(float(draws.mean())) < 0.1, (
        f"суррогат дал IC {draws.mean():.3f} — не нулёвка")


# ─── Состав корзины ─────────────────────────────────────────────────────────
def test_membership_ignores_a_coin_before_its_listing():
    """
    Монета не входит в корзину раньше `history_start + прогрев`.

    Прогрев обязателен: нормировки монеты должны быть посчитаны на её
    собственной истории, а не на первых днях листинга.
    """
    index = pd.date_range("2024-01-01", periods=200, freq="1D", tz="UTC")
    close = pd.DataFrame(1.0, index=index, columns=["BTCUSDT", "TAOUSDT"])
    membership = xs.basket_membership(index, ["BTCUSDT", "TAOUSDT"], close)

    listed = pd.Timestamp("2024-04-01", tz="UTC") + xs.WARMUP
    assert not membership.loc[membership.index < listed, "TAOUSDT"].any()
    assert membership.loc[membership.index >= listed, "TAOUSDT"].all()
    assert membership["BTCUSDT"].all()


def test_a_late_coin_does_not_change_any_earlier_rank():
    """
    Буквальная проверка §0.2 ТЗ: строки TAOUSDT за 2024+ не меняют НИ ОДНОГО
    значения на барах 2022 года.

    Это и есть заглядывание в состав корзины, отобранной задним числом, — то
    самое, которого не поймает ни один существующий тест на look-ahead.
    """
    n = 400
    index = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(3)
    old = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AAVEUSDT"]
    close = pd.DataFrame(
        {t: 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n))) for t in old},
        index=index,
    )
    without = xs.Basket(
        frames={"close": close}, membership=pd.DataFrame(True, index=index, columns=old)
    )
    # Та же панель плюс монета, которой в 2022 году не существовало.
    close_with = close.copy()
    close_with["TAOUSDT"] = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    membership = pd.DataFrame(True, index=index, columns=old + ["TAOUSDT"])
    membership["TAOUSDT"] = False  # прогрев не пройден — в корзину не входит
    with_late = xs.Basket(frames={"close": close_with}, membership=membership)

    a = xs.cross_rank(xs.normalised_return(without, xs.WINDOW_1H), without.membership)
    b = xs.cross_rank(xs.normalised_return(with_late, xs.WINDOW_1H), with_late.membership)
    pd.testing.assert_frame_equal(a, b[old])
    assert b["TAOUSDT"].isna().all()


def test_bar_with_a_small_basket_is_dropped_not_filled():
    """
    Бар, где в корзине меньше `MIN_BASKET` монет, уходит в NaN целиком — а не
    достраивается последним известным значением.

    Протяжка это заглядывание в прошлое, которое выглядит как настоящее: на
    дневном ряде FGI проект уже попался на позиционном сдвиге вместо
    календарного.
    """
    basket = make_panel(300, seed=4)
    basket.membership.iloc[100, 2:] = False  # осталось две монеты
    ranks = xs.cross_rank(xs.normalised_return(basket, xs.WINDOW_1H),
                          basket.membership)
    assert ranks.iloc[100].isna().all()


def test_degenerate_cross_section_gets_the_average_rank():
    """
    Все монеты дали одинаковую доходность → ранги определены (средний), а не
    NaN. «Никто никого не обогнал» — это измерение.
    """
    n = 200
    index = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    same = pd.DataFrame(
        {t: 100.0 * np.exp(np.arange(n) * 0.001) for t in TICKERS}, index=index
    )
    basket = xs.Basket(
        frames={"close": same},
        membership=pd.DataFrame(True, index=index, columns=list(TICKERS)),
    )
    ranks = xs.cross_rank(xs.normalised_return(basket, xs.WINDOW_1H),
                          basket.membership)
    row = ranks.dropna().iloc[-1]
    assert np.allclose(row.to_numpy(), 0.6), row.to_dict()


# ─── Инварианты величин ─────────────────────────────────────────────────────
def test_measures_do_not_look_ahead():
    """
    Величины, посчитанные на префиксе панели, совпадают с посчитанными на
    полной. Единственная функция модуля, смотрящая вперёд, —
    `cross_forward_return`, и она в этот набор не входит.
    """
    basket = make_panel(600, seed=5)
    cut = 400
    prefix = xs.Basket(
        frames={f: v.iloc[:cut] for f, v in basket.frames.items()},
        membership=basket.membership.iloc[:cut],
    )
    full_per, full_market = xs.measures(basket)
    part_per, part_market = xs.measures(prefix)
    for name, frame in part_per.items():
        pd.testing.assert_frame_equal(frame, full_per[name].iloc[:cut])
    for name, series in part_market.items():
        pd.testing.assert_series_equal(series, full_market[name].iloc[:cut])


def test_measures_are_scale_invariant():
    """
    Цена одной монеты × 128 — поперечные величины не изменились.

    Степень двойки, чтобы тест не ловил разъезд округления. Ранги и беты
    безразмерны по построению, но проверяется это всё равно: масштабная
    зависимость означала бы, что где-то потерялась нормировка.
    """
    basket = make_panel(600, seed=6)
    scaled = xs.Basket(
        frames={f: v.copy() for f, v in basket.frames.items()},
        membership=basket.membership,
    )
    for field in ("open", "high", "low", "close"):
        scaled.frames[field]["ETHUSDT"] = scaled.frames[field]["ETHUSDT"] * 128.0

    before_per, _ = xs.measures(basket)
    after_per, _ = xs.measures(scaled)
    for name in ("xs_rank_ret_1h", "xs_rank_ret_1d", "xs_rank_rv"):
        pd.testing.assert_frame_equal(before_per[name], after_per[name], atol=1e-9)


def test_cross_forward_return_has_zero_cross_sectional_mean():
    """
    Цель по построению центрирована по сечению: общий рыночный фактор в ней
    отсутствует. Это формальная причина, по которой замер не воскрешает
    закрытую задачу про направление.
    """
    basket = make_panel(400, seed=8)
    target = xs.cross_forward_return(basket, 16).dropna(how="all")
    assert np.allclose(target.mean(axis=1).dropna().to_numpy(), 0.0, atol=1e-12)


def test_market_wide_measure_is_identical_for_every_coin():
    """
    Общерыночная величина — ОДИН ряд, и это должно быть видно из типа: она
    возвращается Series, а не панелью. Тест держит границу классов из §0.3 ТЗ,
    на которой проект уже ошибся с FGI.
    """
    basket = make_panel(400, seed=9)
    _, market_wide = xs.measures(basket)
    for name, series in market_wide.items():
        assert isinstance(series, pd.Series), name


def test_cross_symbol_correlation_flags_a_shared_series():
    """
    Признак класса работает: величина, одинаковая у всех монет, даёт
    корреляцию около единицы независимо от того, как она названа.
    """
    n = 500
    index = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(10)
    shared = rng.normal(size=n)
    same = pd.DataFrame({t: shared for t in TICKERS}, index=index)
    different = pd.DataFrame({t: rng.normal(size=n) for t in TICKERS}, index=index)
    assert xs.cross_symbol_correlation(same) > 0.99
    assert abs(xs.cross_symbol_correlation(different)) < 0.3


def test_minimum_detectable_ic_matches_the_declared_arithmetic():
    """
    MDE считается по фактическому размеру корзины и длине блока.

    Числа сверяются с посчитанными вручную для боевой панели (203 031 бар при
    `N ≥ 4`, средневзвешенный `N−1` = 3.58): на 4h блок 16 даёт ≈0.013, на 24h
    блок 96 — ≈0.032. Вторая величина выше порога практической значимости 0.02,
    и именно поэтому горизонты задачи C перенесены на 1h/2h/4h.
    """
    sizes = pd.Series([4] * 122247 + [5] * 43764 + [6] * 37020)
    assert xs.minimum_detectable_ic(sizes, 16) == pytest.approx(0.0131, abs=0.001)
    assert xs.minimum_detectable_ic(sizes, 96) == pytest.approx(0.0322, abs=0.001)
    assert xs.minimum_detectable_ic(sizes, 4) < 0.02


def test_heteroscedastic_null_does_not_produce_a_negative_rank_ic():
    """
    Монеты независимы, волатильности РАЗНЫЕ и постоянные, связи между
    волатильностью и будущей доходностью нет вовсе. Проверяется, что стенд не
    выдумывает на этом отрицательный IC для ранга волатильности.

    Тест появился из-за конкретного подозрения. На боевых данных `xs_rank_rv`
    и `beta_basket_1m` дали устойчиво ОТРИЦАТЕЛЬНЫЙ IC, растущий по модулю с
    горизонтом, — а это ровно та форма, которую даёт неравенство Йенсена
    (логарифмическая доходность ниже простой на σ²/2). Оказалось, к ранговой
    метрике оно неприменимо в принципе: вычитание общего среднего по бару —
    одна и та же константа для всех монет, порядок она не меняет, а логарифм
    монотонен, поэтому ранг `xs_fwd_ret` совпадает с рангом простой
    доходности. Пересчёт с `log=False` дал числа, идентичные до четвёртого
    знака, — и это не совпадение, а тождество.

    Механическое смещение у центрирования всё же есть — монета вносит разброс
    в среднее, из которого сама вычитается, — но знак у него
    ПОЛОЖИТЕЛЬНЫЙ (+0.004…+0.012 при разбросе волатильностей в шестнадцать
    раз). То есть наблюдённый отрицательный эффект им не объясняется, а
    занижается. Тест закрепляет знак: если он однажды станет отрицательным,
    все выводы о поперечной доходности придётся пересматривать.
    """
    n = 20000
    sigmas = {"A": 0.001, "B": 0.002, "C": 0.004, "D": 0.008, "E": 0.016}
    rng = np.random.default_rng(7)
    index = pd.date_range("2021-01-01", periods=n, freq="15min", tz="UTC")
    close = pd.DataFrame(
        {t: 100 * np.exp(np.cumsum(rng.normal(0, s, n))) for t, s in sigmas.items()},
        index=index,
    )
    basket = xs.Basket(
        frames={"close": close, "open": close,
                "high": close * 1.001, "low": close * 0.999,
                "quote_volume": pd.DataFrame(1.0, index=index, columns=list(sigmas))},
        membership=pd.DataFrame(True, index=index, columns=list(sigmas)),
    )
    per_symbol, _ = xs.measures(basket)
    target = xs.cross_forward_return(basket, 16)
    ic = float(xs.information_coefficient(per_symbol["xs_rank_rv"], target).mean())
    assert ic > -0.005, (
        f"стенд выдумал отрицательный IC {ic:+.4f} там, где связи нет: "
        f"механическое смещение центрирования сменило знак"
    )


def test_rank_ic_is_the_same_for_log_and_simple_returns():
    """
    Тождество, а не совпадение: ранговый IC не зависит от того, считается цель
    в логарифмах или в простых доходностях.

    Логарифм монотонен, а вычитание среднего по корзине — общая для всех монет
    бара константа; ни то, ни другое порядок внутри бара не меняет. Отсюда
    практический вывод, который стоил одного лишнего прогона: контроль на
    артефакт Йенсена через `--simple-return` для ранговой метрики бесполезен,
    и подозрение такого рода надо снимать рассуждением, а не замером.
    """
    basket = make_panel(2000, seed=15)
    predictor = xs.cross_rank(xs.normalised_return(basket, xs.WINDOW_1D),
                              basket.membership)
    log_target = xs.cross_forward_return(basket, 16, log=True)
    simple_target = xs.cross_forward_return(basket, 16, log=False)

    log_ic = xs.information_coefficient(predictor, log_target)
    simple_ic = xs.information_coefficient(predictor, simple_target)
    pd.testing.assert_series_equal(log_ic, simple_ic, atol=1e-9)


def test_between_symbol_effect_does_not_survive_demeaning():
    """
    Панель, где связи ВНУТРИ монеты нет вовсе, а средние уровни монет
    подобраны так, что поперечный IC заведомо отрицателен.

    Это форма, в которой пришёл единственный «положительный» результат задачи
    C: `xs_rank_rv` и `beta_basket_1m` давали IC −0.037 и −0.040, устойчиво во
    всех эпохах, на отложенной части и обеими нулёвками, — а после вычитания
    средних по монете от эффекта оставалось +0.003 и +0.002. Весь он сидел в
    различиях МЕЖДУ шестью монетами, то есть в шести наблюдениях, отобранных
    задним числом; блочный бутстрап по времени про это не знает и честно
    печатает `p = 0.001` по двумстам тысячам баров.

    Тест закрепляет, что `within_symbol` такую конструкцию гасит.
    """
    n = 4000
    columns = ["A", "B", "C", "D", "E"]
    index = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(21)
    # Предиктор: у каждой монеты свой постоянный уровень плюс общий шум.
    levels = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    predictor = pd.DataFrame(
        levels + rng.normal(0, 0.1, size=(n, len(columns))),
        index=index, columns=columns,
    )
    # Цель: уровень монеты СТРОГО противоположен уровню предиктора, а внутри
    # монеты связи нет — только шум.
    target = pd.DataFrame(
        -levels + rng.normal(0, 0.1, size=(n, len(columns))),
        index=index, columns=columns,
    )

    raw = float(xs.information_coefficient(predictor, target).mean())
    within = float(xs.information_coefficient(
        xs.within_symbol(predictor), xs.within_symbol(target)).mean())

    assert raw < -0.9, f"между монетами связь должна быть видна, а IC {raw:+.3f}"
    assert abs(within) < 0.1, (
        f"после вычитания средних по монете осталось {within:+.3f} — "
        f"процедура не отделяет between от within"
    )
