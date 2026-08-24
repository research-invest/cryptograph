"""
Суррогатные данные: нулёвка для ВСЕГО конвейера, а не для одной функции.

Здесь закреплено ровно то, из-за чего суррогат легко сделать нулёвкой не того
вопроса — и на чём эта конструкция уже один раз ошиблась (2026-08-24, первая
версия модуля).

Ошибка была такая: геометрия бара (`H/C`, `L/C`) бралась у донора ТОЙ ЖЕ
позиции, чтобы сохранить суточный профиль размаха. В результате цель
`range_ratio`, которая считается ровно из `high`/`low`, наследовала реальную
кластеризацию волатильности — и суррогат становился данными, на которых
предсказывать ЕСТЬ что. «Ложное срабатывание» на таком материале не означало
бы ничего.

Правильная конструкция — перестановка ЦЕЛЫХ баров: в новую позицию переезжает
весь бар донора вместе с геометрией и объёмом, а разрушен ровно порядок во
времени. Тесты ниже проверяют именно это свойство, потому что по числам
конвейера отличить одно от другого невозможно.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcproc.analysis import surrogate as sg


def _bars(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    """
    Бары с кластеризацией волатильности И суточным профилем — обе структуры
    нужны, потому что суррогаты по-разному обходятся с каждой.

    Медленная волатильность задана ДЕТЕРМИНИРОВАННОЙ волной с периодом 1000
    баров, а не случайным процессом. Причина практическая: у почти единичного
    AR(1) выборочная автокорреляция на длинных лагах пляшет от зерна, и тест
    ловил бы шум генератора вместо свойства суррогата. Волна даёт ту же
    структуру («волатильность держится сериями») с точно известным ответом.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    slow = 1.0 + 0.8 * np.sin(2 * np.pi * np.arange(n) / 1000.0)
    daily = 1.0 + 0.5 * np.sin(2 * np.pi * index.hour.to_numpy() / 24.0)
    returns = rng.normal(0, 0.002, n) * slow * daily
    close = 100 * np.exp(np.cumsum(returns))
    spread = 0.001 * slow * daily
    return pd.DataFrame({
        "open": close * (1 - spread / 2), "high": close * (1 + spread),
        "low": close * (1 - spread), "close": close,
        "volume": rng.random(n) * 100,
    }, index=index)


def _abs_autocorr(close: pd.Series, lag: int) -> float:
    returns = pd.Series(np.abs(sg.log_returns(close)))
    return float(returns.autocorr(lag))


def _clustering_net_of_season(frame: pd.DataFrame, lag: int = 1) -> float:
    """
    Кластеризация ЗА ВЫЧЕТОМ суточной формы.

    Нужна отдельная величина, потому что сырая автокорреляция |r| смешивает
    две структуры: сезонность и собственно кластеризацию. Сезонная
    перестановка первую сохраняет намеренно, и по сырой автокорреляции её
    работу не увидеть — на ряде, где сезонность крупнее кластеризации, сырая
    величина после неё даже растёт.
    """
    returns = pd.Series(np.abs(sg.log_returns(frame["close"])),
                        index=frame.index[1:])
    hourly = returns.groupby(returns.index.hour).transform("mean")
    return float((returns / hourly).autocorr(lag))


def test_iaaft_destroys_volatility_clustering():
    """
    Смысл IAAFT: спектр и распределение те же, нелинейная структура — нет.

    Если кластеризация переживает суррогат, нулёвка перестаёт быть нулёвкой:
    у цели остаётся ровно то свойство, ради предсказания которого всё и
    делается.
    """
    base = _bars()
    fake = sg.surrogate_bars(base, "iaaft", np.random.default_rng(7))
    assert _clustering_net_of_season(base) > 0.15
    assert abs(_clustering_net_of_season(fake)) < 0.05


def test_block_keeps_clustering_inside_the_block_and_breaks_it_outside():
    """Блочная перестановка мягче IAAFT — и это её единственное назначение."""
    base = _bars()
    fake = sg.surrogate_bars(base, "block", np.random.default_rng(7), block=96)
    assert _clustering_net_of_season(fake) > 0.10, "внутри блока структура жива"
    # Лаг в период медленной волны: у исходного ряда там структура заведомо
    # есть, у переставленного блоками (блок 96 ≪ 1000) её быть не должно.
    assert _clustering_net_of_season(base, 1000) > 0.15
    assert abs(_clustering_net_of_season(fake, 1000)) < 0.05


def test_surrogate_is_a_permutation_of_real_bars():
    """
    Каждый бар суррогата — настоящий бар: та же геометрия, тот же объём.

    Это и есть исправление ошибки первой версии. Проверяется по объёму: он
    переезжает вместе с баром и не пересчитывается, поэтому мультимножество
    объёмов обязано совпадать с исходным.
    """
    base = _bars()
    fake = sg.surrogate_bars(base, "iaaft", np.random.default_rng(3))
    assert sorted(fake["volume"].round(9)) == sorted(base["volume"].round(9))
    assert len(fake) == len(base)
    assert (fake.index == base.index).all()


def test_bar_geometry_is_not_glued_to_its_original_time():
    """
    Размах бара обязан ПЕРЕЕХАТЬ вместе с баром, а не остаться на позиции.

    Регрессия на ошибку первой версии: там доли `H/C` копировались с бара той
    же позиции, и автокорреляция размаха сохранялась полностью — то есть цель
    оставалась предсказуемой, а тест на ложные срабатывания становился
    бессмысленным.
    """
    base = _bars()
    fake = sg.surrogate_bars(base, "iaaft", np.random.default_rng(11))
    real_range = ((base["high"] - base["low"]) / base["close"]).autocorr(1)
    fake_range = ((fake["high"] - fake["low"]) / fake["close"]).autocorr(1)
    assert real_range > 0.3
    assert abs(fake_range) < 0.05


def test_seasonal_permutation_preserves_the_daily_profile():
    """
    Сезонная перестановка — для вопросов, где сезонность разрушать НЕЛЬЗЯ.

    Час дня — крупнейший заведённый предиктор размаха в проекте, и нулёвка,
    которая его тоже ломает, отвечает на другой вопрос. Ровно на этой подмене
    был отозван гейт R деривативов (журнал 47.4).
    """
    base = _bars()
    fake = sg.surrogate_bars(base, "seasonal", np.random.default_rng(5))
    by_hour = lambda frame: ((frame["high"] - frame["low"]) / frame["close"]
                             ).groupby(frame.index.hour).mean()
    correlation = float(np.corrcoef(by_hour(base), by_hour(fake))[0, 1])
    assert correlation > 0.99, "суточный профиль размаха обязан уцелеть"
    # А кластеризация — за вычетом сезонности — обязана быть разрушена.
    assert (_clustering_net_of_season(fake)
            < 0.5 * _clustering_net_of_season(base))


def test_ohlc_order_is_repaired():
    """
    Доли считаются от `close` донора, а `close` бара стал другим — порядок
    H ≥ max(O, C) ≥ min(O, C) ≥ L донор не гарантирует. Бар с `low` выше
    `high` уронил бы расчёт признаков там, где причину искали бы в признаках.
    """
    base = _bars()
    for method in sg.METHODS:
        fake = sg.surrogate_bars(base, method, np.random.default_rng(1))
        assert (fake["high"] >= fake["low"]).all(), method
        assert (fake["high"] >= fake["close"]).all(), method
        assert (fake["low"] <= fake["close"]).all(), method


def test_returns_distribution_is_preserved_exactly():
    """У IAAFT маргинальное распределение доходностей совпадает поточечно."""
    base = _bars()
    fake = sg.surrogate_bars(base, "iaaft", np.random.default_rng(2))
    real = np.sort(sg.log_returns(base["close"]))
    surrogate = np.sort(sg.log_returns(fake["close"]))
    assert np.allclose(real, surrogate, atol=1e-9)


def test_unknown_method_fails_loudly():
    with pytest.raises(ValueError, match="Неизвестный способ"):
        sg.surrogate_bars(_bars(600), "shuffle", np.random.default_rng(0))
