"""
Нижние панели графика: объём, атомы бара и серия признака.

Панель признака читает `features` — те самые числа, на которых обучалась
модель, а не пересчитанный в браузере индикатор. Отсюда два требования, оба
проверяются здесь: набор признаков берётся у МОДЕЛИ прогона (live-прогон
разыменовывается в свой train), а индекс значения в массиве — из состава
набора, записанного в `feature_sets`, а не из порядка колонок в коде.

Третье требование — про атомы: отсутствие строки в `bar_events` и пустой
список атомов на баре различаются. Схлопывание превратило бы «этот бар ещё не
считали» в «на этом баре ничего не происходило».

БД не нужна: fetch_* и model_root подменяются.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from btcproc.admin import queries

LIVE_RUN, TRAIN_RUN = 41, 40
NAMES = ["ret_1h", "rv_rank", "rsi", "taker_bias"]


def _ts(day: int) -> datetime:
    return datetime(2026, 8, day, tzinfo=timezone.utc)


def _bar(day: int, atoms, context) -> dict:
    return {
        "ts": _ts(day), "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
        "volume": 100.0, "group_id": 3.0, "is_transition": False,
        "transition_id": None, "age_bucket": "fresh", "entropy": "low",
        "atoms": atoms, "context_atoms": context,
    }


@pytest.fixture
def model(monkeypatch):
    """Прогон #41 (live) обучен моделью #40 на наборе v1 из NAMES."""
    monkeypatch.setattr(
        queries.runs_repo, "model_root",
        lambda run_id: TRAIN_RUN if run_id == LIVE_RUN else run_id,
    )

    def fake_fetch_one(sql, params=None):
        if "state_models" in sql:
            assert params == (TRAIN_RUN,), "набор берётся у модели, а не у live-прогона"
            return {"feature_ver": "v1"}
        if "feature_sets" in sql:
            return {"names": NAMES}
        return None

    monkeypatch.setattr(queries, "fetch_one", fake_fetch_one)
    return monkeypatch


def test_missing_bar_events_row_is_not_an_empty_atom_list(monkeypatch):
    """
    Хвост свежее последнего прогона событий не размечен вовсе. Это должно
    доезжать до интерфейса как «нет разметки», а не как «нет событий»:
    второе читается как спокойный рынок и врёт.
    """
    monkeypatch.setattr(queries.runs_repo, "model_root", lambda run_id: run_id)
    monkeypatch.setattr(queries, "fetch_one", lambda sql, params=None: None)
    monkeypatch.setattr(
        queries, "fetch_all",
        lambda sql, params=None: (
            [_bar(1, ["volume_spike"], ["asia_session"]), _bar(2, [], []), _bar(3, None, None)]
            if "ohlcv" in sql else []
        ),
    )

    bars = queries.chart_data(TRAIN_RUN, symbol="BTCUSDT")["bars"]

    assert bars[0]["atoms"] == ["volume_spike"]
    assert bars[1]["atoms"] == [], "событий на баре нет — это пустой список"
    assert bars[2]["atoms"] is None, "бар без строки в bar_events — это None"


def test_bars_carry_volume_and_atom_labels(monkeypatch):
    """Объём и русские подписи атомов — то, ради чего появились панели."""
    monkeypatch.setattr(queries.runs_repo, "model_root", lambda run_id: run_id)
    monkeypatch.setattr(queries, "fetch_one", lambda sql, params=None: None)
    monkeypatch.setattr(
        queries, "fetch_all",
        lambda sql, params=None: (
            [_bar(1, ["volume_spike"], ["asia_session"])] if "ohlcv" in sql else []
        ),
    )

    data = queries.chart_data(TRAIN_RUN, symbol="BTCUSDT")

    assert data["bars"][0]["volume"] == 100.0
    # Подписи едут одним словарём на ответ, а не строкой в каждом баре.
    assert data["atom_labels"] == {
        "asia_session": "азиатская сессия", "volume_spike": "всплеск объёма",
    }


def test_indicator_index_comes_from_the_stored_feature_set(model, monkeypatch):
    """
    Значение достаётся как `values[i]`, и `i` обязан считаться по составу
    набора из базы. Порядок колонок в коде мог уехать (новый источник
    признаков), а строки `features` остаются старыми — тогда панель молча
    показывала бы чужой признак.
    """
    seen: list[tuple] = []

    def fake_fetch_all(sql, params=None):
        seen.append((sql, tuple(params)))
        return [{"ts": _ts(1), "value": 0.62}]

    monkeypatch.setattr(queries, "fetch_all", fake_fetch_all)

    out = queries.indicator_series(LIVE_RUN, "BTCUSDT", "rsi")

    sql, params = seen[0]
    assert "values[%s]" in sql
    # rsi третий в наборе, массивы в postgres 1-based.
    assert params[:3] == (3, "BTCUSDT", "v1")
    assert out["points"] == [{"time": int(_ts(1).timestamp()), "value": 0.62}]
    assert out["note"] == ""


def test_indicator_absent_from_the_set_says_so(model, monkeypatch):
    """
    SMC-признаков нет в наборе v1. Запрос такого признака обязан объяснить
    пустую панель — иначе она неотличима от сломанной.
    """
    monkeypatch.setattr(queries, "fetch_all", lambda sql, params=None: [])

    out = queries.indicator_series(LIVE_RUN, "BTCUSDT", "near_ob")

    assert out["points"] == []
    assert "v1" in out["note"]


def test_indicator_series_asks_only_for_the_window(model, monkeypatch):
    """
    Границы окна обязаны уезжать в SQL. Без них панель тянула бы всю историю
    монеты — сотни тысяч строк ради полутора тысяч видимых баров.
    """
    seen: list[tuple] = []
    monkeypatch.setattr(
        queries, "fetch_all",
        lambda sql, params=None: seen.append((sql, tuple(params))) or [],
    )

    queries.indicator_series(LIVE_RUN, "BTCUSDT", "rsi",
                             start="2026-08-01", end="2026-08-02")

    sql, params = seen[0]
    assert sql.count("ts >=") == 1 and sql.count("ts <=") == 1
    assert params[3:5] == ("2026-08-01", "2026-08-02")


def test_indicator_series_is_capped_and_keeps_the_fresh_end(model, monkeypatch):
    """
    Запрос без границ окна (прямое обращение к API) не должен тянуть всю
    историю монеты. Потолок обязан резать СТАРЫЙ конец: без границ нужен
    свежий хвост, а `LIMIT` при сортировке по возрастанию отрезал бы его.
    """
    rows = [{"ts": _ts(day), "value": float(day)} for day in (3, 2, 1)]
    monkeypatch.setattr(queries, "fetch_all", lambda sql, params=None: rows)

    out = queries.indicator_series(LIVE_RUN, "BTCUSDT", "rsi")

    assert [p["value"] for p in out["points"]] == [1.0, 2.0, 3.0], (
        "строки приходят с конца истории и должны разворачиваться по времени"
    )


def test_transition_nan_does_not_reach_the_client(monkeypatch):
    """
    Бар без предыдущего состояния помечен строкой "NaN" — это артефакт pandas,
    а не идентификатор перехода. Наружу он уходил подписью «переход NaN»,
    которая читается как сбой.
    """
    monkeypatch.setattr(queries.runs_repo, "model_root", lambda run_id: run_id)
    monkeypatch.setattr(queries, "fetch_one", lambda sql, params=None: None)
    bar = _bar(1, [], [])
    bar["transition_id"] = "NaN"
    monkeypatch.setattr(
        queries, "fetch_all",
        lambda sql, params=None: [bar] if "ohlcv" in sql else [],
    )

    assert queries.chart_data(TRAIN_RUN, symbol="BTCUSDT")["bars"][0]["transition_id"] is None


def test_feature_without_a_phrase_is_offered_anyway(monkeypatch):
    """
    Признак без строки в словаре имён — нарушение инварианта 7, и его ловит
    тест полноты. Но прятать такой признак из селектора нельзя: молчание здесь
    выглядело бы как «признака нет в наборе», то есть маскировало бы дефект.
    """
    monkeypatch.setattr(queries.runs_repo, "model_root", lambda run_id: run_id)

    def fake_fetch_one(sql, params=None):
        if "state_models" in sql:
            return {"feature_ver": "vX"}
        return {"names": ["rsi", "выдуманный_признак"]}

    monkeypatch.setattr(queries, "fetch_one", fake_fetch_one)

    catalog = queries.indicator_catalog(TRAIN_RUN)
    stray = [item for item in catalog if item["name"] == "выдуманный_признак"]

    assert stray and stray[0]["axis"] == "без описания"


def test_catalog_offers_only_features_of_this_model(model):
    """
    Селектор показывает состав набора модели, а не всё, что умеет считать код:
    иначе половина пунктов давала бы пустую панель.
    """
    catalog = queries.indicator_catalog(LIVE_RUN)
    offered = [item["name"] for item in catalog]

    assert set(offered) == set(NAMES)
    # Порядок и подписи — из словаря имён состояний: панель и имя состояния
    # обязаны говорить об одном и том же одними словами.
    assert offered[0] == "rv_rank", "оси идут в порядке naming.AXES"
    assert {item["axis"] for item in catalog} == {"волатильность", "тренд",
                                                  "импульс", "объём"}
