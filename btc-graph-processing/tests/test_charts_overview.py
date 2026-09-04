"""
Страница «все инструменты»: свечи всех монет одним запросом.

Проверяется то, что ломается молча: окно считается от последнего бара КАЖДОЙ
монеты (общий срез «сейчас минус 12 часов» показал бы отставшую монету пустой
плиткой, хотя данные у неё есть), монета без баров не выпадает из ответа, а
остаётся в нём пустой — иначе на странице просто исчезала бы плитка, — и
изменение цены считается от открытия окна, а не от его первого закрытия.

БД не нужна: fetch_all подменяется.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from btcproc.admin import queries

TICKERS = ["BTCUSDT", "ETHUSDT"]
NOW = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)


def _rows(spec: dict[str, list[tuple[datetime, float, float]]]) -> list[dict]:
    """spec: монета → список (ts, open, close). Максимум ts монеты — её last_ts."""
    rows = []
    for symbol, bars in spec.items():
        last_ts = max(ts for ts, _, _ in bars)
        for ts, open_, close in bars:
            rows.append({"symbol": symbol, "ts": ts, "open": open_,
                         "high": max(open_, close), "low": min(open_, close),
                         "close": close, "last_ts": last_ts})
    return rows


def _call(monkeypatch, rows, tickers=TICKERS, hours=12) -> dict:
    monkeypatch.setattr(queries, "fetch_all", lambda sql, params=None: rows)
    return queries.overview_charts(tickers, hours=hours)


def test_bars_are_grouped_by_symbol(monkeypatch):
    """Один запрос на все монеты — но разложить его обязано по монетам."""
    data = _call(monkeypatch, _rows({
        "BTCUSDT": [(NOW - timedelta(minutes=15), 100.0, 110.0), (NOW, 110.0, 121.0)],
        "ETHUSDT": [(NOW, 10.0, 9.0)],
    }))
    by_symbol = {item["symbol"]: item for item in data["symbols"]}
    assert [len(by_symbol[t]["bars"]) for t in TICKERS] == [2, 1]
    assert by_symbol["BTCUSDT"]["last"] == 121.0


def test_change_is_measured_from_window_open(monkeypatch):
    """
    Изменение — от ОТКРЫТИЯ первого бара окна до закрытия последнего.

    От закрытия первого бара оно потеряло бы движение внутри него: на окне в
    четыре часа это одна свеча из шестнадцати, то есть заметная доля.
    """
    data = _call(monkeypatch, _rows({
        "BTCUSDT": [(NOW - timedelta(minutes=15), 100.0, 110.0), (NOW, 110.0, 120.0)],
    }), tickers=["BTCUSDT"])
    assert data["symbols"][0]["change_pct"] == pytest.approx(20.0)


def test_symbol_without_bars_stays_in_answer(monkeypatch):
    """
    Монета без баров остаётся в ответе пустой.

    Выкинуть её значило бы убрать плитку со страницы — а «монеты не видно» и
    «у монеты нет свежих данных» это разные новости, и вторая как раз та,
    ради которой на страницу и смотрят.
    """
    data = _call(monkeypatch, _rows({"BTCUSDT": [(NOW, 100.0, 101.0)]}))
    empty = [item for item in data["symbols"] if item["symbol"] == "ETHUSDT"][0]
    assert empty["bars"] == [] and empty["last"] is None
    assert empty["change_pct"] is None and empty["lag_minutes"] is None


def test_window_is_clamped_and_reported(monkeypatch):
    """Окно из строки запроса не доверяем: своё число страница должна знать."""
    assert _call(monkeypatch, [], hours=10 ** 6)["hours"] == queries.OVERVIEW_MAX_HOURS
    assert _call(monkeypatch, [], hours=0)["hours"] == 1


def test_no_tickers_makes_no_query(monkeypatch):
    """Пустой реестр не должен уходить в базу с пустым ANY(%s)."""
    def explode(sql, params=None):
        raise AssertionError("запроса быть не должно")

    monkeypatch.setattr(queries, "fetch_all", explode)
    assert queries.overview_charts([], hours=12)["symbols"] == []


def test_lag_is_measured_per_symbol(monkeypatch):
    """
    Отставание — свойство монеты, а не среза.

    Ровно из-за него окно и отсчитывается от последнего бара каждой монеты:
    у HYPEUSDT хвост приезжает через прокси и отстаёт заметно сильнее.
    """
    stale = datetime.now(timezone.utc) - timedelta(hours=5)
    fresh = datetime.now(timezone.utc) - timedelta(minutes=20)
    data = _call(monkeypatch, _rows({
        "BTCUSDT": [(fresh, 100.0, 101.0)],
        "ETHUSDT": [(stale, 10.0, 10.5)],
    }))
    lag = {item["symbol"]: item["lag_minutes"] for item in data["symbols"]}
    assert 19 <= lag["BTCUSDT"] <= 21
    assert 299 <= lag["ETHUSDT"] <= 301


# ─── Разметка плиток: состояния и кандидаты ─────────────────────────────────
# Раскраска приезжает вторым запросом (`overview_states`), и ломается она тише
# всего в двух местах: ключ имени состояния и слой кандидатов.

def _states_call(monkeypatch, *, bar_rows, candidate_rows=(), names=(("2.0",),),
                 root=24, palette=None):
    calls: dict[str, list] = {"sql": []}

    def fake_fetch_all(sql, params=None):
        calls["sql"].append(sql)
        if "FROM candidates" in sql:
            return list(candidate_rows)
        if "FROM market_groups" in sql:
            return [{"group_id": 2.0, "name": "тренд вверх"}]
        if "FROM ohlcv" in sql:
            return list(bar_rows)
        raise AssertionError(f"неожиданный запрос: {sql}")

    monkeypatch.setattr(queries, "fetch_all", fake_fetch_all)
    monkeypatch.setattr(queries, "state_palette",
                        lambda run_id: palette if palette is not None else {2.0: "#abcdef"})
    monkeypatch.setattr(queries.runs_repo, "latest_completed_run",
                        lambda kind="train", symbol=None: {"run_id": root} if root else None)
    monkeypatch.setattr(queries.runs_repo, "active_run",
                        lambda kind=None, symbol=None: None)
    monkeypatch.setattr(queries.runs_repo, "model_root", lambda run_id: root)
    monkeypatch.setattr(queries.runs_repo, "model_run_scope",
                        lambda run_id, alias=None: ("b.run_id = %s", [run_id]))
    monkeypatch.setattr(queries, "model_run_scope",
                        lambda run_id, alias=None: ("b.run_id = %s", [run_id]))
    data = queries.overview_states(["BTCUSDT"], hours=12)
    return data, calls


def _bar(ts, group_id):
    return {"ts": ts, "group_id": group_id, "is_transition": False}


def test_states_paint_bars_from_model_palette(monkeypatch):
    """Цвет бара — цвет его состояния в палитре модели монеты."""
    data, _ = _states_call(monkeypatch, bar_rows=[_bar(NOW, 2.0)])
    bars = data["symbols"]["BTCUSDT"]["bars"]
    assert bars == [{"time": int(NOW.timestamp()), "group_id": 2.0,
                     "color": "#abcdef", "is_transition": False}]


def test_unmarked_bar_is_dropped_not_recoloured(monkeypatch):
    """
    Бар без разметки остаётся обычной свечой.

    Хвост свежее последнего прогона размечен не всегда, и дать такому бару
    любой цвет из палитры значило бы соврать: на плитке подмену не заметить.
    """
    data, _ = _states_call(monkeypatch, bar_rows=[_bar(NOW, None)])
    assert data["symbols"]["BTCUSDT"]["bars"] == []


def test_names_are_pairs_not_a_dict(monkeypatch):
    """
    Имена состояний едут списком пар.

    Словарь с ключом-float в JSON превращается в «2.0», а тот же номер из
    `bars` приезжает как 2 — подпись состояния молча переставала находиться.
    """
    data, _ = _states_call(monkeypatch, bar_rows=[_bar(NOW, 2.0)])
    assert data["symbols"]["BTCUSDT"]["names"] == [{"group_id": 2.0, "name": "тренд вверх"}]


def test_only_issued_strong_and_moderate_candidates(monkeypatch):
    """
    На плитке только выпущенные кандидаты и только STRONG с MODERATE.

    Ретроспективу здесь нечем отличить: на /chart её отделяет форма маркера и
    слово «ретро» в подписи, а на плитке подписей нет. WEAK — вопрос места:
    их сотни, и они закрывают саму раскраску.
    """
    row = {"candidate_id": "c1", "ts": NOW, "research_side": "long",
           "rating": "STRONG", "quality_score": 0.62}
    data, calls = _states_call(monkeypatch, bar_rows=[_bar(NOW, 2.0)], candidate_rows=[row])
    marker = data["symbols"]["BTCUSDT"]["candidates"][0]
    assert marker["shape"] == "arrowUp" and marker["position"] == "belowBar"
    assert marker["rating"] == "STRONG" and marker["score"] == 0.62
    sql = [q for q in calls["sql"] if "FROM candidates" in q][0]
    assert "r.kind = 'live'" in sql, "ретроспективные кандидаты сюда попадать не должны"
    assert "c.rating = ANY(%s)" in sql


def test_symbol_without_model_is_skipped(monkeypatch):
    """
    Монета без обученной модели просто не получает разметки.

    Плитка при этом остаётся: свечи ей нужны и без состояний — иначе новая
    монета до первого train исчезала бы со страницы.
    """
    data, _ = _states_call(monkeypatch, bar_rows=[], root=None)
    assert data["symbols"] == {}
