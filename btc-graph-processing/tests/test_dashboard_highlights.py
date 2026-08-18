"""
Общая сводка дашборда: заметные кандидаты по ВСЕМ монетам.

Все остальные страницы админки работают внутри монеты, выбранной в шапке, —
этот блок единственный смотрит поперёк. Отсюда и то, что здесь закреплено:
в запрос не должен просачиваться фильтр по монете (иначе блок молча
превратится в дубль страницы кандидатов), окно считается от `ts` бара, а не
от момента расчёта, и WEAK в выдачу не попадает.

Плюс имя состояния в таблице «Крупнейшие состояния»: номер `group_id`
перенумеровывается каждым обучением и сам по себе не значит ничего.

БД не нужна: fetch_all подменяется.
"""
from __future__ import annotations

from btcproc.admin import queries


def _capture(monkeypatch) -> list[tuple[str, tuple]]:
    seen: list[tuple[str, tuple]] = []

    def fake_fetch_all(sql, params=None):
        seen.append((" ".join(sql.split()), tuple(params or ())))
        return []

    monkeypatch.setattr(queries, "fetch_all", fake_fetch_all)
    return seen


def test_highlights_do_not_filter_by_symbol(monkeypatch):
    """Смысл блока — увидеть все монеты разом, не перебирая их по очереди."""
    seen = _capture(monkeypatch)
    queries.recent_highlights()

    sql, _ = seen[0]
    assert "symbol" not in sql.split("WHERE")[1], (
        "в общую сводку не должен просачиваться фильтр по монете"
    )
    assert "SELECT candidate_id, symbol" in sql, "монету надо показать в строке"


def test_highlights_take_only_strong_and_moderate(monkeypatch):
    """WEAK — большинство кандидатов; лента из них перестаёт быть сводкой."""
    seen = _capture(monkeypatch)
    queries.recent_highlights(hours=24, limit=50)

    sql, params = seen[0]
    assert "rating = ANY(%s)" in sql
    assert params[0] == ["STRONG", "MODERATE"]
    assert params[1] == 24 and params[2] == 50
    assert "WEAK" not in params[0]


def test_highlights_window_counts_from_bar_time(monkeypatch):
    """
    Окно считается по `ts` (время бара), а не по `created_at`.

    После `train` они расходятся на всю историю: прогон пересчитывает
    кандидатов 2017 года сегодня, и по времени расчёта в «за сутки» попала
    бы вся история разом.
    """
    seen = _capture(monkeypatch)
    queries.recent_highlights()

    sql, _ = seen[0]
    assert "ts >= now() - make_interval(hours => %s)" in sql
    assert "created_at" not in sql
    assert "ORDER BY ts DESC" in sql


def test_top_groups_selects_state_name(monkeypatch):
    """
    Без имени таблица крупнейших состояний — список номеров, а номер
    осмыслен только в паре (symbol, run_id) и живёт до следующего обучения.
    """
    seen = _capture(monkeypatch)
    queries.top_groups(14, 10)

    sql, params = seen[0]
    assert "name" in sql.split("FROM")[0], "имя состояния должно приезжать с числами"
    assert params == (14, 10)
