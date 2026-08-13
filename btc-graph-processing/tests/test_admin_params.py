"""
Разбор query-параметров фильтров.

История вопроса: форма фильтров отправляет незаполненные поля пустыми
строками (`?run=&rating=&min_quality=`). С типизированными параметрами
FastAPI (`int | None`, `float | None`) такой запрос отвечал 422, и фильтрация
на странице кандидатов ломалась целиком — стоило один раз выбрать «все».

Правило: пусто и мусор означают «фильтр не задан», а не ошибку запроса.
"""
from __future__ import annotations

import pytest

from btcproc.admin.app import opt_float, opt_int, opt_str


def test_empty_values_mean_no_filter():
    for value in ("", "   ", None):
        assert opt_str(value) is None
        assert opt_int(value) is None
        assert opt_float(value) is None


def test_normal_values_parsed():
    assert opt_str(" STRONG ") == "STRONG"
    assert opt_int("12") == 12
    assert opt_float("0.75") == 0.75


def test_garbage_does_not_raise():
    """Мусор в числовом фильтре снимает фильтр, а не роняет страницу 422-й."""
    assert opt_int("xyz") is None
    assert opt_float("абв") is None
    assert opt_int("3.9") is None


def test_comma_decimal_is_accepted():
    """Русская раскладка ставит запятую — это не повод терять фильтр."""
    assert opt_float("0,7") == 0.7


def test_negative_and_zero_survive():
    assert opt_int("0") == 0
    assert opt_float("-1.5") == -1.5


def test_model_run_scope_covers_train_and_its_live_runs():
    """
    Регрессия: страницы админки замирали на последнем train.

    Кандидат принадлежит прогону, а прогонов у одной модели много — сам train
    и все live, которые её загрузили. Фильтр `run_id = N` показывал только
    запуск, а не модель, и по мере работы крона видимая часть данных таяла:
    на графике пропадали маркеры, на дашборде — счётчики рейтингов.
    """
    from btcproc.admin import queries

    sql, params = queries.model_run_scope(42)

    # Сам прогон и его потомки — оба условия обязаны быть.
    assert "run_id = %s" in sql
    assert "model_run_id" in sql
    assert params == [42, "42"], "второй параметр сравнивается с JSON-текстом"
    # Число плейсхолдеров совпадает с числом параметров, иначе psycopg2 упадёт.
    assert sql.count("%s") == len(params)


def test_model_run_scope_does_not_widen_to_whole_symbol():
    """
    Расширять до «все кандидаты монеты» нельзя.

    transition_id и group_id осмыслены только внутри своей модели. Кандидат от
    прежнего train показал бы номер перехода из чужой нумерации — молча, без
    какого-либо признака ошибки.
    """
    from btcproc.admin import queries

    sql, _ = queries.model_run_scope(7)
    assert "symbol" not in sql, "область задаётся моделью, а не монетой"


def test_skip_if_busy_is_not_a_failure(monkeypatch):
    """
    Занятая монета при `--skip-if-busy` — пропуск, а не ошибка.

    Разница видна в коде возврата. Раз в неделю `train` занимает монету на
    полчаса, и попавший на него получасовой `live` не должен рапортовать крону
    об ошибке: он ничего не потерял, точка продолжения берётся из данных.
    Без этого расписание сыпало бы ошибками всё время обучения.
    """
    from btcproc import cli
    from btcproc.db import runs as runs_repo

    monkeypatch.setattr(
        runs_repo, "active_run",
        lambda **kw: {"run_id": 5, "kind": "train", "stage": "states", "progress": 0.4},
    )

    with pytest.raises(cli.SkipBusy):
        cli._guard_active_run(force=False, symbol="BTCUSDT", skip_if_busy=True)


def test_busy_without_the_flag_is_an_error(monkeypatch):
    """Человеку за клавиатурой то же самое — ошибка: он хотел запустить."""
    import typer

    from btcproc import cli
    from btcproc.db import runs as runs_repo

    monkeypatch.setattr(
        runs_repo, "active_run",
        lambda **kw: {"run_id": 5, "kind": "train", "stage": "states", "progress": 0.4},
    )

    with pytest.raises(typer.BadParameter):
        cli._guard_active_run(force=False, symbol="BTCUSDT")


def test_force_wins_over_both(monkeypatch):
    from btcproc import cli
    from btcproc.db import runs as runs_repo

    monkeypatch.setattr(runs_repo, "active_run", lambda **kw: {"run_id": 5, "kind": "live",
                                                               "stage": None, "progress": 0.1})
    cli._guard_active_run(force=True, symbol="BTCUSDT", skip_if_busy=True)
    cli._guard_active_run(force=True, symbol="BTCUSDT")


def _render_runs_table(**context) -> str:
    """Рендер общего шаблона таблицы прогонов с заданным контекстом."""
    import datetime as dt
    from pathlib import Path

    from jinja2 import Environment, FileSystemLoader

    root = Path(__file__).resolve().parents[1] / "btcproc" / "admin" / "templates"
    env = Environment(loader=FileSystemLoader(str(root)))

    class Row(dict):
        __getattr__ = dict.get

    now = dt.datetime(2026, 8, 9, 8, 0, tzinfo=dt.timezone.utc)
    row = Row(run_id=7, symbol="BTCUSDT", kind="live", status="done", progress=1.0,
              stage="candidates", started_at=now, finished_at=now)
    return env.get_template("partials/runs_table.html").render(runs=[row], **context)


def test_runs_table_renders_without_pagination():
    """
    Регрессия: дашборд включает тот же шаблон, что и страница прогонов.

    На дашборде таблица показывает шесть последних прогонов, пагинация там не
    нужна и контекста для неё нет. Пока блок пагинации не был закрыт проверкой
    `pages is defined`, сравнение `page_no > 1` встречало Undefined и роняло
    страницу — причём обе сразу, потому что шаблон общий.
    """
    out = _render_runs_table()
    assert "BTCUSDT" in out
    assert "<nav" not in out, "пагинация не должна появляться без контекста"


def test_runs_table_renders_with_pagination():
    out = _render_runs_table(page_no=2, pages=5, total_runs=120, kind="live")
    assert "страница 2 из 5" in out
    assert "page_no=3&kind=live" in out, "фильтр обязан переживать переход по страницам"
    assert "page_no=1&kind=live" in out
