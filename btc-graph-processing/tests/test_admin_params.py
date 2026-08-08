"""
Разбор query-параметров фильтров.

История вопроса: форма фильтров отправляет незаполненные поля пустыми
строками (`?run=&rating=&min_quality=`). С типизированными параметрами
FastAPI (`int | None`, `float | None`) такой запрос отвечал 422, и фильтрация
на странице кандидатов ломалась целиком — стоило один раз выбрать «все».

Правило: пусто и мусор означают «фильтр не задан», а не ошибку запроса.
"""
from __future__ import annotations

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
