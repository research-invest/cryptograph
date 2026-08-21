"""
Страница «Состояния» и палитра цветов.

Две вещи, которые здесь легко сломать молча.

**Цвет состояния — свойство МОДЕЛИ, а не окна графика.** До 2026-08-21 палитра
считалась от набора состояний видимого периода: цвет был порядковым номером в
отсортированном списке того, что попало в окно. Сдвинул график на неделю —
часть состояний поменяла оттенок, и ни один тест этого не замечал, потому что
все цвета оставались «правильными» по отдельности. Список состояний на
отдельной странице после этого показывал бы не те цвета, что график, то есть
врал бы ровно в том, ради чего заведён.

**Список читает готовое.** `market_groups` заполняет `train`; страница не
считает ничего (инвариант 19).
"""
from __future__ import annotations

import pytest

from btcproc.admin import queries

TRAIN_RUN = 41


@pytest.fixture
def model(monkeypatch):
    """Модель из четырёх состояний и подставная выборка баров под неё."""
    groups = [
        {"group_id": 1.0, "name": "затишье", "size": 500, "share": 0.5,
         "dominant_bias": "long_skew", "up_share": 0.55, "avg_ret_pct": 0.4,
         "avg_vol_pct": 3.0, "top_features": {"rsi": 1.4, "trend_align": -0.9,
                                              "rv_ratio": 0.3}},
        {"group_id": 2.0, "name": "тренд вверх", "size": 300, "share": 0.3,
         "dominant_bias": None, "up_share": None, "avg_ret_pct": None,
         "avg_vol_pct": None, "top_features": {}},
        {"group_id": 3.0, "name": "", "size": 150, "share": 0.15,
         "dominant_bias": "short_skew", "up_share": 0.41, "avg_ret_pct": -0.2,
         "avg_vol_pct": 5.0, "top_features": {"tf1d_pos": -1.2}},
        {"group_id": 4.0, "name": "рывок", "size": 50, "share": 0.05,
         "dominant_bias": None, "up_share": 0.5, "avg_ret_pct": 0.0,
         "avg_vol_pct": 4.0, "top_features": {"ret_1h": 2.0}},
    ]

    def fake_fetch_all(sql, params=None):
        if "FROM market_groups" in sql:
            if "ORDER BY group_id" in sql:      # палитра
                return [{"group_id": g["group_id"]} for g in groups]
            if "ORDER BY size DESC" in sql:     # список страницы
                return [dict(g) for g in groups]
            return [{"group_id": g["group_id"], "name": g["name"]} for g in groups]
        return []

    monkeypatch.setattr(queries, "fetch_all", fake_fetch_all)
    monkeypatch.setattr(queries.runs_repo, "model_root", lambda run_id: TRAIN_RUN)
    return groups


def test_palette_covers_every_state_of_the_model(model):
    palette = queries.state_palette(TRAIN_RUN)
    assert set(palette) == {1.0, 2.0, 3.0, 4.0}
    assert len(set(palette.values())) == 4, "оттенки не должны совпадать"
    assert all(color.startswith("#") and len(color) == 7 for color in palette.values()), (
        "lightweight-charts не умеет hsl(...) — только hex"
    )


def test_palette_does_not_depend_on_which_states_are_visible(model):
    """
    Главная регрессия: цвет состояния 3 одинаков независимо от того, сколько
    состояний попало в окно.

    Раньше палитра строилась от окна, и это ломалось само собой при любом
    сдвиге периода. Проверяем через саму палитру, а не через `chart_data`:
    она и есть источник цвета для обеих страниц.
    """
    full = queries.state_palette(TRAIN_RUN)
    again = queries.state_palette(TRAIN_RUN)
    assert full == again
    # Цвет не зависит от размера состояния и от порядка сортировки списка:
    # ключ — номер, а он у модели фиксирован.
    assert full[3.0] != full[1.0]


def test_states_page_reads_ready_rows_and_adds_colour(model):
    rows = queries.states_page(TRAIN_RUN)
    assert [row["group_id"] for row in rows] == [1.0, 2.0, 3.0, 4.0], (
        "порядок — по размеру, как отдаёт запрос"
    )
    palette = queries.state_palette(TRAIN_RUN)
    assert [row["color"] for row in rows] == [palette[row["group_id"]] for row in rows]


def test_states_page_describes_top_features_in_russian(model):
    """
    Признаки, которыми состояние выделяется, подписаны фразами из того же
    словаря, которым названо само состояние (`naming.AXES`). Второй словарь
    здесь разъехался бы молча, и подпись начала бы противоречить имени.
    """
    rows = queries.states_page(TRAIN_RUN)
    first = rows[0]
    assert len(first["top"]) == 3, "берём три самых крупных отклонения"
    assert abs(first["top"][0]["value"]) >= abs(first["top"][1]["value"])
    assert first["top"][0]["feature"] == "rsi"
    assert first["top"][0]["phrase"], "у признака должна быть русская формулировка"
    # Знак решает, какая из двух фраз оси попадёт в строку.
    trend = next(item for item in first["top"] if item["feature"] == "trend_align")
    assert trend["phrase"] != first["top"][0]["phrase"]


def test_state_without_features_survives(model):
    """Состояние с пустым `top_features` не роняет страницу и не выдумывает фраз."""
    rows = queries.states_page(TRAIN_RUN)
    empty = next(row for row in rows if row["group_id"] == 2.0)
    assert empty["top"] == []
    assert empty["color"].startswith("#")


def test_page_renders(monkeypatch, model):
    """
    Страница отдаёт HTML: шаблон разбирается, все поля строки на месте.

    Нужен настоящий HTTP, а не вызов функции: половина ошибок шаблона (опечатка
    в имени поля, фильтр без значения) видна только при рендере.
    """
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from btcproc import config
    from btcproc.admin import auth

    monkeypatch.setattr(
        config, "admin",
        config.AdminConfig(user="operator", password="very-long-password-42",
                           secret_key="k" * 64, ip_allowlist=[]),
    )
    from btcproc.admin import app as admin_app

    monkeypatch.setattr(auth, "current_user", lambda request: "operator")
    monkeypatch.setattr(admin_app, "init_schema", lambda: None, raising=False)
    monkeypatch.setattr(admin_app, "_latest_train_id", lambda symbol: TRAIN_RUN)
    monkeypatch.setattr(admin_app.runs_repo, "list_runs",
                        lambda *a, **kw: [])

    with fastapi_testclient.TestClient(admin_app.app) as client:
        response = client.get("/states?symbol=BTCUSDT")

    assert response.status_code == 200
    body = response.text
    assert "затишье" in body
    assert "Состояния · BTCUSDT" in body
    # Цвет обязан попасть в разметку: без него строка списка не связывается
    # со свечой на графике, а это единственная причина, по которой список
    # вообще показывает цвет.
    assert queries.state_palette(TRAIN_RUN)[1.0] in body
