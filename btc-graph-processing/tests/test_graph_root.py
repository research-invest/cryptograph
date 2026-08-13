"""
Граф состояний в админке при выбранном live-прогоне.

`market_groups` и `transitions` пишет только train; live лишь размечает бары
его моделью. Пока страница графа ходила по голому run_id, выбор live-прогона
в селекторе показывал ПУСТОЙ граф — молча, без единой ошибки. Здесь
закреплено, что запросы графа разыменовывают прогон в его модель через
`runs.model_root`, как это делает раскраска `/chart`.

БД не нужна: fetch_* и model_root подменяются.
"""
from __future__ import annotations

from btcproc.admin import queries

LIVE_RUN, TRAIN_RUN = 16, 14


def _fake_root(run_id: int) -> int:
    return TRAIN_RUN if run_id == LIVE_RUN else run_id


def test_graph_payload_resolves_live_run_to_its_model(monkeypatch):
    seen: list[tuple] = []

    def fake_fetch_all(sql, params):
        seen.append(tuple(params))
        return []

    monkeypatch.setattr(queries.runs_repo, "model_root", _fake_root)
    monkeypatch.setattr(queries, "fetch_all", fake_fetch_all)

    out = queries.graph_payload(LIVE_RUN)

    assert out == {"nodes": [], "edges": []}
    assert seen, "запросы должны были уйти"
    assert all(params[0] == TRAIN_RUN for params in seen), (
        "узлы и рёбра должны браться у train-прогона модели, а не у live"
    )


def test_group_detail_resolves_live_run_to_its_model(monkeypatch):
    seen: list[tuple] = []

    def fake_fetch_one(sql, params):
        seen.append(tuple(params))
        return None

    monkeypatch.setattr(queries.runs_repo, "model_root", _fake_root)
    monkeypatch.setattr(queries, "fetch_one", fake_fetch_one)

    assert queries.group_detail(LIVE_RUN, 7.0) is None
    assert seen == [(TRAIN_RUN, 7.0)]


def test_train_run_passes_through_unchanged(monkeypatch):
    seen: list[tuple] = []

    def fake_fetch_all(sql, params):
        seen.append(tuple(params))
        return []

    monkeypatch.setattr(queries.runs_repo, "model_root", _fake_root)
    monkeypatch.setattr(queries, "fetch_all", fake_fetch_all)

    queries.graph_payload(TRAIN_RUN)

    assert all(params[0] == TRAIN_RUN for params in seen)
