"""
Батчевая запись пачки оценок: эквивалентность поштучной и деградация.

Главный вопрос этих тестов не «быстрее ли стало», а **то же ли записывается**.
Свойства графа накапливаются (`count`, `sample_count`, скользящие средние),
поэтому расхождение батчевого пути с поштучным не всплыло бы ошибкой — граф
просто стал бы другим, и отличить его было бы не от чего.
"""
from __future__ import annotations

import pytest

from src.agent import pipeline
from src.agent.pipeline import _deterministic_evaluation
from src.db import graph_repo
from src.scorer.candidate_scorer import score_candidate


@pytest.fixture
def pairs(make_candidate):
    """Три кандидата: два по одному переходу, один по другому."""
    made = [
        make_candidate(candidate_id="c1", current_group_id=1.0, previous_group_id=2.0,
                       transition_id="2->1", long_outcome_share=0.80),
        make_candidate(candidate_id="c2", current_group_id=1.0, previous_group_id=2.0,
                       transition_id="2->1", long_outcome_share=0.60),
        make_candidate(candidate_id="c3", current_group_id=3.0, previous_group_id=1.0,
                       transition_id="1->3", long_outcome_share=0.70),
    ]
    return [
        (c, _deterministic_evaluation(c, score_candidate(c), []))
        for c in made
    ]


# ─── Свёртка пачки: то, на чём держится равенство с поштучным путём ───────────

def test_aggregate_counts_and_sums(pairs):
    groups, prev_groups, edges = graph_repo._aggregate(pairs)

    by_group = {(r["symbol"], r["group_id"]): r for r in groups}
    assert by_group[("BTCUSDT", 1.0)]["n"] == 2   # c1 и c2
    assert by_group[("BTCUSDT", 3.0)]["n"] == 1

    by_edge = {r["tid"]: r for r in edges}
    assert by_edge["2->1"]["n"] == 2
    assert by_edge["1->3"]["n"] == 1

    # Суммы, а не средние: среднее считает Cypher, и только так формула
    # совпадает с k последовательными инкрементами.
    expected = sum(e.win_rate for c, e in pairs if c.transition_id == "2->1")
    assert by_edge["2->1"]["win_sum"] == pytest.approx(expected)
    expected_qs = sum(e.quality_score for c, e in pairs if c.transition_id == "2->1")
    assert by_edge["2->1"]["qs_sum"] == pytest.approx(expected_qs)


def test_aggregate_keeps_last_value_of_overwritten_fields(make_candidate):
    """
    `dominant_bias` узла и `rarity` ребра поштучный путь ПЕРЕЗАПИСЫВАЕТ каждым
    кандидатом — побеждает последний. Свёртка обязана взять последний в
    порядке пачки, иначе граф разойдётся с последовательным на тех переходах,
    где редкость успела смениться.
    """
    first = make_candidate(candidate_id="c1", transition_rarity="common",
                           historical_bias_context="short_skew")
    last = make_candidate(candidate_id="c2", transition_rarity="rare",
                          historical_bias_context="long_skew")
    batch = [
        (c, _deterministic_evaluation(c, score_candidate(c), []))
        for c in (first, last)
    ]

    groups, _, edges = graph_repo._aggregate(batch)

    assert edges[0]["rarity"] == "rare"
    assert groups[0]["bias"] == "long_skew"


def test_previous_group_row_carries_no_counters(pairs):
    """
    Узел, впервые созданный как «предыдущий», в поштучном пути получает только
    label — без sample_count и без bias. Свёртка не должна дописать их: тогда
    у группы, которая в пачке ни разу не была текущей, появился бы счётчик
    наблюдений из ниоткуда.
    """
    _, prev_groups, _ = graph_repo._aggregate(pairs)

    assert prev_groups
    for row in prev_groups:
        assert set(row) == {"symbol", "group_id", "label"}


def test_batch_mean_matches_sequential_increments():
    """
    Арифметическое основание свёртки: инкрементальное среднее, применённое k
    раз, равно одному шагу с суммой k наблюдений. Именно на этом равенстве
    построен Cypher батчевого запроса, и если оно перестанет выполняться,
    сломается оно молча.
    """
    values = [0.80, 0.60, 0.70, 0.55]
    avg, count = 0.42, 7  # то, что уже лежит на ребре

    seq_avg, seq_count = avg, count
    for value in values:
        seq_avg = (seq_avg * seq_count + value) / (seq_count + 1)
        seq_count += 1

    n = len(values)
    batch_avg = (avg * count + sum(values)) / (count + n)

    assert batch_avg == pytest.approx(seq_avg)
    assert count + n == seq_count


# ─── Деградация: контракт «хранилище упало — оценка выжила» ───────────────────

def test_persist_batch_reports_disabled_storages(pairs):
    assert pipeline._persist_batch(pairs) == {"db": 0, "graph": 0, "redis": 0}


def test_db_batch_failure_falls_back_to_one_by_one(pairs, monkeypatch, caplog):
    """Одна битая пачка не должна уносить с собой остальных кандидатов."""
    monkeypatch.setattr(pipeline, "_USE_DB", True)

    import src.db.candidate_repo as candidate_repo

    def broken_batch(*args, **kwargs):
        raise RuntimeError("postgres недоступен")

    saved: list[str] = []
    monkeypatch.setattr(candidate_repo, "save_evaluations", broken_batch)
    monkeypatch.setattr(
        candidate_repo, "save_evaluation",
        lambda session, c, e: saved.append(c.candidate_id),
    )
    monkeypatch.setattr(candidate_repo, "log_event", lambda *a, **k: None)

    import contextlib

    @contextlib.contextmanager
    def fake_session():
        yield object()

    monkeypatch.setattr("src.db.connection.get_session", fake_session)

    with caplog.at_level("WARNING", logger="src.agent.pipeline"):
        written = pipeline._persist_batch(pairs)

    assert written["db"] == len(pairs)
    assert saved == [c.candidate_id for c, _ in pairs]
    assert "поштучно" in " ".join(r.getMessage() for r in caplog.records)


def test_successful_graph_batch_is_not_repeated_per_candidate(pairs, monkeypatch):
    """
    Самый опасный сценарий батчинга: пачка прошла, а откат отработал сверху.
    Для Neo4j это не идемпотентно — счётчики рёбер посчитали бы всё дважды.
    """
    monkeypatch.setattr(pipeline, "_USE_GRAPH", True)
    monkeypatch.setattr(graph_repo, "upsert_batch", lambda chunk: True)
    monkeypatch.setattr(
        graph_repo, "upsert_from_candidate",
        lambda c, e: pytest.fail("поштучная запись после успешной пачки"),
    )

    assert pipeline._persist_batch(pairs)["graph"] == len(pairs)


def test_unavailable_graph_does_not_retry_per_candidate(pairs, monkeypatch):
    """
    False от `upsert_batch` означает «Neo4j недоступен», а не «пачка кривая».
    Поштучный повтор упрётся в то же самое и только растянет прогон.
    """
    monkeypatch.setattr(pipeline, "_USE_GRAPH", True)
    monkeypatch.setattr(graph_repo, "upsert_batch", lambda chunk: False)
    monkeypatch.setattr(
        graph_repo, "upsert_from_candidate",
        lambda c, e: pytest.fail("повтор при недоступном Neo4j"),
    )

    assert pipeline._persist_batch(pairs)["graph"] == 0


def test_redis_entries_publish_only_strong(pairs, monkeypatch):
    monkeypatch.setattr(pipeline, "_USE_REDIS", True)

    import src.cache.redis_cache as redis_cache

    captured: list[dict] = []

    def fake_persist_batch(entries):
        captured.extend(entries)
        return True

    monkeypatch.setattr(redis_cache, "persist_batch", fake_persist_batch)

    written = pipeline._persist_batch(pairs)

    assert written["redis"] == len(pairs)
    assert len(captured) == len(pairs)
    for entry, (candidate, evaluation) in zip(captured, pairs):
        assert entry["candidate_id"] == candidate.candidate_id
        assert entry["configuration_hash"] == candidate.configuration_hash
        if evaluation.rating == "STRONG":
            assert entry["strong_payload"] is not None
        else:
            assert entry["strong_payload"] is None


def test_run_batch_pipeline_persists_once_for_the_whole_batch(pairs, monkeypatch):
    """
    Батчевый путь обязан звать `_persist_batch` один раз, а не `_persist` на
    кандидата: ради этого всё и делалось.
    """
    from src.agent.pipeline import run_batch_pipeline

    calls: list[int] = []
    monkeypatch.setattr(
        pipeline, "_persist_batch",
        lambda batch: calls.append(len(batch)) or {"db": 0, "graph": 0, "redis": 0},
    )
    monkeypatch.setattr(
        pipeline, "_persist",
        lambda c, e: pytest.fail("поштучная запись в батчевом пути"),
    )

    payloads = [c.model_dump(mode="json") for c, _ in pairs]
    run_batch_pipeline(payloads, use_llm=False, min_quality_score=0.0, save=True)

    assert len(calls) == 1
    assert calls[0] > 0


# ─── Запись в PostgreSQL: одна выборка на монету, одна строка на кандидата ────

class _FakeQuery:
    def __init__(self, log: list):
        self._log = log

    def filter(self, *args):
        self._log.append(args)
        return self

    def all(self):
        return []


class _FakeSession:
    """Сессия ровно в том объёме, в каком её трогает `save_evaluations`."""

    def __init__(self):
        self.added = []
        self.queries = []

    def query(self, *args):
        return _FakeQuery(self.queries)

    def add(self, obj):
        self.added.append(obj)

    def add_all(self, objs):
        self.added.extend(objs)


def test_batch_issues_one_select_per_symbol(pairs, make_candidate):
    import src.db.candidate_repo as candidate_repo

    eth = make_candidate(candidate_id="e1", symbol="ETHUSDT")
    batch = pairs + [(eth, _deterministic_evaluation(eth, score_candidate(eth), []))]

    session = _FakeSession()
    candidate_repo.save_evaluations(session, batch)

    # Две монеты — две выборки, а не по одной на кандидата.
    assert len(session.queries) == 2
    assert len(session.added) == len(batch)


def test_duplicate_candidate_in_batch_writes_one_row(make_candidate):
    """
    Повтор `(symbol, candidate_id)` внутри пачки обязан обновить ту же строку.
    Поштучный путь это переживал сам (второй `session.get` находил первую
    запись после коммита), батчевый — только если помнит уже добавленные.
    """
    import src.db.candidate_repo as candidate_repo

    candidate = make_candidate(candidate_id="dup")
    evaluation = _deterministic_evaluation(candidate, score_candidate(candidate), [])

    session = _FakeSession()
    candidate_repo.save_evaluations(session, [(candidate, evaluation)] * 2)

    assert len(session.added) == 1
