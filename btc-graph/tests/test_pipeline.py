"""
Тесты pipeline: детерминированная оценка, сохранение и деградация при
недоступных хранилищах.

Реальные PostgreSQL / Neo4j / Redis не нужны — соответствующие модули
подменяются заглушками.
"""
from __future__ import annotations

import logging
import types

import pytest

from src.agent import pipeline
from src.agent.pipeline import _deterministic_evaluation, run_batch_pipeline, run_pipeline
from src.models.candidate import CandidateEvaluation
from src.scorer.candidate_scorer import score_candidate
from src.validator.candidate_validator import validate_candidate


@pytest.fixture(autouse=True)
def no_persistence(monkeypatch):
    """
    По умолчанию все внешние хранилища выключены: тесты проверяют логику,
    а не интеграцию. Отдельные тесты включают их обратно точечно.
    """
    monkeypatch.setattr(pipeline, "_USE_DB", False)
    monkeypatch.setattr(pipeline, "_USE_GRAPH", False)
    monkeypatch.setattr(pipeline, "_USE_REDIS", False)


# ─── Детерминированная оценка ─────────────────────────────────────────────────

def test_deterministic_evaluation_matches_scorer(reference_candidate):
    score = score_candidate(reference_candidate)
    flags = validate_candidate(reference_candidate)

    ev = _deterministic_evaluation(reference_candidate, score, flags)

    assert isinstance(ev, CandidateEvaluation)
    assert ev.candidate_id == reference_candidate.candidate_id
    assert ev.quality_score == pytest.approx(score.total)
    assert ev.rating == "STRONG"
    assert ev.direction == "long"
    assert ev.score_statistical == pytest.approx(score.statistical)
    assert ev.score_directional == pytest.approx(score.directional)
    assert ev.score_context == pytest.approx(score.context)
    assert ev.score_rarity == pytest.approx(score.rarity)


def test_win_rate_follows_research_side(make_candidate):
    long_c = make_candidate(research_side="long", long_outcome_share=0.74)
    short_c = make_candidate(research_side="short", long_outcome_share=0.74)

    long_ev = _deterministic_evaluation(long_c, score_candidate(long_c), [])
    short_ev = _deterministic_evaluation(short_c, score_candidate(short_c), [])

    assert long_ev.win_rate == pytest.approx(0.74)
    assert short_ev.win_rate == pytest.approx(0.26)


def test_strengths_and_risks_are_never_empty(make_candidate):
    """Даже у безупречного кандидата список рисков не должен быть пустым списком."""
    perfect = make_candidate(
        research_score=0.99, valid_label_pct=0.99, sample_size=9999,
        monthly_concentration=0.01, repeatability_months=36,
        long_outcome_share=0.9, historical_outcome_skew=0.9,
        long_favorable_adverse_ratio_p70_p80=9.0, context_status="fresh",
        current_group_age_bucket="age_lt_30", trajectory_entropy="low",
        event_rarity_bucket="rare", transition_rarity="rare",
    )
    ev = _deterministic_evaluation(perfect, score_candidate(perfect), [])

    assert ev.strengths
    assert ev.risks == ["Явных рисков не обнаружено"]


def test_weak_candidate_has_no_invented_strengths(make_candidate):
    poor = make_candidate(
        research_score=0.2, valid_label_pct=0.5, sample_size=30,
        monthly_concentration=0.9, repeatability_months=1,
        long_outcome_share=0.5, historical_outcome_skew=0.01,
        long_favorable_adverse_ratio_p70_p80=1.0, context_status="stale",
        current_group_age_bucket="age_gt_120", trajectory_entropy="high",
    )
    ev = _deterministic_evaluation(poor, score_candidate(poor), [])

    assert ev.strengths == ["Нет выраженных сильных сторон"]
    assert ev.rating == "WEAK"


def test_warning_flags_are_passed_through(reference_candidate):
    flags = validate_candidate(reference_candidate)
    ev = _deterministic_evaluation(reference_candidate, score_candidate(reference_candidate), flags)
    assert ev.warning_flags == flags


# ─── run_pipeline ─────────────────────────────────────────────────────────────

def test_run_pipeline_without_llm_and_without_save(reference_payload):
    ev = run_pipeline(reference_payload, use_llm=False, save=False)
    assert ev.candidate_id == "245be5fb0908d59f6e89"
    assert ev.rating == "STRONG"


def test_run_pipeline_accepts_raw_text(reference_payload):
    text = "\n".join(f"{k}: {v}" for k, v in reference_payload.items())
    ev = run_pipeline(text, use_llm=False, save=False)
    assert ev.candidate_id == "245be5fb0908d59f6e89"


def test_run_pipeline_does_not_call_llm_when_disabled(reference_payload, monkeypatch):
    import src.agent.llm_node as llm_node

    def explode(*args, **kwargs):
        raise AssertionError("LLM не должна вызываться при use_llm=False")

    monkeypatch.setattr(llm_node, "evaluate_with_llm", explode)
    assert run_pipeline(reference_payload, use_llm=False, save=False).rating == "STRONG"


def test_run_batch_pipeline_filters_and_dedups(reference_payload):
    strong = dict(reference_payload, candidate_id="strong", candidate_family_key="A")
    twin = dict(reference_payload, candidate_id="twin", candidate_family_key="A")
    weak = dict(
        reference_payload, candidate_id="weak", candidate_family_key="B",
        research_score=0.1, valid_label_pct=0.5, sample_size=40,
        monthly_concentration=0.9, repeatability_months=1,
        long_outcome_share=0.5, historical_outcome_skew=0.01,
        long_favorable_adverse_ratio_p70_p80=1.0,
    )

    result = run_batch_pipeline([strong, twin, weak], use_llm=False,
                                min_quality_score=0.60, save=False)

    ids = [e.candidate_id for e in result]
    assert len(ids) == 1                    # слабый отсеян, близнец схлопнут
    assert ids[0] in {"strong", "twin"}


def test_run_batch_pipeline_on_empty_list():
    assert run_batch_pipeline([], use_llm=False, save=False) == []


# ─── Дедупликация через Redis (регрессия на замечание №6) ────────────────────

@pytest.fixture
def fake_redis(monkeypatch):
    """
    Подменяет redis_cache: хранит хэши и оценки в памяти.

    Ключи — пары (symbol, …), как в настоящем кэше: без символа в ключе
    совпадение configuration_hash между монетами отдало бы ETH готовую
    оценку BTC.
    """
    from src.cache import redis_cache

    hashes: dict[tuple[str, str], str] = {}
    evaluations: dict[tuple[str, str], str] = {}

    def seed(config_hash: str, candidate_id: str, evaluation=None,
             symbol: str = "BTCUSDT") -> None:
        hashes[(symbol, config_hash)] = candidate_id
        if evaluation is not None:
            payload = (
                evaluation if isinstance(evaluation, str)
                else evaluation.model_dump_json()
            )
            evaluations[(symbol, candidate_id)] = payload

    monkeypatch.setattr(pipeline, "_USE_REDIS", True)
    monkeypatch.setattr(redis_cache, "is_hash_cached", lambda s, h: hashes.get((s, h)))
    monkeypatch.setattr(
        redis_cache, "get_cached_evaluation", lambda s, cid: evaluations.get((s, cid))
    )
    return types.SimpleNamespace(hashes=hashes, evaluations=evaluations, seed=seed)


def test_cache_hit_returns_own_evaluation(reference_payload, fake_redis):
    """Оценка того же кандидата берётся из кэша без пересчёта."""
    cached = CandidateEvaluation(
        candidate_id="245be5fb0908d59f6e89", quality_score=0.42, rating="WEAK",
        direction="long", win_rate=0.5, favorable_adverse_ratio=1.0,
        context_freshness="fresh", warning_flags=[], strengths=[], risks=[],
        summary="из кэша",
    )
    fake_redis.seed("0f8928cb2fc1547b", "245be5fb0908d59f6e89", cached)

    ev = run_pipeline(reference_payload, use_llm=False, save=False)

    assert ev.summary == "из кэша"
    assert ev.quality_score == pytest.approx(0.42)


def test_cache_never_returns_another_candidates_evaluation(reference_payload, fake_redis):
    """
    Под одним configuration_hash могут идти разные candidate_id. Раньше кэш
    отдавался без проверки, и клиент получал чужую оценку с чужим id.
    """
    foreign = CandidateEvaluation(
        candidate_id="СОВСЕМ_ДРУГОЙ", quality_score=0.11, rating="WEAK",
        direction="short", win_rate=0.3, favorable_adverse_ratio=None,
        context_freshness="fresh", warning_flags=[], strengths=[], risks=[],
        summary="чужая оценка",
    )
    fake_redis.seed("0f8928cb2fc1547b", "СОВСЕМ_ДРУГОЙ", foreign)

    ev = run_pipeline(reference_payload, use_llm=False, save=False)

    assert ev.candidate_id == "245be5fb0908d59f6e89"
    assert ev.summary != "чужая оценка"
    assert ev.rating == "STRONG"


def test_use_cache_false_forces_recompute(reference_payload, fake_redis):
    """Кэш и сохранение развязаны: use_cache=False пересчитывает всегда."""
    cached = CandidateEvaluation(
        candidate_id="245be5fb0908d59f6e89", quality_score=0.42, rating="WEAK",
        direction="long", win_rate=0.5, favorable_adverse_ratio=1.0,
        context_freshness="fresh", warning_flags=[], strengths=[], risks=[],
        summary="из кэша",
    )
    fake_redis.seed("0f8928cb2fc1547b", "245be5fb0908d59f6e89", cached)

    ev = run_pipeline(reference_payload, use_llm=False, save=False, use_cache=False)

    assert ev.rating == "STRONG"
    assert ev.summary != "из кэша"


def test_corrupted_cache_is_ignored(reference_payload, fake_redis, caplog):
    fake_redis.seed("0f8928cb2fc1547b", "245be5fb0908d59f6e89", "{не json")

    with caplog.at_level(logging.WARNING, logger="src.agent.pipeline"):
        ev = run_pipeline(reference_payload, use_llm=False, save=False)

    assert ev.rating == "STRONG"


def test_candidate_without_hash_is_never_cached(reference_payload, fake_redis):
    reference_payload.pop("configuration_hash")
    assert run_pipeline(reference_payload, use_llm=False, save=False).rating == "STRONG"


def test_dedup_does_not_leak_between_symbols(reference_payload, fake_redis):
    """
    Самая опасная точка мультимонетности: configuration_hash считается по
    конфигурации графа и совпадение между монетами возможно. Без символа
    в ключе ETH получил бы готовую оценку BTC — и выглядел бы при этом
    нормально оценённым кандидатом.
    """
    btc_evaluation = CandidateEvaluation(
        candidate_id="общий_id", symbol="BTCUSDT", quality_score=0.42, rating="WEAK",
        direction="long", win_rate=0.5, favorable_adverse_ratio=1.0,
        context_freshness="fresh", warning_flags=[], strengths=[], risks=[],
        summary="оценка BTC",
    )
    fake_redis.seed("одинаковый_хэш", "общий_id", btc_evaluation, symbol="BTCUSDT")

    eth = dict(reference_payload, candidate_id="общий_id", symbol="ETHUSDT",
               configuration_hash="одинаковый_хэш")
    ev = run_pipeline(eth, use_llm=False, save=False)

    assert ev.summary != "оценка BTC"
    assert ev.symbol == "ETHUSDT"

    # А своя запись той же монеты по-прежнему отдаётся из кэша.
    fake_redis.seed("одинаковый_хэш", "общий_id",
                    btc_evaluation.model_copy(update={"summary": "оценка ETH"}),
                    symbol="ETHUSDT")
    assert run_pipeline(eth, use_llm=False, save=False).summary == "оценка ETH"


# ─── _persist: деградация и логирование (регрессия на замечание №4) ───────────

def test_persist_reports_disabled_storages(reference_candidate):
    ev = _deterministic_evaluation(
        reference_candidate, score_candidate(reference_candidate), []
    )
    assert pipeline._persist(reference_candidate, ev) == {
        "db": None, "graph": None, "redis": None
    }


def test_persist_logs_and_reports_db_failure(reference_candidate, monkeypatch, caplog):
    """
    Раньше сбой записи в PostgreSQL проглатывался через `except Exception: pass`
    и был невидим. Теперь он обязан попасть в лог и в статус.
    """
    monkeypatch.setattr(pipeline, "_USE_DB", True)

    import src.db.candidate_repo as candidate_repo

    def broken_save(*args, **kwargs):
        raise RuntimeError("postgres недоступен")

    monkeypatch.setattr(candidate_repo, "save_evaluation", broken_save)

    ev = _deterministic_evaluation(
        reference_candidate, score_candidate(reference_candidate), []
    )
    with caplog.at_level(logging.WARNING, logger="src.agent.pipeline"):
        status = pipeline._persist(reference_candidate, ev)

    assert status["db"] is False
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "PostgreSQL" in messages
    assert "частично" in messages


def test_persist_failure_does_not_break_evaluation(reference_payload, monkeypatch, caplog):
    """Недоступное хранилище не должно ломать саму оценку — она возвращается клиенту."""
    monkeypatch.setattr(pipeline, "_USE_DB", True)
    monkeypatch.setattr(
        pipeline, "_persist",
        lambda c, e: (_ for _ in ()).throw(AssertionError("не должно вызываться")),
    )
    # save=False → _persist вообще не трогается
    ev = run_pipeline(reference_payload, use_llm=False, save=False)
    assert ev.rating == "STRONG"


def test_persist_success_path(reference_candidate, monkeypatch):
    """Успешная запись в граф отражается в статусе как True."""
    monkeypatch.setattr(pipeline, "_USE_GRAPH", True)

    import src.db.graph_repo as graph_repo
    monkeypatch.setattr(graph_repo, "upsert_from_candidate", lambda c, e: True)

    ev = _deterministic_evaluation(
        reference_candidate, score_candidate(reference_candidate), []
    )
    status = pipeline._persist(reference_candidate, ev)

    assert status["graph"] is True
    assert status["db"] is None


def test_persist_reports_graph_failure(reference_candidate, monkeypatch, caplog):
    """Neo4j вернул False (сервис недоступен) — статус обязан это показать."""
    monkeypatch.setattr(pipeline, "_USE_GRAPH", True)

    import src.db.graph_repo as graph_repo
    monkeypatch.setattr(graph_repo, "upsert_from_candidate", lambda c, e: False)

    ev = _deterministic_evaluation(
        reference_candidate, score_candidate(reference_candidate), []
    )
    with caplog.at_level(logging.WARNING, logger="src.agent.pipeline"):
        status = pipeline._persist(reference_candidate, ev)

    assert status["graph"] is False
    assert any("частично" in r.message for r in caplog.records)
