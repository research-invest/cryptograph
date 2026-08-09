"""
Батч не должен считать score дважды.

`filter_candidates` считает quality_score каждому кандидату, чтобы применить
порог. Затем `_evaluate` считал его же выжившим ещё раз — а у монеты со своим
профилем скоринг двухпроходный (профильный плюс baseline по `_default`).
На батчах в тысячи кандидатов это ровно вдвое больше работы, чем нужно.

Условие корректности: переиспользованная разбивка обязана быть той же самой,
что посчитал бы `_evaluate` сам. Иначе rating и тексты разъедутся со score.
"""
from __future__ import annotations

import json

from src.agent import pipeline
from src.scorer.candidate_scorer import score_candidate


def test_survivors_are_scored_once(monkeypatch, make_candidate):
    calls = []
    real = score_candidate

    def counting(candidate, profile=None):
        calls.append(candidate.candidate_id)
        return real(candidate, profile)

    monkeypatch.setattr("src.filters.candidate_filter.score_candidate", counting)
    monkeypatch.setattr(pipeline, "score_candidate", counting)

    batch = [make_candidate(candidate_id=f"c{i}") for i in range(5)]
    pipeline.run_batch_pipeline(
        [json.loads(c.model_dump_json()) for c in batch],
        use_llm=False, save=False, min_quality_score=0.0,
    )

    assert len(calls) == len(batch), (
        f"score посчитан {len(calls)} раз на {len(batch)} кандидатов — "
        "выжившие пересчитываются повторно"
    )


def test_reused_breakdown_equals_freshly_computed(make_candidate):
    """Оценка батчем совпадает с оценкой поштучно — до последнего знака."""
    candidate = make_candidate(candidate_id="one")
    payload = json.loads(candidate.model_dump_json())

    batched = pipeline.run_batch_pipeline(
        [payload], use_llm=False, save=False, min_quality_score=0.0
    )[0]
    single = pipeline.run_pipeline(
        payload, use_llm=False, save=False, use_cache=False
    )

    assert batched.quality_score == single.quality_score
    assert batched.rating == single.rating
    assert batched.quality_score_baseline == single.quality_score_baseline
    for axis in ("statistical", "directional", "context", "rarity"):
        assert getattr(batched, f"score_{axis}") == getattr(single, f"score_{axis}")
    assert batched.profile_fingerprint == single.profile_fingerprint


def test_filter_still_returns_plain_scores(make_candidate):
    """Публичная форма filter_candidates не изменилась — ею пользуются снаружи."""
    from src.filters.candidate_filter import filter_candidates

    scored = filter_candidates([make_candidate()], min_quality_score=0.0)

    assert len(scored) == 1
    candidate, quality = scored[0]
    assert isinstance(quality, float)
    assert quality == score_candidate(candidate).total
