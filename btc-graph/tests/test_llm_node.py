"""
Тесты LLM-узла: извлечение текста, разбор ответа и деградация при сбоях API.

Сеть не используется — клиент Anthropic подменяется заглушками.
"""
from __future__ import annotations

import logging
import types

import anthropic
import pytest

from src.agent import llm_node
from src.agent.llm_node import _extract_text, evaluate_with_llm
from src.scorer.candidate_scorer import score_candidate


def _block(type_: str, text: str = "") -> types.SimpleNamespace:
    return types.SimpleNamespace(type=type_, text=text)


def _response(*blocks) -> types.SimpleNamespace:
    return types.SimpleNamespace(content=list(blocks))


class _FakeMessages:
    def __init__(self, result):
        self._result = result

    def create(self, **kwargs):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _FakeClient:
    def __init__(self, result):
        self.messages = _FakeMessages(result)


@pytest.fixture
def fake_client(monkeypatch):
    """Подменяет клиента Anthropic на заглушку с заданным ответом или ошибкой."""
    def _install(result):
        monkeypatch.setattr(llm_node, "_get_client", lambda: _FakeClient(result))
    return _install


# ─── _extract_text ────────────────────────────────────────────────────────────

def test_extract_text_takes_text_block():
    assert _extract_text(_response(_block("text", "  привет  "))) == "привет"


def test_extract_text_skips_non_text_blocks():
    """
    При extended thinking или tool use первым идёт нетекстовый блок —
    обращение к content[0].text падало бы (docs/audit_findings.md, №11).
    """
    response = _response(
        _block("thinking"),
        _block("tool_use"),
        _block("text", "ответ"),
    )
    assert _extract_text(response) == "ответ"


def test_extract_text_raises_without_text_block():
    with pytest.raises(ValueError):
        _extract_text(_response(_block("thinking")))


# ─── Успешный путь ────────────────────────────────────────────────────────────

def test_parses_plain_json(reference_candidate, fake_client):
    fake_client(_response(_block("text", '{"strengths":["a"],"risks":["b"],"summary":"c"}')))

    ev = evaluate_with_llm(reference_candidate, score_candidate(reference_candidate), [])

    assert ev.strengths == ["a"]
    assert ev.risks == ["b"]
    assert ev.summary == "c"


def test_parses_json_inside_code_fence(reference_candidate, fake_client):
    fake_client(_response(_block(
        "text", 'Вот результат:\n```json\n{"strengths":["a"],"risks":["b"],"summary":"c"}\n```'
    )))

    ev = evaluate_with_llm(reference_candidate, score_candidate(reference_candidate), [])

    assert ev.summary == "c"


def test_numbers_come_from_scorer_not_llm(reference_candidate, fake_client):
    """LLM не влияет на оценку: quality_score и rating считает скорер."""
    fake_client(_response(_block(
        "text", '{"strengths":[],"risks":[],"summary":"","quality_score":0.01,"rating":"WEAK"}'
    )))
    score = score_candidate(reference_candidate)

    ev = evaluate_with_llm(reference_candidate, score, [])

    assert ev.quality_score == pytest.approx(score.total)
    assert ev.rating == "STRONG"


def test_unparseable_answer_falls_back_to_placeholders(reference_candidate, fake_client):
    fake_client(_response(_block("text", "не json вовсе")))

    ev = evaluate_with_llm(reference_candidate, score_candidate(reference_candidate), [])

    assert ev.rating == "STRONG"
    assert "распарсить" in ev.strengths[0]


def test_short_candidate_gets_no_fa_ratio(make_candidate, fake_client):
    fake_client(_response(_block("text", '{"strengths":[],"risks":[],"summary":"s"}')))
    short_c = make_candidate(research_side="short")

    ev = evaluate_with_llm(short_c, score_candidate(short_c), [])

    assert ev.favorable_adverse_ratio is None
    assert ev.direction == "short"


# ─── Деградация при сбое API (регрессия на замечание №11) ────────────────────

def test_api_error_falls_back_to_deterministic(reference_candidate, fake_client, caplog):
    """
    Сбой Claude не должен терять оценку: числа считает скорер, поэтому
    возвращается детерминированный вариант с пометкой в risks.
    """
    fake_client(anthropic.APIError("сервис недоступен", request=None, body=None)
                if _api_error_takes_kwargs() else anthropic.APIError("сервис недоступен"))

    with caplog.at_level(logging.WARNING, logger="src.agent.llm_node"):
        ev = evaluate_with_llm(reference_candidate, score_candidate(reference_candidate), [])

    assert ev.rating == "STRONG"
    assert ev.quality_score == pytest.approx(0.7783, abs=1e-4)
    assert "без LLM" in ev.risks[0]
    assert any("Claude API недоступен" in r.getMessage() for r in caplog.records)


def test_unexpected_error_falls_back_too(reference_candidate, fake_client, caplog):
    fake_client(RuntimeError("что-то пошло не так"))

    with caplog.at_level(logging.ERROR, logger="src.agent.llm_node"):
        ev = evaluate_with_llm(reference_candidate, score_candidate(reference_candidate), [])

    assert ev.rating == "STRONG"
    assert "без LLM" in ev.risks[0]


def test_fallback_keeps_warning_flags(reference_candidate, fake_client):
    fake_client(RuntimeError("сбой"))
    flags = ["context_status_stale"]

    ev = evaluate_with_llm(reference_candidate, score_candidate(reference_candidate), flags)

    assert ev.warning_flags == flags


def _api_error_takes_kwargs() -> bool:
    """anthropic.APIError в разных версиях SDK принимает разную сигнатуру."""
    import inspect
    try:
        params = inspect.signature(anthropic.APIError).parameters
    except (TypeError, ValueError):
        return False
    return "request" in params
