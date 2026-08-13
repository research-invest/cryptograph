"""
Тесты парсера: dict, JSON-строка, raw text, JSON array и поведение на битых данных.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.models.candidate import AgeBucket, ContextStatus, ResearchSide
from src.parser.candidate_parser import parse_candidate, parse_candidates


# ─── Три формата входа ────────────────────────────────────────────────────────

def test_parse_from_dict(reference_payload):
    c = parse_candidate(reference_payload)
    assert c.candidate_id == "245be5fb0908d59f6e89"
    assert c.research_side == ResearchSide.long
    assert c.current_group_age_bucket == AgeBucket.age_gt_120


def test_parse_from_json_string(reference_payload):
    c = parse_candidate(json.dumps(reference_payload))
    assert c.candidate_id == "245be5fb0908d59f6e89"
    assert c.sample_size == 1339


def test_parse_from_json_array(reference_payload):
    second = dict(reference_payload, candidate_id="second")
    result = parse_candidates(json.dumps([reference_payload, second]))
    assert [c.candidate_id for c in result] == ["245be5fb0908d59f6e89", "second"]


def test_parse_from_list_of_dicts(reference_payload):
    result = parse_candidates([reference_payload])
    assert len(result) == 1


def test_json_and_dict_give_identical_result(reference_payload):
    assert parse_candidate(reference_payload) == parse_candidate(json.dumps(reference_payload))


# ─── Raw text ─────────────────────────────────────────────────────────────────

def _raw_text(payload: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in payload.items())


def test_parse_raw_text(reference_payload):
    c = parse_candidate(_raw_text(reference_payload))
    assert c.candidate_id == "245be5fb0908d59f6e89"
    assert c.sample_size == 1339
    assert c.context_status == ContextStatus.stale


def test_raw_text_accepts_equals_separator(reference_payload):
    text = "\n".join(f"{k} = {v}" for k, v in reference_payload.items())
    assert parse_candidate(text).candidate_id == "245be5fb0908d59f6e89"


def test_raw_text_strips_arrow_comments(reference_payload):
    text = _raw_text(reference_payload).replace(
        "research_score: 0.9571800918456002",
        "research_score: 0.9571800918456002   ← очень высокий",
    )
    assert parse_candidate(text).research_score == pytest.approx(0.9571800918456002)


def test_raw_text_ignores_comments_and_blank_lines(reference_payload):
    text = "# заголовок отчёта\n\n" + _raw_text(reference_payload) + "\n\n# конец\n"
    assert parse_candidate(text).candidate_id == "245be5fb0908d59f6e89"


def test_raw_text_ignores_unknown_keys(reference_payload):
    text = _raw_text(reference_payload) + "\nunknown_field: 123\nсовсем_левая_строка"
    c = parse_candidate(text)
    assert not hasattr(c, "unknown_field")
    assert c.candidate_id == "245be5fb0908d59f6e89"


def test_raw_text_typo_in_key_surfaces_as_missing_field(reference_payload):
    """
    Опечатка в имени поля не игнорируется тихо на уровне результата:
    поле просто не доедет, и Pydantic сообщит, что оно обязательное.
    """
    text = _raw_text(reference_payload).replace("sample_size:", "sampel_size:")
    with pytest.raises(ValidationError) as exc:
        parse_candidate(text)
    assert "sample_size" in str(exc.value)


def test_broken_json_falls_back_to_raw_text_parser(reference_payload):
    """Строка, начинающаяся с '{', но не являющаяся JSON, уходит в текстовый парсер."""
    broken = "{\n" + _raw_text(reference_payload)
    assert parse_candidate(broken).candidate_id == "245be5fb0908d59f6e89"


# ─── Ошибки ───────────────────────────────────────────────────────────────────

def test_missing_required_field_raises(reference_payload):
    del reference_payload["sample_size"]
    with pytest.raises(ValidationError):
        parse_candidate(reference_payload)


def test_optional_fields_may_be_absent(reference_payload):
    for field in ("configuration_hash", "candidate_family_key",
                  "previous_group_id", "primary_event_family"):
        reference_payload.pop(field)
    c = parse_candidate(reference_payload)
    assert c.configuration_hash is None
    assert c.previous_group_id is None


def test_invalid_enum_value_raises(reference_payload):
    reference_payload["context_status"] = "unknown_status"
    with pytest.raises(ValidationError):
        parse_candidate(reference_payload)


@pytest.mark.parametrize("field,bad_value", [
    ("research_score", 1.5),                    # ge=0, le=1
    ("research_score", -0.1),
    ("valid_label_pct", 2.0),
    ("historical_outcome_skew", 1.5),           # ge=-1, le=1
    ("sample_size", -1),                        # ge=0
    ("event_block_row_share", 1.5),
])
def test_out_of_range_values_raise(reference_payload, field, bad_value):
    reference_payload[field] = bad_value
    with pytest.raises(ValidationError):
        parse_candidate(reference_payload)


def test_unsupported_input_type_raises():
    with pytest.raises(TypeError):
        parse_candidate(42)


def test_parse_candidates_rejects_non_list(reference_payload):
    with pytest.raises(ValueError, match="JSON array"):
        parse_candidates(json.dumps(reference_payload))
