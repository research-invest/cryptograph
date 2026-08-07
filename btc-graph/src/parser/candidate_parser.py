"""
Парсер кандидатов из raw-текста или JSON.
Поддерживает три формата: raw text, JSON объект, JSON array.
"""
from __future__ import annotations

import json
import re
from typing import Union

from src.models.candidate import Candidate


_FIELD_MAP = {
    "candidate_id": str,
    "symbol": str,
    "configuration_hash": str,
    "candidate_family_key": str,
    "research_score": float,
    "previous_group_id": float,
    "current_group_id": float,
    "transition_id": str,
    "current_group_age_bucket": str,
    "context_status": str,
    "trajectory_entropy": str,
    "transition_rarity": str,
    "event_block_id": str,
    "primary_event_family": str,
    "event_intensity_bucket": str,
    "event_rarity_bucket": str,
    "signature_atom_count": int,
    "event_family_count": int,
    "event_block_total_rows": int,
    "event_block_row_share": float,
    "horizon": str,
    "sample_size": int,
    "valid_label_count": int,
    "invalid_label_count": int,
    "valid_label_pct": float,
    "repeatability_days": int,
    "repeatability_months": int,
    "monthly_concentration": float,
    "historical_bias_context": str,
    "research_side": str,
    "long_outcome_count": int,
    "short_outcome_count": int,
    "long_outcome_share": float,
    "historical_outcome_skew": float,
    "p70_long_favorable_pct": float,
    "p80_long_adverse_pct": float,
    "long_favorable_adverse_ratio_p70_p80": float,
}


def _parse_raw_text(text: str) -> dict:
    """Парсит свободный текст формата 'ключ: значение'."""
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*(.+)$", line)
        if not match:
            continue
        key = match.group(1).strip()
        raw_value = match.group(2).strip()
        # Убираем trailing комментарии вида  ← ...
        raw_value = re.sub(r"\s*←.*$", "", raw_value).strip()
        if key not in _FIELD_MAP:
            continue
        target_type = _FIELD_MAP[key]
        try:
            result[key] = target_type(raw_value)
        except (ValueError, TypeError):
            result[key] = raw_value
    return result


def parse_candidate(raw: Union[str, dict]) -> Candidate:
    """
    Принимает raw text, JSON-строку или dict.
    Возвращает валидированный объект Candidate.
    """
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("{"):
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                data = _parse_raw_text(stripped)
        else:
            data = _parse_raw_text(stripped)
    else:
        raise TypeError(f"Неподдерживаемый тип входных данных: {type(raw)}")

    return Candidate(**data)


def parse_candidates(raw: Union[str, list]) -> list[Candidate]:
    """
    Принимает JSON array строкой или списком dict.
    Возвращает список валидированных объектов Candidate.
    """
    if isinstance(raw, str):
        data = json.loads(raw.strip())
    else:
        data = raw

    if not isinstance(data, list):
        raise ValueError("Ожидался JSON array, получен не список")

    return [parse_candidate(item) for item in data]
