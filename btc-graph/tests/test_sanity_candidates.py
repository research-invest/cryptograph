"""
Sanity-якоря профилей (правило 5 из docs/step_04_multi_symbol.md).

На каждую заведённую монету в tests/fixtures/sanity_candidates.json лежит один
реальный кандидат с зафиксированным ожидаемым рейтингом. Смысл — поймать
перекалибровку, которая выглядит безобидной правкой одной ступени, а на деле
переводит понятный кандидат через границу рейтинга.

Второй тест обязателен не меньше первого: он не даёт завести монету без якоря.
Профиль без якоря — это калибровка, про которую никто не может сказать, верна
она или нет.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config.profiles import DEFAULT_PROFILE_NAME, get_profile, list_profiles
from src.models.candidate import Candidate
from src.scorer.candidate_scorer import get_rating, score_candidate

FIXTURE = Path(__file__).parent / "fixtures" / "sanity_candidates.json"

ANCHORS: dict[str, dict] = json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("symbol", sorted(ANCHORS))
def test_sanity_candidate_keeps_its_rating(symbol, shipped_profiles):
    """Якорь проверяет то, что поставляется, а не замороженную тестовую копию."""
    anchor = ANCHORS[symbol]
    candidate = Candidate(**anchor["candidate"])
    assert candidate.symbol == symbol, "якорь лежит не под своей монетой"

    profile = shipped_profiles.get_profile(symbol)
    breakdown = score_candidate(candidate, profile)
    rating = get_rating(breakdown.total, profile)

    assert rating == anchor["expected_rating"], (
        f"{symbol}: якорь сменил рейтинг {anchor['expected_rating']} → {rating} "
        f"(quality_score={breakdown.total:.4f}, профиль {profile.name}).\n"
        f"{anchor['note']}\n"
        "Если перекалибровка осознанна — обнови ожидание в фикстуре "
        "и бампни version профиля."
    )


def test_every_profile_has_a_sanity_anchor(shipped_profiles):
    """
    Монета, заведённая без якоря, — это профиль, который нечем проверить.
    Тест падает при добавлении config/symbols/<COIN>.yaml без фикстуры.
    """
    registered = {
        row["key"] for row in shipped_profiles.list_profiles()
        if row["key"] != DEFAULT_PROFILE_NAME.upper()
    }
    missing = registered - set(ANCHORS)

    assert not missing, (
        f"Нет sanity-кандидата для {sorted(missing)}. Положи по одному реальному "
        f"кандидату монеты в {FIXTURE.name} с ожидаемым рейтингом — правило 5 "
        "из docs/step_04_multi_symbol.md."
    )
