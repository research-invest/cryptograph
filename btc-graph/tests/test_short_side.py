"""
Короткая сторона: собственный F/A ratio.

До 2026-08-13 его не было, и следствие было тихим: ось directional
считалась short'у по двум критериям вместо трёх — медианный short получал
по оси 0.100 против 0.400 у long, то есть систематический штраф за сторону
рынка, а не за качество конфигурации.
"""
from __future__ import annotations

import pytest

from src.config.profiles import default_profile
from src.scorer.candidate_scorer import _score_directional, fa_ratio_for


# ─── F/A ratio для short ──────────────────────────────────────────────────────

def test_short_uses_its_own_ratio(make_candidate):
    c = make_candidate(
        research_side="short",
        long_outcome_share=0.3,
        long_favorable_adverse_ratio_p70_p80=3.5,
        short_favorable_adverse_ratio_p70_p80=2.4,
    )
    assert fa_ratio_for(c) == 2.4, "short обязан брать свою сторону, а не long"


def test_long_is_unaffected_by_the_short_field(make_candidate):
    c = make_candidate(
        research_side="long",
        long_favorable_adverse_ratio_p70_p80=3.5,
        short_favorable_adverse_ratio_p70_p80=0.1,
    )
    assert fa_ratio_for(c) == 3.5


def test_missing_short_ratio_still_means_two_criteria(make_candidate):
    """
    Кандидаты, выпущенные до появления поля, и конфигурации без падений в
    выборке приходят с None. Поведение обязано остаться прежним — критерий
    выбывает, а не начисляет ноль.
    """
    c = make_candidate(research_side="short", long_outcome_share=0.3)
    assert c.short_favorable_adverse_ratio_p70_p80 is None
    assert fa_ratio_for(c) is None

    profile = default_profile()
    spec = profile.directional
    from src.scorer.candidate_scorer import _ladder, win_rate_for

    expected = (
        _ladder(win_rate_for(c), spec.win_rate)
        + _ladder(abs(c.historical_outcome_skew), spec.abs_outcome_skew)
    ) / 2
    assert _score_directional(c, profile) == pytest.approx(expected)


def test_short_ratio_adds_the_third_criterion(make_candidate):
    """Появление данных обязано вернуть short третий критерий, а не остаться незамеченным."""
    without = make_candidate(research_side="short", long_outcome_share=0.3)
    with_ratio = make_candidate(
        research_side="short", long_outcome_share=0.3,
        short_favorable_adverse_ratio_p70_p80=9.0,
    )
    profile = default_profile()
    assert _score_directional(with_ratio, profile) > _score_directional(without, profile)
