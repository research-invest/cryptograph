"""
Тесты профильной оценки: то, ради чего затевался шаг 4.

Главная проверяемая мысль — кандидат по монете с короткой историей не обязан
уезжать в WEAK только потому, что его выборка меньше биткоиновой. Линейка
должна быть своя, а межмонетное сравнение — отдельным числом.
"""
from __future__ import annotations

import pytest

from src.config.profiles import default_profile, get_profile
from src.filters.candidate_filter import (
    detect_conflicts,
    filter_candidates,
    select_best_per_family,
)
from src.scorer.candidate_scorer import get_rating, score_candidate


# ─── Профиль меняет оценку ────────────────────────────────────────────────────

def test_short_history_candidate_is_rescued_by_its_profile(make_candidate, eth_profile):
    """
    Выборка 300 случаев за 6 месяцев: для BTC это нижние ступени почти по всем
    статистическим критериям, для монеты с двухлетней историей — верхние.
    """
    c = make_candidate(symbol="ETHUSDT", sample_size=300, repeatability_months=9)

    by_btc = score_candidate(c, default_profile())
    by_eth = score_candidate(c, eth_profile)

    assert by_btc.statistical < by_eth.statistical
    assert by_btc.total < by_eth.total
    # Ось, которой профиль не касался, обязана остаться прежней.
    assert by_btc.context == by_eth.context


def test_rating_bounds_come_from_profile(eth_profile):
    """У монеты своя граница STRONG — сравнивать score с числом напрямую нельзя."""
    assert get_rating(0.72) == "MODERATE"                  # базовый профиль: 0.75
    assert get_rating(0.72, eth_profile) == "STRONG"       # профиль ETH: 0.70

    assert get_rating(0.52) == "WEAK"
    assert get_rating(0.52, eth_profile) == "MODERATE"


def test_breakdown_carries_profile_identity(make_candidate, eth_profile):
    """Каждая оценка помечена профилем и его отпечатком — иначе её не с чем сравнить."""
    b = score_candidate(make_candidate(symbol="ETHUSDT"), eth_profile)

    assert b.profile == "ETHUSDT@2"
    assert b.fingerprint == eth_profile.fingerprint
    assert len(b.fingerprint) == 12


def test_baseline_is_always_the_default_profile(make_candidate, eth_profile):
    """
    quality_score_baseline обязан считаться базовым профилем независимо от
    того, какой профиль применён: это единственное межмонетно сравнимое число.
    """
    c = make_candidate(symbol="ETHUSDT", sample_size=300, repeatability_months=9)

    by_eth = score_candidate(c, eth_profile)
    by_default = score_candidate(c, default_profile())

    assert by_eth.baseline_total == pytest.approx(by_default.total)
    assert by_eth.total != pytest.approx(by_eth.baseline_total)


def test_profile_resolved_from_symbol_when_not_given(make_candidate):
    """Без явного профиля берётся профиль монеты кандидата."""
    assert score_candidate(make_candidate(symbol="BTCUSDT")).profile == "BTCUSDT@1"
    # Неизвестная монета считается базовой калибровкой, а не падает.
    assert score_candidate(make_candidate(symbol="DOGEUSDT")).profile == "_default@1"


def test_win_rate_and_fa_ratio_are_not_profiled(make_candidate, eth_profile):
    """
    Для short ось directional считается по двум критериям при любом профиле:
    неприменимость F/A ratio — свойство данных, а не калибровки.
    """
    from src.scorer.candidate_scorer import _score_directional

    short_c = make_candidate(
        symbol="ETHUSDT", research_side="short",
        long_outcome_share=0.20, historical_outcome_skew=0.5,
        long_favorable_adverse_ratio_p70_p80=0.5,   # был бы 0.1, но игнорируется
    )
    assert _score_directional(short_c, eth_profile) == pytest.approx(1.0)


# ─── Фильтры ──────────────────────────────────────────────────────────────────

@pytest.fixture
def rescued_candidate(make_candidate):
    """
    Кандидат, ради которого и нужен шаг 4: по линейке BTC он WEAK и не проходит
    в батч, по линейке своей монеты — MODERATE и проходит.
    """
    return make_candidate(
        symbol="ETHUSDT",
        sample_size=300, repeatability_months=9,
        long_outcome_share=0.64, historical_outcome_skew=0.28,
        long_favorable_adverse_ratio_p70_p80=2.5,
    )


def test_batch_threshold_comes_from_profile(rescued_candidate, eth_profile):
    """
    Порог попадания в батч — тоже часть калибровки. По базовой линейке
    кандидат набирает 0.478 при пороге 0.60 и выпадает; по своей — 0.572
    при пороге 0.50 и проходит.
    """
    by_default = score_candidate(rescued_candidate, default_profile()).total
    by_eth = score_candidate(rescued_candidate, eth_profile).total
    assert by_default < default_profile().batch.min_quality_score
    assert by_eth >= eth_profile.batch.min_quality_score

    assert filter_candidates([rescued_candidate]) == []
    assert len(filter_candidates([rescued_candidate],
                                profiles={"ETHUSDT": eth_profile})) == 1


def test_rescued_candidate_changes_rating(rescued_candidate, eth_profile):
    """Та же выборка: WEAK по чужой линейке, MODERATE по своей."""
    assert get_rating(score_candidate(rescued_candidate, default_profile()).total) == "WEAK"
    assert get_rating(
        score_candidate(rescued_candidate, eth_profile).total, eth_profile
    ) == "MODERATE"


def test_explicit_threshold_overrides_profiles(make_candidate):
    """Явное число применяется ко всему батчу — для разовых экспериментов."""
    weak = make_candidate(sample_size=10, valid_label_pct=0.5, monthly_concentration=0.9)
    assert filter_candidates([weak], min_quality_score=0.0) != []
    assert filter_candidates([weak], min_quality_score=0.99) == []


def test_family_grouping_is_per_symbol(make_candidate):
    """
    Одинаковый family_key у BTC и ETH — норма: ключ описывает конфигурацию
    внутри графа монеты. Без символа в группировке одна монета гасила бы другую.
    """
    btc = make_candidate(candidate_id="btc-1", symbol="BTCUSDT",
                         candidate_family_key="1.0|42->1|blk|long_skew")
    eth = make_candidate(candidate_id="eth-1", symbol="ETHUSDT",
                         candidate_family_key="1.0|42->1|blk|long_skew")

    best = select_best_per_family([(btc, 0.9), (eth, 0.4)])

    assert {c.candidate_id for c, _ in best} == {"btc-1", "eth-1"}


def test_conflicts_are_per_symbol(make_candidate):
    """
    BTC-long и ETH-short на «42->1» — не противоречие: это разные переходы
    в разных графах. Ложный конфликт здесь стоил бы отброшенных кандидатов.
    """
    btc_long = make_candidate(candidate_id="btc-1", symbol="BTCUSDT",
                              transition_id="42->1", research_side="long")
    eth_short = make_candidate(candidate_id="eth-1", symbol="ETHUSDT",
                               transition_id="42->1", research_side="short")

    types = {c.conflict_type for c in detect_conflicts([btc_long, eth_short])}
    assert "contradictory_direction" not in types

    # Внутри одной монеты конфликт по-прежнему обнаруживается.
    btc_short = make_candidate(candidate_id="btc-2", symbol="BTCUSDT",
                               transition_id="42->1", research_side="short")
    types = {c.conflict_type for c in detect_conflicts([btc_long, btc_short])}
    assert "contradictory_direction" in types


def test_conflict_thresholds_come_from_profile(make_candidate, eth_profile):
    """Сезонная концентрация оценивается порогом монеты, а не общим."""
    relaxed = eth_profile.model_copy(update={
        "validator": eth_profile.validator.model_copy(
            update={"high_monthly_concentration": 0.60}
        )
    })
    c = make_candidate(symbol="ETHUSDT", monthly_concentration=0.45)

    assert "seasonal_concentration" in {r.conflict_type for r in detect_conflicts([c])}
    assert "seasonal_concentration" not in {
        r.conflict_type
        for r in detect_conflicts([c], profiles={"ETHUSDT": relaxed})
    }
