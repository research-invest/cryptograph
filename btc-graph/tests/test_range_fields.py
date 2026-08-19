"""
Поля размаха в схеме кандидата (2026-08-19).

Генератор с этой даты умеет класть в кандидата квантили размаха и
относительный `range_lift` — единственную свою величину, дошедшую до
положительного вердикта на отложенной части (btc-graph-processing, раздел 48).

Здесь проверяется сторона ПРИЁМНИКА: поля обязаны приниматься, обязаны
оставаться необязательными (кандидаты до этой даты их не несут) и обязаны
доезжать до сохраняемого payload'а. На оценку они не влияют — скорер о них не
знает, и это отдельно закреплено тестом ниже.
"""
from __future__ import annotations

import pytest

from src.models.candidate import Candidate, RangeRegime


def test_range_fields_are_optional(make_candidate):
    """
    Кандидат без полей размаха проходит модель, и все четыре равны None.

    Так выглядит всё, что выпущено до 2026-08-19, и всё, что выпущено монетой,
    чья модель не прошла гейт калибровки. None означает «система про размах
    ничего не говорит» — и это не то же самое, что «размах обычный».
    """
    c = make_candidate()
    assert c.expected_range_ratio_p50 is None
    assert c.expected_range_ratio_p90 is None
    assert c.range_lift is None
    assert c.range_regime is None


@pytest.mark.parametrize("regime", ["compressed", "normal", "expanded"])
def test_all_regimes_are_accepted(make_candidate, regime):
    """Три режима генератора обязаны быть известны enum'у приёмника."""
    c = make_candidate(
        expected_range_ratio_p50=1.15, expected_range_ratio_p90=2.30,
        range_lift=1.22, range_regime=regime,
    )
    assert c.range_regime == RangeRegime(regime)
    assert c.expected_range_ratio_p90 > c.expected_range_ratio_p50


def test_unknown_regime_is_refused(make_candidate):
    """
    Чужое значение режима — ошибка валидации, а не тихое сохранение строки.

    Иначе опечатка на стороне генератора доехала бы до интерфейса как режим,
    которого не существует.
    """
    with pytest.raises(Exception):
        make_candidate(range_regime="huge")


def test_negative_range_values_are_refused(make_candidate):
    """
    Размах не бывает отрицательным: `range_ratio` — отношение длин, а
    `range_lift` — отношение двух таких отношений. Отрицательное значение
    означает поломку на стороне генератора, и принимать его нельзя.
    """
    with pytest.raises(Exception):
        make_candidate(expected_range_ratio_p50=-0.5)
    with pytest.raises(Exception):
        make_candidate(range_lift=-1.0)


def test_range_fields_reach_the_stored_payload(make_candidate):
    """
    `model_dump` — то, что уходит в `raw_payload` JSONB. Поля обязаны там быть,
    иначе они потеряются между валидацией и хранением молча.
    """
    c = make_candidate(expected_range_ratio_p50=1.1, range_lift=0.8,
                       range_regime="compressed")
    dumped = c.model_dump()
    assert dumped["expected_range_ratio_p50"] == 1.1
    assert dumped["range_lift"] == 0.8
    assert dumped["range_regime"] == RangeRegime.compressed


def test_range_fields_do_not_change_the_score(make_candidate):
    """
    Оценка кандидата от размаха не зависит — и это осознанно.

    Размах и направление — разные предметы предсказания. Скорер меряет второй;
    подмешать в него первый значило бы смешать величину, которая на отложенной
    части воспроизводится, с величиной, которая нет (разделы 26 и 48 журнала
    генератора). Поля размаха описательные, и если это когда-нибудь изменится,
    тест обязан упасть.
    """
    from src.scorer.candidate_scorer import score_candidate

    plain = make_candidate()
    with_range = make_candidate(expected_range_ratio_p50=3.0,
                                expected_range_ratio_p90=6.0,
                                range_lift=2.5, range_regime="expanded")
    assert score_candidate(plain).total == pytest.approx(
        score_candidate(with_range).total
    )
