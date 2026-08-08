"""
Снапшот-тест скорера: доказывает, что перевод порогов из кода в YAML-профиль
не сдвинул ни одного числа.

Файл tests/fixtures/scorer_snapshot.json снят скриптом
scripts/make_scorer_snapshot.py на реализации ДО рефакторинга (шаг 4.2).
Это единственная защита от тихого сдвига оценок BTC: точечные тесты ступеней
проверяют границы, но не проверяют, что весь набор кандидатов считается так же.

Тест падает в двух случаях:
  * рефакторинг скорера изменил результат — это баг, чинить код;
  * `_default.yaml` осознанно перекалиброван — тогда пересними снапшот и
    обнови README (раздел «Как читать результат оценки»).
Второе должно быть событием, а не побочным эффектом правки.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config.profiles import default_profile, get_profile
from src.scorer.candidate_scorer import score_candidate

SNAPSHOT = Path(__file__).parent / "fixtures" / "scorer_snapshot.json"

AXES = ("statistical", "directional", "context", "rarity", "total")


@pytest.fixture(scope="module")
def expected() -> dict[str, dict]:
    rows = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    return {row["candidate_id"]: row for row in rows}


def test_snapshot_covers_the_whole_set(snapshot_candidates, expected):
    """Снапшот и генератор кандидатов обязаны описывать один и тот же набор."""
    assert len(expected) == len(snapshot_candidates)
    assert set(expected) == {c.candidate_id for c in snapshot_candidates}


def test_default_profile_reproduces_pre_refactor_scores(
    snapshot_candidates, expected, shipped_profiles
):
    """
    Все 200 кандидатов, все четыре оси и итог — до последнего знака.

    Сторожит именно ПОСТАВЛЯЕМЫЙ `_default`: он общая линейка для
    quality_score_baseline, и его правка обесценивает сравнение всех монет
    между собой и во времени.
    """
    mismatches = []
    for candidate in snapshot_candidates:
        actual = score_candidate(candidate, shipped_profiles.default_profile())
        row = expected[candidate.candidate_id]
        for axis in AXES:
            if getattr(actual, axis) != pytest.approx(row[axis], abs=1e-9):
                mismatches.append(
                    f"{candidate.candidate_id}.{axis}: "
                    f"было {row[axis]}, стало {getattr(actual, axis)}"
                )

    assert not mismatches, "Скорер сдвинулся:\n  " + "\n  ".join(mismatches[:20])


def test_btcusdt_has_its_own_calibration(snapshot_candidates, shipped_profiles):
    """
    BTCUSDT больше НЕ пустое наследование: с версии 2 у него своя калибровка
    по выгрузке прогона #17. Раньше здесь проверялось обратное.

    Что важно сохранить: `baseline` считается базовой линейкой независимо от
    профиля монеты, поэтому он обязан отличаться от профильного результата —
    иначе межмонетное сравнение потеряло бы смысл.
    """
    btc = shipped_profiles.get_profile("BTCUSDT")
    base = shipped_profiles.default_profile()

    assert btc.version >= 2
    assert btc.statistical.sample_size.thresholds != base.statistical.sample_size.thresholds

    differing = 0
    for candidate in snapshot_candidates[:50]:
        actual = score_candidate(candidate, btc)
        assert actual.profile == btc.name
        assert actual.baseline_total == pytest.approx(
            score_candidate(candidate, base).total, abs=1e-9
        )
        if actual.total != pytest.approx(actual.baseline_total, abs=1e-9):
            differing += 1

    assert differing > 0, "профиль BTC не отличается от базового — калибровка потерялась"
