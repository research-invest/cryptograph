from __future__ import annotations

import pandas as pd
import pytest

from btcproc.analysis import lift


def test_z_test_matches_reference_example():
    """
    Сверка с численным примером из ТЗ (docs/task_smc_integration.md, 8.2).

        p₁ = 0.810, n₁ =  180
        p₂ = 0.729, n₂ = 1159
        p_pool = 0.740,  SE = 0.0351,  z = 2.31,  p ≈ 0.021
    """
    z, p_value = lift.two_proportion_z(0.810, 180, 0.729, 1159)
    assert z == pytest.approx(2.31, abs=0.01)
    assert p_value == pytest.approx(0.021, abs=0.002)


def test_z_test_is_symmetric_and_degenerate_safe():
    z_up, _ = lift.two_proportion_z(0.8, 100, 0.6, 100)
    z_down, _ = lift.two_proportion_z(0.6, 100, 0.8, 100)
    assert z_up == pytest.approx(-z_down)

    # Пустая группа и нулевая дисперсия — «не отличили», а не падение.
    assert lift.two_proportion_z(0.8, 0, 0.6, 100) == (0.0, 1.0)
    assert lift.two_proportion_z(0.0, 100, 0.0, 100) == (0.0, 1.0)


def test_bonferroni_kills_the_reference_effect():
    """
    Тот же эффект z = 2.31 при 12 тестах поправку не переживает.

    Это главный смысл раздела 8.3: без поправки он выглядел бы находкой.
    """
    _, p_value = lift.two_proportion_z(0.810, 180, 0.729, 1159)
    assert p_value <= 0.05                       # значим сам по себе
    others = [0.9] * 11                          # ещё 11 заведомо пустых тестов
    flags = lift.bonferroni([p_value, *others], alpha=0.05)
    assert flags[0] is False                     # и уже не значим среди двенадцати


def test_benjamini_hochberg_is_less_conservative_than_bonferroni():
    p_values = [0.001, 0.008, 0.02, 0.3, 0.5]
    bh = lift.benjamini_hochberg(p_values, alpha=0.05)
    bonf = lift.bonferroni(p_values, alpha=0.05)
    assert sum(bh) >= sum(bonf)
    assert bh[0] and not bh[-1]
    # BH отвергает всё до наибольшего проходящего ранга включительно.
    assert bh[1]


def test_benjamini_hochberg_rejects_nothing_on_uniform_noise():
    assert not any(lift.benjamini_hochberg([0.2, 0.4, 0.6, 0.8], alpha=0.05))


def test_corrections_handle_empty_input():
    assert lift.bonferroni([]) == []
    assert lift.benjamini_hochberg([]) == []


def _frame(n: int = 400) -> pd.DataFrame:
    """Кандидаты, где good_atom реально сдвигает метрику, а noise_atom — нет."""
    rows = []
    for i in range(n):
        has_good = i % 4 == 0
        has_noise = i % 3 == 0
        rows.append({
            "ts": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=i),
            "atoms": (["good_atom"] if has_good else []) + (["noise_atom"] if has_noise else []),
            # Сдвиг присутствует в обеих половинах истории — holdout его подтвердит.
            "metric": (0.9 if has_good else 0.5) + (0.01 if i % 2 else -0.01),
        })
    return pd.DataFrame(rows)


def test_split_by_time_is_chronological():
    frame = _frame()
    train, test = lift.split_by_time(frame, holdout=0.3)
    assert len(train) == 280 and len(test) == 120
    # Никакого перемешивания: весь train строго раньше всего test.
    assert train["ts"].max() < test["ts"].min()


def test_split_by_time_rejects_bad_holdout():
    with pytest.raises(ValueError):
        lift.split_by_time(_frame(), holdout=0.0)
    with pytest.raises(ValueError):
        lift.split_by_time(_frame(), holdout=1.0)


def test_measure_lift_finds_real_effect_and_ignores_noise():
    results = lift.measure_lift(
        _frame(), atoms=["good_atom", "noise_atom"], correction="bonferroni",
    )
    by_atom = {r.atom: r for r in results}

    assert by_atom["good_atom"].significant
    assert by_atom["good_atom"].lift > 0.3
    assert by_atom["good_atom"].confirmed          # знак удержался на holdout

    assert not by_atom["noise_atom"].significant
    assert not by_atom["noise_atom"].confirmed


def test_measure_lift_drops_atoms_without_enough_observations():
    """
    Редкий атом исключается, а не помечается незначимым: иначе он раздувал бы
    число тестов и штрафовал поправкой те атомы, по которым данные есть.
    """
    frame = _frame()
    # Присваивание списка в ячейку через .loc pandas трактует как набор строк,
    # поэтому колонка пересобирается целиком.
    atoms_column = list(frame["atoms"])
    atoms_column[0] = atoms_column[0] + ["rare_atom"]
    frame["atoms"] = atoms_column

    results = lift.measure_lift(frame, atoms=["good_atom", "rare_atom"], min_group=30)
    assert {r.atom for r in results} == {"good_atom"}


def test_measure_lift_sorted_by_absolute_z():
    results = lift.measure_lift(_frame(), atoms=["good_atom", "noise_atom"])
    assert [abs(r.z) for r in results] == sorted(
        [abs(r.z) for r in results], reverse=True
    )


def test_confirmed_requires_holdout_agreement():
    base = dict(atom="a", n_with=100, n_without=100, z=5.0, p_value=1e-6,
                significant=True)
    flipped = lift.AtomLift(p_with=0.8, p_without=0.5, **base)
    flipped.holdout = lift.AtomLift(p_with=0.4, p_without=0.5, **base)
    assert not flipped.confirmed, "знак лифта развернулся — засчитывать нельзя"

    shrunk = lift.AtomLift(p_with=0.8, p_without=0.5, **base)
    shrunk.holdout = lift.AtomLift(p_with=0.52, p_without=0.5, **base)
    assert not shrunk.confirmed, "лифт усох на порядок — не подтверждение"

    held = lift.AtomLift(p_with=0.8, p_without=0.5, **base)
    held.holdout = lift.AtomLift(p_with=0.75, p_without=0.5, **base)
    assert held.confirmed

    # Без holdout ничего не подтверждено, даже при значимости.
    assert not lift.AtomLift(p_with=0.8, p_without=0.5, **base).confirmed


def test_confirmed_requires_significance():
    result = lift.AtomLift(atom="a", n_with=100, n_without=100, p_with=0.8,
                           p_without=0.5, z=5.0, p_value=1e-6, significant=False)
    result.holdout = lift.AtomLift(atom="a", n_with=100, n_without=100, p_with=0.8,
                                   p_without=0.5, z=5.0, p_value=1e-6)
    assert not result.confirmed


def test_format_table_handles_empty_results():
    assert "Нет атомов" in lift.format_table([])
