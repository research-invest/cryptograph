from __future__ import annotations

import numpy as np
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


# ─── Зависимые наблюдения: блочный бутстрап ─────────────────────────────────
def _ar_flags(n: int, rho: float, rng) -> np.ndarray:
    """Сильно автокоррелированный булев ряд — «атом держится сериями»."""
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal()
    return x > 0


def _dependent_frame(n: int = 3000, seed: int = 7, effect: float = 0.0):
    """
    Признак и метрика автокоррелированы одинаково сильно; связь между ними
    задаётся `effect` (0.0 — связи нет вовсе).

    Это модель ровно нашей ситуации: горизонт 24h при базовом ТФ 15m
    означает, что соседние кандидаты делят почти один и тот же исход.
    """
    rng = np.random.default_rng(seed)
    flags = _ar_flags(n, 0.99, rng)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.99 * noise[i - 1] + rng.normal()
    noise /= np.std(noise)
    metric = (effect * (flags - 0.5) + 0.5 * noise + rng.normal(0, 0.3, n) > 0).astype(float)
    return pd.DataFrame({
        "ts": pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC"),
        "atoms": [["a"] if f else [] for f in flags],
        "metric": metric,
    })


def test_block_length_is_measured_in_rows_not_bars():
    """
    Длина блока — в строках выборки, и перевод из баров не единичный.

    Кандидаты идут не на каждом баре: на BTC один примерно раз в 11 баров.
    Взять 96 (горизонт в барах) как число строк значило бы накрыть блоком
    тысячу баров вместо ста.
    """
    dense = pd.Series(pd.date_range("2020-01-01", periods=2000, freq="15min", tz="UTC"))
    # Кандидат на каждом баре: блок = горизонт в барах.
    assert lift.block_length_rows(dense, horizon_minutes=1440) == 96

    sparse = pd.Series(pd.date_range("2020-01-01", periods=2000, freq="165min", tz="UTC"))
    # Кандидат раз в 11 баров: блок в одиннадцать раз короче.
    assert lift.block_length_rows(sparse, horizon_minutes=1440) == 9

    # Вырожденные входы не роняют замер.
    assert lift.block_length_rows(pd.Series(dtype="datetime64[ns, UTC]"), 1440) == 1
    assert lift.block_length_rows(dense.iloc[:1], 1440) == 1


def test_block_length_ignores_the_truncated_tail():
    """
    У последних строк окно горизонта обрезано концом выборки.

    Усреднять по ним нельзя: на плотном ряду из 2000 строк это давало бы 94
    вместо 96, и чем короче выборка, тем сильнее занижение — а короткий блок
    делает тест анти-консервативным.
    """
    short = pd.Series(pd.date_range("2020-01-01", periods=300, freq="15min", tz="UTC"))
    assert lift.block_length_rows(short, horizon_minutes=1440) == 30  # потолок n//10

    medium = pd.Series(pd.date_range("2020-01-01", periods=1000, freq="15min", tz="UTC"))
    assert lift.block_length_rows(medium, horizon_minutes=1440) == 96


def test_block_length_follows_actual_window_density():
    """
    Считается фактическое число строк в окне, а не производная от среднего
    шага между строками.

    Разница принципиальна на скошенном распределении интервалов — а оно у нас
    именно такое: снимки офсетов дают пачки по четыре строки внутри трёх
    часов, между пачками разрыв. Средний шаг такую выборку описывает плохо.
    """
    # Две эпохи: ранняя редкая (строка раз в 5 часов) и поздняя плотная
    # (раз в полчаса). Ровно то, что наблюдается на боевых данных.
    sparse = pd.date_range("2020-01-01", periods=1000, freq="300min", tz="UTC")
    dense = pd.date_range(sparse[-1] + pd.Timedelta(minutes=30),
                          periods=1000, freq="30min", tz="UTC")
    uneven = pd.Series(sparse.append(dense))

    got = lift.block_length_rows(uneven, horizon_minutes=1440)

    # Расчёт по среднему шагу усредняет разрывы и недооценивает плотную
    # эпоху — именно там, где зависимость сильнее всего.
    span = (uneven.iloc[-1] - uneven.iloc[0]).total_seconds() / 60.0
    naive = round(1440 / (span / (len(uneven) - 1)))
    assert naive < got, f"по среднему {naive}, по факту {got}"
    # В плотной эпохе в сутки укладывается 48 строк, в редкой — около пяти;
    # честное среднее лежит между ними, а не у нижней границы.
    assert 20 <= got <= 40, got


def test_naive_test_finds_an_effect_that_is_not_there():
    """
    Контрольная точка задачи P0-1: на зависимых наблюдениях без всякой связи
    наивный z-тест уверенно «находит» эффект. Если этот тест начнёт падать,
    значит синтетика перестала быть зависимой и остальные проверки ничего
    не проверяют.
    """
    results = lift.measure_lift(
        _dependent_frame(), atoms=["a"], holdout=None, min_group=10,
        horizon_minutes=None,
    )
    assert abs(results[0].z) > 3.0
    assert results[0].p_value < 0.01
    assert results[0].p_boot is None       # бутстрап не просили — не считали


def test_block_bootstrap_rejects_the_spurious_effect():
    """Тот же ряд, но с блочным бутстрапом — значимости нет."""
    results = lift.measure_lift(
        _dependent_frame(), atoms=["a"], holdout=None, min_group=10,
        horizon_minutes=1440, n_boot=400,
    )
    result = results[0]
    assert result.p_boot > 0.05
    assert result.effective_p == result.p_boot
    assert not result.significant


def test_block_bootstrap_keeps_a_real_effect():
    """Сильную настоящую связь бутстрап обязан сохранить — иначе он бесполезен."""
    results = lift.measure_lift(
        _dependent_frame(effect=1.8, seed=11), atoms=["a"], holdout=None,
        min_group=10, horizon_minutes=1440, n_boot=400,
    )
    result = results[0]
    assert result.lift > 0.3
    assert result.p_boot < 0.05
    assert result.significant


def test_bootstrap_p_value_is_never_zero():
    """
    Поправка на единицу: (1 + k) / (1 + B). Ноль означал бы «достоверно
    абсолютно», хотя на деле это «реплик не хватило».
    """
    frame = _dependent_frame(effect=3.0, seed=3)
    flags = np.array([bool(a) for a in frame["atoms"]])
    p = lift.block_bootstrap_p(flags, frame["metric"].to_numpy(), block_length=9, n_boot=50)
    assert p >= 1 / 51
    assert p <= 1.0


def test_bootstrap_is_reproducible_and_independent_of_atom_order():
    """
    Зерно выводится из имени атома, а не из порядка перебора: замер по
    одному атому обязан воспроизводить строку общего замера.
    """
    frame = _dependent_frame(seed=5)
    frame["atoms"] = [list(a) + (["b"] if i % 5 else []) for i, a in enumerate(frame["atoms"])]

    both = {r.atom: r.p_boot for r in lift.measure_lift(
        frame, atoms=["a", "b"], holdout=None, min_group=10,
        horizon_minutes=1440, n_boot=200)}
    alone = {r.atom: r.p_boot for r in lift.measure_lift(
        frame, atoms=["b"], holdout=None, min_group=10,
        horizon_minutes=1440, n_boot=200)}
    assert both["b"] == alone["b"]


def test_degenerate_input_does_not_break_the_bootstrap():
    metric = np.array([1.0, 0.0] * 20)
    assert lift.block_bootstrap_p(np.zeros(40, dtype=bool), metric, 5, n_boot=20) == 1.0
    assert lift.block_bootstrap_p(np.ones(40, dtype=bool), metric, 5, n_boot=20) == 1.0
    # Слишком короткий ряд — «не отличили», а не падение.
    assert lift.block_bootstrap_p(np.array([True, False]), np.array([1.0, 0.0]), 2) == 1.0


def test_thinning_agrees_with_the_bootstrap():
    """
    Прореживание — независимая от бутстрапа проверка того же самого: оно
    выбрасывает данные, но не делает никаких допущений. На ряде без связи
    z должен упасть в область шума.
    """
    frame = _dependent_frame()
    z = lift.thinned_z(frame, "a", horizon_minutes=1440)
    assert z is not None and abs(z) < 2.0

    strong = lift.thinned_z(_dependent_frame(effect=1.8, seed=11), "a", horizon_minutes=1440)
    assert strong is not None and abs(strong) > 2.0


def test_table_prints_both_p_values_and_warns_without_bootstrap():
    frame = _dependent_frame()
    with_boot = lift.measure_lift(frame, atoms=["a"], holdout=None, min_group=10,
                                  horizon_minutes=1440, n_boot=100)
    table = lift.format_table(with_boot)
    assert "p наив." in table and "p блок." in table
    assert "БЛОЧНОМУ" in table

    without = lift.measure_lift(frame, atoms=["a"], holdout=None, min_group=10)
    assert "ВНИМАНИЕ" in lift.format_table(without)


def test_unusable_replicates_do_not_bias_p_downwards():
    """
    Реплика, где все флаги True (или все False), лифта не даёт и в числитель
    попасть не может. Считая её в знаменателе, мы занижали бы p, то есть
    двигали результат в сторону значимости — и сильнее всего у самых редких
    атомов, которые и без того наиболее подозрительны.

    Проверяем на редком атоме: при `n_boot = 200` заметная доля реплик
    вырождается. Если бы они шли в знаменатель, p был бы систематически ниже.
    """
    rng = np.random.default_rng(3)
    n = 600
    flags = np.zeros(n, dtype=bool)
    flags[rng.choice(n, size=12, replace=False)] = True   # 2% строк
    metric = (rng.uniform(size=n) < 0.5).astype(float)

    p = lift.block_bootstrap_p(flags, metric, block_length=30, n_boot=200,
                               rng=np.random.default_rng(1))
    # Связи нет — p обязан быть далеко от значимости, а не «почти прошёл»
    # из-за вырожденных реплик в знаменателе.
    assert 0.05 < p <= 1.0


def test_bootstrap_collects_the_requested_number_of_usable_replicates():
    """
    Знаменатель — число ПРИГОДНЫХ реплик, и оно достигает запрошенного.

    Косвенно: минимально достижимый p равен 1/(1+n_boot). Если бы часть
    реплик терялась, знаменатель был бы меньше и минимум — крупнее.
    """
    rng = np.random.default_rng(5)
    n = 500
    flags = np.zeros(n, dtype=bool)
    flags[:250] = True
    metric = np.concatenate([np.ones(250), np.zeros(250)])   # идеальная связь

    p = lift.block_bootstrap_p(flags, metric, block_length=10, n_boot=100,
                               rng=np.random.default_rng(2))
    assert p == pytest.approx(1 / 101, abs=1e-9)


def test_degenerate_input_terminates_instead_of_looping():
    """
    У вырожденного входа пригодных реплик может не набраться никогда.
    Потолок на попытки обязателен: молча крутиться здесь нельзя.
    """
    flags = np.zeros(40, dtype=bool)
    flags[0] = True                       # один True на сорок строк
    metric = np.tile([1.0, 0.0], 20)
    # Блок во всю выборку — почти каждая реплика вырождается.
    p = lift.block_bootstrap_p(flags, metric, block_length=40, n_boot=50,
                               rng=np.random.default_rng(7))
    assert 0.0 < p <= 1.0


def test_table_warns_when_resolution_is_too_coarse():
    """
    При Бонферрони на многих тестах минимальный достижимый p сопоставим с
    порогом, и результат решает шум Монте-Карло, а не данные. Молчать об
    этом нельзя.
    """
    frame = _dependent_frame(effect=1.8, seed=11)
    results = lift.measure_lift(frame, atoms=["a"], holdout=None, min_group=10,
                                horizon_minutes=1440, n_boot=100)
    # Один тест — порога хватает, предупреждения нет.
    assert "РЕПЛИК МАЛО" not in lift.format_table(results, "bonferroni", 0.05, 100)

    # Сорок пять тестов при тех же ста репликах — уже не хватает.
    padded = results * 45
    table = lift.format_table(padded, "bonferroni", 0.05, 100)
    assert "РЕПЛИК МАЛО" in table
    assert "--n-boot ≥" in table


def test_resolution_rule_matches_the_documented_formula():
    # Бонферрони: нужно B ≥ 4·m/α.
    assert not lift.resolution_is_sufficient(2000, 0.05, 45, "bonferroni")
    assert lift.resolution_is_sufficient(3600, 0.05, 45, "bonferroni")
    # BH мягче: порог не делится на число тестов.
    assert lift.resolution_is_sufficient(2000, 0.05, 45, "bh")
    assert not lift.resolution_is_sufficient(10, 0.05, 45, "bh")
