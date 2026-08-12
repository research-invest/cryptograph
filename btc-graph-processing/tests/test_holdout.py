"""
Проверки валидации на отложенной части (btcproc/analysis/holdout.py).

Замер этот особенный: его результат — основание решать, продолжать ли
полировать систему вообще (Ш0 плана 2026-08-12). Поэтому проверяется не
только «функция считает то, что обещает», но и два свойства, ошибка в
которых сделала бы вывод ложным:

* блочный бутстрап обязан быть КОНСЕРВАТИВНЕЕ наивного теста на зависимых
  наблюдениях — иначе замер найдёт предсказательную силу там, где её нет;
* карты редкости обязаны считаться по префиксу истории — иначе кандидат
  holdout'а знает, чем этот holdout закончился.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcproc.analysis import holdout as ho


def _ts(n: int, minutes: int = 45) -> pd.Series:
    """Хронологический ряд отметок с шагом, как у снимков офсетов."""
    return pd.Series(pd.date_range("2024-01-01", periods=n, freq=f"{minutes}min",
                                   tz="UTC"))


# ─── Попадание и базовые метрики ────────────────────────────────────────────
def test_hit_flags_reads_short_side_backwards():
    """Short попадает, когда бар закрылся ВНИЗ. Это самая дешёвая ошибка знака."""
    side = pd.Series(["long", "long", "short", "short"])
    is_up = pd.Series([True, False, True, False])
    assert list(ho.hit_flags(side, is_up)) == [1.0, 0.0, 0.0, 1.0]


def test_brier_rewards_the_honest_forecast():
    y = np.array([1.0, 1.0, 0.0, 0.0])
    honest = np.array([0.9, 0.8, 0.2, 0.1])
    confident_and_wrong = np.array([0.1, 0.2, 0.8, 0.9])
    assert ho.brier(honest, y) < ho.brier(np.full(4, 0.5), y)
    assert ho.brier(confident_and_wrong, y) > ho.brier(np.full(4, 0.5), y)


def test_calibration_is_diagonal_when_predictions_are_honest():
    """
    Честный прогноз: y ~ Bernoulli(p). Тогда фактическая доля в дециле
    совпадает с предсказанной, и ECE мал.
    """
    rng = np.random.default_rng(0)
    p = rng.uniform(0.3, 0.7, 20_000)
    y = (rng.random(20_000) < p).astype(float)
    table = ho.calibration_bins(p, y)
    assert len(table) == 10
    assert ho.expected_calibration_error(table) < 0.02


def test_calibration_exposes_a_forecast_that_means_nothing():
    """
    Прогноз гуляет от 0.3 до 0.7, а исход всегда монетка. Калибровка обязана
    показать разрыв, растущий к краям, — ровно тот исход «перекос не
    реализуется», ради которого замер и ставится.
    """
    rng = np.random.default_rng(1)
    p = rng.uniform(0.3, 0.7, 20_000)
    y = (rng.random(20_000) < 0.5).astype(float)
    table = ho.calibration_bins(p, y)
    assert ho.expected_calibration_error(table) > 0.05
    # В нижнем дециле факт выше предсказания, в верхнем — ниже.
    assert table.iloc[0]["gap"] > 0
    assert table.iloc[-1]["gap"] < 0


# ─── Значимость ─────────────────────────────────────────────────────────────
def test_estimate_does_not_reject_pure_noise():
    rng = np.random.default_rng(3)
    values = (rng.random(5000) < 0.5).astype(float)
    est = ho.estimate(values, 0.5, block_length=10, n_boot=500, rng=rng)
    assert not est.significant
    assert est.ci_low < 0.5 < est.ci_high


def test_estimate_rejects_a_real_skew():
    rng = np.random.default_rng(4)
    values = (rng.random(5000) < 0.56).astype(float)
    est = ho.estimate(values, 0.5, block_length=10, n_boot=500, rng=rng)
    assert est.significant
    assert est.value > 0.5
    assert est.ci_low > 0.5


def test_block_bootstrap_is_wider_than_the_naive_interval():
    """
    Главная защита замера. Наблюдения зависимы (перекрытие горизонтов и
    снимки офсетов), поэтому наивный биномиальный интервал вокруг accuracy
    занижен. Блочный интервал на серийных данных обязан быть заметно шире —
    иначе вся поправка на зависимость существует только на бумаге.

    Ряд строится сериями по 20 одинаковых значений: это огрубление того, что
    в данных даёт один горизонт, накрывающий полтора десятка кандидатов.
    """
    rng = np.random.default_rng(5)
    blocks = (rng.random(250) < 0.5).astype(float)
    values = np.repeat(blocks, 20)               # 5000 строк, 250 независимых

    est = ho.estimate(values, 0.5, block_length=20, n_boot=1000, rng=rng)
    block_width = est.ci_high - est.ci_low

    naive_se = np.sqrt(0.25 / len(values))
    naive_width = 2 * 1.96 * naive_se
    assert block_width > 3 * naive_width

    # И при таком n «доля 0.5 ± шум» не должна объявляться значимой.
    assert not est.significant


def test_estimate_survives_an_empty_sample():
    est = ho.estimate(np.array([]), 0.5, block_length=5, n_boot=50)
    assert est.n == 0
    assert est.p_value == 1.0


# ─── Разрезы ────────────────────────────────────────────────────────────────
def _frame(hits: np.ndarray, ratings: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"ts": _ts(len(hits)), "hit": hits, "rating": ratings})


def test_contrast_finds_a_group_that_is_really_better():
    rng = np.random.default_rng(6)
    n = 3000
    ratings = ["STRONG" if i % 2 else "WEAK" for i in range(n)]
    hits = np.where(
        np.array(ratings) == "STRONG",
        (rng.random(n) < 0.62).astype(float),
        (rng.random(n) < 0.48).astype(float),
    )
    result = ho.contrast(_frame(hits, ratings), "rating", "STRONG", "WEAK",
                         horizon_minutes=1440, n_boot=400,
                         rng=np.random.default_rng(7))
    assert result["significant"]
    assert result["delta"] > 0.1


def test_contrast_stays_quiet_when_the_rating_means_nothing():
    """
    Если STRONG не лучше WEAK, разрез обязан молчать. Это второй половина
    критерия Ш0: линейка оценки, не различающая ничего, — декоративна.
    """
    rng = np.random.default_rng(8)
    n = 3000
    ratings = ["STRONG" if i % 2 else "WEAK" for i in range(n)]
    hits = (rng.random(n) < 0.52).astype(float)
    result = ho.contrast(_frame(hits, ratings), "rating", "STRONG", "WEAK",
                         horizon_minutes=1440, n_boot=400,
                         rng=np.random.default_rng(9))
    assert not result["significant"]


def test_contrast_does_not_call_a_worse_group_significant():
    """Двусторонний p-value: STRONG ХУЖЕ WEAK — это провал, а не успех."""
    rng = np.random.default_rng(10)
    n = 3000
    ratings = ["STRONG" if i % 2 else "WEAK" for i in range(n)]
    hits = np.where(
        np.array(ratings) == "STRONG",
        (rng.random(n) < 0.40).astype(float),
        (rng.random(n) < 0.60).astype(float),
    )
    result = ho.contrast(_frame(hits, ratings), "rating", "STRONG", "WEAK",
                         horizon_minutes=1440, n_boot=400,
                         rng=np.random.default_rng(11))
    assert result["delta"] < 0
    assert not result["significant"]


def test_by_bucket_marks_thin_groups():
    rng = np.random.default_rng(12)
    hits = (rng.random(200) < 0.5).astype(float)
    labels = ["A"] * 150 + ["B"] * 50
    table = ho.by_bucket(_frame(hits, labels).rename(columns={"rating": "bucket_col"}),
                         "bucket_col", horizon_minutes=1440, n_boot=200,
                         rng=np.random.default_rng(13))
    assert set(table["bucket"]) == {"A", "B"}
    assert bool(table[table["bucket"] == "B"]["thin"].iat[0])


def test_quantile_buckets_survive_a_degenerate_column():
    """У монеты с короткой историей sample_size бывает константой."""
    values = pd.Series([100] * 50)
    assert ho.quantile_buckets(values).nunique() == 1


# ─── Отчёт целиком ──────────────────────────────────────────────────────────
def _measurable(n: int = 4000, accuracy: float = 0.55, seed: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hit = (rng.random(n) < accuracy).astype(float)
    side = np.where(rng.random(n) < 0.5, "long", "short")
    is_up = np.where(side == "long", hit.astype(bool), ~hit.astype(bool))
    return pd.DataFrame({
        "ts": _ts(n),
        "side": side,
        "is_up": is_up,
        "hit": hit,
        "long_outcome_share": np.where(side == "long",
                                       rng.uniform(0.55, 0.7, n),
                                       rng.uniform(0.3, 0.45, n)),
        "sample_size": rng.integers(30, 3000, n),
        "research_score": rng.uniform(0.2, 0.8, n),
        "rating": rng.choice(["STRONG", "MODERATE", "WEAK"], n),
    })


def test_measure_builds_a_complete_report():
    frame = _measurable()
    report = ho.measure(frame, "BTCUSDT", model_run=1,
                        split_ts=pd.Timestamp("2024-01-01", tz="UTC"),
                        horizon_minutes=1440, n_boot=200)
    assert report.n_valid == len(frame)
    assert 0.5 < report.accuracy.value < 0.6
    assert not report.by_rating.empty
    assert len(report.by_sample_size) == 4
    assert "acc_high" in report.strong_vs_weak
    # Рейтинг здесь случайный — различать он не должен ничего.
    assert not report.strong_vs_weak["significant"]
    assert not report.passes
    text = ho.format_report(report)
    assert "directional accuracy" in text
    assert "критерий НЕ пройден" in text


def test_measure_passes_only_when_both_halves_hold():
    """
    Критерий Ш0 — конъюнкция: и accuracy значимо выше 50%, и STRONG лучше
    WEAK. Здесь воспроизведён второй исход из таблицы плана: рейтинг
    различает, а перекос в целом не реализуется. Критерий обязан НЕ пройти —
    иначе он схлопнулся бы до «хоть что-то сработало».
    """
    frame = _measurable(n=6000, accuracy=0.5, seed=21)
    rng = np.random.default_rng(22)
    for rating, rate in (("STRONG", 0.70), ("MODERATE", 0.42), ("WEAK", 0.42)):
        mask = frame["rating"] == rating
        frame.loc[mask, "hit"] = (rng.random(int(mask.sum())) < rate).astype(float)
    frame["is_up"] = np.where(frame["side"] == "long",
                              frame["hit"].astype(bool), ~frame["hit"].astype(bool))

    report = ho.measure(frame, "BTCUSDT", model_run=1,
                        split_ts=pd.Timestamp("2024-01-01", tz="UTC"),
                        horizon_minutes=1440, n_boot=300)
    assert report.strong_vs_weak["significant"]
    assert report.accuracy.value < 0.53
    assert not report.passes


def test_always_long_benchmark_catches_a_bull_market():
    """
    Растущий рынок: система всегда говорит long и «угадывает» в 60%. Первый
    тест (accuracy > 0.5) она проходит, а парное сравнение с «всегда long»
    обязано показать ноль — никакого знания за этим нет.
    """
    rng = np.random.default_rng(23)
    n = 4000
    is_up = rng.random(n) < 0.6
    frame = pd.DataFrame({
        "ts": _ts(n),
        "side": "long",
        "is_up": is_up,
        "hit": is_up.astype(float),
        "long_outcome_share": rng.uniform(0.55, 0.7, n),
        "sample_size": rng.integers(30, 3000, n),
        "research_score": rng.uniform(0.2, 0.8, n),
        "rating": rng.choice(["STRONG", "WEAK"], n),
    })
    report = ho.measure(frame, "BTCUSDT", model_run=1,
                        split_ts=pd.Timestamp("2024-01-01", tz="UTC"),
                        horizon_minutes=1440, n_boot=300)
    assert report.accuracy.value > 0.55
    assert report.versus_always_long.value == pytest.approx(0.0, abs=1e-9)
    assert not report.versus_always_long.significant


# ─── Карты редкости не заглядывают вперёд ───────────────────────────────────
def test_prefix_maps_use_only_the_training_part(pipeline_data):
    """
    Регрессия на утечку. Карта редкости, посчитанная для holdout-валидации,
    обязана совпадать с картой, посчитанной на обрезанной истории, и не
    зависеть от того, что было после границы.
    """
    from btcproc.features import events as ev
    from btcproc.states import graph

    states = pipeline_data["states"]
    events = pipeline_data["events"]
    outcomes = pipeline_data["outcomes"]
    split_ts = states.index[int(len(states) * 0.7)]

    rarity_map, block_map = ho.prefix_maps(states, events, outcomes, split_ts)

    # Эталон: те же функции на данных, которых после границы просто нет.
    past_states = states[states.index < split_ts]
    reference = graph.transition_stats(past_states,
                                       outcomes.reindex(past_states.index))
    expected = dict(zip(reference["transition_id"], reference["rarity"]))
    assert rarity_map == expected

    reference_blocks = ev.block_statistics(events[events.index < split_ts])
    assert set(block_map) == set(reference_blocks["event_block_id"])

    # И главное: карта по ВСЕЙ истории — другая. Если бы она совпадала,
    # тест выше ничего не доказывал бы.
    full = ev.block_statistics(events)
    assert set(full["event_block_id"]) - set(block_map)
