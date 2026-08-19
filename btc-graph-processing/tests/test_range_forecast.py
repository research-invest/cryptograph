"""
Тесты квантильного регрессора размаха (`btcproc/analysis/range_forecast.py`).

Конструкция описана в шапке модуля. Здесь проверяется то, без чего её числам
нельзя верить:

* квантили возвращаются в исходной шкале и упорядочены;
* `range_lift` относителен по построению — общий сдвиг уровня размаха его не
  меняет (иначе он мерил бы волатильность рынка, а не вклад признаков);
* модель на признаках обязана обыгрывать свой бенчмарк там, где связь есть по
  построению, и НЕ обыгрывать на шуме;
* калибровка на честной синтетике попадает в номинал.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcproc.analysis import range_forecast as rf


def _cell(n: int = 12000, signal: float = 1.0, seed: int = 3):
    """
    Синтетика с ИЗВЕСТНЫМ источником сигнала.

    `bench` — липкий уровень (роль HAR-RV), `feature` — независимый от него
    предиктор (роль признака). При `signal=0` признак чистый шум, и модель
    обязана не найти в нём ничего.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-01", periods=n, freq="15min", tz="UTC")
    bench = np.cumsum(rng.normal(0, 0.02, n))
    bench -= bench.mean()
    feature = rng.normal(size=n)
    noise = rng.normal(0, 0.4, n)
    target = pd.Series(0.6 * bench + signal * 0.6 * feature + noise, index=index)
    frame = pd.DataFrame({"bench_col": bench, "feature_col": feature}, index=index)
    return frame, target, ["bench_col"]


def _fit(frame, target, benchmark, train_end: int, seed: int = 42):
    return rf.fit("TESTUSDT", frame, target, benchmark, slice(0, train_end), seed,
                  horizon="24h", normalization="atr14", log=lambda m: None)


def test_quantiles_come_back_in_the_original_scale_and_are_ordered():
    """
    Модель учится в логарифмах, а отдаёт `range_ratio`. Квантиль монотонного
    преобразования — это преобразование квантиля, поэтому обратный ход точен;
    проверяется, что он вообще сделан и что порядок квантилей не нарушен.
    """
    frame, target, benchmark = _cell()
    model = _fit(frame, target, benchmark, 9000)
    predicted = model.predict(frame.iloc[9000:])

    assert (predicted[["p25", "p50", "p75", "p90"]] > 0).all().all(), (
        "значения обязаны быть в шкале range_ratio, то есть положительными"
    )
    heads = predicted[["p25", "p50", "p75", "p90"]].to_numpy()
    assert (np.diff(heads, axis=1) >= -1e-12).all(), "квантили обязаны не убывать"
    bench_heads = predicted[["bench_p25", "bench_p50", "bench_p75", "bench_p90"]]
    assert (np.diff(bench_heads.to_numpy(), axis=1) >= -1e-12).all()


def test_range_lift_is_relative_to_the_benchmark():
    """
    Главное свойство `range_lift`: он не зависит от общего уровня размаха.

    Сдвиг цели на константу в логарифмах — это умножение размаха на константу,
    то есть «рынок стал шире везде». Обе модели съедут одинаково, и отношение
    обязано остаться прежним. Урок 26.4: абсолютное число неотличимо от
    дрейфа, отличимо только сравнение с бенчмарком.
    """
    frame, target, benchmark = _cell()
    base = _fit(frame, target, benchmark, 9000)
    shifted = _fit(frame, target + 0.5, benchmark, 9000)

    lift_base = base.predict(frame.iloc[9000:])["range_lift"]
    lift_shifted = shifted.predict(frame.iloc[9000:])["range_lift"]
    assert lift_shifted.median() == pytest.approx(lift_base.median(), abs=0.05)
    # А сами квантили съехать ОБЯЗАНЫ — иначе сдвиг просто не дошёл до модели.
    p50_base = base.predict(frame.iloc[9000:])["p50"].median()
    p50_shifted = shifted.predict(frame.iloc[9000:])["p50"].median()
    assert p50_shifted > p50_base * 1.3


def test_features_beat_the_benchmark_when_the_signal_is_real():
    """Позитивный контроль: связь есть по построению — модель обязана её найти."""
    frame, target, benchmark = _cell(signal=1.0)
    model = _fit(frame, target, benchmark, 9000)
    predicted = model.predict(frame.iloc[9000:])
    actual = np.exp(target.to_numpy()[9000:])

    from scipy.stats import spearmanr
    rho_model = spearmanr(predicted["p50"], actual).statistic
    rho_bench = spearmanr(predicted["bench_p50"], actual).statistic
    assert rho_model > rho_bench + 0.05


def test_pure_noise_feature_does_not_beat_the_benchmark():
    """
    Негативный контроль: признак не несёт ничего, и превосходства быть не
    должно. Без этого теста «модель работает» неотличимо от «код запустился».
    """
    frame, target, benchmark = _cell(signal=0.0)
    model = _fit(frame, target, benchmark, 9000)
    predicted = model.predict(frame.iloc[9000:])
    actual = np.exp(target.to_numpy()[9000:])

    from scipy.stats import spearmanr
    rho_model = spearmanr(predicted["p50"], actual).statistic
    rho_bench = spearmanr(predicted["bench_p50"], actual).statistic
    assert rho_model <= rho_bench + 0.03
    assert abs(predicted["range_lift"].median() - 1.0) < 0.10


def test_coverage_hits_the_nominal_quantiles():
    """Калибровка: под p25 обязаны оказаться примерно 25% случаев, и так далее."""
    frame, target, benchmark = _cell(n=16000)
    model = _fit(frame, target, benchmark, 12000)
    predicted = model.predict(frame.iloc[12000:])
    actual = np.exp(target.to_numpy()[12000:])

    observed = rf.coverage(actual, predicted)
    for q in rf.QUANTILES:
        assert abs(observed[q] - q) < 0.06, f"квантиль {q}: покрытие {observed[q]:.3f}"
    assert rf.calibration_error(observed) < 0.05


def test_pinball_loss_is_minimised_by_the_true_quantile():
    """
    Потеря пинбола обязана быть минимальна на истинном квантиле — иначе она
    меряет не то, и все сравнения моделей по ней бессмысленны.
    """
    rng = np.random.default_rng(0)
    actual = rng.normal(10.0, 2.0, 20000)
    true_p90 = float(np.quantile(actual, 0.90))
    at_truth = rf.pinball_loss(actual, np.full_like(actual, true_p90), 0.90)
    for wrong in (true_p90 - 1.0, true_p90 + 1.0):
        assert rf.pinball_loss(actual, np.full_like(actual, wrong), 0.90) > at_truth


def test_regime_edges_are_declared_not_fitted():
    """
    Границы режима — константа модуля, а не свойство данных. Тест фиксирует
    и сами значения, и то, что режим считается именно по `range_lift`.
    """
    assert rf.REGIME_EDGES == (0.85, 1.15)
    frame, target, benchmark = _cell()
    model = _fit(frame, target, benchmark, 9000)
    predicted = model.predict(frame.iloc[9000:])

    low, high = rf.REGIME_EDGES
    assert (predicted.loc[predicted["range_lift"] < low, "range_regime"]
            == "compressed").all()
    assert (predicted.loc[predicted["range_lift"] > high, "range_regime"]
            == "expanded").all()


# ─── Интеграция с кандидатом ────────────────────────────────────────────────
def test_view_refuses_bars_the_model_was_trained_on(bars, features):
    """
    Прогноз отдаётся ТОЛЬКО для баров строго после конца обучения.

    Инвариант 4 в применении к модели: на исторических барах боевая версия
    обучалась, и её квантили там — запоминание. Разница не видна ни по одному
    числу, поэтому граница проверяется явно.
    """
    frame, target, benchmark = rf.design_matrix(bars, features, "24h", "atr14", 15)
    cut = int(len(frame) * 0.7)
    model = rf.fit("TESTUSDT", frame, target, benchmark, slice(0, cut), 42,
                   horizon="24h", normalization="atr14", log=lambda m: None)

    view = rf.view(model, bars, features, log=lambda *a: None)
    assert not view.empty
    assert (view.index > model.train_end).all()
    assert list(view.columns) == ["p50", "p90", "range_lift", "range_regime"]


def test_view_refuses_a_changed_feature_set(bars, features):
    """
    Набор признаков разошёлся с моделью — отказ, а не подгонка.

    Молча предсказывать по другому вектору хуже, чем не предсказывать вовсе:
    числа появятся, и отличить их от правильных будет нечем.
    """
    frame, target, benchmark = rf.design_matrix(bars, features, "24h", "atr14", 15)
    cut = int(len(frame) * 0.7)
    model = rf.fit("TESTUSDT", frame, target, benchmark, slice(0, cut), 42,
                   horizon="24h", normalization="atr14", log=lambda m: None)

    said = []
    assert rf.view(model, bars, features.drop(columns=["rsi"]),
                   log=lambda msg, *a: said.append(msg)).empty
    assert said, "отказ обязан быть громким"


def test_candidate_gets_range_fields_only_where_the_view_has_them(pipeline_data):
    """
    Поля размаха появляются у кандидата ровно на тех барах, что есть в
    прогнозе, и равны None везде ещё. Пустота — не ошибка, а честное «система
    про размах этого бара ничего не говорит».
    """
    from btcproc.candidates import builder as cand

    snapshots = pipeline_data["snapshots"]
    rarity = dict(zip(pipeline_data["transitions"]["transition_id"],
                      pipeline_data["transitions"]["rarity"]))
    blocks = pipeline_data["blocks"].set_index("event_block_id").to_dict("index")

    covered = pd.DatetimeIndex(sorted(set(snapshots["ts"]))[-50:])
    view = pd.DataFrame({
        "p50": 1.1, "p90": 2.2, "range_lift": 1.3, "range_regime": "expanded",
    }, index=covered)

    from btcproc import config
    cfg = config.CandidateConfig(min_sample_size=10, min_effective_sample_size=4)
    produced = list(cand.generate(snapshots, rarity, blocks, "BTCUSDT", cfg=cfg,
                                  range_view=view))
    assert produced

    with_fields = [c for c in produced
                   if pd.Timestamp(c["_meta"]["ts"]) in covered]
    without = [c for c in produced
               if pd.Timestamp(c["_meta"]["ts"]) not in covered]
    assert with_fields and without, "нужны обе группы, иначе тест ничего не проверяет"
    for c in with_fields:
        assert c["expected_range_ratio_p50"] == 1.1
        assert c["expected_range_ratio_p90"] == 2.2
        assert c["range_lift"] == 1.3
        assert c["range_regime"] == "expanded"
    for c in without:
        assert all(c[name] is None for name in cand.RANGE_FIELDS)


def test_candidate_without_a_model_carries_empty_range_fields(pipeline_data):
    """`range_view=None` — штатный путь: поля присутствуют и равны None."""
    from btcproc import config
    from btcproc.candidates import builder as cand

    rarity = dict(zip(pipeline_data["transitions"]["transition_id"],
                      pipeline_data["transitions"]["rarity"]))
    blocks = pipeline_data["blocks"].set_index("event_block_id").to_dict("index")
    cfg = config.CandidateConfig(min_sample_size=10, min_effective_sample_size=4)

    produced = list(cand.generate(pipeline_data["snapshots"], rarity, blocks,
                                  "BTCUSDT", cfg=cfg))
    assert produced
    for c in produced[:50]:
        assert all(name in c for name in cand.RANGE_FIELDS)
        assert all(c[name] is None for name in cand.RANGE_FIELDS)


def test_artifact_survives_a_round_trip(bars, features):
    """Сохранение в БД и обратно: модель обязана считать то же самое."""
    frame, target, benchmark = rf.design_matrix(bars, features, "24h", "atr14", 15)
    cut = int(len(frame) * 0.7)
    model = rf.fit("TESTUSDT", frame, target, benchmark, slice(0, cut), 42,
                   horizon="24h", normalization="atr14", log=lambda m: None)

    restored = rf.loads(rf.dumps(model))
    assert restored is not None
    before = model.predict(frame.iloc[cut:cut + 20])
    after = restored.predict(frame.iloc[cut:cut + 20])
    assert (before["p50"].to_numpy() == after["p50"].to_numpy()).all()
    assert (before["range_regime"] == after["range_regime"]).all()


def test_artifact_of_a_foreign_version_is_refused_loudly():
    """
    Артефакт чужой версии формата или чужой sklearn — отказ с записью в лог.

    Пиклы моделей между версиями библиотеки несовместимы, и подняться такой
    артефакт может молча-неправильным. Отказ здесь дешевле.
    """
    import io

    import joblib

    said = []
    buffer = io.BytesIO()
    joblib.dump({"version": "чужая", "sklearn": "0.0", "model": None}, buffer)
    assert rf.loads(buffer.getvalue(), log=lambda msg, *a: said.append(msg)) is None
    assert said

    said.clear()
    assert rf.loads("не пикл вовсе".encode("utf-8"),
                    log=lambda msg, *a: said.append(msg)) is None
    assert said
