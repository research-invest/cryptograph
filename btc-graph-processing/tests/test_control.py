"""
Проверки контрольной модели без графа (btcproc/analysis/control.py, D1).

Замер отвечает на вопрос «потолок в графе или в признаках», и цена ошибки
здесь та же, что у holdout: неверный ответ разворачивает направление работ
целиком. Поэтому проверяется не «функция считает то, что обещает», а три
свойства, ошибка в каждом из которых сделала бы вывод ложным:

* между обучающим и проверочным окном обязан быть зазор в горизонт — без
  него модель учится на исходах, которые потом предсказывает, и находит
  сигнал в чистом шуме;
* на данных с реальной зависимостью модель обязана её найти — иначе замер
  не отличает «сигнала нет» от «мерялка сломана»;
* критерий обязан проваливаться, если хоть одна его половина не выполнена.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcproc.analysis import control


def _ts(n: int, minutes: int = 15) -> pd.Series:
    return pd.Series(pd.date_range("2024-01-01", periods=n, freq=f"{minutes}min",
                                   tz="UTC"))


# ─── Разбиение ──────────────────────────────────────────────────────────────
def test_purged_splits_leave_the_horizon_gap():
    """
    Между концом обучения и началом проверки — ровно `gap` строк.

    Это единственное, что отделяет честный walk-forward от бэктеста, который
    учится на будущем: метка бара t считается по барам t+1…t+gap.
    """
    splits = control.purged_splits(n=10_000, folds=5, gap=96)
    assert splits, "на десяти тысячах строк фолды обязаны построиться"
    for train, test in splits:
        assert test.start - train.stop == 96
        assert train.start == 0                 # окно расширяющееся
        assert test.stop <= 10_000
    # Окна проверки идут вперёд и не перекрываются.
    starts = [test.start for _, test in splits]
    assert starts == sorted(starts)
    for (_, a), (_, b) in zip(splits, splits[1:]):
        assert a.stop <= b.start


def test_purged_splits_refuse_a_history_too_short():
    """Лучше ноль фолдов, чем фолд, где зазор съел проверочное окно."""
    assert control.purged_splits(n=200, folds=5, gap=96) == []


# ─── Модель находит то, что есть ────────────────────────────────────────────
@pytest.fixture(scope="module")
def learnable():
    """
    Синтетика, где связь признака с исходом ЕСТЬ и она нелинейная.

    Нужна как контроль самого контроля: если на таких данных замер не находит
    сигнала, значит сломан он, а не рынок.
    """
    rng = np.random.default_rng(7)
    n = 6000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = rng.normal(scale=0.5, size=n)
    target = (x1 * x2 + noise) > 0
    features = pd.DataFrame({"x1": x1, "x2": x2, "junk": rng.normal(size=n)})
    return features, target


def test_model_finds_a_real_dependency(learnable):
    features, target = learnable
    train = slice(0, 4000)
    test = slice(4100, 6000)

    raw, calibrated = control.fit_predict(features, target, train, test, seed=42, gap=96)

    assert control.roc_auc(raw, target[test].astype(float)) > 0.7
    accuracy = float(((calibrated > 0.5) == target[test]).mean())
    assert accuracy > 0.65


def test_model_finds_nothing_in_pure_noise():
    """Обратная половина того же контроля: на шуме AUC обязан лечь около 0.5."""
    rng = np.random.default_rng(11)
    n = 6000
    features = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    target = rng.random(n) > 0.5

    raw, _ = control.fit_predict(features, target, slice(0, 4000), slice(4100, n),
                                 seed=42, gap=96)
    auc = control.roc_auc(raw, target[slice(4100, n)].astype(float))
    assert 0.45 < auc < 0.55


# ─── Замер и критерий ───────────────────────────────────────────────────────
def _report(predicted: np.ndarray, actual: np.ndarray, **kwargs) -> control.ControlReport:
    n = len(actual)
    return control.measure(
        ts=_ts(n), raw=predicted, predicted=predicted, actual=actual,
        symbol="TESTUSDT", seed=42, split_ts=pd.Timestamp("2024-01-01", tz="UTC"),
        horizon_minutes=1440, n_features=3, n_train=n, n_boot=kwargs.get("n_boot", 200),
    )


def test_criterion_fails_on_a_coin_flip():
    rng = np.random.default_rng(3)
    n = 4000
    actual = rng.random(n) > 0.5
    predicted = np.clip(0.5 + rng.normal(scale=0.02, size=n), 0.01, 0.99)

    report = _report(predicted, actual)
    assert not report.passes
    assert abs(report.accuracy.value - 0.5) < 0.05


def test_criterion_needs_more_than_beating_fifty_percent():
    """
    Рынок, который просто рос: «всегда long» даёт 70%, знания в этом нет.

    Модель, повторяющая дрейф, обязана провалить критерий — парная разница с
    бенчмарком у неё нулевая. Это та же защита, что у валидации графа, и она
    обязана работать одинаково в обоих замерах.
    """
    n = 4000
    rng = np.random.default_rng(5)
    actual = rng.random(n) > 0.3            # 70% вверх
    predicted = np.full(n, 0.7)             # «всегда long», ничего не зная

    report = _report(predicted, actual)
    assert report.accuracy.value > 0.65     # выглядит прекрасно…
    assert report.versus_always_long.value == pytest.approx(0.0, abs=1e-9)
    assert not report.passes                # …и критерий это ловит


def test_collapsed_calibration_is_flagged_not_hidden():
    """
    Изотоника, схлопнувшая прогноз в константу, обязана быть видна в отчёте.

    Иначе замер выглядит как «accuracy равна базовой частоте» без объяснения,
    и читатель решит, что модель что-то предсказывает.
    """
    n = 2000
    rng = np.random.default_rng(13)
    actual = rng.random(n) > 0.5
    report = _report(np.full(n, 0.5167), actual)

    assert report.calibration_collapsed
    assert "изотоника" in control.format_report(report).lower()
