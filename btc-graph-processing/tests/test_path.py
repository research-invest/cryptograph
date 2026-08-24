"""
Асимметрия пути: аналитический якорь и три способа получить её из ничего.

Ценность этого замера целиком в нулёвке, поэтому и тесты про неё.

**Аналитический якорь.** Для процесса без сноса вероятность достичь `+a`
раньше `−b` равна `b/(a+b)` — независимо от волатильности, по замене времени.
При симметричных барьерах это ровно 0.5. Значит на случайном блуждании
разметка ОБЯЗАНА дать 0.5, и любое отклонение — дефект кода, а не свойство
рынка. Это единственный тест в проекте, где ответ известен точно.

**Одновременное касание.** Внутри одного бара оба барьера могут быть задеты,
и порядок в данных отсутствует. Любое правило («считаем, что сначала был
low») внесло бы систематический перекос ровно в измеряемую величину, поэтому
такие случаи идут в отдельную метку.

**Размах по состояниям** — статистика экстремума по полусотне состояний. Она
заметно больше нуля и при полном отсутствии эффекта, и сравнивать её с нулём
нельзя.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from btcproc.analysis import path as pt


def _walk(n: int = 60000, sigma: float = 0.003, drift: float = 0.0,
          seed: int = 0, spread: float = 0.0005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(drift, sigma, n)))
    return pd.DataFrame({
        "open": close, "high": close * (1 + spread), "low": close * (1 - spread),
        "close": close, "volume": 1.0,
    }, index=index)


def test_driftless_walk_gives_exactly_half():
    """
    Аналитический якорь. Ответ известен точно и не зависит ни от
    волатильности, ни от её суточной формы.
    """
    base = _walk()
    sigma = pt.sigma_series(base)
    for k in (1.0, 2.0):
        labels = pt.triple_barrier(base, sigma, k, 96)
        shares = pt.label_shares(labels)
        assert abs(shares["up_share"] - pt.analytic_anchor(k)) < 0.01, k


def test_drift_is_detected():
    """Положительный контроль: снос обязан сдвигать долю вверх."""
    base = _walk(drift=0.0004)
    labels = pt.triple_barrier(base, pt.sigma_series(base), 1.0, 96)
    assert pt.label_shares(labels)["up_share"] > 0.55


def test_simultaneous_touch_is_not_assigned_to_a_side():
    """
    Бар, накрывающий оба барьера, обязан попасть в `ambiguous`. Любое правило
    доопределения внесло бы перекос ровно в измеряемую величину.
    """
    index = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")
    base = pd.DataFrame({
        "open": [100.0] * 4, "high": [100.0, 100.0, 200.0, 100.0],
        "low": [100.0, 100.0, 50.0, 100.0], "close": [100.0] * 4,
        "volume": 1.0,
    }, index=index)
    sigma = pd.Series([0.01] * 4, index=index)
    labels = pt.triple_barrier(base, sigma, 1.0, 2)
    assert labels[1] == pt.LABELS.index("ambiguous")


def test_bars_without_a_full_horizon_have_no_outcome():
    """
    У последних баров горизонт не помещается в историю. Метка `none` там
    означала бы «до барьера не дошли», а на деле мы просто не смотрели.
    """
    base = _walk(n=1000)
    labels = pt.triple_barrier(base, pt.sigma_series(base), 1.0, 96)
    assert (labels[-96:] == -1).all()


def test_entry_bar_itself_is_not_an_outcome():
    """
    Проверка начинается с бара `t+1`: касание внутри бара входа — это
    прошлое, а не исход.
    """
    index = pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC")
    base = pd.DataFrame({
        "open": [100.0] * 3, "high": [200.0, 100.0, 100.0],
        "low": [100.0, 100.0, 100.0], "close": [100.0] * 3, "volume": 1.0,
    }, index=index)
    sigma = pd.Series([0.01] * 3, index=index)
    labels = pt.triple_barrier(base, sigma, 1.0, 1)
    assert labels[0] == pt.LABELS.index("none"), "свой же бар исходом не является"


def test_state_spread_is_large_even_without_any_effect():
    """
    Смысл нулёвки для размаха: при случайных метках состояний размах доли
    между полусотней групп заметно больше нуля просто потому, что это
    максимум минус минимум. Сравнивать его с нулём — значит находить различие
    всегда.
    """
    rng = np.random.default_rng(0)
    n = 60000
    labels = np.where(rng.random(n) < 0.5, pt.LABELS.index("up"),
                      pt.LABELS.index("down")).astype(np.int8)
    groups = rng.integers(0, 50, n).astype(float)
    table = pt.by_state(labels, groups, min_rows=200)
    spread = float(table["up"].max() - table["up"].min())
    assert spread > 0.05, "у случайных меток размах уже больше порога критерия"


def test_by_state_drops_thin_groups():
    rng = np.random.default_rng(1)
    labels = np.full(1000, pt.LABELS.index("up"), dtype=np.int8)
    labels[:500] = pt.LABELS.index("down")
    groups = np.concatenate([np.zeros(990), np.ones(10)])
    table = pt.by_state(labels, groups, min_rows=200)
    assert list(table.index) == [0.0]


def test_seasonal_null_keeps_the_share_near_the_anchor():
    """
    Суррогатная нулёвка на случайном блуждании обязана дать около 0.5 — то
    есть не вносить собственного перекоса. Если вносит, все эффекты замера
    будут измеряться относительно смещённой точки.
    """
    base = _walk(n=20000)
    null = pt.surrogate_shares(base, 1.0, 96, 5, np.random.default_rng(3))
    assert null.size == 5
    assert abs(float(np.mean(null)) - 0.5) < 0.02
