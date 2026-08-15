"""
Общие фикстуры. Тесты не ходят ни в БД, ни в сеть: всё считается на
синтетических барах, у которых заранее известна структура.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_bars(
    n: int = 12000,
    start: str = "2020-01-01",
    freq: str = "15min",
    seed: int = 7,
    regimes: bool = True,
) -> pd.DataFrame:
    """
    Синтетические свечи.

    regimes=True добавляет чередующиеся режимы (тихий рост / волатильное
    падение / боковик) — тогда у кластеризации есть что находить, и тест на
    «граф не выродился в один узел» имеет смысл.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n, freq=freq, tz="UTC")

    if regimes:
        block = max(n // 12, 50)
        drift, vol = [], []
        for i in range(0, n, block):
            phase = (i // block) % 3
            drift.append(np.full(min(block, n - i), [0.0004, -0.0006, 0.0][phase]))
            vol.append(np.full(min(block, n - i), [0.002, 0.006, 0.001][phase]))
        drift = np.concatenate(drift)[:n]
        vol = np.concatenate(vol)[:n]
    else:
        drift = np.zeros(n)
        vol = np.full(n, 0.003)

    returns = rng.normal(drift, vol)
    close = 10000 * np.exp(np.cumsum(returns))
    spread = close * vol

    high = close + np.abs(rng.normal(0, spread))
    low = close - np.abs(rng.normal(0, spread))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = np.abs(rng.normal(100, 30, n)) * (1 + 5 * vol)

    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum.reduce([high, open_, close]),
            "low": np.minimum.reduce([low, open_, close]),
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "trades": rng.integers(50, 500, n),
            "taker_buy_base": volume * rng.uniform(0.35, 0.65, n),
        },
        index=index,
    )


@pytest.fixture(scope="session")
def bars() -> pd.DataFrame:
    return make_bars()


@pytest.fixture(scope="session")
def context(bars) -> dict:
    from btcproc.ingest.bars import resample

    return {tf: resample(bars, tf) for tf in ("1h", "4h", "1d")}


@pytest.fixture(scope="session")
def features(bars, context) -> pd.DataFrame:
    """
    Базовый набор признаков — БЕЗ внешних источников.

    Источники гасятся явно, а не «по умолчанию»: их флаги читаются из
    окружения, и на боевом VPS умолчание означает «включён». Признаки FGI и
    деривативов при этом джойнятся из БД, куда тесты не ходят, — набор
    вышел бы пустым. Раньше это ничем не проявлялось (пустой DataFrame
    сохраняет колонки), а с предохранителем на dropna (аудит 2026-08-15, B4)
    стало честной ошибкой.

    Тестам, которым нужен источник, фикстура не подходит по построению: они
    заводят свою и сами кладут синтетические внешние данные — образцы в
    `test_naming.py::all_features` и `test_smc.py`.
    """
    from _pytest.monkeypatch import MonkeyPatch

    from btcproc.features.builder import build_features

    patch = MonkeyPatch()
    try:
        disable_all_sources(patch)
        return build_features(bars, context)
    finally:
        patch.undo()


# На синтетике 125 дней переходов набирается немного, поэтому порог выборки
# ослаблен: тесты проверяют корректность сборки кандидата, а не статистическую
# значимость (её обеспечивает боевой CAND_MIN_SAMPLE_SIZE).
def test_candidate_config():
    from btcproc import config

    return config.CandidateConfig(min_sample_size=10)


@pytest.fixture(scope="session")
def pipeline_data(bars, context, features) -> dict:
    """
    Прогон всей цепочки на синтетике — база для проверок кандидатов.

    Живёт в conftest, а не в одном файле тестов: цепочка нужна и проверкам
    сборки кандидата, и проверкам мультимонетности, а считается она секунды.

    Признаки берутся ГОТОВОЙ фикстурой, а не пересобираются здесь: набор
    должен быть тем же самым, и гасить источники надо в одном месте, а не в
    двух (см. докстринг фикстуры `features`).
    """
    from btcproc import config
    from btcproc.candidates import builder as cand
    from btcproc.candidates.outcomes import compute_outcomes
    from btcproc.features import builder as feat
    from btcproc.features import events as ev
    from btcproc.states import assign, clustering, graph

    scale = feat.robust_scale_params(features)
    matrix = feat.apply_scale(features, scale)
    # min_group_share=0 — на синтетике порог задаём абсолютным числом, чтобы
    # результат не зависел от длины сгенерированного ряда.
    cfg = config.StatesConfig(
        seed_clusters=4, min_group_size=400, min_group_share=0.0, max_depth=2
    )
    model, labels = clustering.fit_states(matrix, list(features.columns), scale, cfg)

    states = assign.assign_states(features.index, labels)
    events = ev.build_event_blocks(bars).reindex(features.index).dropna(
        subset=["event_block_id"]
    )
    blocks = ev.block_statistics(events)
    outcomes = compute_outcomes(bars).reindex(features.index)
    transitions = graph.transition_stats(states, outcomes)

    snapshots = cand.build_snapshots(states, events, outcomes)
    return {
        "features": features, "states": states, "events": events, "blocks": blocks,
        "outcomes": outcomes, "transitions": transitions, "snapshots": snapshots,
    }


@pytest.fixture(scope="session")
def states_and_features(pipeline_data) -> tuple:
    """Разметка, признаки и исходы одного прогона — для проверок графа и имён."""
    return (
        pipeline_data["states"],
        pipeline_data["features"],
        pipeline_data["outcomes"],
    )


@pytest.fixture(scope="module")
def monkeypatch_module():
    """
    monkeypatch на весь модуль.

    Штатный `monkeypatch` пофункционный, а сборка признаков со включённым SMC
    достаточно дорога, чтобы делать её один раз на файл. Отмена в конце
    обязательна: `config.smc` — глобальный объект, и утёкшая подмена включила
    бы SMC остальным тестам сессии, причём молча.
    """
    from _pytest.monkeypatch import MonkeyPatch

    patcher = MonkeyPatch()
    yield patcher
    patcher.undo()


def disable_all_sources(monkeypatch) -> None:
    """
    Гасит ВСЕ внешние источники сигнала — и атомы, и признаки.

    Тесты, проверяющие метку набора (`feature_version` / `event_version`),
    обязаны фиксировать состав источников целиком: метка собирается из
    состава, а флаги читаются из окружения, поэтому «выключим чужие руками»
    зеленело локально и краснело на VPS каждый раз, когда там включали
    очередной источник (FGI — дважды, deriv — ждал того же, аудит
    2026-08-15, B1).

    Список берётся из реестра, а не перечисляется здесь: четвёртый источник
    попадёт под нейтрализацию сам, без правки тестов. Раздел конфига ищется
    по имени источника — если новый источник назовёт их по-разному, тест
    упадёт с диагнозом, а не пропустит его молча.
    """
    import dataclasses

    from btcproc import config
    from btcproc.features import registry

    for source in registry.sources():
        section = getattr(config, source.name, None)
        assert section is not None, (
            f"у источника {source.name!r} нет одноимённого раздела в config — "
            "нейтрализовать его в тестах нечем"
        )
        monkeypatch.setattr(
            config, source.name,
            dataclasses.replace(section, enabled=False, features_enabled=False),
        )


@pytest.fixture
def sources_off(monkeypatch):
    """Фикстурная форма `disable_all_sources` — для тестов без monkeypatch в теле."""
    disable_all_sources(monkeypatch)
