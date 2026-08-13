"""
Тесты мультимонетности.

Самый ценный здесь — test_configuration_hash_includes_symbol. Без символа
в хэше btc-graph дедуплицирует кандидатов разных монет между собой (TTL
30 минут) и отдаёт ETH готовую оценку BTC. По данным это не диагностируется:
кандидат выглядит нормально оценённым.

Как и остальные тесты проекта, в БД и в сеть не ходят.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from btcproc import config, symbols
from btcproc.candidates import builder as cand
from btcproc.states import clustering


# ─── Реестр монет ─────────────────────────────────────────────────────────────

def test_unknown_ticker_gives_readable_error():
    """
    Опечатка в CLI не должна выглядеть как «нет данных по монете»: сообщение
    обязано назвать проблему и перечислить известные тикеры.
    """
    with pytest.raises(symbols.UnknownSymbolError) as exc:
        symbols.get("XYZUSDT")

    message = str(exc.value)
    assert "XYZUSDT" in message
    assert "symbols.py" in message
    assert "BTCUSDT" in message


def test_lookup_is_case_insensitive():
    assert symbols.get("btcusdt").ticker == "BTCUSDT"
    assert symbols.get("  BtcUsdt ").ticker == "BTCUSDT"


def test_default_symbol_is_registered():
    """
    Дефолт, которого нет в реестре, — это ссылки в никуда на всех страницах
    админки и падение на середине прогона. Проверяется на старте.
    """
    symbols.validate_default()
    assert symbols.default().ticker == config.data.symbol.upper()


def test_symbol_history_start_overrides_env():
    """
    Дата листинга монеты перекрывает общий HISTORY_START: качать SOL с 2017
    означает полсотни лишних 404 на каждый ingest.
    """
    spec = symbols.SymbolSpec("TESTUSDT", "2021-03-01")
    assert spec.start_date() == "2021-03-01"

    # Пустое поле — откат на общий дефолт из .env.
    fallback = symbols.SymbolSpec("TESTUSDT", "")
    assert fallback.start_date() == config.data.history_start


def test_enabled_filters_out_disabled():
    active = {spec.ticker for spec in symbols.enabled()}
    disabled = {s.ticker for s in symbols.SYMBOLS if not s.enabled}
    assert active.isdisjoint(disabled)
    assert "BTCUSDT" in active


def test_states_overrides_are_applied():
    spec = symbols.SymbolSpec(
        "TESTUSDT", "2021-01-01", states_overrides={"min_group_size": 200}
    )
    cfg = spec.states_config()
    assert cfg.min_group_size == 200
    # Не переопределённое пришло из базового конфига.
    assert cfg.max_depth == config.states.max_depth


def test_unknown_states_override_is_rejected():
    """
    Опечатка в states_overrides иначе всплыла бы TypeError'ом в середине
    прогона — после закачки истории и получаса кластеризации.
    """
    bad = symbols.SymbolSpec(
        "TESTUSDT", "2021-01-01", states_overrides={"min_grup_size": 200}
    )
    with pytest.raises(symbols.UnknownSymbolError, match="min_grup_size"):
        symbols._validate_overrides(bad)


# ─── Разбор аргументов CLI ────────────────────────────────────────────────────

def test_resolve_many_defaults_to_env_symbol():
    """Без флагов поведение прежнее — иначе сломались бы make train и cron."""
    assert [s.ticker for s in symbols.resolve_many(None, False)] == [
        symbols.default().ticker
    ]


def test_resolve_many_deduplicates_preserving_order():
    result = symbols.resolve_many(["ETHUSDT", "BTCUSDT", "ETHUSDT"], False)
    assert [s.ticker for s in result] == ["ETHUSDT", "BTCUSDT"]


def test_resolve_many_all_returns_enabled():
    result = symbols.resolve_many(None, True)
    assert [s.ticker for s in result] == [s.ticker for s in symbols.enabled()]


def test_all_and_symbol_together_are_rejected():
    with pytest.raises(symbols.UnknownSymbolError, match="--all"):
        symbols.resolve_many(["BTCUSDT"], True)


# ─── Хэши кандидата (раздел 6 задачи) ─────────────────────────────────────────

@pytest.fixture
def two_symbol_candidates(pipeline_data):
    """Один и тот же набор снимков, выпущенный под двумя разными тикерами."""
    from tests.test_candidates import TEST_CAND_CFG

    rarity = dict(zip(pipeline_data["transitions"]["transition_id"],
                      pipeline_data["transitions"]["rarity"]))
    blocks = pipeline_data["blocks"].set_index("event_block_id").to_dict("index")

    def produce(symbol: str) -> list[dict]:
        return list(cand.generate(
            pipeline_data["snapshots"], rarity, blocks, symbol, cfg=TEST_CAND_CFG
        ))

    btc, eth = produce("BTCUSDT"), produce("ETHUSDT")
    assert btc and eth, "На синтетике не выпущено ни одного кандидата"
    assert len(btc) == len(eth), "Символ не должен влиять на состав кандидатов"
    return btc, eth


def test_configuration_hash_includes_symbol(two_symbol_candidates):
    """
    Регрессия на главную ловушку задачи.

    btc-graph дедуплицирует по configuration_hash с TTL 30 минут и
    переиспользует готовую оценку. Одинаковый переход на BTC и ETH в пределах
    получаса — и ETH получит оценку BTC. Диагностировать это по данным почти
    невозможно: кандидат выглядит нормально оценённым.
    """
    btc, eth = two_symbol_candidates

    for left, right in zip(btc, eth):
        assert left["configuration_hash"] != right["configuration_hash"], (
            "configuration_hash совпал у BTC и ETH — btc-graph переиспользует "
            "чужую оценку"
        )
        assert left["candidate_family_key"] != right["candidate_family_key"], (
            "candidate_family_key совпал — фильтр btc-graph схлопнет семьи "
            "двух монет, погасив кандидата одной из них"
        )

    # Символ должен быть различающим компонентом, а не случайным совпадением:
    # внутри одной монеты хэши по-прежнему детерминированы.
    assert eth[0]["candidate_family_key"].startswith("ETHUSDT|")


def test_candidate_ids_unique_across_symbols(two_symbol_candidates):
    """
    Прогон по одним и тем же снимкам с разными символами не должен давать
    пересечения candidate_id: иначе в общей таблице btc-graph одна монета
    перезапишет другую.
    """
    btc, eth = two_symbol_candidates
    btc_ids = {c["candidate_id"] for c in btc}
    eth_ids = {c["candidate_id"] for c in eth}

    assert not (btc_ids & eth_ids)
    # И внутри монеты id уникальны — иначе тест выше проходил бы вырожденно.
    assert len(btc_ids) == len(btc)


def test_hashes_are_deterministic_within_symbol(pipeline_data):
    """Дважды выпущенный кандидат обязан иметь те же идентификаторы."""
    from tests.test_candidates import TEST_CAND_CFG

    rarity = dict(zip(pipeline_data["transitions"]["transition_id"],
                      pipeline_data["transitions"]["rarity"]))
    blocks = pipeline_data["blocks"].set_index("event_block_id").to_dict("index")

    first = list(cand.generate(pipeline_data["snapshots"], rarity, blocks,
                               "BTCUSDT", cfg=TEST_CAND_CFG))
    second = list(cand.generate(pipeline_data["snapshots"], rarity, blocks,
                                "BTCUSDT", cfg=TEST_CAND_CFG))

    assert [c["candidate_id"] for c in first] == [c["candidate_id"] for c in second]
    assert [c["configuration_hash"] for c in first] == \
           [c["configuration_hash"] for c in second]


def test_altcoin_candidate_passes_btc_graph_schema(two_symbol_candidates):
    """
    Инвариант 5 из CLAUDE.md для не-BTC монеты: кандидат обязан проходить
    pydantic-модель btc-graph без правок.
    """
    from tests.test_candidates import _load_btc_graph_model

    Candidate = _load_btc_graph_model()
    _, eth = two_symbol_candidates

    for item in eth[:100]:
        parsed = Candidate(**cand.strip_meta(item))
        assert parsed.symbol == "ETHUSDT"
        assert parsed.valid_label_count + parsed.invalid_label_count == parsed.sample_size


def test_altcoin_candidate_is_not_flagged_by_btc_graph(two_symbol_candidates):
    """
    Критерий приёмки 7: кандидат по не-BTC символу принимается btc-graph
    без флага про «чужую монету».

    Флаг unknown_symbol_profile при этом допустим и ожидаем, пока у монеты
    нет своего профиля калибровки, — он предупреждает, а не отвергает.
    """
    from tests.test_candidates import _load_btc_graph_model

    Candidate = _load_btc_graph_model()
    try:
        from src.validator.candidate_validator import validate_candidate
    except ImportError:  # pragma: no cover — старый btc-graph без шага 4
        pytest.skip("btc-graph не отдаёт validate_candidate")

    _, eth = two_symbol_candidates
    flags = validate_candidate(Candidate(**cand.strip_meta(eth[0])))

    assert "symbol_not_btcusdt" not in flags
    assert "symbol_not_allowed" not in flags
    assert "symbol_defaulted" not in flags


# ─── Масштабирование порогов кластеризации (раздел 7) ─────────────────────────

def test_min_group_size_scales_with_history():
    """
    На короткой истории эффективный порог заметно ниже, чем на длинной, но
    не ниже абсолютного пола. Без этого граф альткоина выходит грубее
    не из-за рынка, а из-за чужой линейки.
    """
    cfg = config.StatesConfig()      # боевые значения: пол 300, доля 0.0025

    short = clustering.effective_min_group_size(cfg, 70_000)     # ~2 года 15m
    long = clustering.effective_min_group_size(cfg, 300_000)     # BTC с 2017

    assert short < long, (
        "порог не масштабируется: относительная доля перекрыта абсолютным полом "
        "на всём диапазоне реальных историй"
    )
    assert short >= cfg.min_group_size, "абсолютный пол обязан соблюдаться"
    # На совсем коротком ряде работает именно пол, а не доля.
    assert clustering.effective_min_group_size(cfg, 1_000) == cfg.min_group_size


def test_absolute_floor_does_not_swallow_the_share():
    """
    Регрессия на смысл раздела 7: пол не должен быть настолько высоким, чтобы
    max() всегда возвращал его. Раньше пол стоял на 800, а доля давала 750 на
    истории BTC — то есть относительный порог был формальностью и не работал
    ни для одной реальной монеты.
    """
    cfg = config.StatesConfig()
    driven_by_share = clustering.effective_min_group_size(cfg, 300_000)
    assert driven_by_share > cfg.min_group_size


def test_min_group_share_matches_btc_calibration():
    """
    Доля подобрана так, чтобы на истории BTC дать примерно те же ~800,
    на которых калибровался граф: переход на относительный порог не должен
    тихо переехать эталон.
    """
    cfg = config.StatesConfig()
    at_btc_scale = clustering.effective_min_group_size(cfg, 320_000)
    assert 700 <= at_btc_scale <= 900


def test_fit_states_uses_scaled_threshold(monkeypatch):
    """Эффективный порог должен попадать в params модели — иначе прогон не разобрать."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 4))
    cfg = config.StatesConfig(
        seed_clusters=2, min_group_size=100, min_group_share=0.1,
        max_depth=1, silhouette_sample=500,
    )

    model, _ = clustering.fit_states(x, ["a", "b", "c", "d"], {"center": np.zeros(4)}, cfg)

    # 0.1 × 3000 = 300 > абсолютных 100.
    assert model.params["min_group_size"] == 300
    assert model.params["min_group_share"] == 0.1


# ─── live берёт модель своей монеты (раздел 5) ────────────────────────────────

def test_live_picks_model_of_its_symbol(monkeypatch):
    """
    run_live для монеты обязан искать модель среди прогонов ЭТОЙ монеты,
    а не последнего train вообще. Иначе бары ETH размечаются состояниями BTC,
    и обнаруживается это только по бессмысленным кандидатам.
    """
    from btcproc.db import runs as runs_repo
    from btcproc.pipeline import live

    calls: list[tuple] = []

    def fake_latest(kind="train", symbol=None):
        calls.append((kind, symbol))
        return None      # дальше run_live упадёт с понятным сообщением

    monkeypatch.setattr(runs_repo, "latest_completed_run", fake_latest)

    with pytest.raises(RuntimeError, match="ETHUSDT"):
        live.run_live(symbol="ETHUSDT")

    assert calls == [("train", "ETHUSDT")]


def test_live_rejects_model_of_another_symbol(monkeypatch):
    """
    Явно переданный --model-run чужой монеты отвергается: центроиды обучены
    в пространстве признаков конкретного инструмента.
    """
    from btcproc.db import runs as runs_repo
    from btcproc.pipeline import live

    monkeypatch.setattr(
        runs_repo, "get_run",
        lambda run_id: {"run_id": run_id, "symbol": "BTCUSDT", "kind": "train"},
    )

    with pytest.raises(RuntimeError, match="несопоставимы"):
        live.run_live(symbol="ETHUSDT", model_run_id=42)
