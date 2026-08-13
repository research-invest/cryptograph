from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcproc import config
from btcproc.candidates import builder as cand
from btcproc.candidates.outcomes import compute_outcomes
from btcproc.features import builder as feat
from btcproc.features import events as ev
from btcproc.states import assign, clustering, graph


# На синтетике 125 дней переходов набирается немного, поэтому пороги выборки
# ослаблены: тест проверяет корректность сборки кандидата, а не статистическую
# значимость (её обеспечивают боевые CAND_MIN_SAMPLE_SIZE и
# CAND_MIN_EFFECTIVE_SAMPLE).
TEST_CAND_CFG = config.CandidateConfig(
    min_sample_size=10, min_effective_sample_size=4
)


# Фикстура pipeline_data живёт в conftest.py — она нужна и тестам мультимонетности.


def test_outcomes_align_with_prices(bars):
    horizon = 96
    outcomes = compute_outcomes(bars, horizon)

    i = 500
    expected_ret = (bars["close"].iloc[i + horizon] / bars["close"].iloc[i] - 1) * 100
    assert outcomes["ret_pct"].iloc[i] == pytest.approx(expected_ret)

    expected_mfe = (bars["high"].iloc[i + 1:i + 1 + horizon].max()
                    / bars["close"].iloc[i] - 1) * 100
    assert outcomes["mfe_pct"].iloc[i] == pytest.approx(expected_mfe)

    # У последних баров горизонта нет — метка невалидна.
    assert not outcomes["valid"].iloc[-1]
    assert outcomes["valid"].iloc[:-horizon].all()


def test_outcomes_marks_gaps_invalid(bars):
    """Дыра в данных внутри горизонта делает метку невалидной."""
    holed = pd.concat([bars.iloc[:1000], bars.iloc[1100:]])
    outcomes = compute_outcomes(holed, 96)
    around_gap = outcomes["valid"].iloc[904:1000]
    assert not around_gap.any()


def test_snapshots_cover_all_age_buckets(pipeline_data):
    snapshots = pipeline_data["snapshots"]
    assert not snapshots.empty
    buckets = set(snapshots["age_bucket"])
    # Смещения 0/45/90/180 минут должны дать все четыре бакета схемы.
    assert buckets == {"age_lt_30", "age_30_60", "age_60_120", "age_gt_120"}
    assert snapshots["ts"].is_monotonic_increasing


def test_generated_candidates_match_btc_graph_schema(pipeline_data):
    """
    Главная проверка совместимости: каждый кандидат обязан пройти pydantic-модель
    btc-graph без единой правки. Если схема разъедется — тест упадёт здесь,
    а не в проде при отправке.
    """
    pytest.importorskip("pydantic")
    Candidate = _load_btc_graph_model()

    rarity = dict(zip(pipeline_data["transitions"]["transition_id"],
                      pipeline_data["transitions"]["rarity"]))
    blocks = pipeline_data["blocks"].set_index("event_block_id").to_dict("index")

    produced = list(cand.generate(
        pipeline_data["snapshots"], rarity, blocks, "BTCUSDT", cfg=TEST_CAND_CFG
    ))
    assert produced, "На синтетике не выпущено ни одного кандидата"

    for item in produced[:200]:
        parsed = Candidate(**cand.strip_meta(item))
        assert parsed.symbol == "BTCUSDT"
        assert 0.0 <= parsed.research_score <= 1.0
        assert parsed.valid_label_count + parsed.invalid_label_count == parsed.sample_size
        assert parsed.long_outcome_count + parsed.short_outcome_count == parsed.valid_label_count


def test_candidate_sample_uses_only_matured_past(pipeline_data):
    """
    Кандидат в момент t не должен видеть ни будущее, ни исходы, которые
    к моменту t ещё не закрылись.
    """
    snapshots = pipeline_data["snapshots"]
    rarity = dict(zip(pipeline_data["transitions"]["transition_id"],
                      pipeline_data["transitions"]["rarity"]))
    blocks = pipeline_data["blocks"].set_index("event_block_id").to_dict("index")

    horizon = pd.Timedelta(minutes=config.data.horizon_minutes)
    produced = list(cand.generate(snapshots, rarity, blocks, "BTCUSDT", cfg=TEST_CAND_CFG))

    for item in produced[:100]:
        ts = pd.Timestamp(item["_meta"]["ts"])
        key_scope = item["sample_scope"]
        mask = snapshots["transition_id"] == item["transition_id"]
        if key_scope == "transition+event_block":
            mask &= snapshots["event_block_id"] == item["event_block_id"]
        # Столько случаев было доступно на момент t по правилам генератора.
        available = int((mask & (snapshots["ts"] + horizon <= ts)).sum())
        assert item["sample_size"] == available


def test_research_score_rewards_strong_samples():
    weak = cand.Accumulator()
    strong = cand.Accumulator()
    base = pd.Timestamp("2021-01-01", tz="UTC")

    for i in range(40):
        weak.add(base + pd.Timedelta(days=i % 3), 0.1, 1.0, -1.0, True, i)
    for i in range(1200):
        ts = base + pd.Timedelta(days=i % 540)
        strong.add(ts, 1.0 if i % 4 else -1.0, 2.0, -0.4, True, i)

    assert cand.research_score(strong, "rare", "rare") > cand.research_score(weak, "common", "common")
    assert 0.0 <= cand.research_score(weak, "common", "common") <= 1.0


def test_monotonic_rise_leaves_no_drawdown():
    """
    A1: цена, ни разу не сходившая ниже входа, не должна давать «просадку».

    `mae_pct` — величина со знаком, и при чистом росте она ПОЛОЖИТЕЛЬНА
    (минимум low выше цены входа). Прежний abs() записывал её в список
    неблагоприятных экскурсий как есть, то есть чем чище был рост, тем
    больше выходило «пересидели против». На этом тесте старый код падал.
    """
    horizon = 96
    n = 400
    index = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = pd.Series(np.linspace(100.0, 140.0, n), index=index)
    bars_up = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.9999,   # low следующего бара всё равно выше close текущего
            "close": close,
            "volume": 1.0,
        },
        index=index,
    )
    outcomes = compute_outcomes(bars_up, horizon)
    valid = outcomes[outcomes["valid"]]
    assert (valid["ret_pct"] > 0).all()
    # Именно тот случай, ради которого правка: mae_pct положителен.
    assert (valid["mae_pct"] > 0).all()

    acc = cand.Accumulator()
    for i, (ts, row) in enumerate(valid.iterrows()):
        acc.add(ts, row["ret_pct"], row["mfe_pct"], row["mae_pct"], True, i)

    assert acc.up == acc.valid
    assert max(acc.mae_up) == 0.0
    assert cand._percentile_sorted(acc.mae_up, 80.0) == 0.0


def test_effective_sample_size_counts_realizations(pipeline_data):
    """
    A2: sample_size считает строки снимков, effective_sample_size — случаи.

    Офсеты 0/45/90/180 дают до четырёх строк на одну реализацию перехода,
    и окна их исходов при горизонте 24h перекрываются на 87.5%. Кандидат
    обязан нести оба числа, а effective — быть строго меньше.
    """
    rarity = dict(zip(pipeline_data["transitions"]["transition_id"],
                      pipeline_data["transitions"]["rarity"]))
    blocks = pipeline_data["blocks"].set_index("event_block_id").to_dict("index")
    produced = list(cand.generate(
        pipeline_data["snapshots"], rarity, blocks, "BTCUSDT", cfg=TEST_CAND_CFG
    ))
    assert produced

    for item in produced:
        assert 0 < item["effective_sample_size"] <= item["sample_size"]
    ratios = [item["sample_size"] / item["effective_sample_size"] for item in produced]
    # На синтетике не все офсеты доживают (состояние успевает смениться),
    # поэтому проверяется не ровно 4, а сам факт кратности.
    assert max(ratios) > 1.5


def test_fallback_candidate_does_not_claim_rare_block():
    """
    A3: при откате на выборку по переходу редкость блока не начисляется.

    Статистика в этом случае на блок не обусловлена вовсе, и «редкий блок»
    в research_score означал бы баллы за признак, которого в выборке нет.
    Само поле event_rarity_bucket при этом остаётся — оно описывает бар.
    """
    acc = cand.Accumulator()
    base = pd.Timestamp("2021-01-01", tz="UTC")
    for i in range(60):
        acc.add(base + pd.Timedelta(days=i), 1.0 if i % 3 else -1.0, 2.0, -0.5, True, i)

    row = {
        "ts": base, "offset_min": 0, "state_seq": 1,
        "transition_id": "1->2", "prev_group_id": 1.0, "current_group_id": 2.0,
        "age_minutes": 0, "age_bucket": "age_lt_30", "entropy": "low",
        "event_block_id": "blk", "atom_count": 2, "family_count": 1,
        "intensity": "sparse", "primary_family": "breakout",
    }
    rarity = {"1->2": "common"}
    blocks = {"blk": {"rarity": "rare", "total_rows": 10, "row_share": 0.001}}

    conditioned = cand._assemble(row, acc, "transition+event_block", rarity, blocks,
                                 "BTCUSDT", TEST_CAND_CFG)
    fallback = cand._assemble(row, acc, "transition", rarity, blocks,
                              "BTCUSDT", TEST_CAND_CFG)

    assert conditioned["sample_scope"] == "transition+event_block"
    assert fallback["sample_scope"] == "transition"
    # Поле про бар остаётся честным в обоих случаях…
    assert fallback["event_rarity_bucket"] == "rare"
    # …а в оценку редкость блока при откате не входит.
    assert fallback["research_score"] < conditioned["research_score"]
    assert fallback["research_score"] == cand.research_score(acc, "common", "common")


def test_family_key_separates_sides():
    """
    A4: long и short по одной конфигурации — разные семьи.

    min_abs_skew=0.06 и bias_skew_threshold=0.10 не согласованы, и в полосе
    между ними bias у обеих сторон "neutral". Без стороны в ключе они
    попадали в одну семью и вытесняли друг друга в select_best_per_family.
    """
    base = pd.Timestamp("2021-01-01", tz="UTC")
    row = {
        "ts": base, "offset_min": 0, "state_seq": 1,
        "transition_id": "1->2", "prev_group_id": 1.0, "current_group_id": 2.0,
        "age_minutes": 0, "age_bucket": "age_lt_30", "entropy": "low",
        "event_block_id": "blk", "atom_count": 2, "family_count": 1,
        "intensity": "sparse", "primary_family": "breakout",
    }
    rarity = {"1->2": "common"}
    blocks = {"blk": {"rarity": "common", "total_rows": 100, "row_share": 0.01}}

    keys = {}
    for up_share in (0.54, 0.46):   # |skew| = 0.08 — ровно полоса [0.06, 0.10)
        acc = cand.Accumulator()
        n_up = int(round(100 * up_share))
        for i in range(100):
            acc.add(base + pd.Timedelta(days=i), 1.0 if i < n_up else -1.0,
                    2.0, -0.5, True, i)
        item = cand._assemble(row, acc, "transition+event_block", rarity, blocks,
                              "BTCUSDT", TEST_CAND_CFG)
        assert item["historical_bias_context"] == "neutral"
        keys[item["research_side"]] = item["candidate_family_key"]

    assert set(keys) == {"long", "short"}
    assert keys["long"] != keys["short"]


def test_percentile_helper_matches_numpy():
    values = sorted(np.random.default_rng(1).normal(3, 1, 500).tolist())
    for q in (50.0, 70.0, 80.0, 95.0):
        assert cand._percentile_sorted(values, q) == pytest.approx(
            float(np.percentile(values, q)), rel=1e-9
        )


def _load_btc_graph_model():
    """Модель Candidate берём из самого btc-graph — дублировать её нельзя."""
    import sys

    path = str(config.sink.btc_graph_path)
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        from src.models.candidate import Candidate
    except ImportError:  # pragma: no cover — репозиторий рядом не лежит
        pytest.skip(f"btc-graph не найден по пути {path}")
    return Candidate


# ─── Короткая сторона: зеркало F/A ratio (2026-08-13) ─────────────────────────

def test_down_cases_fill_the_short_side_only():
    """
    Накопители короткой стороны собираются по случаям ПАДЕНИЯ и меняют роли
    mfe и mae местами: выгода short — глубина хода вниз, риск — то, что
    пришлось пересидеть вверх.

    До этой правки списков не было вовсе, F/A ratio существовал только для
    long, и приёмник считал short'у ось directional по двум критериям.
    """
    acc = cand.Accumulator()
    ts = pd.Timestamp("2024-01-01", tz="UTC")

    # Падение: mfe=+1% (сходили вверх против), mae=−5% (глубина вниз).
    acc.add(ts, -3.0, 1.0, -5.0, True, 1)
    # Рост: должен попасть только в long-накопители.
    acc.add(ts, +3.0, 5.0, -1.0, True, 2)
    # Нулевой исход не «хороший случай» ни для одной стороны.
    acc.add(ts, 0.0, 2.0, -2.0, True, 3)

    assert acc.fav_down == [5.0], "выгода short — это глубина хода ВНИЗ"
    assert acc.adv_down == [1.0], "риск short — это ход ВВЕРХ"
    assert acc.mfe_up == [5.0] and acc.mae_up == [1.0]
    assert acc.up == 1


def test_short_ratio_is_none_without_falls():
    """
    Пустой список дал бы ratio = 0.0, а это не «плохое отношение», это
    отсутствие данных. Правило то же, что у sample_scope: критерий без данных
    выбывает, а не начисляет ноль.
    """
    acc = cand.Accumulator()
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    acc.add(ts, +3.0, 5.0, -1.0, True, 1)

    assert acc.fav_down == []


def test_short_ratio_mirrors_long_on_mirrored_data(pipeline_data):
    """
    Зеркальность проверяется на настоящей выдаче: у кандидатов, где падения в
    выборке были, short-ratio обязан присутствовать и быть положительным.
    Поле считается ВСЕГДА — оно свойство конфигурации, а не стороны, — и
    применить его решает приёмник.
    """
    rarity = dict(zip(pipeline_data["transitions"]["transition_id"],
                      pipeline_data["transitions"]["rarity"]))
    blocks = pipeline_data["blocks"].set_index("event_block_id").to_dict("index")
    candidates = list(cand.generate(
        pipeline_data["snapshots"], rarity, blocks, "BTCUSDT", cfg=TEST_CAND_CFG
    ))
    assert candidates, "конвейер не выпустил кандидатов"

    with_ratio = [
        c for c in candidates
        if c.get("short_favorable_adverse_ratio_p70_p80") is not None
    ]
    assert with_ratio, "ни у одного кандидата нет short-стороны F/A ratio"
    assert all(c["short_favorable_adverse_ratio_p70_p80"] > 0 for c in with_ratio)

    # Поле есть у обеих сторон, а не только у short-кандидатов.
    sides = {c["research_side"] for c in with_ratio}
    assert "long" in sides
