"""
Сколько кандидатов даёт монета при разных `min_group_size`.

Нужен, когда у монеты мало кандидатов и есть подозрение, что порог дробления
стоит на абсолютном полу. Порог считается как
`max(min_group_size, round(min_group_share * число баров))` — на короткой
истории относительная часть даёт сотню-другую, монета упирается в пол 300,
граф дробится на состояния по 700–900 баров, и переходы перестают добирать
до `CAND_MIN_EFFECTIVE_SAMPLE=30` независимых реализаций. Симптом — мало
кандидатов и ноль `scope=transition+event_block`; лечится
`states_overrides={"min_group_size": N}` в `btcproc/symbols.py`, как уже
сделано у TAOUSDT.

Скрипт **ничего не пишет** — ни в БД, ни в Neo4j: он повторяет шаги train
(признаки → события → исходы → кластеризация → граф → кандидаты) в памяти.
Именно поэтому им можно мерить на боевой копии, не трогая модель монеты.

    cd btc-graph-processing
    PYTHONPATH=. ./.venv/bin/python \
        ../.claude/skills/system-sanity-check/scripts/probe_min_group_size.py \
        PUMPUSDT 300,600,900,1200

ВАЖНО: одна колонка — один прогон, а кластеризация к данным чувствительна
(см. `probe_states_stability.py`). Выбирать порог по одной таблице нельзя —
повторить на двух-трёх срезах истории и брать то, что держится на всех.
"""
from __future__ import annotations

import dataclasses
import sys

from btcproc import config, symbols
from btcproc.candidates import builder as cand
from btcproc.candidates.outcomes import compute_outcomes
from btcproc.features import builder as feat
from btcproc.features import events as ev
from btcproc.ingest import bars
from btcproc.states import assign, clustering, graph


def main(symbol: str, grid: list[int]) -> None:
    spec = symbols.get(symbol)
    start = spec.start_date()

    base = bars.load_ohlcv(symbol, config.data.base_tf, start, None)
    context = {tf: bars.load_ohlcv(symbol, tf, start, None)
               for tf in config.data.context_tfs}
    features = feat.build_features(base, context, symbol=symbol)
    events = (ev.build_event_blocks(base, symbol=symbol)
              .reindex(features.index).dropna(subset=["event_block_id"]))
    blocks = ev.block_statistics(events)
    outcomes = compute_outcomes(base).reindex(features.index)
    scale = feat.robust_scale_params(features)
    matrix = feat.apply_scale(features, scale)

    print(f"{symbol}: {len(base)} баров, {features.shape[1]} признаков, "
          f"{len(blocks)} блоков событий, порог сейчас "
          f"{spec.states_config().min_group_size}")
    print(f"{'mgs':>6} {'состояний':>10} {'ср.размер':>10} {'рёбер':>7} "
          f"{'снимков':>9} {'кандидатов':>11} {'+event_block':>13} {'на 1000 баров':>14}")

    block_meta = blocks.set_index("event_block_id").to_dict("index")
    for mgs in grid:
        cfg = dataclasses.replace(spec.states_config(), min_group_size=mgs)
        model, raw_labels = clustering.fit_states(
            matrix, list(features.columns), scale, cfg=cfg)
        states = assign.assign_states(features.index, raw_labels)
        transitions = graph.transition_stats(states, outcomes)
        snapshots = cand.build_snapshots(states, events, outcomes)
        rarity = dict(zip(transitions["transition_id"], transitions["rarity"]))

        produced = full = 0
        for candidate in cand.generate(snapshots, rarity, block_meta, symbol=symbol):
            produced += 1
            full += candidate.get("sample_scope") == "transition+event_block"

        avg_size = int(states.groupby("group_id").size().mean())
        print(f"{mgs:6d} {model.n_groups:10d} {avg_size:10d} {len(transitions):7d} "
              f"{len(snapshots):9d} {produced:11d} {full:13d} "
              f"{1000 * produced / len(base):14.1f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1].upper(), [int(x) for x in sys.argv[2].split(",")])
