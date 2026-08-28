"""
Насколько число состояний зависит от нескольких лишних баров.

Кластеризация детерминирована (`StatesConfig.random_state = 42`), поэтому
разброс между прогонами — это чувствительность к данным, а не случайность.
Скрипт обрезает хвост истории на N баров и переобучает модель, чтобы разброс
был виден числом, а не ощущением.

Замер 2026-08-28 (локальная копия): у TAOUSDT ОДИН лишний бар менял число
состояний с 25 на 43; по срезам 0/1/10/100/500/1000 вышло 25/43/44/38/50/45.
У BTCUSDT на тех же срезах 58/57/52/43, у AAVEUSDT 23/32/32/33, у PUMPUSDT
41/33/34/36/33. То есть чувствительность общая (инвариант 10), но у монет с
короткой историей запаса данных нет, и разброс сразу переносится в число
кандидатов: прод-прогоны TAOUSDT #42 и #43 отличались одним баром и дали
37 против 44 состояний и 4 987 против 4 008 кандидатов.

Практический вывод, ради которого скрипт и лежит здесь: **«40 состояний
против 46» диагнозом не является**, и подбирать `min_group_size` по одному
прогону нельзя.

Ничего не пишет.

    cd btc-graph-processing
    PYTHONPATH=. ./.venv/bin/python \
        ../.claude/skills/system-sanity-check/scripts/probe_states_stability.py \
        TAOUSDT 0,1,10,100,500
"""
from __future__ import annotations

import sys

from btcproc import config, symbols
from btcproc.features import builder as feat
from btcproc.ingest import bars
from btcproc.states import clustering


def main(symbol: str, trims: list[int]) -> None:
    spec = symbols.get(symbol)
    start = spec.start_date()
    base_full = bars.load_ohlcv(symbol, config.data.base_tf, start, None)
    context_full = {tf: bars.load_ohlcv(symbol, tf, start, None)
                    for tf in config.data.context_tfs}

    print(f"{symbol} min_group_size={spec.states_config().min_group_size}, "
          f"всего баров {len(base_full)}")
    print(f"{'обрезано':>9} {'баров':>8} {'состояний':>10}")
    for trim in trims:
        base = base_full.iloc[: len(base_full) - trim] if trim else base_full
        # Контекстные таймфреймы обрезаются по той же границе: иначе модель
        # увидит через 1h/4h/1d будущее «обрезанного» хвоста.
        context = {tf: df[df.index <= base.index[-1]]
                   for tf, df in context_full.items()}
        features = feat.build_features(base, context, symbol=symbol)
        scale = feat.robust_scale_params(features)
        matrix = feat.apply_scale(features, scale)
        model, _ = clustering.fit_states(
            matrix, list(features.columns), scale, cfg=spec.states_config())
        print(f"{trim:9d} {len(base):8d} {model.n_groups:10d}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1].upper(), [int(x) for x in sys.argv[2].split(",")])
