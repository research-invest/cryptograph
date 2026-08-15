"""
Гейт G для деривативных метрик: монотонность по квантильным бинам на
`range_ratio` — непрерывной цели на выборке ВСЕХ баров, а не бинарной на
кандидатах (docs/tz_deriv_ingest_14-08-26.md §2.2 — обобщение
`analysis/gradation.py` под непрерывную метрику уже сделано и покрыто
позитивным контролем в tests/test_gradation.py).

Ничего не пишет — по образцу measure_fgi_range.py / measure_deriv_range.py.
`range_ratio` считается автономно из баров и ATR14, без правки outcomes.py.

Блок бутстрапа — ОТДЕЛЬНО для каждой из 6 величин (их автокорреляция
разнится на порядки, scripts/deriv_frequencies.py --section autocorr).
Поправка на множественные сравнения — Бонферрони по 6 тестам (после
блочного бутстрапа каждого).

    python3 scripts/measure_deriv_gradation.py --symbol BTCUSDT
    python3 scripts/measure_deriv_gradation.py --all --horizon 24h
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from btcproc import config, symbols  # noqa: E402
from btcproc.analysis.gradation import DEFAULT_BINS, measure_gradation  # noqa: E402
from btcproc.analysis.lift import benjamini_hochberg, bonferroni  # noqa: E402
from btcproc.analysis.range_lift import autocorr_block_rows, forward_range_ratio  # noqa: E402
from btcproc.features import deriv  # noqa: E402
from btcproc.features import indicators as ind  # noqa: E402
from btcproc.ingest import bars, metrics as metrics_ingest  # noqa: E402

HORIZON_BARS = {"24h": 96, "8h": 32, "4h": 16}
AUTOCORR_MAX_LAG_BARS = 5000


def build_frame(symbol: str, horizon_label: str) -> tuple[pd.DataFrame, dict[str, int]]:
    base = bars.load_ohlcv(symbol, config.data.base_tf, None, None)
    if base.empty:
        raise SystemExit(f"Нет баров по {symbol} — сначала ingest.")
    metrics_frame = metrics_ingest.load_deriv_metrics(symbol, config.data.base_tf)
    if metrics_frame.empty:
        raise SystemExit(f"deriv_metrics пуст по {symbol} — сначала ingest-metrics.")

    values = deriv.build_deriv(base, metrics_frame, symbol)
    atr14 = ind.atr(base, 14)
    h_bars = HORIZON_BARS[horizon_label]
    range_ratio = forward_range_ratio(base, atr14, h_bars)

    frame = pd.DataFrame(index=base.index.rename(None))
    frame["ts"] = base.index
    frame["metric"] = range_ratio.to_numpy()
    for name in deriv.FEATURE_CANDIDATES:
        frame[name] = values[name].to_numpy()

    block_rows = {}
    for name in deriv.FEATURE_CANDIDATES:
        series = frame[name].dropna()
        if len(series) < 200:
            block_rows[name] = 96
            continue
        _, rows = autocorr_block_rows(series, bars_per_day=1, floor=0.2,
                                      max_lag_days=AUTOCORR_MAX_LAG_BARS)
        block_rows[name] = rows
    return frame, block_rows


def run_one(symbol: str, horizon_label: str, args) -> list:
    frame, block_rows = build_frame(symbol, horizon_label)
    horizon_minutes = HORIZON_BARS[horizon_label] * config.data.base_minutes

    per_feature = []
    for name in deriv.FEATURE_CANDIDATES:
        res = measure_gradation(
            frame, features=[name], metric_column="metric", ts_column="ts",
            bins=args.bins, correction="none", holdout=args.holdout or None,
            horizon_minutes=horizon_minutes, n_boot=args.n_boot,
            min_block_rows=block_rows.get(name),
        )
        if res:
            per_feature.append(res[0])

    testable = [r for r in per_feature if r.bins]
    if testable:
        p_values = [r.effective_p for r in testable]
        flags = (bonferroni(p_values, args.alpha) if args.correction == "bonferroni"
                else benjamini_hochberg(p_values, args.alpha))
        for r, flag in zip(testable, flags):
            r.significant = flag

    print(f"\n{symbol} ({horizon_label}): {len(frame)} баров, "
          f"базовое среднее range_ratio={frame['metric'].mean():.4f}")
    header = (f"{'признак':<16} {'блок':>6} {'разброс':>10} {'z':>7} "
              f"{'p блочн.':>10} {'монот.':>7} {'значим.':>8} {'holdout':>8} {'итог':>5}")
    print(header)
    print("─" * len(header))
    for r in per_feature:
        if not r.bins:
            print(f"{r.feature:<16} {r.degenerate}")
            continue
        holdout_ok = "" if r.holdout is None else ("да" if (
            r.holdout.spread != 0 and (r.spread > 0) == (r.holdout.spread > 0)
        ) else "нет")
        print(f"{r.feature:<16} {block_rows.get(r.feature, 0):>6} {r.spread:>+10.4f} "
              f"{r.z:>7.2f} {r.effective_p:>10.5f} "
              f"{'да' if r.monotone else 'нет':>7} {'да' if r.significant else 'нет':>8} "
              f"{holdout_ok:>8} {'ДА' if r.confirmed else '—':>5}")

    confirmed = [r.feature for r in per_feature if r.confirmed]
    print(f"\nГейт G пройден: {', '.join(confirmed) if confirmed else 'ни одной величиной'}")
    return per_feature


def main() -> int:
    parser = argparse.ArgumentParser(description="Гейт G — градация деривативных метрик по range_ratio")
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--horizon", choices=list(HORIZON_BARS), default="24h")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--correction", choices=["bonferroni", "bh"], default="bonferroni")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--holdout", type=float, default=0.3)
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args()

    if args.all:
        targets = symbols.tickers(only_enabled=True)
    elif args.symbol:
        targets = args.symbol
    else:
        targets = [config.data.symbol]

    for symbol in targets:
        print(f"\n{'=' * 78}\n=== {symbol}\n{'=' * 78}")
        run_one(symbol, args.horizon, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
