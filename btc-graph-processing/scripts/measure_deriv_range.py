"""
Гейт R для деривативных метрик (docs/tz_deriv_ingest_14-08-26.md §2, §2.1) —
предсказывает ли какая-то из 6 непрерывных величин B2 РАЗМАХ.

Ничего не пишет — ни в БД, ни в конфиг, по образцу measure_fgi_range.py.
`range_ratio` считается автономно из баров и ATR14, без правки outcomes.py и
candidates/builder.py.

Отличие от measure_fgi_range.py — БЕНЧМАРК-РЕШЕНИЕ §2.1 (выбран вариант
«первое»): не «величина СИЛЬНЕЕ rv» (`paired_diff_p`), а «величина ДОБАВЛЯЕТ
к rv» (`partial_r2_gain` — частная корреляция / приращение R² в
`range_ratio ~ rv (+ величина)`). Причина — у ОИ есть содержательный шанс на
значимое СОБСТВЕННОЕ rho (в отличие от FGI, где вся первая половина гейта
была провалена везде), и проиграть `rv` по конструкции цели (тот же ATR в
знаменателе и у range_ratio, и у rv) было бы отвержением по артефакту
нормировки, а не по существу.

Три горизонта (24h/8h/4h — те же, что у FGI, ради сравнимости), три
ЗАФИКСИРОВАННЫЕ ДО ЗАМЕРА эпохи (те же самые эпохи и границы). Блок
бутстрапа считается ОТДЕЛЬНО для каждой из 6 величин — их автокорреляция
разнится на порядки (oi_chg_1h умирает за единицы баров, top_vs_retail не
умирает вовсе в пределах разумного потолка, deriv_frequencies.py --section autocorr).

    python3 scripts/measure_deriv_range.py --symbol BTCUSDT
    python3 scripts/measure_deriv_range.py --all
"""
from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from btcproc import config, symbols  # noqa: E402
from btcproc.analysis.lift import block_length_rows  # noqa: E402
from btcproc.analysis.range_lift import (  # noqa: E402
    autocorr_block_rows,
    forward_range_ratio,
    partial_r2_gain,
    spearman_block_p,
)
from btcproc.features import deriv  # noqa: E402
from btcproc.features import indicators as ind  # noqa: E402
from btcproc.ingest import bars, metrics as metrics_ingest  # noqa: E402

HORIZONS = {"24h": "24h", "8h": "8h", "4h": "4h"}

EPOCHS = [
    ("2018-02..2020-12", "2018-02-01", "2020-12-31"),
    ("2021-01..2022-12", "2021-01-01", "2022-12-31"),
    ("2023-01..2026-08", "2023-01-01", None),
]

N_BOOT = 2000
AUTOCORR_MAX_LAG_BARS = 5000


def _horizon_bars(label: str) -> int:
    unit = label[-1]
    value = int(label[:-1])
    minutes = value * {"m": 1, "h": 60, "d": 1440}[unit]
    return minutes // config.data.base_minutes


def build_frame(symbol: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Бары + все 6 величин B2 + бенчмарк rv. Плюс block_rows на каждую величину."""
    base = bars.load_ohlcv(symbol, config.data.base_tf, None, None)
    if base.empty:
        raise SystemExit(f"Нет баров по {symbol} — сначала ingest.")
    metrics_frame = metrics_ingest.load_deriv_metrics(symbol, config.data.base_tf)
    if metrics_frame.empty:
        raise SystemExit(f"deriv_metrics пуст по {symbol} — сначала ingest-metrics.")

    values = deriv.build_deriv(base, metrics_frame, symbol)
    atr14 = ind.atr(base, 14)
    rv_96 = ind.realized_vol(base["close"], 96)

    frame = pd.DataFrame(index=base.index)
    for name in deriv.FEATURE_CANDIDATES:
        frame[name] = values[name]
    frame["rv_96"] = rv_96
    frame["_atr14"] = atr14
    frame["_close"] = base["close"]
    frame["_high"] = base["high"]
    frame["_low"] = base["low"]

    block_rows = {}
    for name in deriv.FEATURE_CANDIDATES:
        series = frame[name].dropna()
        if len(series) < 200:
            block_rows[name] = 96  # горизонт 24h как минимум
            continue
        _, rows = autocorr_block_rows(series, bars_per_day=1, floor=0.2,
                                      max_lag_days=AUTOCORR_MAX_LAG_BARS)
        block_rows[name] = rows
    return frame, block_rows


def measure(symbol: str, frame: pd.DataFrame, block_rows: dict[str, int],
           predictors: list[str] | None = None, n_boot: int = N_BOOT) -> list[dict]:
    results = []
    predictors = predictors or list(deriv.FEATURE_CANDIDATES)
    for horizon_label in HORIZONS:
        h_bars = _horizon_bars(horizon_label)
        atr14 = frame["_atr14"]
        base_like = frame[["_high", "_low", "_close"]].rename(
            columns={"_high": "high", "_low": "low", "_close": "close"}
        )
        range_ratio = forward_range_ratio(base_like, atr14, h_bars)

        for predictor_name in predictors:
            joined = pd.concat(
                [frame[predictor_name], frame["rv_96"], range_ratio.rename("range_ratio")],
                axis=1,
            ).dropna()

            horizon_minutes = h_bars * config.data.base_minutes
            block_by_horizon = block_length_rows(pd.Series(joined.index), horizon_minutes)
            block = max(block_by_horizon, block_rows.get(predictor_name, 96))

            for epoch_label, start, end in [("вся история", None, None)] + EPOCHS:
                window = joined
                if start:
                    window = window[window.index >= pd.Timestamp(start, tz="UTC")]
                if end:
                    window = window[window.index <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
                if len(window) < 50:
                    continue

                salt = zlib.crc32(f"{symbol}|{predictor_name}|{horizon_label}|{epoch_label}".encode())
                rng = np.random.default_rng([42, salt])
                pred_arr = window[predictor_name].to_numpy()
                rv_arr = window["rv_96"].to_numpy()
                target = window["range_ratio"].to_numpy()

                rho, p_own = spearman_block_p(pred_arr, target, block, n_boot, rng)
                r2_base, r2_full, r_partial, p_gain = partial_r2_gain(
                    pred_arr, rv_arr, target, block, n_boot, rng
                )

                results.append({
                    "symbol": symbol, "predictor": predictor_name,
                    "horizon": horizon_label, "epoch": epoch_label,
                    "n": len(window), "block": block,
                    "rho": rho, "p_own": p_own,
                    "r2_base": r2_base, "r2_full": r2_full,
                    "r_partial": r_partial, "p_gain": p_gain,
                })
    return results


def format_table(rows: list[dict], predictor: str) -> str:
    subset = [r for r in rows if r["predictor"] == predictor]
    header = (f"{'горизонт':<9} {'эпоха':<18} {'n':>7} {'блок':>6} "
              f"{'rho':>8} {'p(rho)':>8} {'ΔR²':>8} {'r_part':>8} {'p(добав.)':>10}  вердикт")
    lines = [header, "─" * len(header)]
    for r in subset:
        sig_own = r["p_own"] <= 0.05
        adds = r["p_gain"] <= 0.05
        verdict = "ГЕЙТ R" if sig_own and adds else ("значим, не добавляет" if sig_own else "не значим")
        delta = r["r2_full"] - r["r2_base"]
        lines.append(
            f"{r['horizon']:<9} {r['epoch']:<18} {r['n']:>7} {r['block']:>6} "
            f"{r['rho']:>+8.4f} {r['p_own']:>8.4f} {delta:>+8.4f} "
            f"{r['r_partial']:>+8.4f} {r['p_gain']:>10.4f}  {verdict}"
        )
    return "\n".join(lines)


def gate_verdict(rows: list[dict]) -> None:
    print("\n── Гейт R: свод по эпохам (без «вся история» — она не входит в критерий) ──")
    for predictor in deriv.FEATURE_CANDIDATES:
        print(f"\n  {predictor}:")
        for horizon_label in HORIZONS:
            by_epoch = [r for r in rows if r["predictor"] == predictor
                       and r["horizon"] == horizon_label and r["epoch"] != "вся история"]
            passed = sum(1 for r in by_epoch if r["p_own"] <= 0.05 and r["p_gain"] <= 0.05)
            verdict = "ПРОЙДЕН" if passed >= 2 else "не пройден"
            print(f"    {horizon_label}: прошло эпох {passed}/{len(by_epoch)} → {verdict}")


def run_one(symbol: str, predictors: list[str] | None = None, n_boot: int = N_BOOT) -> list[dict]:
    frame, block_rows = build_frame(symbol)
    print(f"\n{symbol}: block_rows по автокорреляции —",
          ", ".join(f"{k}={v}" for k, v in block_rows.items()))
    rows = measure(symbol, frame, block_rows, predictors, n_boot)
    for predictor in (predictors or deriv.FEATURE_CANDIDATES):
        print(f"\n=== {predictor} ===")
        print(format_table(rows, predictor))
    gate_verdict(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Гейт R — деривативные метрики против размаха")
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--predictor", action="append",
                        help="Только эти величины B2; по умолчанию все шесть")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()

    if args.all:
        targets = symbols.tickers(only_enabled=True)
    elif args.symbol:
        targets = args.symbol
    else:
        targets = [config.data.symbol]

    for symbol in targets:
        print(f"\n{'=' * 78}\n=== {symbol}\n{'=' * 78}")
        run_one(symbol, args.predictor, args.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
