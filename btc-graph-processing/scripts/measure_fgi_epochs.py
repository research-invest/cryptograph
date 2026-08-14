"""
Гейт L по эпохам (docs/task_fear_greed.md §1.1, §2) — устойчивость лифта FGI
разбиением по ВРЕМЕНИ, а не «знак совпал на трёх монетах» (тот критерий для
FGI вырожден: ряд предиктора один и тот же для всех монет, §1.1).

Границы заданы ДО замера, по внешним событиям, не по результату:

    2018-02 … 2020-12   ранний рынок, доминирование BTC высокое
    2021-01 … 2022-12   цикл роста и обвал, включая LUNA/FTX
    2023-01 … 2026-08   после обвала, ETF-эпоха

Внутри каждой эпохи holdout не берётся — сама эпоха и есть проверка
устойчивости; но блочный бутстрап считается всё равно, с нижней границей
блока по автокорреляции FGI (--block-rows, см. fgi_frequencies.py).

    python3 scripts/measure_fgi_epochs.py --all --block-rows 12000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btcproc import config, symbols  # noqa: E402
from btcproc.analysis import samples  # noqa: E402
from btcproc.analysis.lift import DEFAULT_N_BOOT, measure_lift  # noqa: E402
from btcproc.features import fear_greed as fg  # noqa: E402

EPOCHS = [
    ("2018-02..2020-12", "2018-02-01", "2020-12-31"),
    ("2021-01..2022-12", "2021-01-01", "2022-12-31"),
    ("2023-01..2026-08", "2023-01-01", None),
]


def run_one(symbol: str, args) -> dict:
    model_run = samples.resolve_model_run(symbol, args.run)
    frame = samples.load(symbol, args.metric, model_run, with_atoms=True)
    if frame.empty:
        print(f"Нет данных по {symbol}.")
        return {}
    frame, stale = samples.merge_atom_columns(frame)
    if stale:
        print(f"Пропущено {stale} баров без context_atoms.")

    print(f"\n{symbol}: {len(frame)} кандидатов модели #{model_run}")
    by_atom: dict[str, list] = {a: [] for a in fg.CONTEXT_CANDIDATES}

    for epoch_label, start, end in EPOCHS:
        window = frame[frame["ts"] >= f"{start}T00:00:00Z"]
        if end:
            window = window[window["ts"] <= f"{end}T23:59:59Z"]
        if len(window) < 100:
            print(f"  {epoch_label}: слишком мало строк ({len(window)}), пропуск")
            continue

        results = measure_lift(
            window, atoms=list(fg.CONTEXT_CANDIDATES), metric_column="metric",
            alpha=args.alpha, correction="bh", holdout=None, min_group=args.min_group,
            horizon_minutes=config.data.horizon_minutes, n_boot=args.n_boot,
            min_block_rows=args.block_rows,
        )
        print(f"\n  ── {epoch_label} ({len(window)} строк) ──")
        for r in results:
            by_atom[r.atom].append(r)
            print(f"    {r.atom:<24} lift={r.lift:>+.4f} z={r.z:>6.2f} "
                  f"p_boot={r.p_boot if r.p_boot is not None else float('nan'):.4f} "
                  f"{'значим' if r.significant else ''}")

    print(f"\n  ── Свод по эпохам: {symbol} ──")
    verdict = {}
    for atom, rows in by_atom.items():
        if not rows:
            continue
        signs = {r.lift > 0 for r in rows}
        all_significant = all(r.significant for r in rows) and len(rows) == len(EPOCHS)
        one_sign = len(signs) == 1
        passed = all_significant and one_sign
        verdict[atom] = passed
        print(f"    {atom:<24} эпох={len(rows)}/{len(EPOCHS)} "
              f"знак_один={one_sign} все_значимы={all_significant} → "
              f"{'ПРОШЁЛ' if passed else 'не прошёл'}")
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser(description="Гейт L по эпохам — атомы FGI")
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--run", type=int)
    parser.add_argument("--metric", choices=["long_outcome_share", "realized"],
                        default="realized")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-group", type=int, default=30)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--block-rows", type=int, default=None)
    args = parser.parse_args()

    if args.all:
        targets = symbols.tickers(only_enabled=True)
    elif args.symbol:
        targets = args.symbol
    else:
        targets = [config.data.symbol]

    for symbol in targets:
        run_one(symbol, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
