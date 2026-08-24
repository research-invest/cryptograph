"""
Дрейф распределения признаков: правда ли рынок меняется поквартально.

    python3 scripts/measure_drift.py --all
    python3 scripts/measure_drift.py --symbol BTCUSDT --window-days 90

ТЗ — `crypto-graph/docs/tz_wave_a_24-08-26.md`, задача W. Постановка вопроса,
величина и обе оговорки — шапка `btcproc/analysis/drift.py`.

## Критерий, заявляемый ДО прогона

> Гипотеза «квартал особенный» подтверждается, если расстояние «последний
> квартал против всей предшествующей истории» лежит выше **95-го перцентиля**
> распределения расстояний между случайными непересекающимися парами окон
> той же длины, **на ≥4 монетах из 6**, и превышает его медиану не менее чем
> в **1.5 раза**.

Продуктовая часть считается независимо от исхода критерия: ряд «непохожесть
на собственную историю» описателен и годен для сводки в любом случае. Он
ничего не предсказывает, и подавать его как предупреждение о движении цены
нельзя.

## Что прогон НЕ делает

Не переобучает модель, ничего не пишет в БД. Признаки считает из баров тем же
`features/builder.py`, что и конвейер.
"""
from __future__ import annotations

import argparse
import os
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["SINK_MODE"] = "none"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from btcproc import config, symbols  # noqa: E402
from btcproc.analysis import drift as dr  # noqa: E402
from btcproc.features import builder as feat  # noqa: E402
from btcproc.ingest import bars  # noqa: E402

FROZEN_END = "2026-08-01"


def prepare(symbol: str, end: str) -> pd.DataFrame:
    spec = symbols.get(symbol)
    base = bars.load_ohlcv(symbol, config.data.base_tf, spec.start_date(), end)
    if base.empty:
        raise SystemExit(f"{symbol}: нет баров до {end}")
    context = {tf: bars.load_ohlcv(symbol, tf, spec.start_date(), end)
               for tf in config.data.context_tfs}
    features = feat.build_features(base, context, symbol=symbol)
    return dr.robust_scale(features)


def run_symbol(symbol: str, args) -> dict | None:
    frame = prepare(symbol, args.end)
    window = int(args.window_days * 24 * 60 / config.data.base_minutes)
    if len(frame) < 3 * window:
        print(f"  [{symbol}] истории меньше трёх окон ({len(frame)} строк) — пропуск")
        return None

    rng = np.random.default_rng([2026, zlib.crc32(f"{symbol}|drift".encode())])
    recent = frame.iloc[-window:]
    past = frame.iloc[:-window]
    observed = dr.frame_distance(recent, past)
    null = dr.null_distances(frame, window, args.n_draws, rng)
    if null.size == 0:
        print(f"  [{symbol}] нулёвка не набралась — пропуск")
        return None

    p95 = float(np.percentile(null, 95))
    median = float(np.median(null))
    series = dr.rolling_drift(frame, window, args.step) if args.series else None
    return {
        "symbol": symbol, "n_rows": len(frame), "window": window,
        "observed": observed, "null_median": median, "null_p95": p95,
        "ratio": observed / median if median > 0 else float("inf"),
        "p": float((1 + np.sum(null >= observed)) / (1 + len(null))),
        "passed": observed > p95 and (median <= 0 or observed / median >= 1.5),
        "series": series,
    }


def format_report(results: list[dict], args) -> str:
    if not results:
        return "\nНи одной монеты не посчитано."
    lines = ["", "=" * 96,
             "ДРЕЙФ РАСПРЕДЕЛЕНИЯ ПРИЗНАКОВ: ВЫДЕЛЯЕТСЯ ЛИ ПОСЛЕДНИЙ КВАРТАЛ",
             "=" * 96,
             f"W₁, усреднённое по 32 признакам после robust_scale. Окно "
             f"{args.window_days} дней.",
             "Вопрос НЕ «отличается ли квартал от истории» (отличается всегда), "
             "а «отличается ли",
             "сильнее, чем два случайных куска той же длины внутри самой "
             "истории».",
             "Критерий: выше p95 нулёвки И не менее чем в 1.5 раза выше её "
             "медианы.", "",
             f"{'монета':<10} {'строк':>9} {'окно':>7} {'наблюдение':>11} "
             f"{'нулёвка p50':>12} {'нулёвка p95':>12} {'отношение':>10} "
             f"{'p':>8} {'вердикт':>9}", "─" * 96]
    for r in results:
        lines.append(
            f"{r['symbol']:<10} {r['n_rows']:>9} {r['window']:>7} "
            f"{r['observed']:>11.4f} {r['null_median']:>12.4f} "
            f"{r['null_p95']:>12.4f} {r['ratio']:>10.2f} {r['p']:>8.4f} "
            f"{'да' if r['passed'] else 'нет':>9}"
        )
    passed = sum(1 for r in results if r["passed"])
    lines += ["", f"  ВЕРДИКТ: прошло {passed} монет из {len(results)} "
                  f"(критерий — ≥4 из 6)"]
    lines.append("  " + (
        "ПОСЛЕДНИЙ КВАРТАЛ ДЕЙСТВИТЕЛЬНО ОСОБЕННЫЙ" if passed >= 4 else
        "КВАРТАЛ НЕ ВЫДЕЛЯЕТСЯ среди кварталов вообще — гипотеза «рынок "
        "меняется каждый\n  квартал» в этой форме не подтверждается"))

    if any(r.get("series") is not None for r in results):
        lines += ["", "РЯД «НЕПОХОЖЕСТЬ НА СОБСТВЕННУЮ ИСТОРИЮ» "
                      "(последние точки, продуктовая величина)"]
        for r in results:
            series = r.get("series")
            if series is None or series.empty:
                continue
            tail = series.tail(6)
            pairs = "  ".join(f"{row['ts']:%Y-%m}:{row['distance']:.3f}"
                              for _, row in tail.iterrows())
            lines.append(f"  {r['symbol']:<10} {pairs}")
        lines += ["", "  Величина описательна: она говорит «рынок непохож на "
                      "своё прошлое» и НИЧЕГО",
                  "  не говорит о том, куда пойдёт цена. Подавать её как "
                  "предупреждение нельзя."]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--end", default=FROZEN_END)
    parser.add_argument("--window-days", type=int, default=90)
    parser.add_argument("--n-draws", type=int, default=200)
    parser.add_argument("--step", type=int, default=672,
                        help="шаг ряда дрейфа в барах (по умолчанию неделя)")
    parser.add_argument("--series", action="store_true", default=True)
    parser.add_argument("--no-series", dest="series", action="store_false")
    args = parser.parse_args()

    specs = symbols.resolve_many(args.symbol, args.all)
    print(f"Дрейф распределения. Монет {len(specs)}, окно {args.window_days} дней, "
          f"нулёвка {args.n_draws} пар, граница {args.end}.")
    print("Критерий заявлен ДО запуска — см. шапку скрипта.\n")

    results = []
    for spec in specs:
        print(f"=== {spec.ticker}")
        try:
            result = run_symbol(spec.ticker, args)
        except SystemExit as exc:
            print(f"  пропущена: {exc}")
            continue
        if result:
            print(f"  наблюдение {result['observed']:.4f}, нулёвка "
                  f"{result['null_median']:.4f} (p95 {result['null_p95']:.4f}), "
                  f"{'выделяется' if result['passed'] else 'не выделяется'}")
            results.append(result)

    print(format_report(results, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
