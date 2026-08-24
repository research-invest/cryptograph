"""
Асимметрия пути: есть ли она там, где нет асимметрии знака.

    python3 scripts/measure_path.py --all
    python3 scripts/measure_path.py --symbol BTCUSDT --k 1.5

ТЗ — `crypto-graph/docs/tz_wave_a_24-08-26.md`, задача P. Разметка, обе
нулёвки и правило про одновременное касание — шапка `btcproc/analysis/path.py`.

## Чем это отличается от раздела 26

Там мерился ЗНАК доходности на фиксированном горизонте. Рынок может
систематически ходить «сначала вниз на 1%, потом вверх на 3%» и давать при
этом ровно 50% по знаку. Здесь меряется порядок достижения барьеров `±k·σ_t`
— величина, к которой знак на горизонте не сводится.

## Критерий, заявляемый ДО прогона

> Асимметрия пути признаётся существующей, если доля `up`-меток отличается от
> эмпирической нулёвки на **≥ 0.02 по абсолютной величине** при `p ≤ 0.05`
> после BH по всем ячейкам прогона, **на ≥3 монетах из 6** и **≥2 значениях
> `k` из трёх** (`k ∈ {1, 1.5, 2}`).
>
> Дополнительно, независимо от исхода: та же величина считается ПО
> СОСТОЯНИЯМ графа. Порог — размах доли между состояниями ≥ **0.05** при
> значимом тренд-тесте. Если общая асимметрия отсутствует, а по состояниям
> различается — это результат про граф; если не различается — граф и здесь
> ничего не добавляет.

## Почему нулёвка сохраняет сезонность

Прямой урок отзыва гейта R (47.4): σ меняется по часам суток вместе с
вероятностью дойти до барьера, и нулёвка без часа дня даёт ложную асимметрию
тем же механизмом, каким тот гейт был отозван. Суррогат — перестановка
реальных баров ВНУТРИ бина «час дня × день недели».

Рядом печатается аналитический якорь: для процесса без сноса доля `up` при
симметричных барьерах равна ровно 0.5 независимо от волатильности (замена
времени). Он проверяет код, а не рынок.
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
from btcproc.analysis import markov as mk  # noqa: E402
from btcproc.analysis import path as pt  # noqa: E402
from btcproc.analysis import samples  # noqa: E402
from btcproc.analysis.lift import benjamini_hochberg  # noqa: E402
from btcproc.db import runs as runs_repo  # noqa: E402
from btcproc.db.session import fetch_all  # noqa: E402
from btcproc.ingest import bars  # noqa: E402

FROZEN_END = "2026-08-01"
K_VALUES = (1.0, 1.5, 2.0)
MIN_EFFECT = 0.02
STATE_SPREAD = 0.05


def load_base(symbol: str, end: str) -> pd.DataFrame:
    spec = symbols.get(symbol)
    base = bars.load_ohlcv(symbol, config.data.base_tf, spec.start_date(), end)
    if base.empty:
        raise SystemExit(f"{symbol}: нет баров до {end}")
    return base


def load_states(symbol: str, model_run: int, index: pd.DatetimeIndex) -> np.ndarray | None:
    scope_sql, scope_params = runs_repo.model_run_scope(model_run, "s")
    rows = fetch_all(
        f"SELECT s.ts, s.group_id FROM bar_states s WHERE s.symbol = %s "
        f"AND {scope_sql} ORDER BY s.ts", (symbol, *scope_params))
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    frame = frame.drop_duplicates(subset="ts", keep="last").set_index("ts")
    return frame["group_id"].reindex(index).to_numpy()


def run_cell(symbol: str, base: pd.DataFrame, k: float, args) -> dict:
    horizon_bars = int(config.data.horizon_minutes / config.data.base_minutes)
    sigma = pt.sigma_series(base, args.sigma_window)
    labels = pt.triple_barrier(base, sigma, k, horizon_bars)
    shares = pt.label_shares(labels)

    rng = np.random.default_rng([2026, zlib.crc32(f"{symbol}|{k}".encode())])
    null = pt.surrogate_shares(base, k, horizon_bars, args.n_draws, rng,
                               args.sigma_window)
    if null.size == 0:
        return {"symbol": symbol, "k": k, **shares, "null_mean": float("nan"),
                "effect": float("nan"), "p": 1.0, "labels": labels}

    observed = shares["up_share"]
    effect = observed - float(np.mean(null))
    p_value = float((1 + np.sum(np.abs(null - np.mean(null)) >= abs(effect)))
                    / (1 + len(null)))
    return {"symbol": symbol, "k": k, **shares,
            "null_mean": float(np.mean(null)), "null_sd": float(np.std(null)),
            "effect": effect, "p": p_value, "labels": labels}


def state_table(symbol: str, base: pd.DataFrame, labels: np.ndarray,
                groups: np.ndarray | None, k: float, args) -> dict | None:
    if groups is None:
        return None
    table = pt.by_state(labels, groups)
    if len(table) < 3:
        return None
    spread = float(table["up"].max() - table["up"].min())
    scores = np.arange(len(table), dtype=float)
    counts_total = table["n"].to_numpy(dtype=float)
    counts_up = (table["up"].to_numpy(dtype=float) * counts_total)
    z, p = pt.cochran_armitage(counts_up, counts_total, scores)

    # Нулёвка размаха — обязательна: это статистика экстремума по полусотне
    # состояний, и без неё «размах 0.31» нельзя отличить от «состояний много».
    horizon_bars = int(config.data.horizon_minutes / config.data.base_minutes)
    rng = np.random.default_rng([2026, zlib.crc32(f"{symbol}|{k}|states".encode())])
    null = pt.surrogate_state_spread(base, groups, k, horizon_bars,
                                     args.n_draws, rng, args.sigma_window)
    null_median = float(np.median(null)) if null.size else float("nan")
    p_spread = (float((1 + np.sum(null >= spread)) / (1 + len(null)))
                if null.size else float("nan"))
    holdout = pt.state_holdout(labels, groups)
    return {"n_states": len(table), "spread": spread, "z": z, "p": p,
            "null_spread": null_median, "p_spread": p_spread,
            "holdout": holdout,
            "lowest": (table.index[0], float(table["up"].iloc[0]),
                       int(table["n"].iloc[0])),
            "highest": (table.index[-1], float(table["up"].iloc[-1]),
                        int(table["n"].iloc[-1]))}


def format_report(rows: list[dict], states: dict, args) -> str:
    if not rows:
        return "\nНи одной ячейки не посчитано."
    marks = benjamini_hochberg([r["p"] for r in rows])
    lines = ["", "=" * 104,
             "АСИММЕТРИЯ ПУТИ: ТРОЙНОЙ БАРЬЕР ПРОТИВ СЕЗОННОЙ НУЛЁВКИ",
             "=" * 104,
             f"Барьеры ±k·σ, σ = ATR{args.sigma_window}/close; предел — горизонт "
             f"{config.data.horizon}.",
             "Доля `up` считается среди РАЗРЕШЁННЫХ случаев (up + down).",
             f"Аналитический якорь для процесса без сноса: {pt.analytic_anchor(1.0):.3f} "
             f"— не зависит от волатильности.",
             "Нулёвка — перестановка реальных баров ВНУТРИ бина «час дня × день "
             "недели».", "",
             f"{'монета':<10} {'k':>4} {'размечено':>10} {'решено':>9} "
             f"{'up':>7} {'нулёвка':>8} {'эффект':>8} {'p':>7} {'BH':>4} "
             f"{'без исх.':>9} {'неясно':>8}", "─" * 104]
    for r, mark in zip(rows, marks):
        lines.append(
            f"{r['symbol']:<10} {r['k']:>4.1f} {r['n']:>10} {r['n_decided']:>9} "
            f"{r['up_share']:>7.4f} {r['null_mean']:>8.4f} {r['effect']:>+8.4f} "
            f"{r['p']:>7.4f} {'да' if mark else 'нет':>4} "
            f"{r['none_share']:>9.3f} {r['ambiguous_share']:>8.4f}"
        )

    lines += ["", "ВЕРДИКТ ПО ЗАЯВЛЕННОМУ КРИТЕРИЮ"]
    by_symbol: dict[str, set] = {}
    for r, mark in zip(rows, marks):
        if abs(r["effect"]) >= MIN_EFFECT and mark:
            by_symbol.setdefault(r["symbol"], set()).add(r["k"])
    good = [s for s, ks in by_symbol.items() if len(ks) >= 2]
    lines.append(f"  ячеек с |эффектом| ≥ {MIN_EFFECT} и значимых после BH: "
                 f"{sum(len(v) for v in by_symbol.values())} из {len(rows)}")
    lines.append(f"  монет с ≥2 значениями k: {len(good)} (критерий — ≥3)")
    lines.append("  ВЕРДИКТ: " + (
        "АСИММЕТРИЯ ПУТИ ЕСТЬ" if len(good) >= 3 else
        "АСИММЕТРИИ ПУТИ НЕТ — там же, где нет асимметрии знака"))

    if states:
        lines += ["", "ПО СОСТОЯНИЯМ ГРАФА (вторая половина задачи)",
                  f"{'монета':<10} {'k':>4} {'состояний':>10} {'размах':>8} "
                  f"{'нулёвка':>8} {'p':>7} {'z тренда':>9}   крайние состояния",
                  "─" * 104]
        for (symbol, k), value in sorted(states.items()):
            if value is None:
                continue
            low, high = value["lowest"], value["highest"]
            lines.append(
                f"{symbol:<10} {k:>4.1f} {value['n_states']:>10} "
                f"{value['spread']:>8.3f} {value['null_spread']:>8.3f} "
                f"{value['p_spread']:>7.4f} {value['z']:>9.2f}   "
                f"#{low[0]:g}: {low[1]:.3f} (n={low[2]}) … "
                f"#{high[0]:g}: {high[1]:.3f} (n={high[2]})"
            )
        lines += ["", "ТО ЖЕ НА ОТЛОЖЕННОЙ ЧАСТИ (70/30) — обязательная проверка",
                  "Разметка обучена на ВСЕЙ истории, поэтому доля по состоянию, "
                  "посчитанная там же,",
                  "величина in-sample. Раздел 26 журнала — история ровно про то, "
                  "что бывает дальше.",
                  f"{'монета':<10} {'k':>4} {'общих':>7} {'rho(70,30)':>11} "
                  f"{'размах 70%':>11} {'размах 30%':>11}", "─" * 104]
        for (symbol, k), value in sorted(states.items()):
            holdout = (value or {}).get("holdout") or {}
            if "rho" not in holdout:
                continue
            lines.append(
                f"{symbol:<10} {k:>4.1f} {holdout['n_common']:>7} "
                f"{holdout['rho']:>+11.3f} {holdout['spread_train']:>11.3f} "
                f"{holdout['spread_test']:>11.3f}"
            )

        useful = [v for v in states.values() if v]
        wide = sum(1 for v in useful
                   if v["spread"] >= STATE_SPREAD and v["p_spread"] <= 0.05)
        lines += ["", f"  ячеек с размахом ≥ {STATE_SPREAD} И значимым по "
                      f"нулёвке: {wide} из {len(useful)}",
                  "  «Нулёвка» — тот же размах на сезонных суррогатах при "
                  "неподвижных метках",
                  "  состояний. Сравнивать наблюдённый размах с нулём нельзя: "
                  "это статистика",
                  "  экстремума по полусотне состояний, она заметно больше нуля "
                  "и при полном",
                  "  отсутствии эффекта.",
                  "  z тренда СПРАВОЧНЫЙ: шкала — ранг состояния по самой доле, "
                  "то есть выбрана",
                  "  по данным, и тест анти-консервативен по построению "
                  "(докстринг `cochran_armitage`)."]
        rhos = [(v.get("holdout") or {}).get("rho") for v in useful]
        rhos = [r for r in rhos if r is not None]
        if rhos:
            positive = sum(1 for r in rhos if r > 0.3)
            lines += ["", f"  ячеек с rho(70%, 30%) > 0.3: {positive} из "
                          f"{len(rhos)}; медиана rho "
                          f"{float(np.median(rhos)):+.3f}",
                      "  Низкая rho означает, что порядок состояний по доле "
                      "на отложенной части",
                      "  другой, то есть in-sample различие не переносится и "
                      "продуктом быть не может."]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--end", default=FROZEN_END)
    parser.add_argument("--k", type=float, action="append")
    parser.add_argument("--sigma-window", type=int, default=14)
    parser.add_argument("--n-draws", type=int, default=40)
    parser.add_argument("--run", type=int)
    parser.add_argument("--no-states", dest="states", action="store_false", default=True)
    args = parser.parse_args()

    ks = args.k or list(K_VALUES)
    specs = symbols.resolve_many(args.symbol, args.all)
    print(f"Асимметрия пути. Монет {len(specs)}, k = {ks}, нулёвка "
          f"{args.n_draws} суррогатов, граница {args.end}.")
    print("Критерий заявлен ДО запуска — см. шапку скрипта.\n")

    rows, states = [], {}
    for spec in specs:
        symbol = spec.ticker
        print(f"=== {symbol}")
        try:
            base = load_base(symbol, args.end)
        except SystemExit as exc:
            print(f"  пропущена: {exc}")
            continue
        groups = None
        if args.states:
            try:
                model_run = samples.resolve_model_run(symbol, args.run)
                groups = load_states(symbol, model_run, base.index)
            except SystemExit:
                groups = None
        for k in ks:
            row = run_cell(symbol, base, k, args)
            labels = row.pop("labels")
            print(f"  k={k}: up={row['up_share']:.4f} против нулёвки "
                  f"{row['null_mean']:.4f}, эффект {row['effect']:+.4f} "
                  f"(p={row['p']:.4f})")
            rows.append(row)
            if groups is not None:
                states[(symbol, k)] = state_table(symbol, base, labels, groups,
                                                  k, args)

    print(format_report(rows, states, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
