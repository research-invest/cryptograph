"""
Обучение и проверка квантильного регрессора размаха на признаках.

    python3 scripts/fit_range_forecast.py --all
    python3 scripts/fit_range_forecast.py --symbol BTCUSDT --horizon 24h --save

Конструкция, её вход-выход и обоснование каждого решения — в шапке
`btcproc/analysis/range_forecast.py`. Читать обязательно: без неё
`range_lift` выглядит произвольным отношением, а он здесь главный ответ.

**Ничего не пишет в БД и ничего не отправляет.** С `--save` кладёт обученные
модели файлами в `DATA_DIR/range_forecast/`, и это единственная запись,
которую скрипт делает вообще.

## Чем это отличается от замера 47.6

Тот мерил приращение R² градиентного бустинга внутри purged walk-forward на
обучающей части. Здесь — обычное разбиение 70/30 тем же `holdout.split_bar`,
что у Ш0 и D1, обучение один раз и проверка на отложенных 30%, которых модель
не видела вовсе. Это строже: в walk-forward последний фолд учится почти на
всей истории, здесь — на её первых семидесяти процентах, и между обучением и
проверкой вырезан зазор в горизонт.

Заодно проверяется то, чего замер 47.6 не проверял вовсе: **калибровка**.
Приращение R² говорит, что модель различает случаи; покрытие квантилей
говорит, можно ли верить числу «p90 = 2.3». Замер C3 (47.10) показал, зачем
это нужно порознь: там перцентили конфигурации были откалиброваны прекрасно
(ECE 0.011–0.069) и при этом бесполезны — бенчмарк предсказывал лучше.

## Критерий, заявляемый ДО запуска

> Регрессор годен к переносу в конвейер, если на отложенной части выполнены
> **все три** условия:
>
> 1. приращение out-of-sample R² медианной модели над бенчмарком B2
>    составляет **≥ 0.02** при `p ≤ 0.05` блочным бутстрапом;
> 2. Spearman(p50, факт) значимо ВЫШЕ, чем у бенчмарка, парным блочным
>    тестом (`p ≤ 0.05`);
> 3. покрытие каждого из четырёх квантилей отклоняется от номинала не более
>    чем на **0.05**;
>
> — на ≥2 монетах, ≥2 горизонтах из трёх, обеих нормировках A3 и обоих
> зёрнах бустинга.

Порог 0.02 — тот же, что в критерии C4 ТЗ, и по той же причине: при n_эфф в
тысячи значимость достигается на ничтожных величинах, поэтому рядом с p
обязан стоять порог практической величины. Условие 2 добавлено против
ситуации замера C3, где абсолютная связь была, а превосходства над тривиальным
прогнозом не было. Условие 3 — против обратной: различающая, но
не калиброванная модель даёт числа, которыми нельзя пользоваться.

Условия не пересматриваются по итогам. Непройденный критерий фиксируется в
журнале прямо, как это сделано в разделах 26, 31 и 47.
"""
from __future__ import annotations

import argparse
import os
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ДО импорта btcproc.config: конфиг читает окружение на импорте.
os.environ["SINK_MODE"] = "none"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from btcproc import config, symbols  # noqa: E402
from btcproc.analysis import holdout as ho  # noqa: E402
from btcproc.analysis import range_forecast as rf  # noqa: E402
from btcproc.analysis import range_model as rm  # noqa: E402
from btcproc.analysis.lift import DEFAULT_N_BOOT, benjamini_hochberg  # noqa: E402
from btcproc.analysis.range_lift import paired_diff_p, spearman_block_p  # noqa: E402
from btcproc.features import builder as feat  # noqa: E402
from btcproc.ingest import bars  # noqa: E402

FROZEN_END = "2026-08-01"

#: Пороги критерия, объявленные в шапке и не пересматриваемые.
MIN_DELTA_R2 = 0.02
MAX_COVERAGE_ERROR = 0.05


def prepare(symbol: str, end: str):
    spec = symbols.get(symbol)
    start = spec.start_date()
    base = bars.load_ohlcv(symbol, config.data.base_tf, start, end)
    if base.empty:
        raise SystemExit(f"{symbol}: в БД нет баров до {end}. Сначала ingest.")
    context = {
        tf: bars.load_ohlcv(symbol, tf, start, end) for tf in config.data.context_tfs
    }
    print(f"  баров {len(base)}; признаки…")
    features = feat.build_features(base, context, symbol=symbol)
    print(f"  признаков {features.shape[1]}")
    return base, features


def run_cell(symbol: str, base: pd.DataFrame, features: pd.DataFrame,
             horizon: str, normalization: str, seed: int, args) -> dict | None:
    frame, target, benchmark_names = rf.design_matrix(
        base, features, horizon, normalization, config.data.base_minutes,
        augment_benchmark=getattr(args, "augment_benchmark", False),
    )
    if len(frame) < 20000:
        print(f"  [{symbol} {horizon} {normalization} зерно {seed}] "
              f"строк {len(frame)} — мало")
        return None

    h_bars = rm.horizon_bars(horizon, config.data.base_minutes)
    split_ts = ho.split_bar(frame.index, args.train_frac)
    n_train = int((frame.index < split_ts).sum())
    # Зазор на границе: последние h_bars строк обучения имеют окна исходов,
    # накрывающие начало отложенной части (та же дисциплина, что в D1).
    train = slice(0, max(1, n_train - h_bars))
    test = slice(n_train, len(frame))

    print(f"  [{symbol} {horizon} {normalization} зерно {seed}] "
          f"обучение {train.stop} строк до {split_ts:%Y-%m-%d}, "
          f"отложено {test.stop - test.start}")
    model = rf.fit(symbol, frame, target, benchmark_names, train, seed,
                   horizon, normalization, log=lambda m: None)
    if args.save:
        save_model(model, args)

    holdout_frame = frame.iloc[test]
    predicted = model.predict(holdout_frame)
    actual_log = target.to_numpy(dtype=float)[test]
    actual = np.exp(actual_log)

    # ── Условие 1: приращение OOS R² над бенчмарком ────────────────────────
    # R² считается в логарифмах — той же шкале, в которой меряли 47.6, — и
    # относительно среднего ОБУЧАЮЩЕЙ части: от среднего отложенной было бы
    # подглядыванием.
    train_mean = float(target.to_numpy(dtype=float)[train].mean())
    full_log = np.log(predicted["p50"].to_numpy(dtype=float))
    bench_log = np.log(predicted["bench_p50"].to_numpy(dtype=float))
    ts = frame.index.to_numpy()[test]
    fold = np.ones(len(actual))
    means = np.full(len(actual), train_mean)
    base_fit = rm.FoldPredictions(ts=ts, actual=actual_log, predicted=bench_log,
                                  train_mean=means, fold=fold)
    full_fit = rm.FoldPredictions(ts=ts, actual=actual_log, predicted=full_log,
                                  train_mean=means, fold=fold)
    block = rm.bootstrap_block(pd.Series(ts), horizon)
    rng = np.random.default_rng([42, zlib.crc32(
        f"{symbol}|{horizon}|{normalization}|{seed}".encode())])
    gain = rm.compare(base_fit, full_fit, block, args.n_boot, rng)

    # ── Условие 2: ранжирование лучше бенчмарка ────────────────────────────
    rho_model, rho_bench, p_better = paired_diff_p(
        predicted["p50"].to_numpy(float), predicted["bench_p50"].to_numpy(float),
        actual, block, args.n_boot, rng,
    )
    _, p_own = spearman_block_p(predicted["p50"].to_numpy(float), actual,
                                block, args.n_boot, rng)

    # ── Условие 3: калибровка квантилей ────────────────────────────────────
    observed = rf.coverage(actual, predicted)
    worst = max(abs(observed[q] - q) for q in rf.QUANTILES)

    pinball = {q: rf.pinball_loss(actual, predicted[f"p{int(q * 100)}"].to_numpy(float), q)
               for q in rf.QUANTILES}
    pinball_bench = {
        q: rf.pinball_loss(actual, predicted[f"bench_p{int(q * 100)}"].to_numpy(float), q)
        for q in rf.QUANTILES
    }
    regimes = predicted["range_regime"].value_counts(normalize=True).to_dict()

    row = {
        "symbol": symbol, "horizon": horizon, "norm": normalization, "seed": seed,
        "n_train": train.stop, "n_test": len(actual), "block": block,
        "split": split_ts,
        "r2_bench": gain.r2_base, "r2_full": gain.r2_full,
        "delta": gain.delta, "p_delta": gain.p_value,
        "rho": rho_model, "rho_bench": rho_bench,
        "p_own": p_own, "p_better": p_better,
        "coverage": observed, "coverage_error": worst,
        "ece": rf.calibration_error(observed),
        "pinball_gain": {q: 1.0 - pinball[q] / pinball_bench[q]
                         for q in rf.QUANTILES},
        "regimes": regimes,
        "lift_p10": float(predicted["range_lift"].quantile(0.10)),
        "lift_p90": float(predicted["range_lift"].quantile(0.90)),
    }
    print(f"      ΔR²={gain.delta:+.4f} (p={gain.p_value:.4f}) "
          f"rho {rho_model:+.3f} против {rho_bench:+.3f} (p={p_better:.4f}) "
          f"худшее покрытие {worst:.3f}")
    return row


def save_model(model: rf.RangeForecast, args) -> None:
    import joblib

    directory = Path(config.data.data_dir) / "range_forecast"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (f"{model.symbol}_{model.horizon}_{model.normalization}"
                        f"_{model.seed}.joblib")
    joblib.dump(model, path)
    print(f"      сохранено: {path}")


# ─── Печать ─────────────────────────────────────────────────────────────────
def format_report(rows: list[dict]) -> str:
    if not rows:
        return "\nНи одной ячейки не посчитано."

    marks = benjamini_hochberg([r["p_delta"] for r in rows])
    header = (f"{'монета':<10} {'гор.':>5} {'норм.':>6} {'зерно':>6} {'n_test':>7} "
              f"{'R²(B2)':>8} {'R²(мод.)':>9} {'ΔR²':>8} {'p':>7} {'BH':>3} "
              f"{'rho':>7} {'rho(B2)':>8} {'p(лучше)':>9} {'макс.откл.':>10}")
    lines = ["", "=" * len(header),
             "КВАНТИЛЬНЫЙ РЕГРЕССОР РАЗМАХА НА ПРИЗНАКАХ — ОТЛОЖЕННАЯ ЧАСТЬ",
             "=" * len(header),
             "R² — в логарифмической шкале цели, относительно среднего обучающей",
             "части. rho — Spearman p50 с фактическим range_ratio. «макс.откл.» —",
             "наибольшее отклонение покрытия квантиля от номинала.", "",
             header, "─" * len(header)]
    for r, mark in zip(rows, marks):
        lines.append(
            f"{r['symbol']:<10} {r['horizon']:>5} {r['norm']:>6} {r['seed']:>6} "
            f"{r['n_test']:>7} {r['r2_bench']:>+8.4f} {r['r2_full']:>+9.4f} "
            f"{r['delta']:>+8.4f} {r['p_delta']:>7.4f} {'да' if mark else 'нет':>3} "
            f"{r['rho']:>+7.3f} {r['rho_bench']:>+8.3f} {r['p_better']:>9.4f} "
            f"{r['coverage_error']:>10.3f}"
        )
    lines.append("")
    lines.append(f"Тестов в поправке BH: {len(rows)}.")

    lines.append("")
    lines.append("КАЛИБРОВКА (доля фактических ниже квантиля; идеал — сам квантиль)")
    lines.append(f"  {'монета':<10} {'гор.':>5} {'норм.':>6} {'зерно':>6}  "
                 + "  ".join(f"p{int(q * 100)}" for q in rf.QUANTILES) + "     ECE")
    for r in rows:
        cov = "  ".join(f"{r['coverage'][q]:.2f}" for q in rf.QUANTILES)
        lines.append(f"  {r['symbol']:<10} {r['horizon']:>5} {r['norm']:>6} "
                     f"{r['seed']:>6}  {cov}   {r['ece']:.3f}")
    lines.append(f"  идеал: {'  '.join(f'{q:.2f}' for q in rf.QUANTILES)}")

    lines.append("")
    lines.append("RANGE_LIFT И РЕЖИМЫ (что модель реально говорит оператору)")
    lines.append(f"  {'монета':<10} {'гор.':>5} {'норм.':>6} {'зерно':>6} "
                 f"{'lift p10':>9} {'lift p90':>9}   доли режимов")
    for r in rows:
        shares = ", ".join(f"{name} {r['regimes'].get(name, 0.0):.0%}"
                           for name in rf.REGIME_NAMES)
        lines.append(f"  {r['symbol']:<10} {r['horizon']:>5} {r['norm']:>6} "
                     f"{r['seed']:>6} {r['lift_p10']:>9.3f} {r['lift_p90']:>9.3f}   "
                     f"{shares}")

    lines.append("")
    lines.append("ВЕРДИКТ ПО ЗАЯВЛЕННОМУ КРИТЕРИЮ")
    passed = []
    for r, mark in zip(rows, marks):
        ok1 = r["delta"] >= MIN_DELTA_R2 and mark
        ok2 = r["p_better"] <= 0.05 and r["rho"] > r["rho_bench"]
        ok3 = r["coverage_error"] <= MAX_COVERAGE_ERROR
        r["passed"] = ok1 and ok2 and ok3
        r["conditions"] = (ok1, ok2, ok3)
        if r["passed"]:
            passed.append(r)
    for i, name in enumerate((
        f"1) ΔR² ≥ {MIN_DELTA_R2} и значим после BH",
        "2) ранжирует лучше бенчмарка (парный тест)",
        f"3) покрытие квантилей в пределах {MAX_COVERAGE_ERROR}",
    )):
        got = sum(1 for r in rows if r["conditions"][i])
        lines.append(f"  {name}: {got} ячеек из {len(rows)}")

    by_symbol: dict[str, dict[str, set]] = {}
    for r in passed:
        seen = by_symbol.setdefault(r["symbol"], {"h": set(), "n": set(), "s": set()})
        seen["h"].add(r["horizon"])
        seen["n"].add(r["norm"])
        seen["s"].add(r["seed"])
    good = [s for s, seen in by_symbol.items()
            if len(seen["h"]) >= 2 and len(seen["n"]) >= 2 and len(seen["s"]) >= 2]
    lines.append(f"  все три условия сразу: {len(passed)} ячеек из {len(rows)}; "
                 f"монет с ≥2 горизонтами, обеими нормировками и обоими зёрнами: "
                 f"{len(good)}")
    verdict = ("РЕГРЕССОР ГОДЕН К ПЕРЕНОСУ В КОНВЕЙЕР" if len(good) >= 2
               else "КРИТЕРИЙ НЕ ПРОЙДЕН")
    lines.append(f"  ВЕРДИКТ: {verdict}")
    if len(good) < 2 and passed:
        lines.append("")
        lines.append("  Прошедшие ячейки есть, но условие устойчивости не выполнено.")
        lines.append("  Это НЕ основание переносить модель в конвейер: ровно так в")
        lines.append("  проекте однажды родилась зацепка по ETHUSDT (26.5).")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--horizon", action="append", choices=list(rm.HORIZONS))
    parser.add_argument("--norm", action="append", choices=list(rm.NORMALIZATIONS))
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--end", default=FROZEN_END)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument(
        "--augment-benchmark", action="store_true",
        help="ДИАГНОСТИКА (2026-08-24): добавить в бенчмарк логарифм "
             "знаменателя цели. Отвечает, сколько от приращения R² остаётся, "
             "когда бенчмарк знает про короткое окно нормировки столько же, "
             "сколько модель. Штатный замер идёт БЕЗ флага.")
    parser.add_argument("--save", action="store_true",
                        help="сложить обученные модели в DATA_DIR/range_forecast/")
    parser.add_argument("--dump", help="сложить ячейки в JSON (для общего отчёта)")
    parser.add_argument("--report", nargs="+",
                        help="не считать, а собрать общий отчёт из файлов --dump")
    args = parser.parse_args()

    if args.report:
        import json
        rows: list[dict] = []
        for path in args.report:
            part = json.loads(Path(path).read_text(encoding="utf-8"))
            for row in part:
                row["coverage"] = {float(k): v for k, v in row["coverage"].items()}
                row["pinball_gain"] = {float(k): v
                                       for k, v in row["pinball_gain"].items()}
                rows.append(row)
        rows.sort(key=lambda r: (r["symbol"], r["horizon"], r["norm"], r["seed"]))
        print(format_report(rows))
        return 0

    args.horizon = args.horizon or list(rm.HORIZONS)
    args.norm = args.norm or list(rm.NORMALIZATIONS)
    args.seed = args.seed or [42, 1337]

    specs = symbols.resolve_many(args.symbol, args.all)
    print(f"Регрессор размаха. Монет {len(specs)}, горизонты {args.horizon}, "
          f"нормировки {args.norm}, зёрна {args.seed}, граница {args.end}.")
    print("Критерий заявлен ДО запуска — см. шапку скрипта.\n")

    rows: list[dict] = []
    for spec in specs:
        symbol = spec.ticker
        print(f"\n{'=' * 78}\n=== {symbol}\n{'=' * 78}")
        try:
            base, features = prepare(symbol, args.end)
        except SystemExit as exc:
            print(f"  пропущена: {exc}")
            continue
        for horizon in args.horizon:
            for norm in args.norm:
                for seed in args.seed:
                    try:
                        row = run_cell(symbol, base, features, horizon, norm, seed, args)
                    except ValueError as exc:
                        print(f"  [{symbol} {horizon} {norm}] пропущена: {exc}")
                        continue
                    if row:
                        rows.append(row)

    if args.dump:
        import json
        Path(args.dump).write_text(
            json.dumps(rows, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"\nЯчейки сохранены в {args.dump}")
    print(format_report(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
