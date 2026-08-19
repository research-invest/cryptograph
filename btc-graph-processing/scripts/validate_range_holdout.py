"""
C3: проверка размаха на отложенной части — предсказывает ли КОНФИГУРАЦИЯ
величину `range_ratio` лучше тривиального бенчмарка.

    python3 scripts/validate_range_holdout.py --all
    python3 scripts/validate_range_holdout.py --symbol BTCUSDT --horizon 4h

Задача D `crypto-graph/docs/tz_range_horizons_19-08-26.md` §5, третий пункт
блока C аудита (§C3). **Из задачи D сделана только измерительная половина**:
поля размаха в кандидате не заводятся и схема контракта с btc-graph не
трогается — обоснование в разделе 47.10 журнала.

Ничего не пишет — ни в БД, ни в конфиг. `train` не запускается: история
размечается ГОТОВОЙ моделью через `analysis/replay.label_history`.

## Что именно меряется

То самое, что делал бы продукт, если бы его собрали: `Accumulator` копит
распределение `range_ratio` по ключу конфигурации, кандидат получает
`expected_range_ratio_p50`. Здесь это считается на лету и проверяется честно,
до того как поле заведено.

* **выборка конфигурации — только из ПРЕФИКСА**, то есть из баров до границы
  модели. Иначе перцентиль конфигурации содержит будущее, и число получится
  впечатляющим и ложным;
* **граница — конец обучения самой модели состояний**, а не свежий пересчёт
  70/30. Модели Ш0 (`kind='holdout'`) обучены до своей даты, и всё после неё
  кластеризация не видела. Взять более позднюю границу значило бы отдать
  замеру часть данных, на которых модель уже стояла;
* **ключ конфигурации — тот же, что у кандидата**: `(transition_id,
  event_block_id)` с откатом на `transition_id`, с тем же порогом
  `CAND_MIN_SAMPLE_SIZE`. Конфигурация, по которой выборки не набралось,
  кандидата не даёт — и здесь тоже не участвует;
* **бенчмарк — B2** (HAR-RV + час дня), обученный на том же префиксе. Не
  «скользящая rv», как предлагал аудит: раздел 47 показал, что без часа дня
  бенчмарк бенчмарком не является — на 4h сезонность объясняет втрое больше,
  чем вся волатильностная кластеризация;
* **два зерна кластеризации** — модели Ш0 существуют парами
  (`states_overrides.random_state = 1337` у второй), и обе гоняются одинаково.

## Критерий, заявленный ДО запуска

> Конфигурация предсказывает размах, если Spearman(`expected_range_ratio_p50`,
> фактический `range_ratio`) значимо выше нуля по блочному бутстрапу **И**
> парное сравнение с бенчмарком B2 значимо положительно — на двух монетах из
> трёх и на обоих зёрнах кластеризации.

Порог значимости 0.05, блочный бутстрап общим кодом (`lift.block_length_rows`,
`range_lift.spearman_block_p`, `range_lift.paired_diff_p`), инвариант 11.

**Порога практической величины здесь нет намеренно, и это отличие от задач
B и C ТЗ.** Там сравнивались приращения R², и значимость доставалась почти
даром при n_эфф в тысячи. Здесь вопрос бинарный и предварительный: обгоняет
ли конфигурация бенчмарк вообще. Если обгонит — размер эффекта станет
следующим вопросом, и вот тогда порог придётся объявлять.

Калибровка печатается рядом и в критерий не входит: доля фактических исходов
ниже предсказанных p25/p50/p75/p90 против номинала. Идеал — 0.25/0.50/0.75/0.90.
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
from btcproc.analysis import range_model as rm  # noqa: E402
from btcproc.analysis import replay  # noqa: E402
from btcproc.analysis.lift import DEFAULT_N_BOOT, block_length_rows  # noqa: E402
from btcproc.analysis.range_lift import paired_diff_p, spearman_block_p  # noqa: E402
from btcproc.db import runs as runs_repo  # noqa: E402
from btcproc.db.session import fetch_all  # noqa: E402

FROZEN_END = "2026-08-01"

#: Перцентили распределения конфигурации — те же, что перечислял блок C аудита.
PERCENTILES = (25, 50, 75, 90)


# ─── Модели и граница ───────────────────────────────────────────────────────
def holdout_models(symbol: str) -> list[tuple[str, int, pd.Timestamp]]:
    """
    Модели, обученные на префиксе истории, и граница их знания.

    Берутся прогоны `kind='holdout'` — те, что оставила валидация Ш0
    (раздел 26). Их две на монету, и различаются они ровно зерном
    кластеризации; это и есть требуемое критерием второе зерно, полученное без
    единого переобучения.

    Граница — `params.end` прогона: до неё модель видела данные, после — нет.
    Прогон `train` сюда не годится ни при каких условиях: он обучен на ВСЕЙ
    истории, и «отложенная часть» для него не отложена.
    """
    rows = fetch_all(
        "SELECT r.run_id, r.params FROM runs r JOIN state_models m ON m.run_id = r.run_id "
        "WHERE r.symbol = %s AND r.status = 'done' AND r.kind = 'holdout' "
        "ORDER BY r.run_id",
        (symbol,),
    )
    result = []
    for row in rows:
        params = row["params"] or {}
        end = params.get("end")
        if not end:
            continue
        overrides = params.get("states_overrides") or {}
        seed = overrides.get("random_state", "по умолчанию")
        result.append((f"зерно {seed}", int(row["run_id"]),
                       pd.Timestamp(end).tz_convert("UTC")))
    return result


# ─── Выборка конфигурации ───────────────────────────────────────────────────
def configuration_percentiles(prefix: pd.DataFrame, holdout: pd.DataFrame,
                              min_sample: int) -> pd.DataFrame:
    """
    Перцентили `range_ratio` по ключу конфигурации, посчитанные ПО ПРЕФИКСУ
    и приписанные строкам отложенной части.

    Ключ и порядок отката — буквально как в `candidates/builder.generate`:
    сначала пара «переход + блок событий», при нехватке выборки — переход
    целиком. Строка, для которой не набралось ни того ни другого, выпадает:
    кандидата по ней система тоже не выпустила бы.
    """
    def percentiles_by(keys: pd.Series) -> dict:
        grouped = prefix.groupby(keys)["range_ratio"]
        table = grouped.quantile([p / 100 for p in PERCENTILES]).unstack()
        table.columns = [f"p{p}" for p in PERCENTILES]
        table["n"] = grouped.size()
        return table[table["n"] >= min_sample].to_dict("index")

    full_key = prefix["transition_id"].astype(str) + "|" + prefix["event_block_id"].astype(str)
    by_full = percentiles_by(full_key)
    by_transition = percentiles_by(prefix["transition_id"].astype(str))

    rows = []
    for row in holdout.itertuples():
        key = f"{row.transition_id}|{row.event_block_id}"
        stats = by_full.get(key)
        scope = "transition+event_block"
        if stats is None:
            stats = by_transition.get(str(row.transition_id))
            scope = "transition"
        if stats is None:
            continue
        rows.append({
            "ts": row.ts, "scope": scope, "sample": int(stats["n"]),
            "actual": row.range_ratio,
            **{f"p{p}": stats[f"p{p}"] for p in PERCENTILES},
        })
    return pd.DataFrame(rows)


# ─── Бенчмарк B2 ────────────────────────────────────────────────────────────
def benchmark_prediction(base: pd.DataFrame, horizon: str, normalization: str,
                         split_ts: pd.Timestamp, at: pd.Series) -> pd.Series:
    """
    Прогноз B2 (HAR-RV + час дня), обученный на префиксе, в точках `at`.

    Тот же стек, что в стенде раздела 47, и обучается он на тех же строках,
    что и выборка конфигурации: сравнение честно только при одинаковом
    доступе к прошлому у обеих сторон.
    """
    h_bars = config.data.bars_of_horizon(horizon)
    target = rm.log_target(rm.range_target(base, h_bars, normalization))
    columns = pd.concat(
        [rm.har_columns(base["close"]), rm.seasonal_columns(base.index)], axis=1
    )
    frame = pd.concat([target.rename("target"), columns], axis=1).dropna()
    train = frame[frame.index < split_ts]
    if len(train) < 1000:
        raise SystemExit("на префиксе меньше тысячи строк — бенчмарк не обучить")

    design = np.column_stack([np.ones(len(train)), train[columns.columns].to_numpy(float)])
    beta, *_ = np.linalg.lstsq(design, train["target"].to_numpy(float), rcond=None)

    wanted = frame.reindex(pd.DatetimeIndex(at)).dropna()
    if wanted.empty:
        return pd.Series(dtype=float)
    predicted = np.column_stack(
        [np.ones(len(wanted)), wanted[columns.columns].to_numpy(float)]
    ) @ beta
    return pd.Series(predicted, index=wanted.index)


# ─── Замер ──────────────────────────────────────────────────────────────────
def measure_one(symbol: str, label: str, model_run: int, split_ts: pd.Timestamp,
                args) -> list[dict]:
    played = replay.label_history(symbol, end=args.end, model_run=model_run,
                                  log=lambda msg: print(f"  {msg}"))
    snapshots = played.snapshots
    if snapshots.empty:
        print(f"  [{symbol} {label}] снимков нет")
        return []

    results = []
    for horizon in args.horizon:
        h_bars = config.data.bars_of_horizon(horizon)
        for norm in args.norm:
            target = rm.range_target(played.bars, h_bars, norm)
            frame = snapshots[["ts", "transition_id", "event_block_id"]].copy()
            frame["range_ratio"] = target.reindex(frame["ts"]).to_numpy()
            frame = frame.dropna(subset=["range_ratio"])

            prefix = frame[frame["ts"] < split_ts]
            holdout = frame[frame["ts"] >= split_ts]
            if len(prefix) < 500 or len(holdout) < 200:
                print(f"  [{symbol} {label} {horizon} {norm}] префикс {len(prefix)}, "
                      f"отложенных {len(holdout)} — мало")
                continue

            table = configuration_percentiles(
                prefix, holdout, config.candidates.min_sample_size
            )
            if table.empty or len(table) < 200:
                print(f"  [{symbol} {label} {horizon} {norm}] выборка конфигураций "
                      f"не набралась")
                continue

            benchmark = benchmark_prediction(played.bars, horizon, norm, split_ts,
                                             table["ts"])
            table = table[table["ts"].isin(benchmark.index)].reset_index(drop=True)
            benchmark = benchmark.reindex(pd.DatetimeIndex(table["ts"])).to_numpy()

            actual = table["actual"].to_numpy(dtype=float)
            predicted = table["p50"].to_numpy(dtype=float)
            block = block_length_rows(table["ts"], rm.horizon_minutes(horizon))
            rng = np.random.default_rng([42, zlib.crc32(
                f"{symbol}|{model_run}|{horizon}|{norm}".encode())])

            rho, p_own = spearman_block_p(predicted, actual, block, args.n_boot, rng)
            rho_cfg, rho_bench, p_better = paired_diff_p(
                predicted, benchmark, actual, block, args.n_boot, rng
            )
            coverage = {
                f"p{p}": float((actual <= table[f"p{p}"].to_numpy(float)).mean())
                for p in PERCENTILES
            }
            ece = float(np.mean([
                abs(coverage[f"p{p}"] - p / 100) for p in PERCENTILES
            ]))

            results.append({
                "symbol": symbol, "model": label, "run_id": model_run,
                "horizon": horizon, "norm": norm, "split": split_ts,
                "n": len(table), "block": block,
                "rho": rho, "p_own": p_own,
                "rho_bench": rho_bench, "p_better": p_better,
                "coverage": coverage, "ece": ece,
                "scope_full": float((table["scope"] == "transition+event_block").mean()),
            })
            print(f"  [{symbol} {label} {horizon} {norm}] n={len(table)} блок={block} "
                  f"rho={rho:+.4f} (p={p_own:.4f}) против B2 {rho_bench:+.4f} "
                  f"(p={p_better:.4f}) ECE={ece:.3f}")
    return results


# ─── Печать ─────────────────────────────────────────────────────────────────
def format_report(rows: list[dict]) -> str:
    if not rows:
        return "\nНи одной ячейки не посчитано."
    header = (f"{'монета':<10} {'модель':<18} {'гор.':>5} {'норм.':>6} {'n':>7} "
              f"{'блок':>5} {'rho(конф.)':>11} {'p':>8} {'rho(B2)':>9} "
              f"{'p(лучше)':>9} {'ECE':>6}")
    lines = ["", "=" * len(header),
             "C3: РАЗМАХ НА ОТЛОЖЕННОЙ ЧАСТИ — КОНФИГУРАЦИЯ ПРОТИВ БЕНЧМАРКА B2",
             "=" * len(header),
             "rho(конф.) — Spearman перцентиля p50 конфигурации с фактическим",
             "range_ratio. rho(B2) — то же для бенчмарка «HAR-RV + час дня»,",
             "обученного на том же префиксе. p(лучше) — односторонний блочный тест",
             "«конфигурация сильнее бенчмарка».", "",
             header, "─" * len(header)]
    for r in rows:
        lines.append(
            f"{r['symbol']:<10} {r['model']:<18} {r['horizon']:>5} {r['norm']:>6} "
            f"{r['n']:>7} {r['block']:>5} {r['rho']:>+11.4f} {r['p_own']:>8.4f} "
            f"{r['rho_bench']:>+9.4f} {r['p_better']:>9.4f} {r['ece']:>6.3f}"
        )

    lines.append("")
    lines.append("КАЛИБРОВКА ПЕРЦЕНТИЛЕЙ (доля фактических ниже предсказанного)")
    lines.append(f"  {'монета':<10} {'модель':<18} {'гор.':>5} {'норм.':>6} "
                 + "  ".join(f"p{p}→" for p in PERCENTILES))
    for r in rows:
        cov = "  ".join(f"{r['coverage'][f'p{p}']:.2f}" for p in PERCENTILES)
        lines.append(f"  {r['symbol']:<10} {r['model']:<18} {r['horizon']:>5} "
                     f"{r['norm']:>6} {cov}")
    lines.append(f"  идеал: {'  '.join(f'{p/100:.2f}' for p in PERCENTILES)}")

    lines.append("")
    lines.append("ВЕРДИКТ ПО ЗАЯВЛЕННОМУ КРИТЕРИЮ")
    passed_cells = [r for r in rows if r["p_own"] <= 0.05 and r["rho"] > 0
                    and r["p_better"] <= 0.05]
    by_symbol: dict[str, set] = {}
    for r in passed_cells:
        by_symbol.setdefault(r["symbol"], set()).add(r["model"])
    seeds_in_run = {r["model"] for r in rows}
    good = [s for s, models in by_symbol.items() if models == seeds_in_run]
    lines.append(f"  ячеек прошло обе половины: {len(passed_cells)} из {len(rows)}")
    lines.append(f"  монет, прошедших на ВСЕХ зёрнах ({len(seeds_in_run)}): {len(good)}")
    verdict = ("КОНФИГУРАЦИЯ ПРЕДСКАЗЫВАЕТ РАЗМАХ" if len(good) >= 2
               else "КРИТЕРИЙ НЕ ПРОЙДЕН")
    lines.append(f"  ВЕРДИКТ: {verdict}")
    if len(good) < 2:
        lines.append("")
        lines.append("  Поля размаха в кандидате не заводятся. Промежуточные")
        lines.append("  формулировки запрещены ТЗ: «на одной монете сработало» — это")
        lines.append("  ровно та зацепка по ETHUSDT, которая не повторилась (26.5).")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="C3: размах на отложенной части")
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--horizon", action="append", choices=list(rm.HORIZONS))
    parser.add_argument("--norm", action="append", choices=list(rm.NORMALIZATIONS))
    parser.add_argument("--end", default=FROZEN_END)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    args = parser.parse_args()

    args.horizon = args.horizon or [config.data.horizon]
    args.norm = args.norm or list(rm.NORMALIZATIONS)

    specs = symbols.resolve_many(args.symbol, args.all)
    print(f"C3: размах на отложенной части. Монет {len(specs)}, "
          f"горизонты {args.horizon}, нормировки {args.norm}, граница {args.end}.")
    print("Критерий заявлен ДО запуска — см. шапку скрипта. "
          "train не запускается: разметка готовой моделью.\n")

    rows: list[dict] = []
    for spec in specs:
        symbol = spec.ticker
        models = holdout_models(symbol)
        if not models:
            print(f"\n=== {symbol}: моделей на префиксе истории нет "
                  f"(нужен scripts/validate_holdout.py) — пропущена")
            continue
        print(f"\n{'=' * 78}\n=== {symbol}: моделей {len(models)}\n{'=' * 78}")
        for label, run_id, split_ts in models:
            print(f"  модель #{run_id} ({label}), граница {split_ts:%Y-%m-%d %H:%M}")
            try:
                rows.extend(measure_one(symbol, label, run_id, split_ts, args))
            except (replay.ReplayError, SystemExit) as exc:
                print(f"  пропущена: {exc}")
    print(format_report(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
