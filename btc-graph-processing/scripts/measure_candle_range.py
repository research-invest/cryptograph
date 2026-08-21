"""
Задача A ТЗ `crypto-graph/docs/tz_candle_geometry_20-08-26.md`: добавляют ли
range-оценщики дисперсии что-нибудь к HAR-RV на закрытиях плюс сезонность.

    python3 scripts/measure_candle_range.py --all
    python3 scripts/measure_candle_range.py --symbol BTCUSDT --horizon 4h

**Ничего не пишет — ни в БД, ни в конфиг.** `train` не запускается: замер идёт
на барах, признаки и разметка ему не нужны вовсе.

## Что здесь считается

На каждой ячейке (монета × горизонт × нормировка) сравниваются две модели на
ОДНИХ И ТЕХ ЖЕ строках:

    B2                = HAR-RV(96/672/2880) + sin/cos часа + флаг выходных
    B2 + оценщики     = то же плюс девять колонок range_estimator_columns

Проверка — **разрез 70/30 с зазором в горизонт** (`rm.holdout_forward`), а не
walk-forward: ТЗ требует отложенной части, то есть данных, которых не видел ни
один этап подгонки. Значимость — блочный бутстрап по построчной разности
квадратов ошибок, общим кодом (инвариант 11). Поправка BH — по ВСЕМ ячейкам
прогона, поэтому считать монеты в отдельных процессах и сводить `--report`
можно только так же, как у `measure_range_horizons.py`.

## Три вещи, которые легко прочитать неправильно

**Три оценщика — одна гипотеза.** Паркинсон, Гарман–Класс и Роджерс–Сатчелл
построены из одних и тех же H, L, O, C и коррелированы между собой (правило
зеркальных пар, §3.1 `extending_features.md`). Поэтому в критерий входит
приращение ВСЕГО набора девяти колонок, а вклад каждого оценщика по
отдельности печатается только справочно.

**Третья нормировка обязательна.** Цель делится на ATR, ATR построен из
размахов баров, Паркинсон и Гарман–Класс — тоже. Общая конструкция между
предиктором и знаменателем цели способна дать ЛОЖНОПОЛОЖИТЕЛЬНЫЙ результат, и
обе нормировки A3 от неё не защищают. `rv_h` считается по закрытиям и
конструкцию не разделяет; вывод засчитывается только при совпадении знака и
порядка величины на всех трёх (§2.4 ТЗ).

**Избыточность печатается рядом с приращением.** Агрегаты оценщиков
объясняются колонками `har_columns` на 0.93–0.99 (разведка 2026-08-21), то
есть свободного места 1–7%. `ΔR² = +0.011` над B2 на этом фоне — не то же
самое, что `ΔR² = +0.011` от независимого источника, и колонка `изб.` не даёт
об этом забыть.

## Кого здесь нет

**HYPEUSDT не считается** (§1.8 ТЗ, правка 2026-08-21). Его бары собираются
из тиковых архивов Bybit, и `ingest/bybit.py` строит `open` как
`close.shift(1)`: `ln(C/O)` там равен полной close-to-close доходности бара,
поэтому Гарман–Класс и Роджерс–Сатчелл считают на этой монете другую
величину. Это свойство ДАННЫХ, а не результат, и знаменатель критерия —
пять монет. `--include-hype` считает её отдельной строкой ради полноты
картины; в критерий строка всё равно не входит.
"""
from __future__ import annotations

import argparse
import json
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
from btcproc.analysis.lift import DEFAULT_N_BOOT, benjamini_hochberg  # noqa: E402
from btcproc.ingest import bars  # noqa: E402

#: Тот же замороженный датасет, что у Ш0, D1 и раздела 49. Менять — значит
#: потерять сравнимость всех выводов проекта про размах.
FROZEN_END = "2026-08-01"

#: Монета, чей `open` синтетический, — см. шапку.
SYNTHETIC_OPEN = {"HYPEUSDT"}

#: Порог практической величины, объявленный в §2.5 ТЗ ДО прогона и по итогам
#: НЕ пересматриваемый. Вдвое ниже порога 0.02, которым принят регрессор
#: размаха (49.3), и на порядок выше MDE.
GATE_DELTA = 0.010

#: Доля истории на обучение в разрезе (остальное — отложенная часть).
TRAIN_FRAC = 0.7

#: Ниже этого числа строк ячейка не считается: блочный бутстрап на короткой
#: зависимой выборке меряет собственный шум.
MIN_ROWS = 5000

HAR_NAMES = [f"log_rv_{w}" for w in rm.HAR_WINDOWS]
SEASON_NAMES = [f"hour_{trig}{k}" for k in rm.SEASONAL_HARMONICS
                for trig in ("sin", "cos")] + ["is_weekend"]


def estimator_names(estimator: str | None = None) -> list[str]:
    """Колонки набора целиком или одного оценщика."""
    names = rm.RANGE_ESTIMATORS if estimator is None else (estimator,)
    return [f"log_{name}_{w}" for name in names for w in rm.HAR_WINDOWS]


def seeded_rng(*parts) -> np.random.Generator:
    salt = zlib.crc32("|".join(str(p) for p in parts).encode())
    return np.random.default_rng([42, salt])


# ─── Данные ─────────────────────────────────────────────────────────────────
def load_base(symbol: str, end: str) -> pd.DataFrame:
    spec = symbols.get(symbol)
    base = bars.load_ohlcv(symbol, config.data.base_tf, spec.start_date(), end)
    if base.empty:
        raise SystemExit(f"{symbol}: в БД нет баров до {end}. Сначала ingest.")
    return base


def cell_frame(base: pd.DataFrame, horizon: str, normalization: str) -> pd.DataFrame:
    """
    Строки одной ячейки: цель в логарифме, колонки B2 и девять колонок
    оценщиков — уже без пропусков.

    Пропуски режутся один раз и здесь: базовая и полная модель обязаны
    считаться на ОДНИХ И ТЕХ ЖЕ строках, иначе их R² несравнимы.
    """
    h_bars = rm.horizon_bars(horizon, config.data.base_minutes)
    target = rm.log_target(rm.range_target(base, h_bars, normalization))
    frame = pd.concat(
        [
            target.rename("target"),
            rm.har_columns(base["close"]),
            rm.seasonal_columns(base.index),
            rm.range_estimator_columns(base),
        ],
        axis=1,
    ).dropna()
    frame.attrs["h_bars"] = h_bars
    return frame


def gap_check(base: pd.DataFrame) -> dict:
    """
    Гейт §2.2 ТЗ: величина разрыва между барами.

    Печатается ВЕЛИЧИНА, а не доля неравенства. Доля у 15-минутных баров
    круглосуточной биржи бесполезна — она 53–67% и означает «последняя сделка
    бара и первая сделка следующего прошли по разным ценам», то есть один тик.
    Решает, вырождается ли овернайт-компонента Yang–Zhang, только величина.
    """
    span = (base["high"] - base["low"]).replace(0.0, np.nan)
    gap = (base["open"] - base["close"].shift(1)).abs() / span
    gap = gap.dropna()
    share = float((base["open"] != base["close"].shift(1)).mean())
    flat = int((base["high"] == base["low"]).sum())
    return {
        "share_unequal": share,
        "gap_p50": float(gap.median()) if len(gap) else float("nan"),
        "gap_p90": float(gap.quantile(0.9)) if len(gap) else float("nan"),
        "gap_p99": float(gap.quantile(0.99)) if len(gap) else float("nan"),
        "flat_bars": flat,
        "flat_share": flat / max(len(base), 1),
    }


def redundancy_to_har(frame: pd.DataFrame) -> float:
    """
    Максимальный in-sample R² колонки оценщика на колонках HAR.

    Не гейт и не критерий — знаменатель для чтения приращения. Максимум, а не
    среднее: если хоть одна колонка набора почти целиком пересказывает HAR,
    приращение всего набора надо читать с этой поправкой.
    """
    har = np.column_stack([np.ones(len(frame)), frame[HAR_NAMES].to_numpy(float)])
    best = 0.0
    for name in estimator_names():
        y = frame[name].to_numpy(float)
        beta, *_ = np.linalg.lstsq(har, y, rcond=None)
        resid = y - har @ beta
        variance = float(y.var())
        if variance > 0:
            best = max(best, 1.0 - float(resid.var()) / variance)
    return best


# ─── Замер ──────────────────────────────────────────────────────────────────
def measure_cell(symbol: str, base: pd.DataFrame, horizon: str, norm: str,
                 args) -> dict | None:
    frame = cell_frame(base, horizon, norm)
    if len(frame) < MIN_ROWS:
        print(f"  [{symbol} {horizon} {norm}] строк {len(frame)} — мало, пропущена")
        return None

    target = frame["target"].to_numpy(dtype=float)
    ts = frame.index.to_numpy()
    gap = frame.attrs["h_bars"]
    block = rm.bootstrap_block(frame.index.to_series(), horizon)
    benchmark = frame[HAR_NAMES + SEASON_NAMES].to_numpy(dtype=float)

    def fit(columns: np.ndarray):
        return rm.holdout_forward(
            len(frame), target, ts, gap,
            rm.ols_predictor(columns, target), frac=TRAIN_FRAC,
        )

    base_fit = fit(benchmark)
    full = np.column_stack([benchmark, frame[estimator_names()].to_numpy(float)])
    full_fit = fit(full)
    gain = rm.compare(base_fit, full_fit, block, args.n_boot,
                      seeded_rng(symbol, horizon, norm, "set"))

    # Справочно, в критерий не входит: вклад каждого оценщика по отдельности.
    single: dict[str, float] = {}
    for name in rm.RANGE_ESTIMATORS:
        columns = np.column_stack(
            [benchmark, frame[estimator_names(name)].to_numpy(float)])
        one = rm.compare(base_fit, fit(columns), block, args.n_boot,
                         seeded_rng(symbol, horizon, norm, name))
        single[name] = one.delta

    row = {
        "symbol": symbol, "horizon": horizon, "norm": norm,
        "n": len(frame), "block": block, "n_eff": len(frame) // block,
        "n_holdout": base_fit.n,
        "r2_b2": base_fit.r2, "r2_full": full_fit.r2,
        "delta": gain.delta, "p": gain.p_value,
        "redundancy": redundancy_to_har(frame),
        "synthetic_open": symbol in SYNTHETIC_OPEN,
        **{f"delta_{name}": value for name, value in single.items()},
    }
    print(f"  [{symbol} {horizon} {norm}] n={len(frame)} (holdout {base_fit.n}) "
          f"блок={block} R²(B2)={base_fit.r2:+.4f} +оценщики={full_fit.r2:+.4f} "
          f"Δ={gain.delta:+.4f} p={gain.p_value:.4f} изб.={row['redundancy']:.3f}")
    return row


# ─── Отчёт ──────────────────────────────────────────────────────────────────
def format_gaps(rows: list[dict]) -> str:
    header = (f"{'монета':<10} {'баров':>8} {'≠close[-1]':>11} {'p50':>8} "
              f"{'p90':>8} {'p99':>8} {'H==L':>6} {'доля':>8}")
    lines = ["", "ГЕЙТ ДАННЫХ (§2.2 и §3.2 ТЗ)",
             "Разрыв — |open − close[-1]| в долях размаха бара. Решает величина,",
             "а не доля неравенства: у круглосуточной биржи она всегда велика.", "",
             header, "─" * len(header)]
    for r in rows:
        lines.append(
            f"{r['symbol']:<10} {r['bars']:>8} {r['share_unequal']:>10.1%} "
            f"{r['gap_p50']:>8.4f} {r['gap_p90']:>8.4f} {r['gap_p99']:>8.4f} "
            f"{r['flat_bars']:>6} {r['flat_share']:>8.4%}"
        )
    lines += [
        "",
        "Медиана заметно ниже 0.05 размаха бара → овернайт-компонента "
        "Yang–Zhang вырождается,",
        "и §1.6 ТЗ (не считать Yang–Zhang) остаётся в силе по существу.",
    ]
    return "\n".join(lines)


def format_cells(rows: list[dict]) -> str:
    scored = [r for r in rows if not r["synthetic_open"]]
    marks = dict(zip(
        [id(r) for r in scored],
        benjamini_hochberg([r["p"] for r in scored]),
    ))
    header = (f"{'монета':<10} {'гор.':>5} {'норм.':>6} {'n':>7} {'блок':>5} "
              f"{'n_эфф':>7} {'R²(B2)':>8} {'R²(+оц)':>8} {'ΔR²':>8} {'p':>7} "
              f"{'BH':>4} {'изб.':>6} {'порог':>6}")
    lines = ["", "ЗАДАЧА A: RANGE-ОЦЕНЩИКИ СВЕРХ БЕНЧМАРКА B2",
             "Отложенная часть 70/30 с зазором в горизонт. Цель — log(range_ratio).",
             f"Порог практической величины ΔR² ≥ {GATE_DELTA:.3f}, объявлен ДО прогона.",
             "«изб.» — максимальный in-sample R² колонки оценщика на колонках HAR:",
             "во сколько предиктор уже содержится в бенчмарке.", "",
             header, "─" * len(header)]
    for r in rows:
        mark = marks.get(id(r))
        bh = "—" if mark is None else ("да" if mark else "нет")
        passed = (mark and r["delta"] >= GATE_DELTA) if mark is not None else False
        lines.append(
            f"{r['symbol']:<10} {r['horizon']:>5} {r['norm']:>6} {r['n']:>7} "
            f"{r['block']:>5} {r['n_eff']:>7} {r['r2_b2']:>+8.4f} "
            f"{r['r2_full']:>+8.4f} {r['delta']:>+8.4f} {r['p']:>7.4f} {bh:>4} "
            f"{r['redundancy']:>6.3f} {'ДА' if passed else 'нет':>6}"
        )
    lines.append("")
    lines.append("Вклад отдельных оценщиков (справочно, в критерий не входит):")
    lines.append(f"{'монета':<10} {'гор.':>5} {'норм.':>6} "
                 f"{'Δ Паркинсон':>12} {'Δ Гарман–Класс':>15} {'Δ Роджерс–С.':>13}")
    for r in rows:
        lines.append(f"{r['symbol']:<10} {r['horizon']:>5} {r['norm']:>6} "
                     f"{r['delta_p']:>+12.4f} {r['delta_gk']:>+15.4f} "
                     f"{r['delta_rs']:>+13.4f}")
    return "\n".join(lines)


def verdict(rows: list[dict], norms: list[str], horizons: list[str]) -> str:
    """
    Критерий §2.5 ТЗ, заявленный ДО прогона и здесь только проверяемый:
    ΔR² ≥ 0.010 при p ≤ 0.05 после BH — на ≥3 монетах из пяти, ≥2 горизонтах
    из трёх и ВСЕХ трёх нормировках.
    """
    scored = [r for r in rows if not r["synthetic_open"]]
    if not scored:
        return "\nВЕРДИКТ: считать нечего."
    marks = benjamini_hochberg([r["p"] for r in scored])
    passed = [r for r, mark in zip(scored, marks)
              if mark and r["delta"] >= GATE_DELTA]

    by_norm = {norm: {r["symbol"] for r in passed if r["norm"] == norm}
               for norm in norms}
    symbols_all = {r["symbol"] for r in passed}
    horizons_all = {r["horizon"] for r in passed}

    ok_norms = all(len(by_norm.get(norm, set())) >= 3 for norm in norms)
    ok_symbols = len(symbols_all) >= 3
    ok_horizons = len(horizons_all) >= 2
    verdict_ok = ok_norms and ok_symbols and ok_horizons

    lines = ["", "=" * 78, "ВЕРДИКТ ЗАДАЧИ A (критерий §2.5 ТЗ, заявлен до прогона)", "=" * 78,
             f"  ячеек в зачёте {len(scored)} (HYPEUSDT исключён по §1.8), "
             f"прошло {len(passed)}",
             f"  монет с прохождением: {len(symbols_all)} из 5 — "
             f"{'да' if ok_symbols else 'НЕТ'} ({', '.join(sorted(symbols_all)) or '—'})",
             f"  горизонтов: {len(horizons_all)} из {len(horizons)} — "
             f"{'да' if ok_horizons else 'НЕТ'} ({', '.join(sorted(horizons_all)) or '—'})"]
    for norm in norms:
        got = by_norm.get(norm, set())
        lines.append(f"  нормировка {norm}: монет {len(got)} — "
                     f"{'да' if len(got) >= 3 else 'НЕТ'} ({', '.join(sorted(got)) or '—'})")
    lines.append("")
    lines.append(f"  ИТОГ: критерий {'ПРОЙДЕН' if verdict_ok else 'НЕ ПРОЙДЕН'}")
    if not verdict_ok:
        lines += [
            "",
            "  Отрицательный результат записывается в журнал так же подробно, как",
            "  записался бы положительный, и закрывает половину задачи B заранее",
            "  (§2.5 ТЗ): если агрегированная геометрия бара не добавляет к",
            "  волатильности ничего, от внутрибарных долей на том же горизонте",
            "  ждать нечего.",
        ]
    return "\n".join(lines)


def report(collected: dict) -> None:
    if collected.get("gaps"):
        print(format_gaps(collected["gaps"]))
    if collected.get("cells"):
        rows = collected["cells"]
        print(format_cells(rows))
        print(verdict(rows,
                      sorted({r["norm"] for r in rows}),
                      sorted({r["horizon"] for r in rows})))


def _json_scalar(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def merge_dumps(paths: list[str]) -> dict:
    merged: dict[str, list[dict]] = {}
    for path in paths:
        chunk = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, rows in chunk.items():
            merged.setdefault(key, []).extend(rows)
    for rows in merged.values():
        rows.sort(key=lambda r: (r.get("symbol", ""), r.get("horizon", ""),
                                 r.get("norm", "")))
    return merged


# ─── Точка входа ────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Range-оценщики дисперсии сверх бенчмарка размаха "
                    "(задача A ТЗ по геометрии свечи)")
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--horizon", action="append", choices=list(rm.HORIZONS))
    parser.add_argument("--norm", action="append",
                        choices=list(rm.NORMALIZATIONS) + list(rm.EXTRA_NORMALIZATIONS),
                        help="по умолчанию все три (§2.4 ТЗ): atr14, atr_h, rv_h")
    parser.add_argument("--include-hype", action="store_true",
                        help="считать HYPEUSDT отдельной строкой (в критерий "
                             "не входит, §1.8 ТЗ)")
    parser.add_argument("--end", default=FROZEN_END,
                        help="замороженная граница; менять нельзя без потери "
                             "сравнимости с разделами 26, 31, 47 и 49")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--dump", help="сложить посчитанные ячейки в файл JSON")
    parser.add_argument("--report", nargs="+",
                        help="не считать, а собрать общий отчёт из файлов --dump")
    args = parser.parse_args()

    if args.report:
        report(merge_dumps(args.report))
        return 0

    args.horizon = args.horizon or list(rm.HORIZONS)
    args.norm = args.norm or list(rm.NORMALIZATIONS) + list(rm.EXTRA_NORMALIZATIONS)

    specs = symbols.resolve_many(args.symbol, args.all)
    if not args.include_hype:
        skipped = [s.ticker for s in specs if s.ticker in SYNTHETIC_OPEN]
        specs = [s for s in specs if s.ticker not in SYNTHETIC_OPEN]
        for ticker in skipped:
            print(f"{ticker} пропущена: у неё синтетический open (§1.8 ТЗ). "
                  f"Посчитать отдельной строкой — --include-hype.")

    print(f"Задача A: range-оценщики. Монет {len(specs)}, горизонты "
          f"{args.horizon}, нормировки {args.norm}, граница {args.end}.")
    print("Направление здесь не меряется вовсе: оно закрыто разделом 26.4.\n")

    collected: dict[str, list[dict]] = {"gaps": [], "cells": []}
    for spec in specs:
        symbol = spec.ticker
        print(f"\n{'=' * 78}\n=== {symbol}\n{'=' * 78}")
        try:
            base = load_base(symbol, args.end)
        except SystemExit as exc:
            print(f"  пропущена: {exc}")
            continue
        print(f"  баров {len(base)} "
              f"({base.index[0]:%Y-%m-%d}…{base.index[-1]:%Y-%m-%d})")
        collected["gaps"].append({"symbol": symbol, "bars": len(base),
                                  **gap_check(base)})
        for horizon in args.horizon:
            for norm in args.norm:
                row = measure_cell(symbol, base, horizon, norm, args)
                if row is not None:
                    collected["cells"].append(row)

    if args.dump:
        Path(args.dump).write_text(
            json.dumps(collected, ensure_ascii=False, default=_json_scalar),
            encoding="utf-8")
        print(f"\nЯчейки сохранены в {args.dump}")
    report(collected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
