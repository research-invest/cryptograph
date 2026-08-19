"""
Стенд «размах × горизонт × бенчмарк» — задачи A, B, C ТЗ
`crypto-graph/docs/tz_range_horizons_19-08-26.md`.

    python3 scripts/measure_range_horizons.py --all
    python3 scripts/measure_range_horizons.py --symbol BTCUSDT --horizon 4h --horizon 24h
    python3 scripts/measure_range_horizons.py --all --task deriv
    python3 scripts/measure_range_horizons.py --all --task features --task states

Тяжёлые задачи считаются по монете в отдельном процессе, а отчёт собирается
одним вызовом — поправка BH обязана считаться по ВСЕМ ячейкам прогона, а не
помонетно (иначе она мягче заявленной):

    for S in BTCUSDT ETHUSDT …; do
      python3 scripts/measure_range_horizons.py --symbol $S --task deriv \
        --dump /tmp/deriv_$S.json &
    done; wait
    python3 scripts/measure_range_horizons.py --report /tmp/deriv_*.json

**Ничего не пишет — ни в БД, ни в конфиг** (образец: `measure_deriv_range.py`).
`train` не запускается ни для чего: задачи A–C идут на барах и на ГОТОВОЙ
разметке `bar_states`. Перенумерация `group_id` снесла бы граф монеты в Neo4j
(инвариант 13, раздел 25 журнала).

Методика, стек бенчмарков и три решения, которые легко принять неправильно, —
в шапке `btcproc/analysis/range_model.py`. Читать её обязательно: без неё числа
ниже читаются не в той шкале.

## Четыре задачи одного стенда

* `--task bench` (по умолчанию) — таблица R² трёх уровней бенчмарка на всех
  монетах и горизонтах плюс вклад сезонности отдельным числом. Это же ответ
  задачи C3: насколько размах описывается ОДНИМ временем суток и днём недели.
  Роль C3 — не найти эффект, а дать честную формулировку тривиального
  результата, чтобы он не был потом выдан за находку графа.
* `--task deriv` — задача B: перепроверка гейта R деривативов с бенчмарком B2
  вместо одиночной `rv`. Регрессия к разделу 36 и первоочередная задача ТЗ.
* `--task features` — задача C1: контрольная модель на размахе, прямой аналог
  D1. Если признаки ничего не добавляют, вопрос про граф не имеет смысла.
* `--task states` — задача C2: готовая разметка (`group_id`, `event_block_id`)
  как предиктор поверх B2, целевым кодированием ВНУТРИ фолда.

Критерии обеих задач заявлены в ТЗ ДО прогона и здесь только проверяются:
гейт R — `ΔR² ≥ 0.005` при `p ≤ 0.05` после BH, на ≥2 эпохах и ≥2 монетах, на
обеих нормировках; «система добавляет к тривиальному» — приращение OOS R² над
B2 `≥ 0.02` при `p ≤ 0.05` после BH, на ≥2 монетах, ≥2 горизонтах, обоих
зёрнах и обеих нормировках. Пороги практической величины здесь не украшение:
при n_эфф в тысячи значимость достигается на ничтожных числах (урок 36).
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

FROZEN_END = "2026-08-01"

#: Эпохи задачи B — те же и с теми же границами, что в measure_deriv_range.py.
#: Менять их нельзя: иначе перепроверка сравнивала бы себя не с разделом 36.
EPOCHS = [
    ("2018-02..2020-12", "2018-02-01", "2020-12-31"),
    ("2021-01..2022-12", "2021-01-01", "2022-12-31"),
    ("2023-01..2026-08", "2023-01-01", None),
]

#: Пороги практической величины, объявленные в ТЗ и не пересматриваемые.
GATE_R_DELTA = 0.005
SYSTEM_ADDS_DELTA = 0.02


# ─── Данные ─────────────────────────────────────────────────────────────────
def load_base(symbol: str, end: str) -> pd.DataFrame:
    spec = symbols.get(symbol)
    base = bars.load_ohlcv(symbol, config.data.base_tf, spec.start_date(), end)
    if base.empty:
        raise SystemExit(f"{symbol}: в БД нет баров до {end}. Сначала ingest.")
    return base


def cell_frame(base: pd.DataFrame, horizon: str, normalization: str,
               extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Строки одной ячейки замера: цель в логарифме, колонки B1/B2 и (при
    необходимости) дополнительные предикторы — уже без пропусков.

    Пропуски режутся один раз и здесь, а не в каждой модели: базовая и полная
    модель обязаны считаться на ОДНИХ И ТЕХ ЖЕ строках, иначе их R²
    несравнимы, и `FoldPredictions.assert_aligned` этого не поймает — длины
    совпадут, а строки будут разные.
    """
    h_bars = rm.horizon_bars(horizon, config.data.base_minutes)
    target = rm.log_target(rm.range_target(base, h_bars, normalization))

    parts = [
        target.rename("target"),
        rm.har_columns(base["close"]),
        rm.seasonal_columns(base.index),
        rm.weekday_columns(base.index),
    ]
    if extra is not None and not extra.empty:
        parts.append(extra.reindex(base.index))
    frame = pd.concat(parts, axis=1).dropna()
    frame.attrs["h_bars"] = h_bars
    return frame


HAR_NAMES = [f"log_rv_{w}" for w in rm.HAR_WINDOWS]
SEASON_NAMES = [f"hour_{trig}{k}" for k in rm.SEASONAL_HARMONICS
                for trig in ("sin", "cos")] + ["is_weekend"]
DOW_NAMES = [f"dow_{d}" for d in range(1, 7)]


def level_columns(frame: pd.DataFrame, level: str) -> np.ndarray:
    names = {"B0": [], "B1": HAR_NAMES, "B2": HAR_NAMES + SEASON_NAMES,
             "time": SEASON_NAMES + DOW_NAMES}[level]
    if not names:
        return np.zeros((len(frame), 0))
    return frame[names].to_numpy(dtype=float)


def seeded_rng(*parts) -> np.random.Generator:
    salt = zlib.crc32("|".join(str(p) for p in parts).encode())
    return np.random.default_rng([42, salt])


# ─── Задача A + C3: таблица бенчмарков ──────────────────────────────────────
def task_bench(symbol: str, base: pd.DataFrame, args) -> list[dict]:
    rows: list[dict] = []
    for horizon in args.horizon:
        for norm in args.norm:
            frame = cell_frame(base, horizon, norm)
            if len(frame) < 5000:
                print(f"  [{symbol} {horizon} {norm}] строк {len(frame)} — "
                      f"мало, ячейка пропущена")
                continue
            target = frame["target"].to_numpy(dtype=float)
            ts = frame.index.to_numpy()
            gap = frame.attrs["h_bars"]
            block = rm.bootstrap_block(frame.index.to_series(), horizon)

            fitted = {}
            for level in ("B0", "B1", "B2", "time"):
                columns = level_columns(frame, level)
                fitted[level] = rm.walk_forward(
                    len(frame), target, ts, gap, args.folds,
                    rm.ols_predictor(columns, target),
                )

            season = rm.compare(fitted["B1"], fitted["B2"], block, args.n_boot,
                                seeded_rng(symbol, horizon, norm, "season"))
            time_only = rm.compare(fitted["B0"], fitted["time"], block, args.n_boot,
                                   seeded_rng(symbol, horizon, norm, "time"))
            rows.append({
                "symbol": symbol, "horizon": horizon, "norm": norm,
                "n": len(frame), "block": block, "n_eff": len(frame) // block,
                "r2_b0": fitted["B0"].r2, "r2_b1": fitted["B1"].r2,
                "r2_b2": fitted["B2"].r2, "r2_time": fitted["time"].r2,
                "drift": fitted["B0"].drift,
                "season_gain": season.delta, "season_p": season.p_value,
                "time_gain": time_only.delta, "time_p": time_only.p_value,
            })
            print(f"  [{symbol} {horizon} {norm}] n={len(frame)} блок={block} "
                  f"R²(B1)={fitted['B1'].r2:+.4f} R²(B2)={fitted['B2'].r2:+.4f} "
                  f"вклад сезонности {season.delta:+.4f} (p={season.p_value:.4f})")
    return rows


def format_bench(rows: list[dict]) -> str:
    header = (f"{'монета':<10} {'гор.':>5} {'норм.':>6} {'n':>8} {'блок':>5} "
              f"{'n_эфф':>7} {'R²(B1)':>8} {'R²(B2)':>8} {'Δсез':>8} {'p':>8} "
              f"{'BH':>3} {'R²(время)':>10} {'дрейф':>8}")
    lines = ["", "ТАБЛИЦА БЕНЧМАРКОВ (out-of-sample R² величины log(range_ratio))",
             "R²(B0) равен нулю по построению: B0 — это и есть среднее обучающей части.",
             "«дрейф» — R², который даёт ЗНАНИЕ среднего проверочного окна; цена ошибки",
             "«считать R² от среднего test» (ловушка 5 ТЗ).", "",
             header, "─" * len(header)]
    marks = benjamini_hochberg([r["season_p"] for r in rows])
    for r, mark in zip(rows, marks):
        lines.append(
            f"{r['symbol']:<10} {r['horizon']:>5} {r['norm']:>6} {r['n']:>8} "
            f"{r['block']:>5} {r['n_eff']:>7} {r['r2_b1']:>+8.4f} "
            f"{r['r2_b2']:>+8.4f} {r['season_gain']:>+8.4f} {r['season_p']:>8.4f} "
            f"{'да' if mark else 'нет':>3} {r['r2_time']:>+10.4f} {r['drift']:>+8.4f}"
        )
    lines.append("")
    lines.append(f"Тестов в поправке BH: {len(rows)} (вклад сезонности, по ячейкам).")
    lines.append("")
    lines.append("C3 — СЕЗОННОСТЬ КАК САМОСТОЯТЕЛЬНЫЙ ОТВЕТ")
    lines.append("Колонка R²(время) — модель ТОЛЬКО из часа дня (две гармоники) и дня")
    lines.append("недели, без единой рыночной величины. Это тривиальный результат, и")
    lines.append("называть его надо тривиальным: всё, что система найдёт сверх B2,")
    lines.append("обязано сравниваться именно с этим числом, а не с нулём.")
    return "\n".join(lines)


# ─── Задача B: гейт R деривативов сверх B2 ──────────────────────────────────
def task_deriv(symbol: str, base: pd.DataFrame, args) -> list[dict]:
    from btcproc.features import deriv
    from btcproc.ingest import metrics as metrics_ingest

    metrics_frame = metrics_ingest.load_deriv_metrics(symbol, config.data.base_tf)
    if metrics_frame.empty:
        print(f"  [{symbol}] deriv_metrics пуст — монета пропущена")
        return []
    values = deriv.build_deriv(base, metrics_frame, symbol)
    extra = values[list(deriv.FEATURE_CANDIDATES)]

    predictors = args.predictor or list(deriv.FEATURE_CANDIDATES)
    # Автокорреляция каждой величины — один раз на монету, а не на ячейку:
    # перебор лагов дорог, а от горизонта и нормировки не зависит.
    autocorr_rows = {name: rm.predictor_autocorr_rows(extra[name])
                     for name in predictors}
    print("  блок по автокорреляции: "
          + ", ".join(f"{k}={v}" for k, v in autocorr_rows.items()))
    rows: list[dict] = []
    for horizon in args.horizon:
        for norm in args.norm:
            benchmark_frame = cell_frame(base, horizon, norm)
            if benchmark_frame.empty:
                continue
            for name in predictors:
                # Пропуски режутся ПО ОДНОМУ предиктору, а не по всем шести
                # сразу: в measure_deriv_range.py выборка тоже своя у каждой
                # величины, и общий dropna обрезал бы её по самой поздней из
                # шести — числа перестали бы сравниваться с разделом 36.
                frame = benchmark_frame.join(extra[name], how="inner").dropna()
                if frame.empty:
                    continue
                block = rm.bootstrap_block(frame.index.to_series(), horizon,
                                           autocorr_rows[name])
                for epoch, start, end in [("вся история", None, None)] + EPOCHS:
                    window = frame
                    if start:
                        window = window[window.index >= pd.Timestamp(start, tz="UTC")]
                    if end:
                        window = window[window.index <= pd.Timestamp(end, tz="UTC")
                                        + pd.Timedelta(days=1)]
                    if len(window) < 500:
                        continue
                    target = window["target"].to_numpy(dtype=float)
                    predictor = window[name].to_numpy(dtype=float)
                    cell = {"symbol": symbol, "predictor": name, "horizon": horizon,
                            "norm": norm, "epoch": epoch, "n": len(window),
                            "block": block}
                    for level in ("B1", "B2"):
                        rng = seeded_rng(symbol, name, horizon, norm, epoch, level)
                        benchmark = level_columns(window, level)
                        r2_base, r2_full, r_partial, p_gain = rm.partial_r2_gain_matrix(
                            predictor, benchmark, target, block, args.n_boot, rng,
                            rank_benchmark_columns=len(HAR_NAMES),
                        )
                        cell[f"delta_{level}"] = r2_full - r2_base
                        cell[f"p_{level}"] = p_gain
                        cell[f"partial_{level}"] = r_partial
                    eaten = (1.0 - cell["delta_B2"] / cell["delta_B1"]
                             if cell["delta_B1"] > 0 else float("nan"))
                    cell["eaten"] = eaten
                    rows.append(cell)
                    print(f"  [{symbol} {name} {horizon} {norm} {epoch}] "
                          f"ΔR²(B1)={cell['delta_B1']:+.5f} "
                          f"ΔR²(B2)={cell['delta_B2']:+.5f} "
                          f"p(B2)={cell['p_B2']:.4f} съедено={eaten:.0%}")
    return rows


def format_deriv(rows: list[dict]) -> str:
    """Таблицы по величинам плюс вердикт гейта R по заявленному критерию."""
    if not rows:
        return "\nЗадача B: ни одной ячейки не посчитано."
    graded = [r for r in rows if r["epoch"] != "вся история"]
    marks = dict(zip(
        [id(r) for r in graded],
        benjamini_hochberg([r["p_B2"] for r in graded]),
    ))

    lines = ["", "=" * 100,
             "ЗАДАЧА B: ГЕЙТ R ДЕРИВАТИВОВ С СЕЗОННЫМ КОНТРОЛЕМ",
             "=" * 100,
             "ΔR² — приращение частного R² на рангах сверх бенчмарка, in-sample, как",
             "в разделе 36. Менялся только бенчмарк: B1 = HAR-RV, B2 = HAR-RV + час.",
             "«съедено» — какая доля приращения сверх B1 исчезает при добавлении часа.",
             f"Критерий подтверждения (заявлен ДО прогона): ΔR² ≥ {GATE_R_DELTA} при "
             f"p ≤ 0.05 после BH,",
             "на ≥2 эпохах и ≥2 монетах, на обеих нормировках.", ""]

    predictors = sorted({r["predictor"] for r in rows})
    for name in predictors:
        lines.append(f"\n=== {name} ===")
        header = (f"{'монета':<10} {'гор.':>5} {'норм.':>6} {'эпоха':<18} {'n':>7} "
                  f"{'блок':>5} {'ΔR²(B1)':>9} {'ΔR²(B2)':>9} {'p(B2)':>8} {'BH':>3} "
                  f"{'съедено':>8}")
        lines.append(header)
        lines.append("─" * len(header))
        for r in [x for x in rows if x["predictor"] == name]:
            mark = marks.get(id(r))
            eaten = "—" if not np.isfinite(r["eaten"]) else f"{r['eaten']:.0%}"
            lines.append(
                f"{r['symbol']:<10} {r['horizon']:>5} {r['norm']:>6} "
                f"{r['epoch']:<18} {r['n']:>7} {r['block']:>5} "
                f"{r['delta_B1']:>+9.5f} {r['delta_B2']:>+9.5f} {r['p_B2']:>8.4f} "
                f"{('да' if mark else 'нет') if mark is not None else '—':>3} "
                f"{eaten:>8}"
            )

    lines.append("")
    lines.append(f"Тестов в поправке BH: {len(graded)} "
                 f"(ячейки по эпохам; «вся история» в критерий не входит).")
    lines.append("")
    lines.append("ВЕРДИКТ ПО ЗАЯВЛЕННОМУ КРИТЕРИЮ")
    lines.append("Критерий ТЗ сформулирован ДЛЯ ВЕЛИЧИНЫ — и подтверждение, и отзыв "
                 "считаются")
    lines.append("по величине отдельно. Общий котёл из шести величин размыл бы "
                 "именно тот")
    lines.append("случай, ради которого ставилась задача: гейт R раздела 36 прошла "
                 "ОДНА из них.")
    lines.append("")
    # Монета, у которой в данных вообще меньше двух эпох, правилу «≥2 эпох»
    # не подчиняется: у неё нет второй эпохи, а не «не прошла» её. Ловушка 11
    # ТЗ (HYPEUSDT с одной эпохой, TAOUSDT с историей с 2024-04).
    epochs_available: dict[str, set] = {}
    for r in rows:
        if r["epoch"] != "вся история":
            epochs_available.setdefault(r["symbol"], set()).add(r["epoch"])
    structural = sorted(s for s, e in epochs_available.items() if len(e) < 2)

    verdicts: dict[str, str] = {}
    for name in predictors:
        cells = [r for r in rows if r["predictor"] == name
                 and r["epoch"] != "вся история" and r["symbol"] not in structural]
        passed = [r for r in cells
                  if r["delta_B2"] >= GATE_R_DELTA and marks.get(id(r))]
        by_symbol: dict[str, set] = {}
        for r in passed:
            by_symbol.setdefault(r["symbol"], set()).add((r["epoch"], r["norm"]))
        good_symbols = [
            s for s, seen in by_symbol.items()
            if len({e for e, _ in seen}) >= 2 and len({n for _, n in seen}) >= 2
        ]
        eaten_values = [r["eaten"] for r in cells if np.isfinite(r["eaten"])]
        eaten = float(np.median(eaten_values)) if eaten_values else float("nan")

        if len(good_symbols) >= 2:
            verdict = "ПОДТВЕРЖДЁН"
        elif not passed and np.isfinite(eaten) and eaten > 0.5:
            verdict = "ОТОЗВАН"
        else:
            verdict = "не подтверждён и не отозван"
        verdicts[name] = verdict
        eaten_text = "—" if not np.isfinite(eaten) else f"{eaten:.0%}"
        lines.append(f"  {name:<16} ячеек прошло {len(passed):>2}/{len(cells)}, "
                     f"монет с ≥2 эпохами и обеими нормировками: {len(good_symbols)}, "
                     f"медиана съеденного {eaten_text:>4} → {verdict}")

    if structural:
        lines.append("")
        lines.append(f"  Структурно неприменимы (эпох в данных меньше двух): "
                     f"{', '.join(structural)}.")
        lines.append("  Их ячейки посчитаны и напечатаны, но в вердикт не входят: "
                     "у монеты нет")
        lines.append("  второй эпохи, а не «она её не прошла».")

    lines.append("")
    lines.append("ЧТО ЭТО ЗНАЧИТ ДЛЯ РАЗДЕЛА 36")
    key = "oi_chg_1h"
    if verdicts.get(key) == "ОТОЗВАН":
        lines.append(f"Гейт R держался на ОДНОЙ величине — {key}. Здесь она не даёт "
                     f"ни одной")
        lines.append("ячейки сверх практического порога при бенчмарке с часом дня, и "
                     "больше половины")
        lines.append("её эффекта сверх B1 съедает именно сезонность. Оба условия "
                     "отзыва выполнены")
        lines.append("вместе, а значит источник деривативных метрик не имеет ни "
                     "одного пройденного")
        lines.append("содержательного гейта. Контекстные атомы остаются: они на "
                     "гейт R и не опирались.")
    elif verdicts.get(key) == "ПОДТВЕРЖДЁН":
        lines.append(f"{key} подтверждён и при бенчмарке с сезонностью — вывод "
                     f"раздела 36 устоял.")
    else:
        lines.append(f"{key}: критерий подтверждения не пройден, но и условие отзыва "
                     f"выполнено не целиком —")
        lines.append("это нулевой результат без объяснения, и записывать его как "
                     "отзыв нельзя.")
    return "\n".join(lines)


# ─── Задача C1: контрольная модель на размахе ───────────────────────────────
def task_features(symbol: str, base: pd.DataFrame, args) -> list[dict]:
    from btcproc.features import builder as feat

    spec = symbols.get(symbol)
    context = {
        tf: bars.load_ohlcv(symbol, tf, spec.start_date(), args.end)
        for tf in config.data.context_tfs
    }
    print(f"  [{symbol}] признаки…")
    features = feat.build_features(base, context)
    print(f"  [{symbol}] признаков {features.shape[1]}, строк {len(features)}")

    rows: list[dict] = []
    for horizon in args.horizon:
        for norm in args.norm:
            frame = cell_frame(base, horizon, norm, features)
            if len(frame) < 5000:
                continue
            target = frame["target"].to_numpy(dtype=float)
            ts = frame.index.to_numpy()
            gap = frame.attrs["h_bars"]
            block = rm.bootstrap_block(frame.index.to_series(), horizon)
            benchmark = level_columns(frame, "B2")
            full = np.column_stack([benchmark, frame[list(features.columns)].to_numpy(float)])

            for seed in args.seed:
                base_fit = rm.walk_forward(len(frame), target, ts, gap, args.folds,
                                           rm.boosted_predictor(benchmark, target, seed))
                full_fit = rm.walk_forward(len(frame), target, ts, gap, args.folds,
                                           rm.boosted_predictor(full, target, seed))
                gain = rm.compare(base_fit, full_fit, block, args.n_boot,
                                  seeded_rng(symbol, horizon, norm, seed, "features"))
                rows.append({
                    "symbol": symbol, "horizon": horizon, "norm": norm, "seed": seed,
                    "n": len(frame), "block": block,
                    "r2_base": gain.r2_base, "r2_full": gain.r2_full,
                    "delta": gain.delta, "p": gain.p_value,
                    "n_features": features.shape[1],
                })
                print(f"  [{symbol} {horizon} {norm} зерно {seed}] "
                      f"R²(B2 бустингом)={gain.r2_base:+.4f} "
                      f"+признаки={gain.r2_full:+.4f} Δ={gain.delta:+.4f} "
                      f"p={gain.p_value:.4f}")
    return rows


# ─── Задача C2: разметка графа как предиктор ────────────────────────────────
def state_runs(symbol: str) -> list[tuple[str, int]]:
    """
    Прогоны, разметку которых можно взять: штатная модель монеты и все модели
    валидации на отложенной части, если они в базе есть.

    Второе зерно КЛАСТЕРИЗАЦИИ берётся именно отсюда, а не переобучением:
    `train` запускать нельзя (инвариант 13), а прогоны Ш0 отличаются ровно
    зерном (`states_overrides.random_state`) на одном и том же периоде — это и
    есть требуемое ТЗ повторение на втором зерне.
    """
    from btcproc.db.session import fetch_all

    rows = fetch_all(
        "SELECT r.run_id, r.kind, r.params FROM runs r "
        "JOIN state_models m ON m.run_id = r.run_id "
        "WHERE r.symbol = %s AND r.status = 'done' ORDER BY r.run_id",
        (symbol,),
    )
    result = []
    for row in rows:
        overrides = (row["params"] or {}).get("states_overrides") or {}
        seed = overrides.get("random_state", "по умолчанию")
        result.append((f"{row['kind']}#{row['run_id']} зерно {seed}", int(row["run_id"])))
    return result


def load_labels(symbol: str, run_id: int, end: str) -> pd.DataFrame:
    """
    Разметка одной модели: `group_id` из `bar_states` и `event_block_id` из
    `bar_events`.

    Выборка по `model_run_scope`, а не по `run_id`: одна модель живёт в своём
    `train` и во всех `live`, которые её загрузили, и фильтр по одному
    прогону молча обрезал бы разметку свежим периодом (раздел 17).
    """
    from btcproc.db import runs as runs_repo
    from btcproc.db.session import fetch_all

    scope_sql, params = runs_repo.model_run_scope(run_id)
    states = pd.DataFrame(fetch_all(
        f"SELECT ts, group_id FROM bar_states WHERE symbol = %s AND ts <= %s "
        f"AND {scope_sql} ORDER BY ts",
        (symbol, end, *params),
    ))
    events = pd.DataFrame(fetch_all(
        "SELECT ts, event_block_id FROM bar_events WHERE symbol = %s AND ts <= %s "
        "ORDER BY ts",
        (symbol, end),
    ))
    if states.empty:
        return pd.DataFrame()
    states["ts"] = pd.to_datetime(states["ts"], utc=True)
    states = states.drop_duplicates("ts").set_index("ts")
    frame = states[["group_id"]].astype(float)
    # `event_block_id` — СТРОКА («event_block_723621»), а не число: это метка
    # комбинации signature-атомов, и её числовое чтение сломало бы кадр целиком.
    # Отсутствие блока заполняется отдельной меткой, а не NaN: «на этом баре
    # ни одного signature-атома» — полноценная категория, и выбрасывать такие
    # бары значило бы мерить размах только на событийных барах.
    if not events.empty:
        events["ts"] = pd.to_datetime(events["ts"], utc=True)
        events = events.drop_duplicates("ts").set_index("ts")
        frame = frame.join(events[["event_block_id"]], how="left")
    else:
        frame["event_block_id"] = None
    frame["event_block_id"] = frame["event_block_id"].fillna("без блока").astype(str)
    return frame


def task_states(symbol: str, base: pd.DataFrame, args) -> list[dict]:
    runs = state_runs(symbol)
    if not runs:
        print(f"  [{symbol}] нет ни одной модели состояний — монета пропущена")
        return []

    rows: list[dict] = []
    for label, run_id in runs:
        labels = load_labels(symbol, run_id, args.end)
        if labels.empty:
            print(f"  [{symbol} {label}] разметки нет")
            continue
        for horizon in args.horizon:
            for norm in args.norm:
                frame = cell_frame(base, horizon, norm, labels)
                frame = frame.dropna(subset=["group_id"])
                if len(frame) < 5000:
                    continue
                target = frame["target"].to_numpy(dtype=float)
                ts = frame.index.to_numpy()
                gap = frame.attrs["h_bars"]
                block = rm.bootstrap_block(frame.index.to_series(), horizon)
                benchmark = level_columns(frame, "B2")
                groups = frame["group_id"].to_numpy()
                # Метки блоков → целые коды. `factorize` — просто нумерация
                # уникальных строк, сведений о цели в ней нет; целевое
                # кодирование считается ниже и только по обучающей части фолда.
                blocks_codes = pd.factorize(frame["event_block_id"])[0]
                seed = args.seed[0]

                def predict_full(train, test, _b=benchmark, _t=target,
                                 _g=groups, _bl=blocks_codes, _s=seed):
                    encoded = np.column_stack([
                        rm.target_encode(_g, _t, train),
                        rm.target_encode(_bl, _t, train),
                    ])
                    columns = np.column_stack([_b, encoded])
                    model = rm.make_regressor(_s)
                    model.fit(columns[train], _t[train])
                    return model.predict(columns[test])

                base_fit = rm.walk_forward(len(frame), target, ts, gap, args.folds,
                                           rm.boosted_predictor(benchmark, target, seed))
                full_fit = rm.walk_forward(len(frame), target, ts, gap, args.folds,
                                           predict_full)
                gain = rm.compare(base_fit, full_fit, block, args.n_boot,
                                  seeded_rng(symbol, horizon, norm, run_id, "states"))
                rows.append({
                    "symbol": symbol, "run": label, "run_id": run_id,
                    "horizon": horizon, "norm": norm, "n": len(frame), "block": block,
                    "n_states": int(pd.Series(groups).nunique()),
                    "r2_base": gain.r2_base, "r2_full": gain.r2_full,
                    "delta": gain.delta, "p": gain.p_value,
                })
                print(f"  [{symbol} {label} {horizon} {norm}] "
                      f"состояний {pd.Series(groups).nunique()} "
                      f"R²(B2)={gain.r2_base:+.4f} +разметка={gain.r2_full:+.4f} "
                      f"Δ={gain.delta:+.4f} p={gain.p_value:.4f}")
    return rows


# ─── Печать задач C ─────────────────────────────────────────────────────────
def format_gain_table(rows: list[dict], title: str, key: str) -> str:
    if not rows:
        return f"\n{title}: ни одной ячейки не посчитано."
    marks = benjamini_hochberg([r["p"] for r in rows])
    header = (f"{'монета':<10} {key:<26} {'гор.':>5} {'норм.':>6} {'n':>8} "
              f"{'R²(B2)':>8} {'R²(полн.)':>10} {'Δ':>8} {'p':>8} {'BH':>3}")
    lines = ["", "=" * len(header), title, "=" * len(header),
             "Δ — приращение out-of-sample R² над B2 на ОДНИХ И ТЕХ ЖЕ строках.",
             f"Порог практической величины (заявлен ДО прогона): Δ ≥ {SYSTEM_ADDS_DELTA}.",
             "", header, "─" * len(header)]
    for r, mark in zip(rows, marks):
        lines.append(
            f"{r['symbol']:<10} {str(r[key]):<26} {r['horizon']:>5} {r['norm']:>6} "
            f"{r['n']:>8} {r['r2_base']:>+8.4f} {r['r2_full']:>+10.4f} "
            f"{r['delta']:>+8.4f} {r['p']:>8.4f} {'да' if mark else 'нет':>3}"
        )
    lines.append("")
    lines.append(f"Тестов в поправке BH: {len(rows)}.")

    # Монета, у которой вариантов меньше двух (например, единственная модель
    # состояний в базе), правилу «на обоих зёрнах» не подчиняется: у неё нет
    # второго варианта, а не «он не прошёл». Ловушка 11 ТЗ в применении к C2.
    variants_available: dict[str, set] = {}
    for r in rows:
        variants_available.setdefault(r["symbol"], set()).add(r[key])
    structural = sorted(s for s, v in variants_available.items() if len(v) < 2)

    passed = [r for r, mark in zip(rows, marks)
              if mark and r["delta"] >= SYSTEM_ADDS_DELTA
              and r["symbol"] not in structural]
    by_symbol: dict[str, dict[str, set]] = {}
    for r in passed:
        seen = by_symbol.setdefault(r["symbol"], {"horizon": set(), "norm": set(),
                                                  "variant": set()})
        seen["horizon"].add(r["horizon"])
        seen["norm"].add(r["norm"])
        seen["variant"].add(r[key])
    good = [s for s, seen in by_symbol.items()
            if len(seen["horizon"]) >= 2 and len(seen["norm"]) >= 2
            and len(seen["variant"]) >= 2]
    lines.append("")
    lines.append(f"Ячеек прошло порог и BH: {len(passed)} из {len(rows)}; "
                 f"монет с ≥2 горизонтами, обеими нормировками и ≥2 вариантами "
                 f"{key}: {len(good)}.")
    if structural:
        lines.append(f"Структурно неприменимы (вариантов «{key}» в данных меньше "
                     f"двух): {', '.join(structural)} —")
        lines.append("их ячейки напечатаны, но в вердикт не входят.")
    lines.append(
        "ВЕРДИКТ ПО КРИТЕРИЮ C4: "
        + ("СИСТЕМА ДОБАВЛЯЕТ К ТРИВИАЛЬНОМУ БЕНЧМАРКУ" if len(good) >= 2
           else "НЕ ДОБАВЛЯЕТ")
    )
    if len(good) < 2:
        lines.append("Промежуточные формулировки («есть тенденция», «на одной монете "
                     "сработало»)")
        lines.append("запрещены ТЗ: ровно так однажды родилась зацепка по ETHUSDT, "
                     "не повторившаяся")
        lines.append("на второй монете. Задача D не начинается.")
    return "\n".join(lines)


def _json_scalar(value):
    """
    Числа numpy в JSON. Без этого `default=str` превратил бы `np.float64` в
    СТРОКУ, и собранный отчёт молча печатал бы форматирование по строкам
    вместо чисел — ошибка, которая не падает, а портит таблицу.
    """
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


def report(collected: dict[str, list[dict]]) -> None:
    """
    Печать отчётов по собранным ячейкам.

    Вынесена отдельно, потому что поправка BH обязана считаться по ВСЕМ
    ячейкам одного прогона, а монеты считаются параллельно, разными
    процессами. Каждый процесс складывает свои ячейки в файл (`--dump`), а
    общий отчёт собирается одним вызовом (`--report`) — иначе поправка
    оказалась бы помонетной, то есть более мягкой, чем заявлено в ТЗ.
    """
    if collected.get("bench"):
        print(format_bench(collected["bench"]))
    if "deriv" in collected:
        print(format_deriv(collected["deriv"]))
    if "features" in collected:
        print(format_gain_table(
            collected["features"],
            "ЗАДАЧА C1: ПРИЗНАКИ БЕЗ ГРАФА ПРОТИВ B2 (контрольная модель на размахе)",
            "seed"))
    if "states" in collected:
        print(format_gain_table(
            collected["states"],
            "ЗАДАЧА C2: РАЗМЕТКА ГРАФА КАК ПРЕДИКТОР ПОВЕРХ B2",
            "run"))


def merge_dumps(paths: list[str]) -> dict[str, list[dict]]:
    """Слияние файлов `--dump` в один набор ячеек, по задачам."""
    merged: dict[str, list[dict]] = {}
    for path in paths:
        part = json.loads(Path(path).read_text(encoding="utf-8"))
        for task, rows in part.items():
            merged.setdefault(task, []).extend(rows)
    for rows in merged.values():
        rows.sort(key=lambda r: (r.get("predictor", ""), r.get("symbol", ""),
                                 r.get("horizon", ""), r.get("norm", ""),
                                 str(r.get("epoch", r.get("run", r.get("seed", ""))))))
    return merged


# ─── Точка входа ────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Стенд «размах × горизонт × бенчмарк» (ТЗ 2026-08-19)")
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--horizon", action="append", choices=list(rm.HORIZONS))
    parser.add_argument("--norm", action="append", choices=list(rm.NORMALIZATIONS),
                        help="нормировка цели; по умолчанию обе (A3)")
    parser.add_argument("--task", action="append",
                        choices=["bench", "deriv", "features", "states"])
    parser.add_argument("--predictor", action="append",
                        help="только эти величины деривативов (задача B)")
    parser.add_argument("--end", default=FROZEN_END,
                        help="замороженная граница; менять нельзя без потери "
                             "сравнимости с разделами 26, 31 и 36")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, action="append",
                        help="зерно бустинга; по умолчанию 42 и 1337")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--dump", help="сложить посчитанные ячейки в файл JSON")
    parser.add_argument("--report", nargs="+",
                        help="не считать, а собрать общий отчёт из файлов --dump")
    args = parser.parse_args()

    if args.report:
        report(merge_dumps(args.report))
        return 0

    args.horizon = args.horizon or list(rm.HORIZONS)
    args.norm = args.norm or list(rm.NORMALIZATIONS)
    args.task = args.task or ["bench"]
    args.seed = args.seed or [42, 1337]

    specs = symbols.resolve_many(args.symbol, args.all)
    print(f"Стенд размаха. Монет {len(specs)}, горизонты {args.horizon}, "
          f"нормировки {args.norm}, задачи {args.task}, граница {args.end}.")
    print("Методика — шапка btcproc/analysis/range_model.py. Направление здесь "
          "не меряется: оно закрыто разделом 26.4.\n")

    collected: dict[str, list[dict]] = {task: [] for task in args.task}
    for spec in specs:
        symbol = spec.ticker
        print(f"\n{'=' * 78}\n=== {symbol}\n{'=' * 78}")
        try:
            base = load_base(symbol, args.end)
        except SystemExit as exc:
            print(f"  пропущена: {exc}")
            continue
        print(f"  баров {len(base)} ({base.index[0]:%Y-%m-%d}…{base.index[-1]:%Y-%m-%d})")
        for task in args.task:
            runner = {"bench": task_bench, "deriv": task_deriv,
                      "features": task_features, "states": task_states}[task]
            collected[task].extend(runner(symbol, base, args))

    if args.dump:
        Path(args.dump).write_text(
            json.dumps(collected, ensure_ascii=False, default=_json_scalar),
            encoding="utf-8")
        print(f"\nЯчейки сохранены в {args.dump} — свести их в общий отчёт: "
              f"--report {args.dump} …")
    report(collected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
