"""
Задачи C и D ТЗ `crypto-graph/docs/tz_cross_section_20-08-26.md`:
предсказывают ли поперечные величины относительную доходность (C) и размах (D).

    python3 scripts/measure_cross_section.py               # обе задачи
    python3 scripts/measure_cross_section.py --task C
    python3 scripts/measure_cross_section.py --task D --symbol BTCUSDT

**Ничего не пишет — ни в БД, ни в конфиг.** Ни `train`, ни новых таблиц, ни
единого похода в чужой API.

## Задача C: относительная доходность

Цель — `xs_fwd_ret(i, t, H)`: доходность монеты вперёд минус средняя по
корзине на баре `t`. Величина с нулевым средним по сечению по построению, то
есть общий рыночный фактор из неё вычтен. Это формальная причина, по которой
замер **не воскрешает закрытую задачу про направление** (26.3, 26.4): вопрос
не «пойдёт ли цена вверх», а «обгонит ли монета корзину». `is_up` здесь не
используется, «всегда long» бенчмарком не является, directional accuracy не
считается.

Метрика — `IC(t)`, ранговая корреляция Спирмена МЕЖДУ МОНЕТАМИ на одном баре,
усреднённая по времени. Значимость — **двумя независимыми нулёвками**:

* блочный бутстрап по временно́му ряду `IC(t)` — общим кодом, `IC(t)` это
  обычный временной ряд, и вся дисциплина проекта к нему применима;
* **суррогатная**: перестановка предиктора внутри каждого `ts`, то есть между
  монетами. Сохраняет и временну́ю структуру, и поперечное распределение, и
  рушит ровно проверяемую связь.

Обе обязаны согласоваться. Их расхождение — результат ПРО ИЗМЕРИТЕЛЬ, и
записывается он именно так.

Горизонты — 1h / 2h / 4h (правка ТЗ 2026-08-21). При прежних 4h/12h/24h MDE на
двух ячейках из трёх был выше порога практической величины 0.02, то есть
ячейки выбрасывались бы правилом самого ТЗ, и условие «знак совпал на ≥2
горизонтах» стало бы невыполнимым.

## Задача D: размах

Цель — `range_ratio`, бенчмарк — только B2 (HAR-RV + час дня + выходные),
измеритель — `range_lift.partial_r2_gain`. Горизонты здесь прежние
(4h/12h/24h): задача про размах, и сравнимость с разделами 47 и 49 важнее.

Помонетные величины (H1–H3) меряются помонетно, общерыночные (H4–H5) —
**эпохами**, потому что ряд у них один на все монеты, и «знак совпал на трёх
монетах» для них не подтверждение, а один замер, показанный трижды.

## Чего этот скрипт не делает

Не заводит признаки, не трогает флаги, не строит общую модель состояний.
И главное: **любой положительный результат помечается «верхняя граница,
корзина отобрана задним числом»** и основанием для заведения величины в
конвейер не является. Шесть монет выбраны владельцем в 2026 году, то есть
про них уже известно, что они дожили и остались ликвидными; основание
появится только после повторения на вселенной, заданной механическим
правилом.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["SINK_MODE"] = "none"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from btcproc import config, symbols  # noqa: E402
from btcproc.analysis import cross_section as xs  # noqa: E402
from btcproc.analysis import range_model as rm  # noqa: E402
from btcproc.analysis.lift import (  # noqa: E402
    DEFAULT_N_BOOT, benjamini_hochberg, block_length_rows,
)
from btcproc.analysis.range_lift import partial_r2_gain  # noqa: E402
from btcproc.ingest import bars  # noqa: E402

FROZEN_END = "2026-08-01"

#: Горизонты задачи C — после правки ТЗ 2026-08-21 (см. шапку).
HORIZONS_C = ("1h", "2h", "4h")

#: Порог практической величины среднего IC, объявлен в §4.4 ТЗ ДО прогона.
IC_THRESHOLD = 0.02

#: Порог задачи D — тот же, что у range-оценщиков, и вдвое ниже порога 0.02,
#: которым принят регрессор размаха (49.3).
RANGE_THRESHOLD = 0.010

#: Эпохи — те же и с теми же границами, что в measure_deriv_range.py и
#: measure_range_horizons.py. Менять нельзя: иначе разрез по времени
#: сравнивал бы себя не с предыдущими замерами.
EPOCHS = [
    ("2018-02..2020-12", "2018-02-01", "2020-12-31"),
    ("2021-01..2022-12", "2021-01-01", "2022-12-31"),
    ("2023-01..2026-08", "2023-01-01", None),
]

#: Группы гипотез §2.3 ТЗ. Поправка на множественность применяется ПО ГРУППАМ:
#: внутри H1 две величины — это один эффект на двух горизонтах, внутри H2
#: `beta` и `idio_ret` связаны по построению (второе считается через первое).
GROUPS = {
    "H1": ("xs_rank_ret_1h", "xs_rank_ret_1d"),
    "H2": ("beta_basket_1m", "idio_ret_1d"),
    "H3": ("xs_rank_rv",),
    "H4": ("basket_dispersion",),
    "H5": ("btc_share_chg_1d",),
}
MARKET_WIDE = {"basket_dispersion", "btc_share_chg_1d"}


def group_of(measure: str) -> str:
    for name, members in GROUPS.items():
        if measure in members:
            return name
    return "?"


def seeded_rng(*parts) -> np.random.Generator:
    salt = zlib.crc32("|".join(str(p) for p in parts).encode())
    return np.random.default_rng([42, salt])


# ─── Задача C ───────────────────────────────────────────────────────────────
def measure_ic(predictor: pd.DataFrame, target: pd.DataFrame, ts: pd.Series,
               horizon: str, sizes: pd.Series, args, tag: str) -> dict | None:
    """
    Одна ячейка задачи C: средний IC, обе нулёвки, MDE рядом с числом.

    MDE печатается ВСЕГДА и до интерпретации: ячейка, где наблюдённый эффект
    меньше минимально детектируемого, смысла не имеет независимо от p-value.
    """
    ic = xs.information_coefficient(predictor, target).dropna()
    if len(ic) < 500:
        return None
    # Длина блока — максимум из перекрытия окон и автокорреляции самого ряда
    # IC (§4.2 ТЗ). Только по горизонту тест анти-консервативен: предиктор
    # считается на скользящем окне, и IC наследует его липкость.
    block = xs.ic_block_length(ic, ts, rm.horizon_minutes(horizon))
    values = ic.to_numpy(dtype=float)
    observed = float(values.mean())

    # Блочная нулёвка. Знак заранее не заявлен, поэтому двусторонне: считаем
    # одностороннюю в сторону наблюдённого знака и удваиваем.
    rng = seeded_rng(tag, horizon, "block")
    oriented = values if observed >= 0 else -values
    p_block = min(1.0, 2.0 * rm.block_mean_p(oriented, block, args.n_boot, rng))

    # Суррогатная нулёвка: перестановка предиктора внутри каждого бара.
    surrogate_rng = seeded_rng(tag, horizon, "surrogate")
    draws = xs.surrogate_ic(predictor, target, surrogate_rng,
                            draws=args.n_surrogate, block=block)
    extreme = int(np.sum(np.abs(draws) >= abs(observed)))
    p_surrogate = (1 + extreme) / (1 + len(draws))

    # IC внутри монеты: тот же расчёт после вычитания средних уровней каждой
    # монеты. Отделяет «когда монета волатильнее обычного, она отстаёт» от
    # «какие монеты в среднем волатильнее и в среднем отстают». Второе — шесть
    # наблюдений, и значимость по двумстам тысячам баров к нему неприменима.
    within = float(xs.information_coefficient(
        xs.within_symbol(predictor), xs.within_symbol(target), min_basket
    ).mean())

    mde = xs.minimum_detectable_ic(sizes.reindex(ic.index).dropna(), block)
    return {
        "ic_within": within,
        "horizon": horizon, "n": len(ic), "block": block,
        "n_eff": len(ic) // block, "ic": observed,
        "p_block": p_block, "p_surrogate": p_surrogate,
        "mde": mde, "powered": bool(np.isfinite(mde) and mde <= IC_THRESHOLD),
        "surrogate_mean": float(draws.mean()), "surrogate_std": float(draws.std()),
    }


def task_c(basket: xs.Basket, per_symbol: dict[str, pd.DataFrame],
           market_wide: dict[str, pd.Series], args) -> list[dict]:
    """
    Задача C. При `--lag-bars N` предиктор сдвигается на N баров назад,
    то есть между окном предиктора и началом цели остаётся зазор.

    Зазор — не часть ТЗ, а обязательная проверка ЧТЕНИЯ результата, если
    эффект найдётся на коротком горизонте. Поперечный разворот на 15-минутных
    барах бывает двух совершенно разных природ: экономической (кто-то
    перекупил, цена возвращается) и микроструктурной (последняя сделка бара
    прошла по одной стороне спреда, следующая по другой). Вторая на бумаге
    выглядит как первая, но торговать её нельзя вовсе — и различает их именно
    зазор: отскок от спреда живёт один бар, экономический возврат — дольше.
    """
    rows: list[dict] = []
    sizes = basket.size()
    ts = basket.index.to_series()
    split = rm.MIN_HOLDOUT_PART  # только чтобы не молчать при короткой панели
    cut = int(len(basket.index) * 0.7)

    for measure, frame in per_symbol.items():
        if args.measure and measure not in args.measure:
            continue
        for horizon in args.horizon_c:
            h_bars = rm.horizon_bars(horizon, config.data.base_minutes)
            target = xs.cross_forward_return(basket, h_bars, log=args.log_return)
            predictor = frame.shift(args.lag_bars) if args.lag_bars else frame
            cell = measure_ic(predictor, target, ts, horizon, sizes, args, measure)
            if cell is None:
                continue
            cell.update({"measure": measure, "group": group_of(measure),
                         "scope": "все данные", "lag": args.lag_bars})
            rows.append(cell)
            print(f"  [{measure} {horizon}] IC={cell['ic']:+.4f} "
                  f"MDE={cell['mde']:.4f} p_блок={cell['p_block']:.4f} "
                  f"p_сурр={cell['p_surrogate']:.4f} n={cell['n']}")

            # Отложенная часть — условие (5) критерия.
            if cut > split and len(basket.index) - cut > split:
                tail = basket.index[cut:]
                cell_out = measure_ic(predictor.loc[tail], target.loc[tail],
                                      tail.to_series(), horizon,
                                      sizes.loc[tail], args, measure + "|hold")
                if cell_out is not None:
                    cell_out.update({"measure": measure, "group": group_of(measure),
                                     "scope": "holdout 30%", "lag": args.lag_bars})
                    rows.append(cell_out)

            # Эпохи — условие (4).
            for label, start, end in EPOCHS:
                window = basket.index[
                    (basket.index >= pd.Timestamp(start, tz="UTC"))
                    & (basket.index <= pd.Timestamp(end or FROZEN_END, tz="UTC"))
                ]
                if len(window) < 5000:
                    continue
                cell_epoch = measure_ic(predictor.loc[window], target.loc[window],
                                        window.to_series(), horizon,
                                        sizes.loc[window], args,
                                        f"{measure}|{label}")
                if cell_epoch is not None:
                    cell_epoch.update({"measure": measure, "group": group_of(measure),
                                       "scope": label, "lag": args.lag_bars})
                    rows.append(cell_epoch)
    return rows


# ─── Задача D ───────────────────────────────────────────────────────────────
def task_d(basket: xs.Basket, per_symbol: dict[str, pd.DataFrame],
           market_wide: dict[str, pd.Series], args) -> list[dict]:
    """
    Поперечные величины против размаха СОБСТВЕННОЙ монеты, сверх B2.

    Помонетные величины берут свою колонку панели, общерыночные — один и тот
    же ряд для каждой монеты; устойчивость у вторых считается по эпохам, а не
    по монетам, и в отчёте это разные строки критерия.
    """
    rows: list[dict] = []
    for ticker in args.symbol:
        spec = symbols.get(ticker)
        base = bars.load_ohlcv(ticker, config.data.base_tf, spec.start_date(),
                               args.end)
        if base.empty:
            continue
        print(f"  [{ticker}] баров {len(base)}")
        for horizon in args.horizon_d:
            h_bars = rm.horizon_bars(horizon, config.data.base_minutes)
            block = block_length_rows(base.index.to_series(),
                                      rm.horizon_minutes(horizon))
            for norm in args.norm:
                target = rm.range_target(base, h_bars, norm)
                benchmark_frame = pd.concat(
                    [rm.har_columns(base["close"]),
                     rm.seasonal_columns(base.index)], axis=1)
                # Бенчмарк для partial_r2_gain — один ряд, поэтому B2 сжимается
                # в его собственный прогноз (OLS по обучающей половине не нужен:
                # измеритель ранговый и сравнивает связь, а не точность).
                benchmark = _collapse_benchmark(benchmark_frame, target)

                for measure, source in _measures_for(ticker, per_symbol, market_wide):
                    frame = pd.concat(
                        [source.rename("x"), benchmark.rename("b"),
                         target.rename("y")], axis=1).dropna()
                    if len(frame) < 5000:
                        continue
                    r2_base, r2_full, r_partial, p_gain = partial_r2_gain(
                        frame["x"].to_numpy(float), frame["b"].to_numpy(float),
                        frame["y"].to_numpy(float), block, args.n_boot,
                        seeded_rng(ticker, horizon, norm, measure),
                    )
                    rows.append({
                        "symbol": ticker, "measure": measure,
                        "group": group_of(measure), "horizon": horizon,
                        "norm": norm, "n": len(frame), "block": block,
                        "r2_base": r2_base, "r2_full": r2_full,
                        "delta": r2_full - r2_base, "r_partial": r_partial,
                        "p": p_gain,
                        "market_wide": measure in MARKET_WIDE,
                    })
                    print(f"    [{ticker} {horizon} {norm} {measure}] "
                          f"ΔR²={r2_full - r2_base:+.4f} p={p_gain:.4f}")
    return rows


def _collapse_benchmark(frame: pd.DataFrame, target: pd.Series) -> pd.Series:
    """
    Свести матрицу B2 к одному ряду — его собственному прогнозу цели.

    `partial_r2_gain` принимает бенчмарк вектором (он ранговый и считает
    частную корреляцию по формуле для двух предикторов). Прогноз B2 —
    единственный честный способ сжать матрицу в вектор: любая одиночная
    колонка B2 была бы неполным бенчмарком, а без часа дня бенчмарк в этой
    задаче бенчмарком не является (раздел 47).

    Регрессия здесь in-sample — как и весь `partial_r2_gain`: вопрос стоит
    «есть ли у предиктора остаток информации сверх бенчмарка», а не «насколько
    точна модель».
    """
    aligned = pd.concat([frame, target.rename("__y")], axis=1).dropna()
    columns = np.column_stack([np.ones(len(aligned)),
                               aligned[frame.columns].to_numpy(float)])
    beta, *_ = np.linalg.lstsq(columns, aligned["__y"].to_numpy(float), rcond=None)
    return pd.Series(columns @ beta, index=aligned.index)


def _measures_for(ticker: str, per_symbol: dict[str, pd.DataFrame],
                  market_wide: dict[str, pd.Series]):
    for measure, frame in per_symbol.items():
        if ticker in frame.columns:
            yield measure, frame[ticker]
    for measure, series in market_wide.items():
        yield measure, series


# ─── Отчёт ──────────────────────────────────────────────────────────────────
def format_c(rows: list[dict]) -> str:
    main = [r for r in rows if r["scope"] == "все данные"]
    marks = dict(zip([id(r) for r in main],
                     benjamini_hochberg([max(r["p_block"], r["p_surrogate"])
                                         for r in main])))
    header = (f"{'величина':<18} {'гр.':>4} {'разрез':<18} {'гор.':>5} {'n':>8} "
              f"{'блок':>5} {'IC':>8} {'MDE':>7} {'p блок':>8} {'p сурр':>8} "
              f"{'BH':>4} {'мощн.':>6} {'IC внутри':>10}")
    lines = ["", "ЗАДАЧА C: ОТНОСИТЕЛЬНАЯ ДОХОДНОСТЬ",
             "Цель — доходность монеты минус средняя по корзине на том же баре.",
             "Общий рыночный фактор из неё вычтен по построению: это НЕ замер",
             "направления (§0.4 ТЗ).",
             f"Порог практической величины |IC| ≥ {IC_THRESHOLD}, заявлен ДО прогона.",
             "", header, "─" * len(header)]
    for r in rows:
        mark = marks.get(id(r))
        bh = "—" if mark is None else ("да" if mark else "нет")
        lines.append(
            f"{r['measure']:<18} {r['group']:>4} {r['scope']:<18} "
            f"{r['horizon']:>5} {r['n']:>8} {r['block']:>5} {r['ic']:>+8.4f} "
            f"{r['mde']:>7.4f} {r['p_block']:>8.4f} {r['p_surrogate']:>8.4f} "
            f"{bh:>4} {'да' if r['powered'] else 'НЕТ':>6} "
            f"{r.get('ic_within', float('nan')):>+10.4f}"
        )
    lines += [
        "",
        "«IC внутри» — тот же IC после вычитания средних уровней каждой монеты.",
        "Разница между ним и обычным IC показывает, сколько эффекта приходится на",
        "различия МЕЖДУ монетами: это шесть наблюдений, а не двести тысяч баров, и",
        "значимость, посчитанная по барам, к ним неприменима.",
    ]
    return "\n".join(lines)


def verdict_c(rows: list[dict]) -> str:
    main = [r for r in rows if r["scope"] == "все данные"]
    if not main:
        return "\nВЕРДИКТ C: считать нечего."
    marks = benjamini_hochberg([max(r["p_block"], r["p_surrogate"]) for r in main])
    lines = ["", "=" * 78,
             "ВЕРДИКТ ЗАДАЧИ C (критерий §4.4 ТЗ, заявлен до прогона)", "=" * 78,
             f"  ячеек {len(main)}, тестов в поправке {len(main)}"]
    passed_any = False
    for measure in sorted({r["measure"] for r in main}):
        cells = [(r, m) for r, m in zip(main, marks) if r["measure"] == measure]
        # Условие «эффект внутримонетный» добавлено к критерию 2026-08-21 по
        # результату замера: без него критерий проходили величины, у которых
        # весь эффект сидел в различиях между шестью монетами.
        strong = [r for r, m in cells
                  if m and abs(r["ic"]) >= IC_THRESHOLD and r["powered"]
                  and abs(r.get("ic_within", 0.0)) >= IC_THRESHOLD
                  and np.sign(r.get("ic_within", 0.0)) == np.sign(r["ic"])]
        horizons = {r["horizon"] for r in strong}
        signs = {np.sign(r["ic"]) for r in strong}
        epochs = [r for r in rows if r["measure"] == measure
                  and r["scope"] not in ("все данные", "holdout 30%")
                  and abs(r["ic"]) >= IC_THRESHOLD]
        epoch_signs = {np.sign(r["ic"]) for r in epochs}
        holdout = [r for r in rows if r["measure"] == measure
                   and r["scope"] == "holdout 30%"]
        ok = (len(horizons) >= 2 and len(signs) == 1
              and len(epochs) >= 2 and len(epoch_signs) == 1
              and holdout and np.sign(holdout[0]["ic"]) in signs)
        passed_any = passed_any or ok
        lines.append(
            f"  {measure:<18} ячеек в зачёте {len(strong)}, горизонтов "
            f"{len(horizons)}, эпох со знаком {len(epochs)} — "
            f"{'ПРОШЛА' if ok else 'не прошла'}")
    lines.append("")
    if passed_any:
        lines += [
            "  ВНИМАНИЕ: любой пройденный критерий — ВЕРХНЯЯ ГРАНИЦА эффекта.",
            "  Корзина отобрана задним числом (§0.2 ТЗ), и основанием заводить",
            "  величину в конвейер результат НЕ является. Основание появится",
            "  только после повторения на вселенной по механическому правилу.",
        ]
    else:
        lines += [
            "  Ни одна величина не прошла. Записывается так же прямо, как раздел 26",
            "  записал собственный отрицательный результат: поперечная ось",
            "  проверена, и это четвёртый независимый источник, упершийся в ту же",
            "  границу. Строка «поперечное сечение» переезжает из волны 4",
            "  ideas_math в §8 «чего сознательно не предлагаю».",
        ]
    return "\n".join(lines)


def format_d(rows: list[dict]) -> str:
    marks = dict(zip([id(r) for r in rows],
                     benjamini_hochberg([r["p"] for r in rows])))
    header = (f"{'монета':<10} {'величина':<18} {'гр.':>4} {'гор.':>5} "
              f"{'норм.':>6} {'n':>8} {'R²(B2)':>8} {'ΔR²':>8} {'r част.':>8} "
              f"{'p':>7} {'BH':>4} {'порог':>6}")
    lines = ["", "ЗАДАЧА D: ПОПЕРЕЧНЫЕ ВЕЛИЧИНЫ ПРОТИВ РАЗМАХА",
             "Бенчмарк — только B2 (HAR-RV + час дня + выходные). Без часа дня",
             "бенчмарк в этой задаче бенчмарком не является (раздел 47).",
             f"Порог ΔR² ≥ {RANGE_THRESHOLD}, заявлен ДО прогона.",
             "", header, "─" * len(header)]
    for r in rows:
        mark = marks.get(id(r))
        passed = mark and r["delta"] >= RANGE_THRESHOLD
        lines.append(
            f"{r['symbol']:<10} {r['measure']:<18} {r['group']:>4} "
            f"{r['horizon']:>5} {r['norm']:>6} {r['n']:>8} {r['r2_base']:>8.4f} "
            f"{r['delta']:>+8.4f} {r['r_partial']:>+8.4f} {r['p']:>7.4f} "
            f"{'да' if mark else 'нет':>4} {'ДА' if passed else 'нет':>6}"
        )
    return "\n".join(lines)


def verdict_d(rows: list[dict]) -> str:
    if not rows:
        return "\nВЕРДИКТ D: считать нечего."
    marks = benjamini_hochberg([r["p"] for r in rows])
    passed = [r for r, m in zip(rows, marks)
              if m and r["delta"] >= RANGE_THRESHOLD]
    lines = ["", "=" * 78, "ВЕРДИКТ ЗАДАЧИ D (критерий §5 ТЗ, заявлен до прогона)",
             "=" * 78, f"  ячеек {len(rows)}, прошло {len(passed)}"]
    for measure in sorted({r["measure"] for r in rows}):
        cells = [r for r in passed if r["measure"] == measure]
        coins = {r["symbol"] for r in cells}
        horizons = {r["horizon"] for r in cells}
        norms = {r["norm"] for r in cells}
        market = measure in MARKET_WIDE
        ok = (len(coins) >= 3 and len(horizons) >= 2 and len(norms) == 2
              if not market else
              len(horizons) >= 2 and len(norms) == 2)
        note = " (общерыночная: монеты подтверждением не являются)" if market else ""
        lines.append(f"  {measure:<18} монет {len(coins)}, горизонтов "
                     f"{len(horizons)}, нормировок {len(norms)} — "
                     f"{'ПРОШЛА' if ok else 'не прошла'}{note}")
    return "\n".join(lines)


def _json_scalar(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


# ─── Точка входа ────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Поперечное сечение: относительная доходность и размах")
    parser.add_argument("--task", action="append", choices=["C", "D"])
    parser.add_argument("--symbol", action="append",
                        help="монеты задачи D; по умолчанию все активные")
    parser.add_argument("--horizon-c", action="append", choices=list(HORIZONS_C))
    parser.add_argument("--horizon-d", action="append", choices=list(rm.HORIZONS))
    parser.add_argument("--norm", action="append", choices=list(rm.NORMALIZATIONS))
    parser.add_argument("--end", default=FROZEN_END)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--measure", action="append",
                        help="считать только эти величины (по умолчанию все)")
    parser.add_argument("--simple-return", dest="log_return", action="store_false",
                        default=True,
                        help="цель в ПРОСТОЙ доходности вместо логарифмической: "
                             "контроль на артефакт Йенсена, из-за которого любой "
                             "предиктор волатильности получает отрицательный IC")
    parser.add_argument("--lag-bars", type=int, default=0,
                        help="зазор между окном предиктора и началом цели, в "
                             "барах: отличает экономический разворот от "
                             "отскока по спреду")
    parser.add_argument("--n-surrogate", type=int, default=200,
                        help="реплик суррогатной нулёвки (перестановка внутри ts)")
    parser.add_argument("--dump")
    args = parser.parse_args()

    args.task = args.task or ["C", "D"]
    args.horizon_c = args.horizon_c or list(HORIZONS_C)
    args.horizon_d = args.horizon_d or list(rm.HORIZONS)
    args.norm = args.norm or list(rm.NORMALIZATIONS)
    args.symbol = args.symbol or [s.ticker for s in symbols.enabled()]

    print(f"Поперечное сечение. Граница {args.end}, корзина ≥ {xs.MIN_BASKET}, "
          f"задачи {args.task}"
          + (f", зазор предиктора {args.lag_bars} бар" if args.lag_bars else "")
          + (", цель — ЛОГАРИФМИЧЕСКАЯ доходность."
             if args.log_return else ", цель — ПРОСТАЯ доходность (контроль)."))
    print("Направление в исходной формулировке не меряется: цель поперечная, "
          "общий рыночный фактор из неё вычтен (§0.4 ТЗ).\n")

    basket = xs.load_basket(end=args.end)
    print(f"Панель: {len(basket.index)} баров, {len(basket.tickers)} монет, "
          f"{basket.index[0]:%Y-%m-%d}…{basket.index[-1]:%Y-%m-%d}, "
          f"средний размер корзины {float(basket.size().mean()):.2f}")
    per_symbol, market_wide = xs.measures(basket)

    collected: dict[str, list[dict]] = {}
    if "C" in args.task:
        print("\n=== Задача C: относительная доходность ===")
        collected["C"] = task_c(basket, per_symbol, market_wide, args)
    if "D" in args.task:
        print("\n=== Задача D: размах ===")
        collected["D"] = task_d(basket, per_symbol, market_wide, args)

    if args.dump:
        Path(args.dump).write_text(
            json.dumps(collected, ensure_ascii=False, default=_json_scalar),
            encoding="utf-8")
        print(f"\nЯчейки сохранены в {args.dump}")

    if collected.get("C"):
        print(format_c(collected["C"]))
        print(verdict_c(collected["C"]))
    if collected.get("D"):
        print(format_d(collected["D"]))
        print(verdict_d(collected["D"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
