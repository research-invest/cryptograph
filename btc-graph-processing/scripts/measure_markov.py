"""
Граф состояний как марковская модель: три вопроса одним заходом.

    python3 scripts/measure_markov.py --all
    python3 scripts/measure_markov.py --symbol BTCUSDT --task m1

ТЗ — `crypto-graph/docs/tz_wave_a_24-08-26.md`, задача M. Математика,
устройство нулёвок и обе ловушки (смесь моделей, самопереходы) — шапка
`btcproc/analysis/markov.py`.

## Три критерия, заявленные ДО прогона

> **M1, марковость.** Разметка признаётся марковской, если implied
> timescales трёх медленнейших процессов выходят на плато — колебание не
> более **20%** на `τ ∈ [2τ*, 5τ*]` — и CK-кривые лежат внутри блочной
> доверительной полосы, на **≥4 монетах из 6**.
>
> **M2, порядок.** Если наблюдаемая CMI выше 95-го перцентиля нулёвки на ≥4
> монетах из 6 **и** превышает её медиану не менее чем в **1.5 раза** —
> глубины пары недостаточно, есть основание ставить вопрос о тройке. Иначе
> вопрос закрывается ссылкой на этот замер.
>
> **M3, независимый контроль.** Совпадение `is_transition` со сменами BOCPD
> в пределах ±2 баров встречается значимо чаще случайного (`p ≤ 0.05` по
> сдвиговой нулёвке) и превышает случайный уровень не менее чем в **1.5
> раза**, на ≥4 монетах из 6. Иначе `is_transition` — дребезг процедуры
> кластеризации, а не наблюдаемое событие рынка.

Критерии не пересматриваются по итогам.

## Что прогон НЕ делает

Не переобучает модель, ничего не пишет в БД. Читает `bar_states` одной
моделью (`runs.model_run_scope`) и, для M3, `features` той же версии.
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
from btcproc.analysis import samples  # noqa: E402
from btcproc.db import runs as runs_repo  # noqa: E402
from btcproc.db.session import fetch_all  # noqa: E402

#: Лаги, на которых строится кривая implied timescales, в барах базового ТФ.
#: Геометрическая сетка: плато ищется в логарифмической шкале, и равномерная
#: сетка тратила бы почти все точки на правый конец.
LAGS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 288)

#: Во сколько раз ±2 бара. Допуск M3 — из ТЗ; он равен получасу при 15m.
TOLERANCE = 2

STATES_SQL = """
SELECT s.ts, s.group_id, s.is_transition
FROM bar_states s
WHERE s.symbol = %s AND {scope}
ORDER BY s.ts
"""


def load_states(symbol: str, model_run: int) -> pd.DataFrame:
    scope_sql, scope_params = runs_repo.model_run_scope(model_run, "s")
    rows = fetch_all(STATES_SQL.format(scope=scope_sql), (symbol, *scope_params))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    # Разметка одной модели может лежать в нескольких прогонах, и на границах
    # окон live бары повторяются. Дубли схлопываются по времени: одна и та же
    # минута обязана иметь одно состояние, иначе счётчики переходов удвоятся
    # в тех местах, где прогоны перекрылись.
    frame = frame.drop_duplicates(subset="ts", keep="last").reset_index(drop=True)
    return frame


def as_codes(group_id: pd.Series) -> tuple[np.ndarray, int]:
    """`group_id` → плотные коды 0..K−1. Номера состояний не обязаны быть
    подряд идущими: модель нумерует по наполнению, а часть номеров может не
    встретиться в этой разметке вовсе."""
    codes, uniques = pd.factorize(group_id)
    return codes.astype(np.int64), len(uniques)


def gap_stats(frame: pd.DataFrame) -> dict:
    """
    Разрывы в ряду — обязательный гейт данных.

    Матрица переходов при лаге τ считает пары `(t, t+τ)` по ПОЗИЦИЯМ, а не по
    времени. Если в разметке дыра (монета не торговалась, прогон не покрыл
    участок), пара через дыру склеит два далеко отстоящих состояния и
    отзовётся в спектре ложным медленным процессом. Печатается доля таких пар;
    если она заметна, результат читать нельзя.
    """
    step = pd.Timedelta(minutes=config.data.base_minutes)
    deltas = frame["ts"].diff().dropna()
    broken = int((deltas > step).sum())
    return {"rows": len(frame), "gaps": broken,
            "gap_share": broken / max(len(frame) - 1, 1),
            "max_gap_hours": float(deltas.max().total_seconds() / 3600)
            if len(deltas) else 0.0}


def task_m1(symbol: str, codes: np.ndarray, n_states: int, args) -> dict:
    curve = mk.timescale_curve(codes, list(LAGS), n_states, count=3)
    # τ* — самый короткий лаг, на котором время первого процесса определено.
    defined = curve.dropna(subset=["t1"])
    if defined.empty:
        return {"symbol": symbol, "curve": curve, "tau_star": None,
                "deviation": {}, "ck": pd.DataFrame(), "passed": False,
                "reason": "ни на одном лаге время процесса не определено"}
    tau_star = int(defined["lag"].iloc[0])
    lag_from, lag_to = 2 * tau_star, 5 * tau_star

    deviation = {column: mk.plateau_deviation(curve, column, lag_from, lag_to)
                 for column in ("t1", "t2", "t3")}
    ck = mk.chapman_kolmogorov(codes, max(tau_star, 1), [2, 3, 4, 5], n_states, 3)

    rng = np.random.default_rng([2026, zlib.crc32(f"{symbol}|m1".encode())])
    band = mk.bootstrap_lambda_band(codes, max(tau_star, 1), n_states,
                                    config.data.bars_per("1d"), args.n_boot_band, rng)

    # Рост по ВСЕЙ кривой — величина, которой в критерии нет, но без которой
    # его результат читается неверно. τ* здесь операционализирован как
    # «самый короткий лаг, где время определено», и на липкой разметке это
    # всегда 1, то есть окно [2τ*, 5τ*] вырождается в [2, 5] — пять баров, на
    # которых плато держится у чего угодно. Отношение времени на самом
    # длинном определённом лаге к самому короткому отвечает на тот же вопрос
    # без этой зависимости от операционализации.
    defined_t1 = curve.dropna(subset=["t1"])
    growth = (float(defined_t1["t1"].iloc[-1] / defined_t1["t1"].iloc[0])
              if len(defined_t1) >= 2 else float("nan"))

    plateau_ok = all(not np.isnan(v) and v <= 0.20 for v in deviation.values())
    # CK внутри полосы: предсказанное λ^k обязано попасть в ДИ наблюдённого.
    # Полоса считается на базовом лаге, поэтому для k>1 она переносится
    # возведением границ в ту же степень — монотонное преобразование, порядок
    # сохраняется.
    ck_ok = True
    if not ck.empty:
        limits = band.set_index("process")
        for _, row in ck.iterrows():
            lo = limits.loc[row["process"], "lo"] ** row["k"]
            hi = limits.loc[row["process"], "hi"] ** row["k"]
            if not (lo <= row["observed"] <= hi):
                ck_ok = False
                break
    return {"symbol": symbol, "curve": curve, "tau_star": tau_star,
            "lag_window": (lag_from, lag_to), "deviation": deviation,
            "ck": ck, "band": band, "plateau_ok": plateau_ok, "ck_ok": ck_ok,
            "growth": growth,
            "passed": plateau_ok and ck_ok, "reason": ""}


def task_m2(symbol: str, codes: np.ndarray, n_states: int, args) -> dict:
    rng = np.random.default_rng([2026, zlib.crc32(f"{symbol}|m2".encode())])
    out = {"symbol": symbol}
    for name, sequence in (("бары", codes), ("скачки", mk.jump_chain(codes))):
        if len(sequence) < 100:
            out[name] = None
            continue
        observed = mk.conditional_mutual_information(
            sequence[:-2], sequence[1:-1], sequence[2:], n_states)
        block = (config.data.bars_per("1d") if name == "бары"
                 else max(2, len(sequence) // 200))
        null = mk.cmi_null(sequence, n_states, block, args.n_boot_cmi, rng)
        median = float(np.median(null))
        out[name] = {
            "observed": observed, "null_median": median,
            "null_p95": float(np.percentile(null, 95)),
            "ratio": observed / median if median > 0 else float("inf"),
            "n": len(sequence), "block": block,
        }
    # Вердикт считается ПО КАЖДОМУ ряду отдельно, а не «оба сразу».
    # Критерий M2 в ТЗ не сказал, как объединять два ряда, а объединять их
    # нельзя: ряд по барам сконструирован так, что мерит прежде всего
    # липкость (шапка `markov.py`), и требовать от него прохождения значило бы
    # закрыть вопрос по причине, к вопросу не относящейся.
    for name in ("бары", "скачки"):
        value = out.get(name)
        if value:
            value["passed"] = (value["observed"] > value["null_p95"]
                               and value["ratio"] >= 1.5)
    out["passed"] = bool(out.get("скачки") and out["скачки"]["passed"])
    return out


FEATURES_SQL = """
SELECT f.ts, f.values
FROM features f
WHERE f.symbol = %s AND f.version = %s
ORDER BY f.ts
"""


def task_m3(symbol: str, frame: pd.DataFrame, model_run: int, args) -> dict:
    """BOCPD по первой главной компоненте признаков против переходов графа."""
    version = fetch_all(
        "SELECT feature_ver FROM state_models WHERE run_id = %s", (model_run,))
    if not version:
        return {"symbol": symbol, "skipped": "нет state_models для прогона"}
    rows = fetch_all(FEATURES_SQL, (symbol, version[0]["feature_ver"]))
    if not rows:
        return {"symbol": symbol, "skipped": "нет сохранённых признаков этой версии"}

    features = pd.DataFrame(
        [row["values"] for row in rows],
        index=pd.to_datetime([row["ts"] for row in rows], utc=True))
    features = features.loc[~features.index.duplicated(keep="last")]
    aligned = frame.set_index("ts").join(features, how="inner")
    if len(aligned) < 5000:
        return {"symbol": symbol, "skipped": f"мало общих баров ({len(aligned)})"}

    matrix = aligned[features.columns].to_numpy(dtype=float)
    # Первая главная компонента. Стандартизация обязательна: признаки уже
    # приведены robust_scale внутри модели, но здесь берутся сырые значения
    # из features, и без неё компонента описывала бы признак с наибольшим
    # масштабом, а не наибольшую общую изменчивость.
    matrix = (matrix - matrix.mean(axis=0)) / np.where(
        matrix.std(axis=0) > 0, matrix.std(axis=0), 1.0)
    _, _, right = np.linalg.svd(matrix - matrix.mean(axis=0), full_matrices=False)
    component = matrix @ right[0]
    component = (component - component.mean()) / component.std()

    hazard = 1.0 / max(args.hazard_bars, 2)
    probability = mk.bocpd_run_length(component, hazard=hazard,
                                      max_run=args.max_run)
    bocpd_points = mk.pick_changepoints(probability, args.cp_share,
                                        args.cp_min_distance)
    graph_points = np.flatnonzero(aligned["is_transition"].to_numpy())
    if len(graph_points) == 0 or len(bocpd_points) == 0:
        return {"symbol": symbol, "skipped": "нет точек для сравнения"}

    observed = mk.overlap_rate(graph_points, np.sort(bocpd_points), TOLERANCE)
    rng = np.random.default_rng([2026, zlib.crc32(f"{symbol}|m3".encode())])
    null = mk.shift_null(graph_points, np.sort(bocpd_points), TOLERANCE,
                         len(aligned), args.n_boot_shift, rng)
    p_value = float((1 + np.sum(null >= observed)) / (1 + len(null)))
    median = float(np.median(null))
    return {
        "symbol": symbol, "n_bars": len(aligned),
        "n_graph": len(graph_points), "n_bocpd": len(bocpd_points),
        "observed": observed, "null_median": median,
        "ratio": observed / median if median > 0 else float("inf"),
        "p": p_value,
        "passed": p_value <= 0.05 and (median <= 0 or observed / median >= 1.5),
    }


def format_m1(results: list[dict]) -> str:
    lines = ["", "=" * 96,
             "M1. IMPLIED TIMESCALES И ТЕСТ ЧЕПМЕНА — КОЛМОГОРОВА",
             "=" * 96,
             "Матрица переходов при ФИКСИРОВАННОМ лаге (это не `transitions` "
             "конвейера —",
             "там переход считается по событию смены состояния). Время в барах "
             "базового ТФ.",
             "Критерий: плато в пределах 20% на τ ∈ [2τ*, 5τ*] и CK внутри "
             "блочной полосы.", "",
             f"{'монета':<10} {'состояний':>10} {'τ*':>4} {'окно плато':>12} "
             f"{'откл. t1':>9} {'откл. t2':>9} {'откл. t3':>9} {'CK макс.':>9} "
             f"{'рост t1':>8} {'вердикт':>9}"]
    lines.append("─" * 96)
    for r in results:
        if r.get("tau_star") is None:
            lines.append(f"{r['symbol']:<10} {r.get('reason', '')}")
            continue
        window = f"[{r['lag_window'][0]}, {r['lag_window'][1]}]"
        dev = r["deviation"]
        ck_max = float(r["ck"]["error"].max()) if not r["ck"].empty else float("nan")
        lines.append(
            f"{r['symbol']:<10} {r['n_states']:>10} {r['tau_star']:>4} "
            f"{window:>12} " + " ".join(
                f"{dev[c]:>9.3f}" if not np.isnan(dev[c]) else f"{'н/д':>9}"
                for c in ("t1", "t2", "t3"))
            + f" {ck_max:>9.3f} {r.get('growth', float('nan')):>8.1f}"
              f" {'ДА' if r['passed'] else 'нет':>9}"
        )

    lines += ["", "КРИВЫЕ (время медленнейшего процесса, в барах)"]
    for r in results:
        if r.get("curve") is None or r["curve"].empty:
            continue
        pairs = "  ".join(
            f"{int(row['lag'])}:{row['t1']:.0f}" if not np.isnan(row["t1"])
            else f"{int(row['lag'])}:—"
            for _, row in r["curve"].iterrows())
        lines.append(f"  {r['symbol']:<10} {pairs}")
    passed = sum(1 for r in results if r.get("passed"))
    lines += ["", f"  ВЕРДИКТ M1: прошло {passed} монет из {len(results)} "
                  f"(критерий — ≥4 из 6)"]
    lines.append("  " + ("РАЗМЕТКА МАРКОВСКАЯ" if passed >= 4 else
                         "РАЗМЕТКА НЕ МАРКОВСКАЯ — счётчики переходов описывают "
                         "не процесс"))
    lines += ["",
              "  «Рост t1» — во сколько раз время медленнейшего процесса на "
              "самом длинном",
              "  определённом лаге больше, чем на самом коротком. У марковской "
              "разметки он",
              "  равен единице ПО ОПРЕДЕЛЕНИЮ: плато и означает независимость "
              "от лага.",
              "  Эта величина в критерии не заявлена и вердикта не меняет — она "
              "показывает,",
              "  насколько далеко ответ от границы, потому что окно [2τ*, 5τ*] "
              "на липкой",
              "  разметке вырождается в пять баров (τ* всегда 1) и само по себе "
              "мало что значит."]
    return "\n".join(lines)


def format_m2(results: list[dict]) -> str:
    lines = ["", "=" * 96,
             "M2. ПОРЯДОК МАРКОВОСТИ: НУЖЕН ЛИ `prev_prev` В КЛЮЧЕ",
             "=" * 96,
             "I(S_{t+1}; S_{t−1} | S_t) в натах. Оценка СМЕЩЕНА ВВЕРХ на "
             "конечной выборке —",
             "сравнивать с нулём нельзя, сравнение идёт с нулёвкой той же "
             "оценки.",
             "Критерий: выше p95 нулёвки И не менее чем в 1.5 раза выше её "
             "медианы.", "",
             f"{'монета':<10} {'ряд':<8} {'длина':>8} {'CMI':>9} "
             f"{'нулёвка p50':>12} {'нулёвка p95':>12} {'отношение':>10} "
             f"{'вердикт':>9}", "─" * 96]
    for r in results:
        for name in ("бары", "скачки"):
            value = r.get(name)
            if not value:
                continue
            ok = value["passed"]
            lines.append(
                f"{r['symbol']:<10} {name:<8} {value['n']:>8} "
                f"{value['observed']:>9.4f} {value['null_median']:>12.4f} "
                f"{value['null_p95']:>12.4f} {value['ratio']:>10.2f} "
                f"{'да' if ok else 'нет':>9}"
            )
    by_bars = sum(1 for r in results if (r.get("бары") or {}).get("passed"))
    passed = sum(1 for r in results if r.get("passed"))
    lines += ["",
              f"  По барам прошло {by_bars} монет из {len(results)}; "
              f"по цепи скачков — {passed}.",
              "  Вердикт считается ПО ЦЕПИ СКАЧКОВ. Ряд по барам мерит прежде "
              "всего липкость:",
              "  состояние держится десятками баров, S_{t−1} почти всегда равно "
              "S_t, и оценка",
              "  на нём выходит НИЖЕ нулёвки — не потому, что памяти нет, а "
              "потому, что в этом",
              "  ряду её негде увидеть. Предупреждение об этом стоит в шапке "
              "`markov.py`",
              "  и написано ДО прогона.", "",
              f"  ВЕРДИКТ M2: прошло {passed} монет из {len(results)} "
              f"(критерий — ≥4 из 6)"]
    lines.append("  " + ("ГЛУБИНЫ ПАРЫ НЕДОСТАТОЧНО — есть основание ставить "
                         "вопрос о тройке" if passed >= 4 else
                         "ПАРЫ ДОСТАТОЧНО — вопрос о `prev_prev` закрывается "
                         "ссылкой на этот замер"))
    return "\n".join(lines)


def format_m3(results: list[dict]) -> str:
    lines = ["", "=" * 96,
             "M3. BOCPD КАК НЕЗАВИСИМЫЙ КОНТРОЛЬ",
             "=" * 96,
             "Вторая разметка, не пользующаяся НИ ОДНОЙ деталью кластеризации.",
             "«Случайно» рядом с «совпало» — не украшение: если помечено много "
             "баров, случайный",
             "уровень подходит к единице и тест теряет мощность независимо от "
             "данных.",
             "Доля переходов графа, рядом с которыми (±2 бара) есть смена "
             "режима по BOCPD.",
             "Нулёвка — циклический сдвиг (не перестановка: оба ряда идут "
             "сериями).", "",
             f"{'монета':<10} {'баров':>9} {'переходов':>10} {'точек CP':>9} "
             f"{'совпало':>9} {'случайно':>9} {'отношение':>10} {'p':>8} "
             f"{'вердикт':>9}", "─" * 96]
    for r in results:
        if r.get("skipped"):
            lines.append(f"{r['symbol']:<10} пропущена: {r['skipped']}")
            continue
        lines.append(
            f"{r['symbol']:<10} {r['n_bars']:>9} {r['n_graph']:>10} "
            f"{r['n_bocpd']:>9} {r['observed']:>9.3f} {r['null_median']:>9.3f} "
            f"{r['ratio']:>10.2f} {r['p']:>8.4f} "
            f"{'да' if r['passed'] else 'нет':>9}"
        )
    counted = [r for r in results if not r.get("skipped")]
    passed = sum(1 for r in counted if r.get("passed"))
    lines += ["", f"  ВЕРДИКТ M3: прошло {passed} монет из {len(counted)} "
                  f"(критерий — ≥4 из 6)"]
    lines.append("  " + ("`is_transition` — наблюдаемое событие: независимый "
                         "детектор видит его тоже" if passed >= 4 else
                         "`is_transition` НЕ подтверждается независимым "
                         "детектором"))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--run", type=int)
    parser.add_argument("--task", action="append", choices=["m1", "m2", "m3"])
    parser.add_argument("--n-boot-band", type=int, default=200)
    parser.add_argument("--n-boot-cmi", type=int, default=100)
    parser.add_argument("--n-boot-shift", type=int, default=500)
    parser.add_argument("--hazard-bars", type=int, default=500,
                        help="ожидаемая длина режима в барах для BOCPD")
    parser.add_argument("--max-run", type=int, default=1500)
    parser.add_argument("--cp-share", type=float, default=0.02,
                        help="доля баров, объявляемых точками смены по BOCPD")
    parser.add_argument("--cp-min-distance", type=int, default=24,
                        help="минимальный разнос точек смены в барах; без него "
                             "всплеск даёт десятки соседних точек и тест теряет "
                             "мощность (см. pick_changepoints)")
    args = parser.parse_args()

    tasks = args.task or ["m1", "m2", "m3"]
    specs = symbols.resolve_many(args.symbol, args.all)
    print(f"Марковость графа. Монет {len(specs)}, задачи {tasks}.")
    print("Критерии заявлены ДО запуска — см. шапку скрипта.\n")

    m1_rows, m2_rows, m3_rows = [], [], []
    for spec in specs:
        symbol = spec.ticker
        print(f"=== {symbol}")
        try:
            model_run = samples.resolve_model_run(symbol, args.run)
        except SystemExit as exc:
            print(f"  пропущена: {exc}")
            continue
        frame = load_states(symbol, model_run)
        if frame.empty:
            print("  пропущена: нет разметки")
            continue
        gaps = gap_stats(frame)
        codes, n_states = as_codes(frame["group_id"])
        print(f"  модель #{model_run}, баров {gaps['rows']}, состояний "
              f"{n_states}, разрывов {gaps['gaps']} "
              f"({gaps['gap_share']:.4%}, максимум {gaps['max_gap_hours']:.1f} ч)")

        if "m1" in tasks:
            row = task_m1(symbol, codes, n_states, args)
            row["n_states"] = n_states
            m1_rows.append(row)
            print(f"  M1: τ*={row.get('tau_star')}, плато "
                  f"{'ок' if row.get('plateau_ok') else 'нет'}, CK "
                  f"{'ок' if row.get('ck_ok') else 'нет'}")
        if "m2" in tasks:
            row = task_m2(symbol, codes, n_states, args)
            m2_rows.append(row)
            print(f"  M2: {'да' if row['passed'] else 'нет'}")
        if "m3" in tasks:
            row = task_m3(symbol, frame, model_run, args)
            m3_rows.append(row)
            print(f"  M3: {row.get('skipped') or ('да' if row['passed'] else 'нет')}")

    if m1_rows:
        print(format_m1(m1_rows))
    if m2_rows:
        print(format_m2(m2_rows))
    if m3_rows:
        print(format_m3(m3_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
