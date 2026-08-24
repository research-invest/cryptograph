"""
Суррогатный прогон конвейера размаха: сколько «находок» даёт сама процедура.

    python3 scripts/surrogate_range.py --symbol BTCUSDT --replicas 20
    python3 scripts/surrogate_range.py --symbol BTCUSDT --method block

ТЗ — `crypto-graph/docs/tz_wave_a_24-08-26.md`, задача S. Устройство суррогата
и обоснование каждого решения — шапка `btcproc/analysis/surrogate.py`.

## Что именно проверяется

Берётся **неизменённый** стенд `fit_range_forecast.run_cell` — те же
гиперпараметры бустинга, то же разбиение 70/30 с зазором в горизонт, тот же
блочный бутстрап, тот же критерий из трёх условий, — и подаётся на вход
суррогатная история вместо настоящей. Признаки пересчитываются на суррогате:
иначе проверялся бы не конвейер, а только его хвост.

Это единственная работа, способная **опровергнуть** единственный
положительный вердикт проекта (раздел 49). Пока она не сделана, цена всех
остальных вердиктов неизвестна.

## Критерий, заявляемый ДО запуска

> Вердикт раздела 49 сохраняет силу, если
>
> 1. **доля суррогатных реплик, прошедших все три условия критерия, не
>    превышает 0.10** (при 20 репликах — не более двух), и
> 2. **медианное ΔR² суррогата не превышает 0.005** — четверти порога
>    практической величины 0.02.
>
> Если проходит больше — раздел 49 подлежит перечтению, а регрессор размаха
> отключению из конвейера до выяснения. Записывается прямо, каким бы
> неудобным ни было.

**MDE, объявляемый до прогона.** При 20 репликах разрешение по доле — 0.05.
Прогон способен различить «около нуля» и «заметно больше номинала» и НЕ
способен различить 0.05 и 0.10. Этого достаточно для поставленного вопроса и
недостаточно для точной оценки уровня теста; так и записывается.

## Чего этот прогон не делает

Не обучает боевых моделей, ничего не пишет ни в БД, ни в файлы, не трогает
`range_models`. Прогон читает бары и считает.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ДО импорта btcproc.config: конфиг читает окружение на импорте.
os.environ["SINK_MODE"] = "none"

import numpy as np  # noqa: E402

from btcproc import config, symbols  # noqa: E402
from btcproc.analysis import range_model as rm  # noqa: E402
from btcproc.analysis import surrogate as sg  # noqa: E402
from btcproc.analysis.lift import DEFAULT_N_BOOT  # noqa: E402
from btcproc.features import builder as feat  # noqa: E402
from btcproc.ingest import bars  # noqa: E402

import fit_range_forecast as frf  # noqa: E402

#: Порог доли прошедших реплик и порог медианного ΔR², оба из критерия выше.
MAX_PASS_SHARE = 0.10
MAX_MEDIAN_DELTA = 0.005


class Args:
    """Минимальный носитель параметров для `frf.run_cell` — он их ждёт объектом."""

    def __init__(self, train_frac: float, n_boot: int, augment: bool = False):
        self.train_frac = train_frac
        self.n_boot = n_boot
        self.save = False
        self.augment_benchmark = augment


def cell_verdict(row: dict) -> tuple[bool, tuple[bool, bool, bool]]:
    """
    Три условия критерия раздела 49 для ОДНОЙ ячейки.

    BH здесь не применяется намеренно, и это делает тест **консервативным в
    нужную сторону**: у реального прогона поправка считалась по 72 ячейкам и
    только ужесточала условие 1. Суррогатная реплика оценивается сама по себе,
    то есть ей дают пройти легче, чем настоящей. Если и так не проходит —
    вывод крепче.
    """
    ok1 = row["delta"] >= frf.MIN_DELTA_R2 and row["p_delta"] <= 0.05
    ok2 = row["p_better"] <= 0.05 and row["rho"] > row["rho_bench"]
    ok3 = row["coverage_error"] <= frf.MAX_COVERAGE_ERROR
    return (ok1 and ok2 and ok3), (ok1, ok2, ok3)


def run_replica(symbol: str, base, replica: int, method: str, args, cli) -> dict | None:
    seed_material = f"{symbol}|{method}|{replica}".encode()
    rng = np.random.default_rng([2026, zlib.crc32(seed_material)])
    fake = sg.surrogate_bars(base, method, rng,
                             block=rm.horizon_bars(cli.horizon, config.data.base_minutes))

    # Контекстные ТФ пересобираются ИЗ СУРРОГАТА, а не берутся настоящими:
    # настоящий старший ТФ рядом с суррогатным базовым — это готовая утечка
    # реальной структуры в нулёвку, причём именно той структуры (медленная
    # волатильность), ради разрушения которой суррогат и делается.
    context = {tf: _resample(fake, tf) for tf in config.data.context_tfs}
    features = feat.build_features(fake, context, symbol=symbol)

    return frf.run_cell(symbol, fake, features, cli.horizon, cli.norm,
                        cli.seed, args)


def _resample(base, tf: str):
    """Старший ТФ из базового — тем же способом, что и у настоящих данных."""
    rule = tf.replace("m", "min").replace("h", "h").replace("d", "D")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum"}
    columns = {c: how for c, how in agg.items() if c in base.columns}
    out = base.resample(rule, label="left", closed="left").agg(columns).dropna()
    return out


def format_report(rows: list[dict], method: str, cli) -> str:
    if not rows:
        return "\nНи одной реплики не посчитано."

    verdicts = [cell_verdict(r) for r in rows]
    passed = [r for r, (ok, _) in zip(rows, verdicts) if ok]
    deltas = np.array([r["delta"] for r in rows], dtype=float)
    share = len(passed) / len(rows)

    header = (f"{'#':>3} {'n_test':>7} {'R²(B2)':>9} {'R²(мод.)':>10} {'ΔR²':>9} "
              f"{'p':>7} {'rho':>7} {'rho(B2)':>8} {'p(лучше)':>9} "
              f"{'откл.':>7} {'усл.':>6}")
    lines = ["", "=" * len(header),
             f"СУРРОГАТНЫЙ ПРОГОН КОНВЕЙЕРА РАЗМАХА — {method.upper()}",
             "=" * len(header),
             f"{cli.symbol} {cli.horizon} {cli.norm} зерно {cli.seed}; "
             f"реплик {len(rows)}.",
             "Данные, в которых предсказывать нечего. Всё, что здесь «проходит»,",
             "порождено процедурой, а не рынком.", "",
             header, "─" * len(header)]
    for i, (r, (ok, conds)) in enumerate(zip(rows, verdicts), start=1):
        marks = "".join("+" if c else "·" for c in conds)
        lines.append(
            f"{i:>3} {r['n_test']:>7} {r['r2_bench']:>+9.4f} {r['r2_full']:>+10.4f} "
            f"{r['delta']:>+9.4f} {r['p_delta']:>7.4f} {r['rho']:>+7.3f} "
            f"{r['rho_bench']:>+8.3f} {r['p_better']:>9.4f} "
            f"{r['coverage_error']:>7.3f} {marks:>6}"
        )

    lines += ["", "УСЛОВИЯ ПО ОТДЕЛЬНОСТИ (сколько реплик прошло каждое)"]
    for i, name in enumerate((
        f"1) ΔR² ≥ {frf.MIN_DELTA_R2} и p ≤ 0.05",
        "2) ранжирует лучше бенчмарка",
        f"3) покрытие квантилей в пределах {frf.MAX_COVERAGE_ERROR}",
    )):
        got = sum(1 for _, conds in verdicts if conds[i])
        lines.append(f"  {name}: {got} из {len(rows)}")

    lines += ["", "ВЕРДИКТ ПО ЗАЯВЛЕННОМУ КРИТЕРИЮ",
              f"  прошли все три условия: {len(passed)} из {len(rows)} "
              f"(доля {share:.2f}, порог {MAX_PASS_SHARE})",
              f"  медианное ΔR²: {np.median(deltas):+.4f} "
              f"(порог {MAX_MEDIAN_DELTA}); разброс "
              f"[{deltas.min():+.4f}, {deltas.max():+.4f}]",
              f"  MDE по доле при {len(rows)} репликах: {1 / len(rows):.2f} — "
              f"различить 0.05 и 0.10 этот прогон не может"]
    ok = share <= MAX_PASS_SHARE and float(np.median(deltas)) <= MAX_MEDIAN_DELTA
    lines.append("  ВЕРДИКТ: " + (
        "ВЫВОД РАЗДЕЛА 49 УСТОЯЛ — процедура не порождает такой эффект сама"
        if ok else
        "ВЫВОД РАЗДЕЛА 49 ПОД ВОПРОСОМ — процедура порождает эффект на пустых данных"
    ))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--horizon", default="24h", choices=list(rm.HORIZONS))
    parser.add_argument("--norm", default="atr14", choices=list(rm.NORMALIZATIONS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", default="iaaft",
                        choices=list(sg.METHODS) + ["both"],
                        help="both = iaaft + block, то есть два способа, "
                             "разрушающих ВСЮ структуру. `seasonal` сюда не "
                             "входит намеренно: он сохраняет суточный профиль "
                             "и предназначен для задачи P, где сезонность "
                             "разрушать нельзя")
    parser.add_argument("--replicas", type=int, default=20)
    parser.add_argument("--end", default=frf.FROZEN_END)
    parser.add_argument("--start", default=None,
                        help="начало окна; по умолчанию — дата листинга монеты")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument(
        "--augment-benchmark", action="store_true",
        help="добавить в бенчмарк логарифм знаменателя цели — диагностика "
             "информационной асимметрии, см. rf.denominator_column")
    parser.add_argument("--dump", help="сложить реплики в JSON")
    cli = parser.parse_args()

    spec = symbols.get(cli.symbol)
    start = cli.start or spec.start_date()
    base = bars.load_ohlcv(cli.symbol, config.data.base_tf, start, cli.end)
    if base.empty:
        raise SystemExit(f"{cli.symbol}: нет баров до {cli.end}")

    # `both` — это ровно два способа, разрушающих всю структуру. Писать здесь
    # `list(sg.METHODS)` нельзя: `seasonal` сохраняет суточный профиль по
    # построению, и как нулёвка для «предсказывать нечего» он не годится.
    methods = ["iaaft", "block"] if cli.method == "both" else [cli.method]
    print(f"Суррогатный прогон. {cli.symbol} {cli.horizon} {cli.norm} "
          f"зерно {cli.seed}; баров {len(base)}; способы {methods}; "
          f"реплик {cli.replicas} на способ.")
    print("Критерий и MDE заявлены ДО запуска — см. шапку скрипта.\n")

    args = Args(cli.train_frac, cli.n_boot, cli.augment_benchmark)
    everything: dict[str, list[dict]] = {}
    for method in methods:
        rows: list[dict] = []
        for replica in range(cli.replicas):
            started = time.time()
            print(f"  [{method} {replica + 1}/{cli.replicas}] суррогат, признаки, обучение…")
            row = run_replica(cli.symbol, base, replica, method, args, cli)
            if row:
                row["replica"] = replica
                row["method"] = method
                rows.append(row)
            print(f"      {time.time() - started:.0f} с")
        everything[method] = rows
        print(format_report(rows, method, cli))

    if cli.dump:
        import json
        Path(cli.dump).write_text(
            json.dumps(everything, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"\nРеплики сохранены в {cli.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
