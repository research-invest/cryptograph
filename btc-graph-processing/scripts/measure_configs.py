"""
FDR по всем конфигурациям: есть ли хоть одна значимая.

    python3 scripts/measure_configs.py --all
    python3 scripts/measure_configs.py --symbol BTCUSDT --alpha 0.05

ТЗ — `crypto-graph/docs/tz_wave_a_24-08-26.md`, задача F. Методика,
устройство нулёвки и объяснение, почему p-value не берётся прямо из реплик, —
шапка `btcproc/analysis/configs.py`.

## Предсказание, записанное ДО прогона

> При FDR 10% выживут **единицы или ноль** конфигураций на монету.

Предсказание записано именно так, чтобы результат можно было засчитать. Если
выживут десятки — предсказание не сбылось, и это тоже результат, который
надо печатать словами, а не пересматривать порог.

## Как читается любой исход

* **выжили единицы** — вот они, с их `n` и долей; их разбирают штучно, как
  отдельные находки, а не как поток кандидатов;
* **ноль** — фильтр кандидатов честнее заменить на разметку: система
  показывает, что было, и не делает вида, что отбирает.

Ни один из двух исходов не является поводом что-либо чинить в генераторе.

## Что прогон НЕ делает

Не переобучает модель, ничего не пишет в БД, не трогает кандидатов. Читает
`bar_states`, `bar_events` и `outcomes` одной моделью состояний
(`runs.model_run_scope`) и считает.
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
from btcproc.analysis import configs as cf  # noqa: E402
from btcproc.analysis import samples  # noqa: E402
from btcproc.analysis.lift import (  # noqa: E402
    DEFAULT_N_BOOT, benjamini_hochberg, block_length_rows,
)
from btcproc.db import runs as runs_repo  # noqa: E402
from btcproc.db.session import fetch_all  # noqa: E402

SQL = """
SELECT s.ts,
       s.transition_id || '|' || e.event_block_id AS key,
       CASE WHEN o.is_up THEN 1.0 ELSE 0.0 END    AS outcome
FROM bar_states s
JOIN bar_events e ON e.symbol = s.symbol AND e.ts = s.ts
JOIN outcomes  o ON o.symbol = s.symbol AND o.ts = s.ts AND o.horizon = %s
WHERE s.symbol = %s
  AND s.is_transition
  AND s.transition_id IS NOT NULL
  AND o.valid
  AND o.is_up IS NOT NULL
  AND {scope}
ORDER BY s.ts
"""


def load(symbol: str, model_run: int) -> pd.DataFrame:
    scope_sql, scope_params = runs_repo.model_run_scope(model_run, "s")
    rows = fetch_all(SQL.format(scope=scope_sql),
                     (config.data.horizon, symbol, *scope_params))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return frame


def run_symbol(symbol: str, args) -> dict | None:
    model_run = samples.resolve_model_run(symbol, args.run)
    frame = load(symbol, model_run)
    if frame.empty:
        print(f"  [{symbol}] нет реализаций переходов с исходами — пропуск")
        return None

    keys = frame["key"].to_numpy()
    outcomes = frame["outcome"].to_numpy(dtype=float)
    block = block_length_rows(frame["ts"], config.data.horizon_minutes)
    rng = np.random.default_rng([2026, zlib.crc32(symbol.encode())])

    table = cf.observed_z(keys, outcomes, args.min_rows)
    if table.empty:
        print(f"  [{symbol}] ни один ключ не набрал {args.min_rows} реализаций")
        return None

    scale, scale_rows = cf.null_scale(keys, outcomes, block, args.n_boot, rng,
                                      args.min_rows)
    p_values = cf.scaled_p_values(table, scale)
    table = table.assign(p=p_values)
    rejected = benjamini_hochberg(list(table["p"]), args.alpha)
    table = table.assign(bh=rejected)

    # Негативный контроль — та же процедура на блочно переставленных исходах.
    control_keys, control_outcomes = cf.permuted_control(keys, outcomes, block, rng,
                                                         args.min_rows)
    control_table = cf.observed_z(control_keys, control_outcomes, args.min_rows)
    control_p = cf.scaled_p_values(control_table, scale)
    control_rejected = sum(benjamini_hochberg(list(control_p), args.alpha))

    return {
        "symbol": symbol, "model_run": model_run,
        "n_rows": len(frame), "n_keys_total": int(frame["key"].nunique()),
        "n_keys_tested": len(table), "base_rate": cf.base_rate(outcomes),
        "block": block, "table": table, "scale": scale_rows,
        "survivors": int(sum(rejected)), "control_survivors": int(control_rejected),
        "span": (frame["ts"].min(), frame["ts"].max()),
    }


def format_report(results: list[dict], args) -> str:
    if not results:
        return "\nНи одной монеты не посчитано."

    lines = ["", "=" * 100,
             "ЗНАЧИМЫЕ КОНФИГУРАЦИИ ПОСЛЕ ПОПРАВКИ НА МНОЖЕСТВЕННОСТЬ",
             "=" * 100,
             f"Ключ — (transition_id, event_block_id). Одна строка = одна "
             f"РЕАЛИЗАЦИЯ перехода,",
             f"а не снимок. Порог выборки {args.min_rows}; FDR по Бенджамини — "
             f"Хохбергу при α = {args.alpha}.",
             "Предсказание, записанное до прогона: выживут единицы или ноль.", "",
             f"{'монета':<10} {'модель':>7} {'реализ.':>9} {'ключей':>8} "
             f"{'тестир.':>8} {'ставка':>8} {'блок':>6} {'выжило':>7} "
             f"{'контроль':>9}",
             "─" * 100]
    for r in results:
        lines.append(
            f"{r['symbol']:<10} {r['model_run']:>7} {r['n_rows']:>9} "
            f"{r['n_keys_total']:>8} {r['n_keys_tested']:>8} "
            f"{r['base_rate']:>8.3f} {r['block']:>6} {r['survivors']:>7} "
            f"{r['control_survivors']:>9}"
        )

    lines += ["", "РАЗДУТИЕ ДИСПЕРСИИ НУЛЁВКИ (σ null-z по бинам размера выборки)",
              "σ ≈ 1 означало бы, что зависимость наблюдений несущественна."]
    for r in results:
        parts = [f"{int(row['bin']) if row['bin'] < 10**8 else '∞':>6}:"
                 f"{row['sigma']:.2f}" for _, row in r["scale"].iterrows()]
        lines.append(f"  {r['symbol']:<10} " + "  ".join(parts))

    lines += ["", "СИЛЬНЕЙШИЕ КЛЮЧИ (до пяти на монету, независимо от вердикта)"]
    for r in results:
        lines.append(f"  {r['symbol']} (базовая ставка {r['base_rate']:.3f}):")
        head = r["table"].head(5)
        if head.empty:
            lines.append("    — нет ключей, прошедших порог выборки")
        for key, row in head.iterrows():
            lines.append(
                f"    {key:<28} n={int(row['n']):>4} доля={row['share']:.3f} "
                f"({row['diff']:+.3f}) z={row['z']:+.2f} p={row['p']:.2e} "
                f"{'ВЫЖИЛ' if row['bh'] else ''}"
            )

    total = sum(r["survivors"] for r in results)
    control = sum(r["control_survivors"] for r in results)
    lines += ["", "ВЕРДИКТ",
              f"  выжило конфигураций всего: {total} "
              f"(на негативном контроле: {control})"]
    if control > total:
        lines.append("  ВНИМАНИЕ: контроль дал больше основного прогона — "
                     "процедура анти-консервативна, читать результат нельзя.")
    elif total == 0:
        lines.append("  НИ ОДНОЙ. Предсказание сбылось в самой сильной форме.")
        lines.append("  Следствие, записанное в ТЗ до прогона: фильтр кандидатов")
        lines.append("  честнее заменить на разметку — система показывает, что")
        lines.append("  было, и не делает вида, что отбирает.")
    elif total <= 3 * len(results):
        lines.append("  ЕДИНИЦЫ. Предсказание сбылось. Выжившие ключи разбираются")
        lines.append("  штучно, как отдельные находки, а не как поток кандидатов.")
    else:
        lines.append("  ДЕСЯТКИ — предсказание НЕ сбылось. Порог не пересматривать:")
        lines.append("  результат записывается как есть и требует отдельного разбора.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--run", type=int, help="прогон модели; по умолчанию последний train")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--min-rows", type=int, default=cf.MIN_ROWS)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT // 4,
                        help="реплик для оценки раздутия дисперсии; здесь их "
                             "нужно на порядок меньше, чем для p-value")
    parser.add_argument("--dump", help="сложить таблицы ключей в CSV-каталог")
    args = parser.parse_args()

    specs = symbols.resolve_many(args.symbol, args.all)
    print(f"FDR по конфигурациям. Монет {len(specs)}, α = {args.alpha}, "
          f"порог выборки {args.min_rows}.")
    print("Предсказание заявлено ДО запуска — см. шапку скрипта.\n")

    results = []
    for spec in specs:
        print(f"=== {spec.ticker}")
        try:
            result = run_symbol(spec.ticker, args)
        except SystemExit as exc:
            print(f"  пропущена: {exc}")
            continue
        if result:
            print(f"  реализаций {result['n_rows']}, ключей "
                  f"{result['n_keys_total']} (тестируется {result['n_keys_tested']}), "
                  f"выжило {result['survivors']}")
            results.append(result)

    if args.dump:
        directory = Path(args.dump)
        directory.mkdir(parents=True, exist_ok=True)
        for r in results:
            r["table"].to_csv(directory / f"{r['symbol']}_configs.csv")
        print(f"\nТаблицы ключей сохранены в {directory}")

    print(format_report(results, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
