"""
Замер лифта по атомам событий на накопленных кандидатах.

Отвечает на один вопрос: отличаются ли исходы кандидатов, у бара которых был
атом A, от исходов всех остальных. Методика — docs/task_smc_integration.md,
раздел 8; реализация — btcproc/analysis/lift.py.

    python3 scripts/measure_atom_lift.py                      # BTC, доли из выборки
    python3 scripts/measure_atom_lift.py --symbol ETHUSDT
    python3 scripts/measure_atom_lift.py --metric realized    # фактические исходы
    python3 scripts/measure_atom_lift.py --correction bh --alpha 0.1
    python3 scripts/measure_atom_lift.py --context-only       # только фоновые атомы

Две метрики, и разница между ними принципиальная:

  long_outcome_share (по умолчанию) — то, что записано в кандидате: доля
    исторических аналогов, закрывшихся вверх. Это **ожидание системы**, а не
    факт. Метрика из ТЗ, но у неё есть слабое место: кандидаты пересекаются
    по своим историческим выборкам, поэтому n завышает число независимых
    наблюдений, и тест анти-консервативен. Значимость по ней — повод
    посмотреть внимательнее, а не вывод.

  realized — фактический исход бара кандидата из processing.outcomes
    (is_up на горизонте). Одно независимое наблюдение на бар, честный
    Бернулли. Считать выводы стоит по ней; выборка меньше, потому что
    берутся только созревшие метки (valid).

Фоновые атомы участвуют в замере наравне с signature — ради этого чинилась
колонка context_atoms. До этой правки контекст в БД не сохранялся, и на
старых барах он NULL: такие строки скрипт исключает и говорит, сколько их.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from btcproc import config, symbols  # noqa: E402
from btcproc.analysis.lift import format_table, measure_lift  # noqa: E402
from btcproc.db import runs as runs_repo  # noqa: E402
from btcproc.db.session import fetch_all  # noqa: E402
from btcproc.features import events as ev  # noqa: E402

# Кандидат и бар связаны напрямую: candidates.ts — колонка, а не только
# payload._meta.ts, поэтому джойн идёт по (symbol, ts) без разбора JSON.
QUERY_EXPECTED = """
SELECT c.ts,
       (c.payload->>'long_outcome_share')::float8 AS metric,
       e.atoms,
       e.context_atoms
FROM candidates c
JOIN bar_events e ON e.symbol = c.symbol AND e.ts = c.ts
WHERE c.symbol = %s
  AND c.payload->>'long_outcome_share' IS NOT NULL
  AND {scope}
ORDER BY c.ts
"""

QUERY_REALIZED = """
SELECT c.ts,
       CASE WHEN o.is_up THEN 1.0 ELSE 0.0 END AS metric,
       e.atoms,
       e.context_atoms
FROM candidates c
JOIN bar_events e ON e.symbol = c.symbol AND e.ts = c.ts
JOIN outcomes o ON o.symbol = c.symbol AND o.ts = c.ts AND o.horizon = %s
WHERE c.symbol = %s
  AND o.valid
  AND o.is_up IS NOT NULL
  AND {scope}
ORDER BY c.ts
"""


def resolve_model_run(symbol: str, run_id: int | None) -> int:
    """Прогон, задающий модель. По умолчанию — последний успешный train монеты."""
    if run_id:
        return run_id
    latest = runs_repo.latest_completed_run("train", symbol)
    if not latest:
        raise SystemExit(f"У {symbol} нет завершённого train — мерить нечего.")
    return int(latest["run_id"])


def load(symbol: str, metric: str, model_run: int) -> pd.DataFrame:
    """
    Кандидаты ОДНОЙ модели состояний.

    Ограничение обязательно, и это не осторожность, а исправление ошибки.
    Замер без него брал всех кандидатов монеты, а в базе их выпускали разные
    модели: полные прогоны и частичные (`train --start ...`). Частичные
    покрывают только свежий период, поэтому скапливаются в holdout и удваивают
    там плотность кандидатов.

    Как это выглядело: на BTC ни один атом не пережил holdout, корреляция
    лифтов между половинами −0.09, знак совпал у 32% атомов — хуже подбрасывания
    монеты. На ETH и SOL, где лишних прогонов не было, корреляция +0.63 и +0.61.
    Списать это на «рынок BTC изменился» было очень легко и совершенно неверно.
    """
    scope_sql, scope_params = runs_repo.model_run_scope(model_run)
    scope_sql = scope_sql.replace("run_id", "c.run_id", 1)
    if metric == "realized":
        sql = QUERY_REALIZED.format(scope=scope_sql)
        rows = fetch_all(sql, (config.data.horizon, symbol, *scope_params))
    else:
        sql = QUERY_EXPECTED.format(scope=scope_sql)
        rows = fetch_all(sql, (symbol, *scope_params))
    return pd.DataFrame(rows)


def consolidate(per_symbol: dict[str, list]) -> None:
    """
    Сводка по монетам: где атом прошёл, где нет и совпал ли знак.

    Это более сильная проверка, чем holdout внутри одной монеты. Holdout
    отделяет позднюю историю от ранней, но рынок один и тот же; три монеты —
    три независимых рынка. Эффект, который держится на всех трёх с одним
    знаком, гораздо труднее объяснить подгонкой.
    """
    symbols_list = list(per_symbol)
    atoms: dict[str, dict] = {}
    for symbol, results in per_symbol.items():
        for r in results:
            atoms.setdefault(r.atom, {})[symbol] = r

    def rank(name: str) -> tuple:
        """Сначала прошедшие, затем согласные по знаку, затем по силе эффекта."""
        rows = atoms[name]
        passed = sum(1 for r in rows.values() if r.confirmed)
        one_sign = 0 if len({r.lift > 0 for r in rows.values()}) == 1 else 1
        return (-passed, one_sign, -max(abs(r.z) for r in rows.values()))

    header = (f"\n{'атом':<24}"
              + "".join(f"{s.replace('USDT',''):>22}" for s in symbols_list)
              + f"{'прошёл':>9} {'знак':>7}")
    print("\n" + "=" * 78)
    print("СВОДКА ПО МОНЕТАМ — лифт (holdout), «ДА» = прошёл всё")
    print("=" * 78)
    print(header)
    print("─" * len(header))

    for name in sorted(atoms, key=rank):
        rows = atoms[name]
        cells = ""
        for symbol in symbols_list:
            r = rows.get(symbol)
            if r is None:
                cells += f"{'—':>22}"
                continue
            mark = "ДА" if r.confirmed else ("зн." if r.significant else "")
            cells += f"{r.lift:>+9.4f} ({r.holdout.lift:+.3f}){mark:>4}" if r.holdout \
                else f"{r.lift:>+16.4f}{mark:>6}"
        passed = sum(1 for r in rows.values() if r.confirmed)
        signs = {r.lift > 0 for r in rows.values()}
        agree = "один" if len(signs) == 1 else "разный"
        print(f"{name:<24}{cells}{passed:>6}/{len(rows)} {agree:>7}")

    print("\n«зн.» — значим после поправки, но holdout не подтвердил.")
    print("Столбец «знак» — совпало ли направление лифта на всех монетах.")


def run_one(symbol: str, args) -> list | None:
    """Замер по одной монете. Печатает таблицу, возвращает результаты."""
    model_run = resolve_model_run(symbol, args.run)
    frame = load(symbol, args.metric, model_run)
    if frame.empty:
        print(f"Нет данных по {symbol}. Нужен прогон train или live.")
        return None

    # NULL в context_atoms — бар размечен прогоном до появления колонки.
    stale = int(frame["context_atoms"].isna().sum())
    if stale:
        frame = frame[frame["context_atoms"].notna()]
        print(f"Пропущено {stale} баров без context_atoms (размечены до правки).")
        if frame.empty:
            print("Ни одного бара с контекстом. Нужен прогон после правки.")
            return None

    frame["atoms"] = [
        list(row["atoms"]) + list(row["context_atoms"])
        for _, row in frame.iterrows()
    ]

    if args.context_only:
        atoms = list(ev.CONTEXT_ATOM_LIST)
    elif args.signature_only:
        atoms = list(ev.SIGNATURE_ATOMS)
    else:
        atoms = list(ev.ATOMS)

    results = measure_lift(
        frame,
        atoms=atoms,
        metric_column="metric",
        alpha=args.alpha,
        correction=args.correction,
        holdout=args.holdout or None,
        min_group=args.min_group,
    )

    span = f"{frame['ts'].min():%Y-%m-%d} … {frame['ts'].max():%Y-%m-%d}"
    print(f"\n{symbol}: {len(frame)} кандидатов модели #{model_run}, {span}")
    print(f"Метрика: {args.metric}"
          + ("  (ожидание системы, не факт — см. шапку скрипта)"
             if args.metric == "long_outcome_share" else "  (фактические исходы)"))
    print(f"Базовая доля: {frame['metric'].mean():.4f}\n")
    print(format_table(results, args.correction, args.alpha))

    confirmed = [r.atom for r in results if r.confirmed]
    print(f"\nПрошли всё: {', '.join(confirmed) if confirmed else 'ни одного'}")
    if not confirmed:
        print("Отрицательный результат — валидный результат (раздел 14 ТЗ).")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Лифт по атомам событий")
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true",
                        help="Все активные монеты и сводка по ним")
    parser.add_argument("--run", type=int,
                        help="Прогон-модель; по умолчанию последний train монеты")
    parser.add_argument("--metric", choices=["long_outcome_share", "realized"],
                        default="long_outcome_share")
    parser.add_argument("--correction", choices=["bonferroni", "bh", "none"],
                        default="bonferroni")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--holdout", type=float, default=0.3,
                        help="Доля хвоста по времени на подтверждение; 0 — без него")
    parser.add_argument("--min-group", type=int, default=30,
                        help="Минимум наблюдений в каждой из групп")
    parser.add_argument("--context-only", action="store_true",
                        help="Только фоновые атомы")
    parser.add_argument("--signature-only", action="store_true",
                        help="Только атомы, входящие в event_block_id")
    args = parser.parse_args()

    if args.all:
        targets = symbols.tickers(only_enabled=True)
    elif args.symbol:
        targets = args.symbol
    else:
        targets = [config.data.symbol]

    per_symbol = {}
    for symbol in targets:
        results = run_one(symbol, args)
        if results:
            per_symbol[symbol] = results

    if not per_symbol:
        return 1
    if len(per_symbol) > 1:
        consolidate(per_symbol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
