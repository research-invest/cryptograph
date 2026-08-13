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
    python3 scripts/measure_atom_lift.py --thinned            # + sanity check
    python3 scripts/measure_atom_lift.py --no-bootstrap       # как мерили до 08-11

Две метрики, и разница между ними принципиальная:

  long_outcome_share (по умолчанию) — то, что записано в кандидате: доля
    исторических аналогов, закрывшихся вверх. Это **ожидание системы**, а не
    факт: она отвечает на вопрос «коррелирует ли атом с исторически
    перекошенными конфигурациями», а не «предсказывает ли атом исход».

  realized — фактический исход бара кандидата из processing.outcomes
    (is_up на горизонте). Свободна от зацикливания на ожиданиях системы,
    поэтому выводы делаются по ней; выборка меньше, потому что берутся
    только созревшие метки (valid).

**Ни одна из двух не даёт независимых наблюдений.** Раньше здесь было
написано, что `realized` — «одно независимое наблюдение на бар, честный
Бернулли»; это неверно. Горизонт 24h при базовом ТФ 15m равен 96 барам, а
кандидаты идут в среднем каждые ~11 баров: исходы соседних кандидатов
перекрываются почти полностью. Плюс снимки офсетов
(`SNAPSHOT_OFFSETS_MIN = 0, 45, 90, 180`) дают до четырёх кандидатов на одну
реализацию перехода. Обеим метрикам нужен блочный бутстрап — он считается
по умолчанию, и именно по нему определяется значимость (см. шапку
`btcproc/analysis/lift.py`).

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
from btcproc.analysis import samples  # noqa: E402
from btcproc.analysis.lift import (  # noqa: E402
    DEFAULT_N_BOOT,
    block_length_rows,
    format_table,
    measure_lift,
)
from btcproc.features import events as ev  # noqa: E402


def consolidate(per_symbol: dict[str, list]) -> None:
    """
    Сводка по монетам: где атом прошёл, где нет и совпал ли знак.

    Проверка сильнее, чем holdout внутри одной монеты: holdout отделяет
    позднюю историю от ранней, но рынок под ними один.

    Чего она НЕ даёт — трёх независимых подтверждений. Дневные доходности
    BTC и ETH исторически коррелированы на 0.8+, BTC и SOL немногим слабее;
    три сильно коррелированных инструмента дают эффективно полтора
    независимых наблюдения, а не три. Поэтому «совпало на 3 из 3» нельзя
    считать как p³ — это одно свидетельство, а не перемножение трёх.
    Реально независимую проверку дал бы инструмент из другого класса
    активов, слабо связанный с BTC; это отдельная задача.

    Что проверка ловит и ради чего остаётся обязательной — грубую подгонку
    под один инструмент: эффект, живущий только на BTC, почти наверняка
    артефакт его конкретной истории.
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
    model_run = samples.resolve_model_run(symbol, args.run)
    frame = samples.load(symbol, args.metric, model_run, with_atoms=True)
    if frame.empty:
        print(f"Нет данных по {symbol}. Нужен прогон train или live.")
        return None

    # NULL в context_atoms — бар размечен прогоном до появления колонки.
    frame, stale = samples.merge_atom_columns(frame)
    if stale:
        print(f"Пропущено {stale} баров без context_atoms (размечены до правки).")
    if frame.empty:
        print("Ни одного бара с контекстом. Нужен прогон после правки.")
        return None

    if args.context_only:
        atoms = list(ev.CONTEXT_ATOM_LIST)
    elif args.signature_only:
        atoms = list(ev.SIGNATURE_ATOMS)
    else:
        atoms = list(ev.ATOMS)

    # Горизонт берётся из конфига, а не хардкодится: он задаёт и длину блока
    # бутстрапа, и ширину окна прореживания.
    horizon_minutes = None if args.no_bootstrap else config.data.horizon_minutes

    results = measure_lift(
        frame,
        atoms=atoms,
        metric_column="metric",
        alpha=args.alpha,
        correction=args.correction,
        holdout=args.holdout or None,
        min_group=args.min_group,
        horizon_minutes=horizon_minutes,
        n_boot=args.n_boot,
        thinned=args.thinned,
    )

    span = f"{frame['ts'].min():%Y-%m-%d} … {frame['ts'].max():%Y-%m-%d}"
    print(f"\n{symbol}: {len(frame)} кандидатов модели #{model_run}, {span}")
    print(f"Метрика: {args.metric}"
          + ("  (ожидание системы, не факт — см. шапку скрипта)"
             if args.metric == "long_outcome_share" else "  (фактические исходы)"))
    if horizon_minutes:
        block = block_length_rows(frame["ts"], horizon_minutes)
        print(f"Горизонт {config.data.horizon} = {config.data.horizon_bars} баров; "
              f"блок бутстрапа {block} строк, реплик {args.n_boot}")
    print(f"Базовая доля: {frame['metric'].mean():.4f}\n")
    print(format_table(results, args.correction, args.alpha, args.n_boot))

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
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT,
                        help="Реплик блочного бутстрапа")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Только наивный z-тест — режим до 2026-08-11. "
                             "Он анти-консервативен, годится лишь для сверки "
                             "со старыми замерами")
    parser.add_argument("--thinned", action="store_true",
                        help="Плюс медиана z по прореживанию непересекающимися "
                             "окнами (дорого, независимая проверка бутстрапа)")
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
