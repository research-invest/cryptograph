"""
Считает фон состояний прогонам, сделанным до появления таблицы `state_context`.

С 2026-08-16 фон (насколько часто контекстный атом встречается внутри
состояния против всей истории) считает `train` и кладёт готовым. Прогоны,
сделанные раньше, фона в базе не имеют, и админка досчитывает его лениво —
по первому открытию узла графа, один раз на прогон. Скрипт делает то же
самое заранее, чтобы первый оператор после выкатки не ждал.

    python3 scripts/backfill_state_context.py            # все train без фона
    python3 scripts/backfill_state_context.py --run 1304 # один прогон
    python3 scripts/backfill_state_context.py --all      # пересчитать и готовые
    python3 scripts/backfill_state_context.py --dry-run  # только показать

--all нужен после `scripts/backfill_context_atoms.py`: тот меняет состав
`bar_events.context_atoms`, а фон посчитан по прежнему составу и сам об этом
не узнает.

Пересчёт безопасен и идемпотентен: величина выводится из уже размеченной
истории, переобучения не требует, `group_id` не трогает.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btcproc.db import repo  # noqa: E402
from btcproc.db.session import fetch_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Фон состояний для старых прогонов")
    parser.add_argument("--run", type=int, help="Только этот прогон")
    parser.add_argument("--all", action="store_true",
                        help="Пересчитать и те прогоны, где фон уже посчитан")
    parser.add_argument("--dry-run", action="store_true", help="Ничего не считать")
    args = parser.parse_args()

    # Только train: `market_groups` и узлы графа пишет он один, у live-прогона
    # фону состояний не на чем висеть.
    where = ["r.kind = 'train'", "r.status = 'done'"]
    params: list = []
    if args.run:
        where.append("r.run_id = %s")
        params.append(args.run)
    if not args.all:
        where.append("c.run_id IS NULL")

    runs = fetch_all(
        "SELECT r.run_id, r.symbol FROM runs r "
        "LEFT JOIN state_context_runs c ON c.run_id = r.run_id "
        f"WHERE {' AND '.join(where)} ORDER BY r.run_id",
        params,
    )
    if not runs:
        print("Нечего считать — фон есть у всех завершённых train-прогонов.")
        return 0

    print(f"Прогонов к расчёту: {len(runs)}")
    for row in runs:
        if args.dry_run:
            print(f"  #{row['run_id']} {row['symbol']}")
            continue
        started = time.time()
        stats = repo.save_state_context(row["run_id"], row["symbol"])
        print(f"  #{row['run_id']} {row['symbol']}: {stats['bars']} баров, "
              f"{stats['rows']} строк, {time.time() - started:.1f} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
