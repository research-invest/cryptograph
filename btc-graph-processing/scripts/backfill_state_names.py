"""
Проставляет имена состояниям прогонов, сделанных до появления колонки `name`.

Пересчитывать граф для этого не нужно: имя выводится из `top_features`
и `dominant_bias`, а они уже лежат в `market_groups`. То есть скрипт читает
ту же самую статистику, из которой имя посчиталось бы при прогоне.

Идемпотентен: по умолчанию трогает только строки без имени.

    python3 scripts/backfill_state_names.py            # только пустые
    python3 scripts/backfill_state_names.py --all      # пересчитать все
    python3 scripts/backfill_state_names.py --dry-run  # только показать

--all нужен, когда меняется словарь формулировок в btcproc/states/naming.py.
По умолчанию его не применяем: имя прогона — часть его результата, и менять
задним числом подписи у старых прогонов стоит осознанно.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btcproc.db.session import connect, fetch_all  # noqa: E402
from btcproc.states.naming import describe_state, label_for  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Имена состояний для старых прогонов")
    parser.add_argument("--all", action="store_true",
                        help="Пересчитать имена и там, где они уже есть")
    parser.add_argument("--dry-run", action="store_true", help="Ничего не записывать")
    parser.add_argument("--run", type=int, help="Только этот прогон")
    args = parser.parse_args()

    where = [] if args.all else ["name IS NULL"]
    params: list = []
    if args.run:
        where.append("run_id = %s")
        params.append(args.run)
    clause = f" WHERE {' AND '.join(where)}" if where else ""

    rows = fetch_all(
        "SELECT g.run_id, g.group_id, g.top_features, g.dominant_bias, r.symbol "
        f"FROM market_groups g LEFT JOIN runs r ON r.run_id = g.run_id{clause} "
        "ORDER BY g.run_id, g.size DESC",
        params,
    )
    if not rows:
        print("Нечего заполнять — у всех состояний уже есть имена.")
        return 0

    updates, shown_runs = [], set()
    for row in rows:
        name = describe_state(row["top_features"], row["dominant_bias"])
        updates.append((name, row["run_id"], row["group_id"]))
        if row["run_id"] not in shown_runs:
            shown_runs.add(row["run_id"])
            print(f"\n=== прогон #{row['run_id']} ({row['symbol'] or '?'}) ===")
        if len(shown_runs) <= 5:
            print(f"  {label_for(float(row['group_id']), name)}")

    print(f"\nСостояний к обновлению: {len(updates)} в {len(shown_runs)} прогонах")
    if args.dry_run:
        print("--dry-run: ничего не записано")
        return 0

    with connect() as conn, conn.cursor() as cur:
        cur.executemany(
            "UPDATE market_groups SET name = %s WHERE run_id = %s AND group_id = %s",
            updates,
        )
    print(f"Записано: {len(updates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
