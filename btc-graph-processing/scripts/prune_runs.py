"""
Убирает из базы лишние модели состояний, оставляя по одной на монету.

Зачем. Каждый `train` создаёт новую модель, и `group_id` с `transition_id`
новой модели несопоставимы со старыми. Пока моделей несколько, любая выборка
«все кандидаты монеты» смешивает несравнимое — и это не гипотетическая
опасность: на базе разработчика такая смесь из шести моделей развалила замер
лифта и родила несуществующую «аномалию BTC» (development_log.md, 16.3).

Особенно вредны частичные прогоны (`train --start ...`): они покрывают только
свежий период, поэтому их кандидаты скапливаются в конце истории и ломают
разбиение на обучение и holdout.

Что делает. Для каждой монеты оставляет **одну** модель — завершённый `train`
с наибольшим покрытием истории (при равенстве — свежий), — и все `live`,
которые её загрузили. Остальные прогоны удаляет вместе со всем, что на них
ссылается.

    python3 scripts/prune_runs.py                 # только показать
    python3 scripts/prune_runs.py --apply         # удалить
    python3 scripts/prune_runs.py --symbol BTCUSDT --apply
    python3 scripts/prune_runs.py --keep 4 --apply # оставить ещё и модель #4

По умолчанию **ничего не удаляет**: печатает, что осталось бы и что ушло.

Это операция для базы разработчика. На боевом контуре модель обычно одна,
и если скрипт предлагает там что-то удалить — сначала разберись почему.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btcproc import config, symbols  # noqa: E402
from btcproc.db.session import connect, fetch_all  # noqa: E402

# Таблицы, где run_id — часть данных прогона. Порядок важен: runs удаляется
# последней, иначе подзапрос «прогоны этой модели» перестанет работать.
RUN_TABLES = [
    "candidates",
    "bar_states",
    "market_groups",
    "transitions",
    "event_blocks",
    "state_models",
    "runs",
]


def models(symbol: str) -> list[dict]:
    """Завершённые train-прогоны монеты с мерой покрытия истории."""
    return fetch_all(
        """
        SELECT r.run_id, r.started_at, r.params->>'start' AS start_param,
               (SELECT count(*) FROM bar_states s WHERE s.run_id = r.run_id) AS bars,
               (SELECT count(*) FROM candidates c WHERE c.run_id = r.run_id) AS candidates,
               (SELECT count(*) FROM runs l
                 WHERE l.params->>'model_run_id' = r.run_id::text) AS live_runs
        FROM runs r
        WHERE r.kind = 'train' AND r.status = 'done' AND r.symbol = %s
        ORDER BY r.started_at DESC
        """,
        (symbol,),
    )


def plan(symbol: str, keep_extra: set[int]) -> tuple[int | None, list[int]]:
    """Возвращает (модель к сохранению, список прогонов к удалению)."""
    found = models(symbol)
    if not found:
        return None, []

    # Покрытие важнее свежести: частичный прогон может быть новее полного,
    # но модель по куску истории хуже модели по всей.
    keep = max(found, key=lambda m: (m["bars"], m["started_at"]))["run_id"]

    keep_ids = {keep, *keep_extra}
    doomed = []
    for row in fetch_all(
        "SELECT run_id, kind, params->>'model_run_id' AS model FROM runs WHERE symbol = %s",
        (symbol,),
    ):
        run_id = row["run_id"]
        if run_id in keep_ids:
            continue
        # live сохраняется вместе со своей моделью.
        if row["kind"] == "live" and row["model"] and int(row["model"]) in keep_ids:
            continue
        doomed.append(run_id)
    return keep, doomed


def describe(symbol: str, keep: int | None, doomed: list[int]) -> None:
    print(f"\n=== {symbol} ===")
    found = models(symbol)
    if not found:
        print("  train-прогонов нет")
        return
    for m in found:
        mark = "ОСТАВИТЬ" if m["run_id"] == keep else "удалить"
        window = m["start_param"] or "вся история"
        print(f"  train #{m['run_id']:<4} {mark:<9} баров {m['bars']:>7}, "
              f"кандидатов {m['candidates']:>6}, live-прогонов {m['live_runs']:>3}, "
              f"с {window}")
    print(f"  к удалению прогонов всего: {len(doomed)}")


def delete(run_ids: list[int]) -> dict[str, int]:
    """Удаляет прогоны и всё, что на них ссылается. Одной транзакцией."""
    if not run_ids:
        return {}
    removed: dict[str, int] = {}
    with connect() as conn, conn.cursor() as cur:
        for table in RUN_TABLES:
            cur.execute(f"DELETE FROM {table} WHERE run_id = ANY(%s)", (run_ids,))
            removed[table] = cur.rowcount
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Оставить по одной модели на монету")
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--keep", action="append", type=int,
                        help="Номер прогона, который сохранить дополнительно")
    parser.add_argument("--apply", action="store_true",
                        help="Действительно удалить; без него — только показать")
    args = parser.parse_args()

    targets = args.symbol or symbols.tickers(only_enabled=True)
    keep_extra = set(args.keep or [])

    all_doomed: list[int] = []
    for symbol in targets:
        keep, doomed = plan(symbol, keep_extra)
        describe(symbol, keep, doomed)
        all_doomed.extend(doomed)

    if not all_doomed:
        print("\nБаза уже чистая: по одной модели на монету.")
        return 0

    print(f"\nВсего к удалению: {len(all_doomed)} прогонов")
    if not args.apply:
        print("Это сухой прогон. Чтобы удалить, добавь --apply")
        return 0

    removed = delete(all_doomed)
    print("\nУдалено строк:")
    for table, n in removed.items():
        print(f"  {table:<16} {n:>8}")
    print(f"\nБаза: {config.db.url.rsplit('@', 1)[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
