"""
Еженедельное обслуживание базы: убрать накопившееся и вернуть место.

Зачем это отдельно от `prune_runs.py`. Тот скрипт — инструмент для рук:
он умеет сносить целые модели вместе с кандидатами и по умолчанию делает
сухой прогон. Гонять такое по крону нельзя. Здесь наоборот — узкий набор
безопасных операций, рассчитанный на то, что его запускают без присмотра
и результат читают потом в логе.

Что делает:

1. **Пропускает всё, если идёт прогон.** Обслуживание не должно соревноваться
   за диск и блокировки с расчётом. Мёртвые прогоны при этом убираются
   (`runs.reap_if_stale`) — иначе один OOM останавливал бы и обслуживание тоже.
2. **Чистит разметку старых live-прогонов** — та же логика, что у
   `prune_runs.py --bar-states`: остаются train и N последних live на монету.
   Кандидаты, граф, статистика и сами прогоны не трогаются.
3. **Вакуумит.** `VACUUM ANALYZE` всегда — он дешёвый и не берёт блокировок.
   `VACUUM FULL` — только если раздутость перешла порог: он переписывает
   таблицу целиком под ACCESS EXCLUSIVE, и еженедельно платить этим за
   несколько мегабайт незачем.
4. **Печатает размеры и свободное место.** Это не украшательство: разметка
   еженедельного `train` (312 тыс. строк на монету) не удаляется никем и
   остаётся главным источником роста. Решение, что с ней делать, отложено до
   того момента, когда место станет проблемой, — и увидеть приближение этого
   момента можно только по такому отчёту.

    python3 scripts/maintenance.py              # сделать
    python3 scripts/maintenance.py --dry-run    # только показать
    python3 scripts/maintenance.py --force      # не смотреть на идущие прогоны

Крон: раз в неделю, в середине недели. Не в воскресенье — там еженедельный
train, который идёт часами, и обслуживание просто пропускалось бы каждый раз.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btcproc import config, symbols  # noqa: E402
from btcproc.db import runs as runs_repo  # noqa: E402
from btcproc.db.session import connect, fetch_one  # noqa: E402
from prune_runs import BAR_STATES_KEEP_LIVE, delete_bar_states, fat_live_runs  # noqa: E402

# Таблицы, за которыми следим. bar_states — единственная, где бывают массовые
# удаления; остальные в отчёте ради ранней диагностики роста.
WATCHED = ["bar_states", "candidates", "ohlcv", "features", "bar_events", "outcomes"]

# Пороги для VACUUM FULL. Любой из трёх — достаточное основание.
#
# 1. Мы только что много удалили. Это основной случай: удаления — вообще
#    единственный заметный источник раздутости здесь, и происходят они прямо
#    в этом же прогоне. 200 тысяч строк — это уже десятки мегабайт, которые
#    иначе останутся в файле навсегда: новые live пишут по паре сотен строк
#    и переиспользуют освободившееся место разве что за годы.
FULL_AFTER_DELETED_ROWS = 200_000

# 2. Мёртвые строки скопились помимо нас.
FULL_DEAD_RATIO = 0.20

# 3. Файл раздут с прошлых раз. Нужен отдельно от (1) и (2), потому что оба
#    слепы к уже случившейся раздутости: autovacuum обнуляет счётчик мёртвых
#    строк, а место ОС не возвращает — таблица так и остаётся большим файлом
#    с редкими живыми строками, и по n_dead_tup это не видно вообще.
#
#    Число откалибровано замером, а не взято из головы: сразу после
#    VACUUM FULL у bar_states выходит ~167 байт на строку вместе с индексами
#    (613 МБ на 3.67 млн строк, три монеты). 250 — это примерно полтора
#    таких значения, то есть треть файла в дырах. Меньший порог гонял бы
#    перезапись под ACCESS EXCLUSIVE ради процентов, больший — копил бы
#    гигабайты незамеченными.
FULL_BYTES_PER_ROW = 250


def table_stats(table: str) -> dict:
    row = fetch_one(
        """
        SELECT pg_total_relation_size(c.oid)          AS bytes,
               pg_size_pretty(pg_total_relation_size(c.oid)) AS pretty,
               coalesce(s.n_live_tup, 0)              AS live,
               coalesce(s.n_dead_tup, 0)              AS dead
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
        WHERE n.nspname = %s AND c.relname = %s
        """,
        (config.db.schema, table),
    )
    return row or {"bytes": 0, "pretty": "—", "live": 0, "dead": 0}


def dead_ratio(stats: dict) -> float:
    total = stats["live"] + stats["dead"]
    return stats["dead"] / total if total else 0.0


def bytes_per_row(stats: dict) -> float:
    """
    Сколько места занимает одна живая строка вместе с индексами.

    Прямая мера раздутости файла, в отличие от `dead_ratio`: та обнуляется,
    как только отработает autovacuum, а место при этом ОС не возвращается.
    На пустой таблице бессмысленна — возвращаем 0, чтобы не звать FULL зря.
    """
    return stats["bytes"] / stats["live"] if stats["live"] else 0.0


def vacuum(table: str, full: bool) -> None:
    """
    VACUUM не выполняется внутри транзакции — нужен autocommit.

    FULL берёт ACCESS EXCLUSIVE: на время перезаписи таблица недоступна и
    начавшийся прогон будет ждать. Поэтому он и под порогом, а сам скрипт
    стоит в кроне подальше от расчётов.

    Параллельные воркеры отключаются намеренно. Postgres в docker получает
    `/dev/shm` в 64 МБ (дефолт демона, в compose не переопределён), а
    параллельная чистка индексов просит под сотню — и VACUUM падает с
    «could not resize shared memory segment: No space left on device».
    Ошибка выглядит как кончившийся диск, хотя на диске гигабайты свободных.
    Обслуживанию параллелизм не нужен: оно идёт раз в неделю и никого не ждёт.
    """
    mode = "FULL, ANALYZE" if full else "ANALYZE"
    with connect(autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("SET max_parallel_maintenance_workers = 0")
        cur.execute("SET max_parallel_workers_per_gather = 0")
        cur.execute(f"VACUUM ({mode}) {config.db.schema}.{table}")


def prune_live_states(keep_live: int, dry_run: bool) -> int:
    doomed: list[int] = []
    rows = 0
    # ВСЕ монеты реестра, включая выключенные: у выключенной новых прогонов
    # нет, но накопленная разметка старых live никуда не делась — если её
    # обходить, она останется до повторного включения монеты.
    for symbol in symbols.tickers():
        fat = fat_live_runs(symbol, keep_live)
        if not fat:
            continue
        symbol_rows = sum(run["bars"] for run in fat)
        doomed.extend(run["run_id"] for run in fat)
        rows += symbol_rows
        print(f"  {symbol}: {len(fat)} live-прогонов, {symbol_rows} строк разметки")

    if not doomed:
        print("  разметки старых live-прогонов нет")
        return 0
    if dry_run:
        print(f"  (сухой прогон) удалили бы {rows} строк из {len(doomed)} прогонов")
        return 0
    removed = delete_bar_states(doomed)
    print(f"  удалено {removed} строк из {len(doomed)} прогонов")
    return removed


def report() -> None:
    print("\nРазмеры:")
    for table in WATCHED:
        st = table_stats(table)
        if not st["bytes"]:
            continue
        print(f"  {table:<14} {st['pretty']:>10}   живых строк {st['live']:>10}, "
              f"мёртвых {st['dead']:>9} ({dead_ratio(st):.0%})")

    db = fetch_one("SELECT pg_size_pretty(pg_database_size(current_database())) AS s")
    usage = shutil.disk_usage("/")
    print(f"\n  база: {db['s']}")
    print(f"  диск: занято {usage.used // 2**30} ГБ из {usage.total // 2**30} ГБ, "
          f"свободно {usage.free // 2**30} ГБ")

    # Главный источник роста после того, как live перестал писать всю историю.
    # Молча он не проявится: место кончится через год, а не завтра.
    train_rows = fetch_one(
        "SELECT coalesce(sum(n), 0) AS n FROM ("
        "  SELECT count(*) AS n FROM bar_states s"
        "  JOIN runs r ON r.run_id = s.run_id AND r.kind = 'train'"
        "  GROUP BY s.run_id) t"
    )
    print(f"\n  из них разметка train-прогонов: {train_rows['n']} строк — она "
          f"не удаляется никем\n  и растёт на ~312 тыс. строк на монету при "
          f"каждом еженедельном переобучении.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Еженедельное обслуживание БД")
    parser.add_argument("--dry-run", action="store_true",
                        help="Ничего не менять, только показать")
    parser.add_argument("--force", action="store_true",
                        help="Работать, даже если идёт прогон")
    parser.add_argument("--keep-live", type=int, default=BAR_STATES_KEEP_LIVE,
                        help=f"Сколько последних live на монету оставить "
                             f"с разметкой (по умолчанию {BAR_STATES_KEEP_LIVE})")
    parser.add_argument("--full", action="store_true",
                        help="VACUUM FULL безусловно, не глядя на пороги")
    args = parser.parse_args()

    print(f"Обслуживание базы {config.db.url.rsplit('@', 1)[-1]}, "
          f"схема {config.db.schema}")

    # Идущий прогон — причина отложить, а не повод падать: следующий запуск
    # через неделю сделает то же самое, ничего не накопится критично.
    active = runs_repo.active_runs()
    alive = [run for run in active if not runs_repo.reap_if_stale(run)]
    if alive and not args.force:
        names = ", ".join(f"#{r['run_id']} {r.get('symbol') or '?'}" for r in alive)
        print(f"Идут прогоны: {names} — обслуживание отложено до следующего раза.")
        return 0

    before = table_stats("bar_states")

    print("\nРазметка старых live-прогонов:")
    removed = prune_live_states(args.keep_live, args.dry_run)

    if args.dry_run:
        report()
        print("\nЭто сухой прогон, вакуум не запускался.")
        return 0

    # Раздутость файла меряется ПОСЛЕ удаления: строки уже ушли, а место — нет,
    # и именно эту разницу FULL и возвращает.
    after_delete = table_stats("bar_states")
    reasons = []
    if args.full:
        reasons.append("принудительно")
    if removed >= FULL_AFTER_DELETED_ROWS:
        reasons.append(f"удалено {removed} строк")
    if dead_ratio(before) >= FULL_DEAD_RATIO:
        reasons.append(f"мёртвых строк {dead_ratio(before):.0%}")
    if bytes_per_row(after_delete) >= FULL_BYTES_PER_ROW:
        reasons.append(f"{bytes_per_row(after_delete):.0f} байт на строку "
                       f"при норме ~167")

    full = bool(reasons)
    why = ", ".join(reasons) if full else (
        f"в норме: {dead_ratio(before):.0%} мёртвых, "
        f"{bytes_per_row(after_delete):.0f} байт на строку"
    )
    print(f"\nВакуум: {'FULL' if full else 'обычный'} — {why}")
    # Сорвавшийся вакуум не должен съедать отчёт: место и рост таблиц —
    # ровно то, ради чего скрипт стоит в кроне, и в логе это нужнее всего
    # именно тогда, когда что-то пошло не так.
    failed = None
    try:
        vacuum("bar_states", full)
        after = table_stats("bar_states")
        print(f"  bar_states: {before['pretty']} → {after['pretty']}")
    except Exception as exc:  # noqa: BLE001 — диагноз важнее трейсбека
        failed = exc
        print(f"  ВАКУУМ НЕ ВЫПОЛНЕН: {type(exc).__name__}: {exc}")

    report()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
