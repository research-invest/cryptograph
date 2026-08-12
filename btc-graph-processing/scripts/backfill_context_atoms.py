"""
Заполняет bar_events.context_atoms на барах, размеченных до появления колонки.

Переобучать ничего не нужно, и в этом весь смысл скрипта: блок событий
считается из одних только баров — ни модель состояний, ни признаки, ни run_id
в него не входят. Значит контекст можно досчитать из ohlcv и дописать, не
трогая ни group_id, ни накопленный в Neo4j граф.

Заодно скрипт проверяет главный критерий приёмки правки: пересчитанный
event_block_id обязан совпасть с сохранённым. Если разошёлся — правка сломала
смысл исторических блоков, и это надо увидеть до записи, а не после.

    python3 scripts/backfill_context_atoms.py --dry-run          # только сверка
    python3 scripts/backfill_context_atoms.py --symbol BTCUSDT
    python3 scripts/backfill_context_atoms.py --all

Идемпотентен: по умолчанию берёт только бары с NULL в context_atoms.
Записывает через тот же repo.save_events, что и прогон, — upsert по
(symbol, ts) обновляет все неключевые колонки. Строки, которых в bar_events
нет (бары прогрева, не дошедшие до features.index), не добавляются: скрипт
чинит существующее, а не расширяет разметку.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from btcproc import config, symbols  # noqa: E402
from btcproc.db import repo  # noqa: E402
from btcproc.db.session import fetch_all  # noqa: E402
from btcproc.features import events as ev  # noqa: E402
from btcproc.ingest import bars  # noqa: E402


def process(symbol: str, only_missing: bool, dry_run: bool) -> bool:
    """Возвращает False, если сверка блоков разошлась — тогда писать нельзя."""
    print(f"\n=== {symbol} ===")

    stored = fetch_all(
        "SELECT ts, event_block_id, context_atoms IS NULL AS missing "
        "FROM bar_events WHERE symbol = %s ORDER BY ts",
        (symbol,),
    )
    if not stored:
        print("  В bar_events нет строк — сначала нужен прогон.")
        return True

    stored_frame = pd.DataFrame(stored).set_index("ts")
    stored_frame.index = pd.DatetimeIndex(stored_frame.index)
    missing = int(stored_frame["missing"].sum())
    print(f"  Строк в bar_events: {len(stored_frame)}, без контекста: {missing}")

    if only_missing and missing == 0:
        print("  Нечего заполнять.")
        return True

    base = bars.load_ohlcv(symbol, config.data.base_tf, None, None)
    if base.empty:
        print("  Нет баров в ohlcv.")
        return True
    print(f"  Баров в ohlcv: {len(base)} ({base.index[0].date()}…{base.index[-1].date()})")

    recomputed = ev.build_event_blocks(base)

    # ── Сверка: блоки не должны были измениться ──
    common = recomputed.index.intersection(stored_frame.index)
    print(f"  Общих баров: {len(common)}")
    if len(common) == 0:
        print("  ! Пересечения по времени нет — сверять нечего, записи не будет.")
        return True

    mismatch = (
        recomputed.loc[common, "event_block_id"].to_numpy()
        != stored_frame.loc[common, "event_block_id"].to_numpy()
    )
    n_mismatch = int(mismatch.sum())
    if n_mismatch:
        share = n_mismatch / len(common)
        print(f"  ✗ event_block_id РАЗОШЁЛСЯ на {n_mismatch} барах ({share:.2%})")
        examples = common[mismatch][:3]
        for ts in examples:
            print(f"      {ts}: было {stored_frame.loc[ts, 'event_block_id']}, "
                  f"стало {recomputed.loc[ts, 'event_block_id']}")
        # Расхождение на самых первых барах ожидаемо: прогон мог идти с --start,
        # и у скользящих окон был другой разогрев. Массовое — это поломка.
        if share > 0.01:
            print("  Записи не будет: сначала разобраться, почему блоки поехали.")
            return False
        print("  Доля мала — похоже на разный разогрев окон у краёв истории.")
    else:
        print("  ✓ event_block_id совпал на всех общих барах")

    target = common
    if only_missing:
        target = common[stored_frame.loc[common, "missing"].to_numpy()]
    print(f"  К записи: {len(target)} строк")
    if len(target) == 0 or dry_run:
        print("  --dry-run: ничего не записано" if dry_run else "  Нечего писать.")
        return True

    written = repo.save_events(recomputed.loc[target], symbol)
    print(f"  Записано: {written}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Бэкфилл context_atoms в bar_events")
    parser.add_argument("--symbol", action="append", help="Монета; можно несколько раз")
    parser.add_argument("--all", action="store_true", help="Все активные монеты")
    parser.add_argument("--dry-run", action="store_true", help="Только сверка, без записи")
    parser.add_argument("--force", action="store_true",
                        help="Переписать контекст и там, где он уже есть")
    args = parser.parse_args()

    if args.all:
        targets = symbols.tickers(only_enabled=True)
    elif args.symbol:
        targets = args.symbol
    else:
        targets = [config.data.symbol]

    ok = True
    for symbol in targets:
        ok &= process(symbol, only_missing=not args.force, dry_run=args.dry_run)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
