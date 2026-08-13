"""
Сверка двух выгрузок баров: питоновской (боевой код btcproc) и go-шной.

Замер скорости без этой сверки бессмысленен: быстрее всего работает
реализация, считающая не то. Поэтому вывод сравнивается построчно, а
расхождения печатаются с указанием бара и поля.

Побитового равенства НЕ требуется и не ожидается: pandas суммирует объёмы в
отсортированном порядке, Go — в порядке файла, и последние биты мантиссы
расходятся. Порог задан относительный; всё, что крупнее, — настоящее
расхождение логики, а не арифметики.

  python3 bench/compare.py py.csv go.csv [--rtol 1e-9]
"""
from __future__ import annotations

import argparse
import csv
import sys

FIELDS = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"]


def load(path: str) -> dict[int, dict[str, float]]:
    with open(path) as fh:
        return {int(row["ts"]): {k: float(row[k]) for k in FIELDS} for row in csv.DictReader(fh)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("left")
    ap.add_argument("right")
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    a, b = load(args.left), load(args.right)

    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    if only_a or only_b:
        print(f"РАЗНЫЙ НАБОР БАРОВ: только слева {len(only_a)}, только справа {len(only_b)}")
        for ts in (only_a + only_b)[: args.show]:
            print(f"  ts={ts}")

    worst: dict[str, float] = {f: 0.0 for f in FIELDS}
    bad: list[str] = []
    for ts in sorted(set(a) & set(b)):
        for f in FIELDS:
            x, y = a[ts][f], b[ts][f]
            scale = max(abs(x), abs(y), 1e-12)
            rel = abs(x - y) / scale
            worst[f] = max(worst[f], rel)
            if rel > args.rtol:
                bad.append(f"  ts={ts} {f}: {x!r} vs {y!r} (отн. {rel:.2e})")

    print(f"баров сверено: {len(set(a) & set(b))}")
    print("максимальное относительное расхождение по полям:")
    for f in FIELDS:
        print(f"  {f:<16} {worst[f]:.2e}")

    if bad:
        print(f"\nПРЕВЫШЕН ПОРОГ {args.rtol:g}: {len(bad)} случаев")
        for line in bad[: args.show]:
            print(line)
        return 1
    if only_a or only_b:
        return 1
    print(f"\nOK: выгрузки совпадают в пределах {args.rtol:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
