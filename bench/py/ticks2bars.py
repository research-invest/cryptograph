"""
Эталон: тиковый архив Bybit → бары 15m БОЕВЫМ кодом btcproc.

Здесь намеренно нет своей реализации. Меряется ровно то, что работает в
`ingest --symbol HYPEUSDT`: `bybit.parse_ticks` + `bybit.ticks_to_bars`.
Замер собственной копии логики измерял бы качество копии, а не системы.

Вывод: бары в CSV на stdout (или в --out), метрики прогона в JSON на stderr —
чтобы `compare.py` мог сверить и то и другое, не разбирая текст.

  python3 bench/py/ticks2bars.py ARCHIVE.csv.gz [ARCHIVE2.csv.gz ...] --out bars.csv

Несколько архивов обрабатываются последовательно и в хронологическом порядке
имён, с передачей `prev_close` через стык месяцев, — как это делает
`sync_history`.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "btc-graph-processing"))

from btcproc.ingest import bybit  # noqa: E402  (после правки sys.path)

# Формат чисел в выгрузке. 10 знаков после запятой — заведомо больше, чем
# несёт цена или объём Bybit, поэтому округление здесь ничего не скрывает,
# а сверка с Go читает одинаковые строки.
FMT = "{:.10f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("archives", nargs="+")
    ap.add_argument("--out", default="-")
    ap.add_argument("--symbol", default="HYPEUSDT")
    ap.add_argument("--tf", default="15m")
    args = ap.parse_args()

    t_start = time.perf_counter()
    t_parse = 0.0
    ticks_total = 0
    frames = []
    prev_close: float | None = None

    for path in args.archives:
        payload = Path(path).read_bytes()

        t0 = time.perf_counter()
        ticks = bybit.parse_ticks(payload)
        t_parse += time.perf_counter() - t0
        ticks_total += len(ticks)

        frame = bybit.ticks_to_bars(ticks, args.symbol, args.tf, prev_close=prev_close)
        if not frame.empty:
            prev_close = float(frame["close"].iloc[-1])
            frames.append(frame)

    elapsed = time.perf_counter() - t_start

    out = sys.stdout if args.out == "-" else open(args.out, "w")
    rows = 0
    with out:
        out.write("ts,open,high,low,close,volume,quote_volume,trades,taker_buy_base\n")
        for frame in frames:
            for row in frame.itertuples(index=False):
                out.write(
                    f"{int(row.ts.timestamp() * 1000)},"
                    + ",".join(
                        FMT.format(float(v))
                        for v in (row.open, row.high, row.low, row.close,
                                  row.volume, row.quote_volume)
                    )
                    + f",{int(row.trades)},{FMT.format(float(row.taker_buy_base))}\n"
                )
                rows += 1

    # ru_maxrss на macOS в байтах, на Linux в килобайтах — различаем по порядку.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 1e6 if rss > 1e7 else rss / 1e3

    json.dump(
        {
            "impl": "python",
            "archives": len(args.archives),
            "ticks": ticks_total,
            "bars": rows,
            "parse_sec": round(t_parse, 3),
            "total_sec": round(elapsed, 3),
            "peak_rss_mb": round(rss_mb, 1),
        },
        sys.stderr,
    )
    sys.stderr.write("\n")


if __name__ == "__main__":
    main()
