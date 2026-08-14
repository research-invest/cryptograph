"""
Гейт R (docs/task_fear_greed.md §2, §5.3) — предсказывает ли FGI РАЗМАХ.

Ничего не пишет — ни в БД, ни в конфиг, по образцу measure_zone_timeframe.py.
`range_ratio` считается автономно из баров и ATR14 (btcproc/features/indicators.py),
без правки outcomes.py и candidates/builder.py — блок C аудита не начат, и
ждать его не нужно.

Три горизонта (24h/8h/4h — те же, на которых мерилось направление в разделе
26.4 журнала, ради сравнимости), три ЗАФИКСИРОВАННЫЕ ДО ЗАМЕРА эпохи:

    2018-02 … 2020-12   ранний рынок
    2021-01 … 2022-12   цикл роста и обвал (LUNA/FTX)
    2023-01 … 2026-08   после обвала, ETF-эпоха

Метрика — Spearman(fgi_level, range_ratio), бенчмарк — скользящая rv
последних 96 баров (indicators.realized_vol(close, 96)). Гейт R (§2):
значим по блочному бутстрапу И значимо сильнее бенчмарка в ДВУХ эпохах из
трёх — иначе источник отвергается целиком, кроме описательной роли.

    python3 scripts/measure_fgi_range.py --symbol BTCUSDT
    python3 scripts/measure_fgi_range.py --all
"""
from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from btcproc import config, symbols  # noqa: E402
from btcproc.analysis.range_lift import (  # noqa: E402
    autocorr_block_rows,
    forward_range_ratio,
    paired_diff_p,
    spearman_block_p,
)
from btcproc.features import fear_greed as fg  # noqa: E402
from btcproc.features import indicators as ind  # noqa: E402
from btcproc.ingest import bars, external  # noqa: E402

HORIZONS = {"24h": "24h", "8h": "8h", "4h": "4h"}

EPOCHS = [
    ("2018-02..2020-12", "2018-02-01", "2020-12-31"),
    ("2021-01..2022-12", "2021-01-01", "2022-12-31"),
    ("2023-01..2026-08", "2023-01-01", None),
]

N_BOOT = 2000


def _horizon_bars(label: str) -> int:
    unit = label[-1]
    value = int(label[:-1])
    minutes = value * {"m": 1, "h": 60, "d": 1440}[unit]
    return minutes // config.data.base_minutes


def build_frame(symbol: str) -> tuple[pd.DataFrame, int]:
    """Бары + fgi_level + бенчмарк rv, готовые к джойну по горизонту. Плюс block_rows FGI."""
    base = bars.load_ohlcv(symbol, config.data.base_tf, None, None)
    if base.empty:
        raise SystemExit(f"Нет баров по {symbol} — сначала ingest.")
    daily = external.load_external_daily(external.FEAR_GREED_SERIES)
    if daily.empty:
        raise SystemExit("external_daily пуст — сначала `fetch-external`.")

    values = fg.build_fear_greed(base, daily)
    atr14 = ind.atr(base, 14)
    rv_96 = ind.realized_vol(base["close"], 96)

    frame = pd.DataFrame(index=base.index)
    frame["fgi_level"] = values["fgi_level"]
    frame["rv_96"] = rv_96
    frame["_atr14"] = atr14
    frame["_close"] = base["close"]
    frame["_high"] = base["high"]
    frame["_low"] = base["low"]

    bars_per_day = config.data.bars_per("1d")
    _, block_rows = autocorr_block_rows(daily["value"].astype(float), bars_per_day)
    return frame, block_rows


def measure(symbol: str, frame: pd.DataFrame, block_rows: int) -> list[dict]:
    results = []
    for horizon_label in HORIZONS:
        h_bars = _horizon_bars(horizon_label)
        atr14 = frame["_atr14"]
        base_like = frame[["_high", "_low", "_close"]].rename(
            columns={"_high": "high", "_low": "low", "_close": "close"}
        )
        range_ratio = forward_range_ratio(base_like, atr14, h_bars)

        joined = pd.concat(
            [frame["fgi_level"], frame["rv_96"], range_ratio.rename("range_ratio")], axis=1
        ).dropna()

        from btcproc.analysis.lift import block_length_rows

        horizon_minutes = h_bars * config.data.base_minutes
        block_by_horizon = block_length_rows(pd.Series(joined.index), horizon_minutes)
        block = max(block_by_horizon, block_rows)

        for epoch_label, start, end in [("вся история", None, None)] + EPOCHS:
            window = joined
            if start:
                window = window[window.index >= pd.Timestamp(start, tz="UTC")]
            if end:
                window = window[window.index <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
            if len(window) < 50:
                continue

            # crc32, а не hash(): хэш строк в Python рандомизирован между
            # процессами (PYTHONHASHSEED), и замер не воспроизводился бы от
            # запуска к запуску — при том что все прочие замеры проекта сидят
            # на фиксированном seed именно ради перепроверки чисел.
            salt = zlib.crc32(f"{symbol}|{horizon_label}|{epoch_label}".encode())
            rng = np.random.default_rng([42, salt])
            fgi_arr = window["fgi_level"].to_numpy()
            rv_arr = window["rv_96"].to_numpy()
            target = window["range_ratio"].to_numpy()

            rho_fgi, p_fgi = spearman_block_p(fgi_arr, target, block, N_BOOT, rng)
            rho_fgi2, rho_rv, p_beats = paired_diff_p(fgi_arr, rv_arr, target, block, N_BOOT, rng)

            results.append({
                "symbol": symbol, "horizon": horizon_label, "epoch": epoch_label,
                "n": len(window), "block": block,
                "rho_fgi": rho_fgi, "p_fgi": p_fgi,
                "rho_rv": rho_rv, "p_beats_rv": p_beats,
            })
    return results


def format_table(rows: list[dict]) -> str:
    header = (f"{'горизонт':<10} {'эпоха':<18} {'n':>7} {'блок':>6} "
              f"{'rho(FGI)':>9} {'p(FGI)':>8} {'rho(rv)':>9} {'p(FGI>rv)':>10}  вердикт")
    lines = [header, "─" * len(header)]
    for r in rows:
        sig_fgi = r["p_fgi"] <= 0.05
        beats = r["p_beats_rv"] <= 0.05 and abs(r["rho_fgi"]) > abs(r["rho_rv"])
        verdict = "ГЕЙТ R" if sig_fgi and beats else ("значим, но не сильнее rv" if sig_fgi else "не значим")
        lines.append(
            f"{r['horizon']:<10} {r['epoch']:<18} {r['n']:>7} {r['block']:>6} "
            f"{r['rho_fgi']:>+9.4f} {r['p_fgi']:>8.4f} {r['rho_rv']:>+9.4f} "
            f"{r['p_beats_rv']:>10.4f}  {verdict}"
        )
    return "\n".join(lines)


def gate_verdict(rows: list[dict]) -> None:
    print("\n── Гейт R: свод по эпохам (без «вся история» — она не входит в критерий) ──")
    for horizon_label in HORIZONS:
        by_epoch = [r for r in rows if r["horizon"] == horizon_label and r["epoch"] != "вся история"]
        passed = sum(1 for r in by_epoch
                     if r["p_fgi"] <= 0.05 and r["p_beats_rv"] <= 0.05
                     and abs(r["rho_fgi"]) > abs(r["rho_rv"]))
        verdict = "ПРОЙДЕН" if passed >= 2 else "НЕ ПРОЙДЕН"
        print(f"{horizon_label}: прошло эпох {passed}/{len(by_epoch)} → {verdict}")


def run_one(symbol: str) -> list[dict]:
    frame, block_rows = build_frame(symbol)
    print(f"\n{symbol}: block_rows по автокорреляции FGI = {block_rows}")
    rows = measure(symbol, frame, block_rows)
    print(format_table(rows))
    gate_verdict(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Гейт R — FGI против размаха")
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        targets = symbols.tickers(only_enabled=True)
    elif args.symbol:
        targets = args.symbol
    else:
        targets = [config.data.symbol]

    for symbol in targets:
        print(f"\n{'=' * 78}\n=== {symbol}\n{'=' * 78}")
        run_one(symbol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
