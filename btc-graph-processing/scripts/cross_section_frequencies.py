"""
Гейты D, N и F поперечного сечения — задачи A и B ТЗ
`crypto-graph/docs/tz_cross_section_20-08-26.md`.

    python3 scripts/cross_section_frequencies.py
    python3 scripts/cross_section_frequencies.py --section data --section mde
    python3 scripts/cross_section_frequencies.py --section novelty --symbol BTCUSDT

**Ничего не пишет — ни в БД, ни в конфиг.** Ни `train`, ни новых таблиц, ни
единого похода в чужой API: шесть монет уже лежат в `processing.ohlcv` на
общей 15-минутной сетке, и в этом весь аргумент «самое недооценённое» из §5.1
`ideas_math`.

## Разделы

    data      гейт D: глубина корзины по времени, размер корзины по барам,
              выравнивание сетки, дыры, чувствительность к взвешиванию
    mde       минимально детектируемый IC на каждом горизонте — ДО любого
              замера эффекта, потому что ячейка слабее MDE интерпретации не
              подлежит независимо от p-value
    freq      гейт F: распределения по годам и по монетам, размер корзины
              рядом с каждой строкой
    class     признак класса: максимальная попарная корреляция ряда МЕЖДУ
              монетами. Печатается для КАЖДОЙ величины, включая заявленные
              как помонетные, — величина, случайно оказавшаяся общерыночной,
              обязана ловиться здесь, а не в выводах (урок FGI, §0.3 ТЗ)
    novelty   гейт N: out-of-sample R² регрессии величины на все 32 базовых
              признака, порог 0.80. Парная корреляция как основание запрещена

Эффект здесь не меряется вовсе: задачи C и D ставятся отдельно и только после
того, как хотя бы одна величина прошла гейты.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["SINK_MODE"] = "none"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from btcproc import config, symbols  # noqa: E402
from btcproc.analysis import cross_section as xs  # noqa: E402
from btcproc.analysis.lift import block_length_rows  # noqa: E402
from btcproc.analysis.range_model import horizon_minutes  # noqa: E402

FROZEN_END = "2026-08-01"

#: Порог гейта N — тот же, что применялся к деривативам.
NOVELTY_GATE_R2 = 0.80

#: Порог практической величины среднего IC (§4.4 ТЗ). Здесь он нужен только
#: для того, чтобы сопоставить его с MDE: ячейка, где MDE выше порога,
#: непроходима в принципе, и узнать это надо ДО замера.
IC_THRESHOLD = 0.02

#: Горизонты задачи C после правки 2026-08-21. Прежние 4h/12h/24h делали
#: критерий невыполнимым: на двух из трёх MDE был выше порога 0.02.
HORIZONS_C = ("1h", "2h", "4h")
#: Горизонты задачи D — про размах, и там они те же, что в разделах 47 и 49.
HORIZONS_D = ("4h", "12h", "24h")


# ─── Гейт D: данные ─────────────────────────────────────────────────────────
def section_data(basket: xs.Basket, raw: dict[str, pd.DataFrame]) -> None:
    print("\n── Гейт D: данные ──────────────────────────────────────────────")

    print("\nГлубина корзины по времени (из symbols.py, проверено фактом):")
    print(f"{'монета':<10} {'history_start':>14} {'первый бар':>12} "
          f"{'последний':>12} {'баров':>9}")
    for ticker, frame in raw.items():
        spec = symbols.get(ticker)
        first, last = frame.index[0].date(), frame.index[-1].date()
        print(f"{ticker:<10} {spec.start_date():>14} {str(first):>12} "
              f"{str(last):>12} {len(frame):>9}")

    sizes = basket.size()
    print(f"\nРазмер корзины по барам (после отсева N < {xs.MIN_BASKET}):")
    print(f"{'N':>3} {'баров':>9} {'первый':>12} {'последний':>12}")
    for size, group in sizes.groupby(sizes):
        print(f"{int(size):>3} {len(group):>9} "
              f"{str(group.index[0].date()):>12} "
              f"{str(group.index[-1].date()):>12}")
    print(f"Итого баров в замере: {len(sizes)}, средневзвешенный N−1 = "
          f"{float((sizes - 1).mean()):.2f}")

    print("\nВыравнивание сетки и дыры:")
    print(f"{'монета':<10} {'вне сетки':>10} {'дыр':>6} {'макс. дыра':>18} "
          f"{'площадка':>12}")
    step = pd.Timedelta(minutes=config.data.base_minutes)
    for ticker, frame in raw.items():
        # Через компоненты времени, а не через int64: на pandas 3 индекс по
        # умолчанию в МИКРОсекундах, и привычное деление на 1e9 дало бы
        # «вне сетки» почти на каждом баре — ошибка бесшумная и правдоподобная.
        index = frame.index
        off_grid = int((
            (index.minute % config.data.base_minutes != 0)
            | (index.second != 0)
            | (index.microsecond != 0)
        ).sum())
        deltas = frame.index.to_series().diff().dropna()
        holes = deltas[deltas != step]
        biggest = str(holes.max()) if len(holes) else "—"
        print(f"{ticker:<10} {off_grid:>10} {len(holes):>6} {biggest:>18} "
              f"{symbols.get(ticker).venue:>12}")
    print("Ненулевая доля рассинхрона — находка про данные, а не мелочь: она")
    print("означает, что джойн панели по ts молча выбросит эти бары.")

    print("\nЧувствительность к взвешиванию (равновесная корзина — в выводах,")
    print("объёмная — только как проверка): доля BTC в обороте корзины")
    volume = basket.masked(basket.frames["quote_volume"])
    rolled = volume.rolling(xs.WINDOW_1D, min_periods=xs.WINDOW_1D // 2).sum()
    total = rolled.sum(axis=1, skipna=True)
    share = (rolled["BTCUSDT"] / total.where(total > 0)).dropna()
    print(f"  p10={share.quantile(0.1):.3f} медиана={share.median():.3f} "
          f"p90={share.quantile(0.9):.3f}")
    print("  Объёмное взвешивание превратило бы корзину в BTC с добавками,")
    print("  то есть в общерыночный ряд — поэтому оно и не берётся в выводы.")


# ─── Мощность ───────────────────────────────────────────────────────────────
def section_mde(basket: xs.Basket) -> None:
    print("\n── Мощность: MDE среднего IC ДО любого замера эффекта ──────────")
    sizes = basket.size()
    ts = basket.index.to_series()
    print(f"{'горизонт':>9} {'блок':>6} {'n_эфф':>9} {'MDE':>8} "
          f"{'порог 0.02':>12} {'задача':>9}")
    for horizon, task in [(h, "C") for h in HORIZONS_C] + \
                         [(h, "D") for h in HORIZONS_D if h not in HORIZONS_C]:
        block = block_length_rows(ts, horizon_minutes(horizon))
        mde = xs.minimum_detectable_ic(sizes, block)
        verdict = "проходима" if mde < IC_THRESHOLD else "НЕДОСТАТОЧНО МОЩНАЯ"
        print(f"{horizon:>9} {block:>6} {len(sizes) // block:>9} {mde:>8.4f} "
              f"{verdict:>12} {task:>9}")
    print("Ячейка, где наблюдённый эффект меньше MDE, интерпретации не подлежит")
    print("независимо от p-value (§4.3 ТЗ).")


# ─── Гейт F: распределения ──────────────────────────────────────────────────
def _describe(series: pd.Series) -> str:
    quantiles = series.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    return "  ".join(f"{q:>7.3f}" for q in quantiles)


def section_freq(per_symbol: dict[str, pd.DataFrame],
                 market_wide: dict[str, pd.Series], basket: xs.Basket) -> None:
    print("\n── Гейт F: распределения ───────────────────────────────────────")
    sizes = basket.size()
    header = f"{'величина':<18} {'год':>6} {'N':>4} {'n':>8}   " \
             f"{'p10':>7}  {'p25':>7}  {'p50':>7}  {'p75':>7}  {'p90':>7}"

    print("\nПо годам (величина, чьё распределение гуляет в разы между эпохами,")
    print("измеряет режим, а не рынок):")
    print(header)
    print("─" * len(header))
    for name, frame in per_symbol.items():
        stacked = frame.stack(future_stack=True).dropna()
        for year, group in stacked.groupby(stacked.index.get_level_values(0).year):
            mean_n = float(sizes[sizes.index.year == year].mean())
            print(f"{name:<18} {year:>6} {mean_n:>4.1f} {len(group):>8}   "
                  f"{_describe(group)}")
    for name, series in market_wide.items():
        clean = series.dropna()
        for year, group in clean.groupby(clean.index.year):
            mean_n = float(sizes[sizes.index.year == year].mean())
            print(f"{name:<18} {year:>6} {mean_n:>4.1f} {len(group):>8}   "
                  f"{_describe(group)}")

    print("\nПо монетам (доли и ранги безразмерны, поэтому сильное расхождение")
    print("между BTC и SOL было бы находкой, а не нормой):")
    print(f"{'величина':<18} {'монета':<10} {'n':>8}   {'p10':>7}  {'p25':>7}  "
          f"{'p50':>7}  {'p75':>7}  {'p90':>7}")
    for name, frame in per_symbol.items():
        for ticker in frame.columns:
            clean = frame[ticker].dropna()
            if len(clean) < 100:
                continue
            print(f"{name:<18} {ticker:<10} {len(clean):>8}   {_describe(clean)}")


def section_class(per_symbol: dict[str, pd.DataFrame]) -> None:
    print("\n── Признак класса: корреляция ряда МЕЖДУ монетами ──────────────")
    print("Значение около 1.0 означает общерыночную величину, как бы она ни")
    print("называлась. Урок FGI: один ряд, показанный шесть раз, даёт ОДИН")
    print("замер, а не шесть подтверждений.\n")
    print(f"{'величина':<18} {'макс. попарная r':>18} {'класс':>16}")
    for name, frame in per_symbol.items():
        correlation = xs.cross_symbol_correlation(frame)
        klass = "ОБЩЕРЫНОЧНАЯ" if correlation > 0.9 else "помонетная"
        print(f"{name:<18} {correlation:>18.3f} {klass:>16}")


# ─── Гейт N: новизна ────────────────────────────────────────────────────────
def section_novelty(per_symbol: dict[str, pd.DataFrame],
                    market_wide: dict[str, pd.Series],
                    tickers: list[str], end: str) -> None:
    from btcproc.analysis.novelty import novelty_r2
    from btcproc.features import builder as feat
    from btcproc.ingest import bars

    print("\n── Гейт N: новизна против всего вектора (порог 0.80) ───────────")
    print("Парная корреляция как основание запрещена: источник, наполовину")
    print("пересобранный из имеющихся величин, проходит порог |r| > 0.9 легко")
    print("при множественной R² 0.85+.\n")
    print(f"{'величина':<18} {'монета':<10} {'n':>8} {'R²':>8} {'вердикт':>22}")

    for ticker in tickers:
        spec = symbols.get(ticker)
        base = bars.load_ohlcv(ticker, config.data.base_tf, spec.start_date(), end)
        context = {tf: bars.load_ohlcv(ticker, tf, spec.start_date(), end)
                   for tf in config.data.context_tfs}
        features = feat.build_features(base, context, symbol=ticker)

        for name, frame in per_symbol.items():
            if ticker not in frame.columns:
                continue
            series = frame[ticker].reindex(features.index)
            if series.notna().sum() < 500:
                print(f"{name:<18} {ticker:<10} {series.notna().sum():>8} "
                      f"{'—':>8} {'мало наблюдений':>22}")
                continue
            r2, _model, top, _cols, index, _cut = novelty_r2(
                series.dropna(), features.loc[series.dropna().index])
            verdict = "ГЕЙТ N НЕ ПРОЙДЕН" if r2 >= NOVELTY_GATE_R2 else "пройден"
            print(f"{name:<18} {ticker:<10} {len(index):>8} {r2:>8.4f} "
                  f"{verdict:>22}")
            print(f"{'':<18} топ-3: "
                  f"{', '.join(f'{n}={v:.3f}' for n, v in top[:3])}")

        # Общерыночные считаются один раз — на признаках любой монеты они дадут
        # разные числа, поэтому берётся первая по списку и это проговаривается.
        if ticker == tickers[0]:
            for name, series in market_wide.items():
                aligned = series.reindex(features.index).dropna()
                if len(aligned) < 500:
                    continue
                r2, _model, top, _cols, index, _cut = novelty_r2(
                    aligned, features.loc[aligned.index])
                verdict = "ГЕЙТ N НЕ ПРОЙДЕН" if r2 >= NOVELTY_GATE_R2 else "пройден"
                print(f"{name:<18} {'(общая)':<10} {len(index):>8} {r2:>8.4f} "
                      f"{verdict:>22}")
                print(f"{'':<18} признаки {ticker}, топ-3: "
                      f"{', '.join(f'{n}={v:.3f}' for n, v in top[:3])}")


# ─── Точка входа ────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Гейты D, N и F поперечного сечения (ТЗ 2026-08-20)")
    parser.add_argument("--section", action="append",
                        choices=["data", "mde", "freq", "class", "novelty"])
    parser.add_argument("--symbol", action="append",
                        help="для гейта N: на признаках каких монет считать")
    parser.add_argument("--start", default=None,
                        help="начало окна; по умолчанию с первого бара, отсев "
                             "по размеру корзины делает своё дело сам")
    parser.add_argument("--end", default=FROZEN_END)
    args = parser.parse_args()
    sections = args.section or ["data", "mde", "freq", "class"]

    from btcproc.ingest import bars

    print(f"Поперечное сечение. Граница {args.end}, минимальная корзина "
          f"{xs.MIN_BASKET}, прогрев {xs.WARMUP}.")
    print("Направление в исходной формулировке здесь не меряется: цель "
          "поперечная, общий рыночный фактор из неё вычтен (§0.4 ТЗ).\n")

    raw = {}
    for spec in symbols.enabled():
        frame = bars.load_ohlcv(spec.ticker, config.data.base_tf,
                                args.start or spec.start_date(), args.end)
        if not frame.empty:
            raw[spec.ticker] = frame
    if not raw:
        raise SystemExit("в базе нет баров — сначала ingest")

    basket = xs.load_basket(end=args.end, start=args.start)
    print(f"Панель: {len(basket.index)} баров, {len(basket.tickers)} монет, "
          f"{basket.index[0]:%Y-%m-%d}…{basket.index[-1]:%Y-%m-%d}")

    per_symbol, market_wide = xs.measures(basket)

    if "data" in sections:
        section_data(basket, raw)
    if "mde" in sections:
        section_mde(basket)
    if "freq" in sections:
        section_freq(per_symbol, market_wide, basket)
    if "class" in sections:
        section_class(per_symbol)
    if "novelty" in sections:
        tickers = args.symbol or ["BTCUSDT", "SOLUSDT"]
        section_novelty(per_symbol, market_wide, tickers, args.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
