"""
Поток B: работают ли зонные концепции SMC на СТАРШЕМ таймфрейме.

Гипотеза, которую проверяет замер: «зоны работают, но не на 15-минутке».
Практика SMC строит ордер-блоки, имбалансы и премиум/дисконт на H1, H4 и D1,
а у нас всё считалось на базовом ТФ, где `SMC_SWING_RIGHT = 3` означает
подтверждение свинга за 45 минут. Зона, построенная на таком свинге, к
H4-ордер-блоку отношения не имеет — это микроструктурный шум с тем же именем.
Вопрос стоял открытым ещё в первом ТЗ по SMC и первым замером закрыт не был.

    python3 scripts/measure_zone_timeframe.py --symbol BTCUSDT
    python3 scripts/measure_zone_timeframe.py --all --metric realized
    python3 scripts/measure_zone_timeframe.py --tf 15m --tf 4h --tf 1d

Как считается. Детекторы `smc.build_smc` гоняются по барам КАЖДОГО
таймфрейма отдельно, результат подмешивается в базовый ТФ через `shift(1)` +
`ffill` — тем же механизмом, что старшие ТФ в `features/_context_features`.
`shift(1)` обязателен: значение старшего бара становится известно только
после его закрытия, без сдвига в признак попадала бы информация из будущего
относительно 15m-бара.

Скрипт **ничего не пишет** — ни в БД, ни в конфиг. Он отвечает на вопрос
«растёт ли лифт с таймфреймом», и только. Если растёт — это самостоятельный
результат, и тогда обсуждается перенос зонных детекторов на старший ТФ; если
нет — гипотеза «дело в таймфрейме» закрыта.

Чего здесь делать нельзя: подбирать таймфрейм по результату лифта и
объявлять победителя. Порядок ТФ задан заранее и мал (три-четыре значения),
поправка на множественные сравнения считается по всем проверенным парам
(атом × ТФ), а не по одному лучшему.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from btcproc import config, symbols  # noqa: E402
from btcproc.analysis import samples  # noqa: E402
from btcproc.analysis.gradation import format_table as format_gradation  # noqa: E402
from btcproc.analysis.gradation import measure_gradation  # noqa: E402
from btcproc.analysis.lift import DEFAULT_N_BOOT, format_table, measure_lift  # noqa: E402
from btcproc.features import smc  # noqa: E402
from btcproc.ingest import binance  # noqa: E402

#: Зонная половина SMC — то, о чём идёт спор. Структурные атомы (bos/choch)
#: сюда не входят: они первый замер прошли и меряются отдельно.
ZONE_ATOMS = [
    "in_bullish_ob",
    "in_bearish_ob",
    "in_breaker",
    "in_unfilled_fvg",
    "in_discount",
    "in_premium",
    "sweep_high",
    "sweep_low",
]

ZONE_FEATURES = [
    "near_ob",
    "ob_age_norm",
    "ob_touch_count_norm",
    "near_fvg_bull",
    "near_fvg_bear",
    "fvg_fill_pct",
    "premium_discount",
    "near_liquidity",
]

DEFAULT_TFS = ("15m", "1h", "4h")


def zone_values(symbol: str, timeframe: str, base_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    SMC-величины таймфрейма, приведённые к индексу базовых баров.

    Для базового ТФ сдвиг не нужен: величина на баре t уже считается только
    из баров до t включительно (дисциплина `smc.py`). Для старших —
    `shift(1)` обязателен, иначе на 15-минутный бар попадёт значение ещё не
    закрытого 4-часового.
    """
    bars = binance.load_ohlcv(symbol, timeframe, None, None)
    if bars.empty:
        raise SystemExit(
            f"Нет баров {symbol} на {timeframe}. Старшие ТФ агрегируются из "
            f"базового: python3 -m btcproc.cli ingest --symbol {symbol}"
        )
    values = smc.build_smc(bars)
    if timeframe != config.data.base_tf:
        values = values.shift(1)
    return values.reindex(base_index, method="ffill")


def run_one(symbol: str, args) -> None:
    model_run = samples.resolve_model_run(symbol, args.run)
    frame = samples.load(symbol, args.metric, model_run)
    if frame.empty:
        print(f"Нет данных по {symbol}. Нужен прогон train или live.")
        return

    print(f"\n{'=' * 78}\n=== {symbol}: зоны против таймфрейма\n{'=' * 78}")
    print(f"{len(frame)} кандидатов модели #{model_run}, метрика {args.metric}, "
          f"базовая доля {frame['metric'].mean():.4f}")

    horizon_minutes = None if args.no_bootstrap else config.data.horizon_minutes
    base_index = pd.DatetimeIndex(frame["ts"].unique()).sort_values()

    summary: dict[str, dict] = {}
    for timeframe in args.tf or list(DEFAULT_TFS):
        values = zone_values(symbol, timeframe, base_index)
        joined = frame.join(values, on="ts").dropna(subset=ZONE_FEATURES)
        if joined.empty:
            print(f"\n[{timeframe}] после джойна не осталось строк — пропуск.")
            continue

        # Булевы зоны меряются как атомы, непрерывные — как градация: у
        # первых градации нет вовсе, у вторых бинарный флаг её теряет.
        joined["atoms"] = [
            [name for name in ZONE_ATOMS if bool(row[name])]
            for _, row in joined[ZONE_ATOMS].iterrows()
        ]

        print(f"\n── {timeframe}: булевы зоны ───────────────────────────────────")
        atom_results = measure_lift(
            joined, atoms=ZONE_ATOMS, metric_column="metric",
            alpha=args.alpha, correction=args.correction,
            holdout=args.holdout or None, min_group=args.min_group,
            horizon_minutes=horizon_minutes, n_boot=args.n_boot,
        )
        print(format_table(atom_results, args.correction, args.alpha, args.n_boot))

        print(f"\n── {timeframe}: градация зон ──────────────────────────────────")
        feature_results = measure_gradation(
            joined, features=ZONE_FEATURES, metric_column="metric",
            bins=args.bins, alpha=args.alpha, correction=args.correction,
            holdout=args.holdout or None,
            horizon_minutes=horizon_minutes, n_boot=args.n_boot,
        )
        print(format_gradation(feature_results, args.correction, args.alpha, args.n_boot))

        summary[timeframe] = {
            "atoms": {r.atom: r for r in atom_results},
            "features": {r.feature: r for r in feature_results},
        }

    _compare(summary)


def _compare(summary: dict[str, dict]) -> None:
    """
    Главная таблица: растёт ли эффект с таймфреймом.

    Именно тренд по ТФ, а не «где значимо»: гипотеза потока B — «зоны
    работают, но не на базовом ТФ», и её подтверждает монотонный рост
    величины эффекта, а не одиночное срабатывание на каком-то из ТФ.
    Одиночное как раз ожидаемо при трёх-четырёх проверенных ТФ.
    """
    if len(summary) < 2:
        return
    tfs = list(summary)

    print(f"\n{'─' * 78}\nСРАВНЕНИЕ ПО ТАЙМФРЕЙМАМ (лифт / разброс, ДА = прошёл всё)")
    header = f"{'детектор':<24}" + "".join(f"{tf:>18}" for tf in tfs)
    print(header)
    print("─" * len(header))

    for name in ZONE_ATOMS:
        cells = ""
        for tf in tfs:
            r = summary[tf]["atoms"].get(name)
            cells += f"{'—':>18}" if r is None else \
                f"{r.lift:>+13.4f}{('ДА' if r.confirmed else ('зн.' if r.significant else '')):>5}"
        print(f"{name:<24}{cells}")
    print()
    for name in ZONE_FEATURES:
        cells = ""
        for tf in tfs:
            r = summary[tf]["features"].get(name)
            cells += f"{'—':>18}" if r is None or not r.bins else \
                f"{r.spread:>+13.4f}{('ДА' if r.confirmed else ('зн.' if r.significant else '')):>5}"
        print(f"{name:<24}{cells}")

    print("\nЧитать как: гипотезу «зоны работают, но не на базовом ТФ» "
          "подтверждает РОСТ величины эффекта с таймфреймом,\nа не одиночное "
          "срабатывание на каком-то из них — одиночное ожидаемо при трёх "
          "проверенных ТФ само по себе.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Зонные детекторы SMC по таймфреймам")
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--run", type=int,
                        help="Прогон-модель; по умолчанию последний train монеты")
    parser.add_argument("--tf", action="append",
                        help=f"Таймфреймы; по умолчанию {DEFAULT_TFS}")
    parser.add_argument("--metric", choices=["long_outcome_share", "realized"],
                        default="realized")
    parser.add_argument("--correction", choices=["bonferroni", "bh", "none"],
                        default="bh")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--holdout", type=float, default=0.3)
    parser.add_argument("--min-group", type=int, default=30)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--no-bootstrap", action="store_true")
    args = parser.parse_args()

    if args.all:
        targets = symbols.tickers(only_enabled=True)
    elif args.symbol:
        targets = args.symbol
    else:
        targets = [config.data.symbol]

    for symbol in targets:
        run_one(symbol, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
