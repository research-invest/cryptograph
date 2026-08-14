"""
Поток A: лифт по ГРАДАЦИИ непрерывных признаков, а не по бинарным атомам.

Первый замер SMC мерил булевы флаги: `in_unfilled_fvg` — «цена внутри
незакрытого имбаланса», и всё. Свежесть зоны, её размер в ATR, доля уже
закрытого разрыва, номер касания — вся градация живёт в непрерывных
признаках `near_*`, `fvg_fill_pct`, `ob_age_norm`, `premium_discount`, а они
в замере не участвовали вовсе: признаки выключены флагом
`SMC_FEATURES_ENABLED`. То есть мерилась худшая треть того, что реализовано.

Этот скрипт меряет остальные две трети — и **не требует включать признаки**.
Значения считаются прямо здесь, `smc.build_smc` по барам монеты, и
приджойниваются к кандидатам по `ts`. Модель состояний не трогается,
переобучение не нужно, калибровка дробления ни при чём. Поэтому поток A идёт
параллельно ломающему пакету, а не после него.

    python3 scripts/measure_feature_gradation.py --symbol BTCUSDT
    python3 scripts/measure_feature_gradation.py --all --metric realized
    python3 scripts/measure_feature_gradation.py --bins 3      # терцили
    python3 scripts/measure_feature_gradation.py --feature near_ob --feature ob_age_norm

Проверяется не разница двух групп, а **монотонность тренда** по квантильным
бинам (Cochran–Armitage, `btcproc/analysis/gradation.py`). Это существенно
более сильное свидетельство: один удачно попавший порог объясняется
случайностью легко, монотонность по пяти бинам — трудно.

Значимость считается по блочному бутстрапу: горизонты кандидатов
перекрываются ровно так же, как в замере атомов (шапка
`btcproc/analysis/lift.py`).

Чего этот замер НЕ делает: не подбирает пороги детекторов под результат.
Пороги калибруются по частоте срабатывания отдельно; подгонка порога под
лифт — это подгонка, и её результат ничего не стоит.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from btcproc import config, symbols  # noqa: E402
from btcproc.analysis import samples  # noqa: E402
from btcproc.analysis.gradation import (  # noqa: E402
    DEFAULT_BINS,
    format_table,
    measure_gradation,
)
from btcproc.analysis.lift import DEFAULT_N_BOOT, block_length_rows  # noqa: E402
from btcproc.features import smc  # noqa: E402
from btcproc.ingest import bars  # noqa: E402

FGI_FEATURES = ["fgi_level", "fgi_residual"]


def attach_features(frame: pd.DataFrame, symbol: str,
                    names: list[str]) -> pd.DataFrame:
    """
    Приджойнить SMC-величины к кандидатам по времени бара.

    Считаем через `smc.build_smc` напрямую, а не через флаг: замер обязан
    работать при выключенном `SMC_FEATURES_ENABLED` — в этом весь смысл
    потока A. Величины на баре t считаются только из баров до t включительно
    (дисциплина модуля `smc.py`), поэтому джойн по `ts` заглядывания вперёд
    не создаёт.
    """
    base = bars.load_ohlcv(symbol, config.data.base_tf, None, None)
    if base.empty:
        raise SystemExit(f"Нет баров по {symbol} — сначала ingest.")

    values = smc.build_smc(base)[names]
    joined = frame.join(values, on="ts")

    missing = int(joined[names[0]].isna().sum())
    if missing:
        joined = joined.dropna(subset=names)
        print(f"Пропущено {missing} кандидатов без бара в ohlcv "
              f"(бары монеты не догружены за этот период).")
    return joined


def attach_fgi_features(frame: pd.DataFrame, symbol: str,
                        names: list[str]) -> pd.DataFrame:
    """
    То же самое для Fear & Greed: fgi_level напрямую из джойна дневного ряда,
    fgi_residual — остаток OLS на 32 базовых признака (docs/task_fear_greed.md
    §1.2), коэффициенты обучены на первых 70% истории и применены ко всей —
    см. btcproc/analysis/fgi_novelty.py.

    Используется, только если гейт N дал R² в [0.5, 0.8): «все замеры
    дублируются на остатке» (§1.2) — если работает ценовая половина индекса,
    а не он сам, монотонность у fgi_level будет, а у fgi_residual пропадёт.
    """
    from btcproc.analysis.fgi_novelty import compute_fgi_residual
    from btcproc.features import builder as feat
    from btcproc.features import fear_greed as fg
    from btcproc.ingest import external

    base = bars.load_ohlcv(symbol, config.data.base_tf, None, None)
    if base.empty:
        raise SystemExit(f"Нет баров по {symbol} — сначала ingest.")
    daily = external.load_external_daily(external.FEAR_GREED_SERIES)
    if daily.empty:
        raise SystemExit("external_daily пуст — сначала `fetch-external`.")

    values = fg.build_fear_greed(base, daily)[["fgi_level"]]
    if "fgi_residual" in names:
        context = {tf: bars.load_ohlcv(symbol, tf, None, None) for tf in config.data.context_tfs}
        features = feat.build_features(base, context)
        values = values.reindex(features.index)
        values["fgi_residual"] = compute_fgi_residual(values["fgi_level"], features)

    joined = frame.join(values[names], on="ts")
    missing = int(joined[names[0]].isna().sum())
    if missing:
        joined = joined.dropna(subset=names)
        print(f"Пропущено {missing} кандидатов без значения FGI "
              f"(период до начала истории индекса, 2018-02-01, или вне прогрева признаков).")
    return joined


def run_one(symbol: str, args) -> list | None:
    model_run = samples.resolve_model_run(symbol, args.run)
    frame = samples.load(symbol, args.metric, model_run)
    if frame.empty:
        print(f"Нет данных по {symbol}. Нужен прогон train или live.")
        return None

    if args.source == "fgi":
        names = args.feature or list(FGI_FEATURES)
        unknown = [n for n in names if n not in FGI_FEATURES]
        if unknown:
            raise SystemExit(f"Неизвестные признаки: {', '.join(unknown)}")
        frame = attach_fgi_features(frame, symbol, names)
    else:
        names = args.feature or list(smc.FEATURE_CANDIDATES)
        unknown = [n for n in names if n not in smc.FEATURE_CANDIDATES]
        if unknown:
            raise SystemExit(f"Неизвестные признаки: {', '.join(unknown)}")
        frame = attach_features(frame, symbol, names)

    if frame.empty:
        print(f"После джойна с барами по {symbol} ничего не осталось.")
        return None

    horizon_minutes = None if args.no_bootstrap else config.data.horizon_minutes
    results = measure_gradation(
        frame,
        features=names,
        metric_column="metric",
        bins=args.bins,
        alpha=args.alpha,
        correction=args.correction,
        holdout=args.holdout or None,
        horizon_minutes=horizon_minutes,
        n_boot=args.n_boot,
        min_block_rows=args.block_rows,
    )

    span = f"{frame['ts'].min():%Y-%m-%d} … {frame['ts'].max():%Y-%m-%d}"
    print(f"\n{symbol}: {len(frame)} кандидатов модели #{model_run}, {span}")
    print(f"Метрика: {args.metric}"
          + ("  (ожидание системы, не факт)" if args.metric == "long_outcome_share"
             else "  (фактические исходы)"))
    if horizon_minutes:
        block = block_length_rows(frame["ts"], horizon_minutes)
        if args.block_rows:
            block = max(block, args.block_rows)
        print(f"Горизонт {config.data.horizon}; блок бутстрапа {block} строк, "
              f"реплик {args.n_boot}")
    print(f"Базовая доля: {frame['metric'].mean():.4f}, бинов: {args.bins}\n")
    print(format_table(results, args.correction, args.alpha, args.n_boot))

    confirmed = [r.feature for r in results if r.confirmed]
    print(f"\nПрошли всё: {', '.join(confirmed) if confirmed else 'ни одного'}")
    if not confirmed:
        print("Отрицательный результат — валидный результат.")
    return results


def consolidate(per_symbol: dict[str, list]) -> None:
    """
    Сводка по монетам: где тренд прошёл и совпало ли направление.

    Три монеты — НЕ три независимых теста: BTC, ETH и SOL коррелированы на
    0.8+ по дневным доходностям. Совпадение знака на всех трёх ловит грубую
    подгонку под один инструмент, но перемножать вероятности нельзя.
    """
    symbols_list = list(per_symbol)
    features: dict[str, dict] = {}
    for symbol, results in per_symbol.items():
        for r in results:
            features.setdefault(r.feature, {})[symbol] = r

    header = (f"\n{'признак':<24}"
              + "".join(f"{s.replace('USDT', ''):>18}" for s in symbols_list)
              + f"{'прошёл':>9} {'знак':>8}")
    print("\n" + "=" * 78)
    print("СВОДКА ПО МОНЕТАМ — разброс (holdout), «ДА» = прошёл всё")
    print("=" * 78)
    print(header)
    print("─" * len(header))

    for name in sorted(features):
        rows = features[name]
        cells = ""
        for symbol in symbols_list:
            r = rows.get(symbol)
            if r is None or not r.bins:
                cells += f"{'—':>18}"
                continue
            mark = "ДА" if r.confirmed else ("зн." if r.significant else "")
            cells += f"{r.spread:>+11.4f}{mark:>7}"
        passed = sum(1 for r in rows.values() if r.confirmed)
        signs = {r.spread > 0 for r in rows.values() if r.bins}
        agree = "один" if len(signs) == 1 else "разный"
        print(f"{name:<24}{cells}{passed:>6}/{len(rows)} {agree:>8}")

    print("\n«зн.» — значим после поправки, но монотонность или holdout не сошлись.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Квантильный лифт по непрерывным SMC-признакам (поток A)")
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true",
                        help="Все активные монеты и сводка по ним")
    parser.add_argument("--run", type=int,
                        help="Прогон-модель; по умолчанию последний train монеты")
    parser.add_argument("--metric", choices=["long_outcome_share", "realized"],
                        default="realized",
                        help="По умолчанию realized — выводы делаются по ней")
    parser.add_argument("--source", choices=["smc", "fgi"], default="smc",
                        help="Чьи признаки мерить: SMC (12 штук) или "
                             "Fear & Greed (fgi_level, fgi_residual)")
    parser.add_argument("--feature", action="append",
                        help="Только эти признаки; по умолчанию все признаки источника")
    parser.add_argument("--block-rows", type=int, default=None,
                        help="Нижняя граница длины блока бутстрапа (max с блоком "
                             "по горизонту) — для FGI берётся из fgi_frequencies.py "
                             "--section autocorr")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--correction", choices=["bonferroni", "bh", "none"],
                        default="bh",
                        help="Двенадцать тестов — BH мягче и здесь уместнее")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--holdout", type=float, default=0.3,
                        help="Доля хвоста по времени на подтверждение; 0 — без него")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Только наивный тест — он анти-консервативен")
    args = parser.parse_args()

    if args.all:
        targets = symbols.tickers(only_enabled=True)
    elif args.symbol:
        targets = args.symbol
    else:
        targets = [config.data.symbol]

    per_symbol = {}
    for symbol in targets:
        results = run_one(symbol, args)
        if results:
            per_symbol[symbol] = results

    if not per_symbol:
        return 1
    if len(per_symbol) > 1:
        consolidate(per_symbol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
