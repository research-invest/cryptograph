"""
D1: контрольная модель без графа состояний.

    python3 scripts/control_model.py --all
    python3 scripts/control_model.py --symbol BTCUSDT --seed 42 --seed 1337
    python3 scripts/control_model.py --all --folds 5 --importance

Задача 6 порядка работ ТЗ `crypto-graph/docs/tz_btcproc_audit_13-08-2026.md`.
Методика, интерпретация и **критерий, заявленный до запуска**, — в шапке
`btcproc/analysis/control.py`. Читать её обязательно: без критерия замер
превращается в разглядывание чисел.

Коротко, что происходит:

1. бары монеты до `--end` (та же замороженная граница 2026-08-01);
2. те же 32 признака тем же `features/builder.py` — со `shift(1)` для
   старших ТФ, то есть без заглядывания вперёд;
3. та же целевая `is_up` на горизонте 24h, только валидные метки;
4. то же разбиение 70/30, буквально тем же кодом (`holdout.split_bar`);
5. purged walk-forward CV на обучающей части, изотоническая калибровка на её
   хвосте, замер на отложенной части теми же метриками и тем же блочным
   бутстрапом, что и валидация графа.

**В БД ничего не пишется, в btc-graph ничего не уходит.** Модель состояний
здесь не нужна вовсе — в этом весь смысл контроля.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ДО импорта btcproc.config: конфиг читает окружение на импорте.
os.environ["SINK_MODE"] = "none"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from btcproc import config, symbols  # noqa: E402
from btcproc.analysis import control, holdout as ho  # noqa: E402
from btcproc.analysis.lift import DEFAULT_N_BOOT  # noqa: E402
from btcproc.candidates.outcomes import compute_outcomes  # noqa: E402
from btcproc.features import builder as feat  # noqa: E402
from btcproc.ingest import bars  # noqa: E402

FROZEN_END = "2026-08-01"


def prepare(symbol: str, end: str) -> tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
    """
    Признаки, целевая и индекс — только строки с валидной меткой исхода.

    Невалидные (хвост истории без полного горизонта и участки с дырами)
    режутся здесь, а не в метриках: там `is_up` был бы None, и молчаливое
    приведение к False означало бы «модель ошиблась» вместо «факта нет».
    """
    spec = symbols.get(symbol)
    start = spec.start_date()

    base = bars.load_ohlcv(symbol, config.data.base_tf, start, end)
    if base.empty:
        raise SystemExit(f"{symbol}: в БД нет баров до {end}. Сначала ingest.")
    context = {
        tf: bars.load_ohlcv(symbol, tf, start, end) for tf in config.data.context_tfs
    }

    print(f"[{symbol}] баров {len(base)}; признаки…")
    features = feat.build_features(base, context)
    outcomes = compute_outcomes(base).reindex(features.index)

    valid = outcomes["valid"].fillna(False).to_numpy(dtype=bool)
    valid &= features.notna().all(axis=1).to_numpy()
    features = features[valid]
    target = outcomes.loc[valid, "ret_pct"].to_numpy(dtype=float) > 0

    print(f"[{symbol}] пригодных строк {len(features)} "
          f"({features.shape[1]} признаков), доля «вверх» {target.mean():.4f}")
    return features, target, features.index


def run_symbol(symbol: str, args: argparse.Namespace) -> list[control.ControlReport]:
    features, target, index = prepare(symbol, args.end)

    try:
        split_ts = ho.split_bar(index, args.train_frac)
    except ValueError as exc:
        raise SystemExit(f"{symbol}: {exc}.") from exc

    is_train = index < split_ts
    n_train = int(is_train.sum())
    gap = config.data.horizon_bars

    train_features = features[is_train]
    train_target = target[is_train]
    # Зазор и на границе train/holdout: последние `gap` строк обучения имеют
    # окна исходов, накрывающие начало отложенной части.
    fit_slice = slice(0, max(1, n_train - gap))
    test_slice = slice(n_train, len(features))

    print(f"[{symbol}] граница {split_ts:%Y-%m-%d %H:%M}: обучение {n_train}, "
          f"holdout {len(features) - n_train}, зазор {gap} баров")

    reports = []
    for seed in args.seed:
        print(f"[{symbol}] зерно {seed}: purged walk-forward "
              f"({args.folds} фолдов)…")
        folds = control.cross_validate(
            train_features.reset_index(drop=True), train_target, gap, args.folds, seed
        )
        for fold in folds:
            print(f"    фолд {fold.index}: accuracy {fold.accuracy:.4f}, "
                  f"AUC {fold.auc:.4f}, skill {fold.brier_skill:+.4f}")

        print(f"[{symbol}] зерно {seed}: обучение на всей train-части "
              f"и замер на holdout…")
        raw, predicted = control.fit_predict(
            features.reset_index(drop=True), target, fit_slice, test_slice,
            seed, gap=gap,
        )
        report = control.measure(
            ts=pd.Series(index[test_slice]),
            raw=raw, predicted=predicted,
            actual=target[test_slice],
            symbol=symbol, seed=seed, split_ts=split_ts,
            horizon_minutes=config.data.horizon_minutes,
            n_features=features.shape[1], n_train=n_train,
            n_boot=args.n_boot,
        )
        report.folds = folds
        if args.importance:
            print(f"[{symbol}] зерно {seed}: важность признаков…")
            report.feature_importance = control.permutation_importance(
                features.reset_index(drop=True), target, fit_slice, test_slice, seed
            )
        print()
        print(control.format_report(report))
        reports.append(report)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--end", default=FROZEN_END)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--folds", type=int, default=control.DEFAULT_FOLDS)
    parser.add_argument(
        "--seed", type=int, action="append",
        help="зерно бустинга; указывать дважды — инвариант 10 требует "
             "воспроизведения на двух зёрнах (по умолчанию 42 и 1337)",
    )
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--importance", action="store_true",
                        help="посчитать важность признаков перемешиванием")
    args = parser.parse_args()
    args.seed = args.seed or [42, 1337]

    specs = symbols.resolve_many(args.symbol, args.all)
    print(f"D1: контрольная модель без графа. Монет {len(specs)}, "
          f"зёрна {args.seed}, граница {args.end}.")
    print("Критерий заявлен ДО запуска — см. шапку btcproc/analysis/control.py.\n")

    reports: list[control.ControlReport] = []
    for spec in specs:
        try:
            reports.extend(run_symbol(spec.ticker, args))
        except SystemExit as exc:  # noqa: BLE001 — одна монета не роняет замер
            print(f"[{spec.ticker}] пропущена: {exc}")
    print(control.format_verdict(reports))


if __name__ == "__main__":
    main()
