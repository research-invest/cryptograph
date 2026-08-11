"""
Калибровка `STATES_SPLIT_GAIN_SIGMA` по СТАБИЛЬНОСТИ числа состояний.

Критерий калибровки — не «столько же состояний, сколько было». Прежний
абсолютный порог давал у ETH 29 → 26 → 42 состояния на трёх прогонах, и
воспроизводить это число незачем: оно само по себе было артефактом. Критерий —
устойчивость: одна и та же монета на окнах разной длины должна давать близкое
число состояний. Если не даёт, значит решения принимаются шумом у границы, и
любое сравнение «до/после» на такой калибровке ничего не значит.

    python3 scripts/calibrate_split_gain.py --all
    python3 scripts/calibrate_split_gain.py --symbol BTCUSDT --sigma 0.5 --sigma 1.0
    python3 scripts/calibrate_split_gain.py --all --start 2022-01-01   # быстрее, грубее

Что делает: для каждой монеты берёт три окна (полная история, минус месяц,
минус два месяца), для каждого значения `split_gain_sigma` обучает модель
состояний и печатает число состояний по окнам и их разброс.

Разброс считается как (max − min) / mean. Целевое значение — не больше 15%;
оно назначено, а не выведено из требований, и это стоит помнить при чтении
таблицы.

Второй раздел — контроль на шуме: то же число признаков, но SMC-блок
перемешан по строкам. Если после калибровки шум перестаёт обрушивать граф,
а настоящие признаки держат состояния на уровне базы, — вопрос о включении
источника можно ставить заново. Он же отвечает на вопрос, ради которого всё
затевалось: калибровка обязана перестать зависеть от размерности, иначе
каждый следующий источник — ончейн, индексы, деривативы — упрётся в ту же
стену.

Скрипт НИЧЕГО не пишет: ни в БД, ни в файлы. Числа переносятся в
development_log.md руками, вместе с выводом.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataclasses  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from btcproc import config, symbols  # noqa: E402
from btcproc.features import builder as feat  # noqa: E402
from btcproc.features import smc  # noqa: E402
from btcproc.ingest import binance  # noqa: E402
from btcproc.states import clustering  # noqa: E402

DEFAULT_SIGMAS = (0.5, 0.75, 1.0, 1.5, 2.0)

#: Целевой разброс числа состояний между окнами. Назначен, а не выведен.
TARGET_SPREAD = 0.15

#: Разумный коридор числа состояний. Верх — из соображения «граф должен
#: читаться человеком», низ — из «по состоянию должна набираться выборка».
SANE_RANGE = (15, 60)


def windows(base: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """
    Три окна разной длины: полная история, минус месяц, минус два.

    Отрезается ХВОСТ, а не начало: длина истории меняется, а её эпоха —
    нет. Отрезав начало, мы сравнивали бы 2017 год с 2019-м, то есть
    мерили бы смену режима рынка вместо устойчивости порога.
    """
    end = base.index[-1]
    return [
        ("полная", base),
        ("−1 мес", base[base.index <= end - pd.DateOffset(months=1)]),
        ("−2 мес", base[base.index <= end - pd.DateOffset(months=2)]),
    ]


def build(symbol: str, start: str | None, enabled: bool) -> pd.DataFrame:
    """Признаки монеты при заданном состоянии флага SMC."""
    saved = config.smc
    config.smc = config.SMCConfig(enabled=enabled, features_enabled=enabled)
    try:
        base = binance.load_ohlcv(symbol, config.data.base_tf, start, None)
        if base.empty:
            raise SystemExit(f"Нет баров по {symbol} — сначала ingest.")
        context = {
            tf: binance.load_ohlcv(symbol, tf, start, None)
            for tf in config.data.context_tfs
        }
        return feat.build_features(base, context)
    finally:
        config.smc = saved


def n_states(features: pd.DataFrame, symbol: str, sigma: float) -> tuple[int, float]:
    """Число состояний при данном пороге и время расчёта."""
    scale = feat.robust_scale_params(features)
    matrix = feat.apply_scale(features, scale)
    cfg = dataclasses.replace(symbols.get(symbol).states_config(),
                              split_gain_sigma=sigma)
    started = time.perf_counter()
    model, _ = clustering.fit_states(matrix, list(features.columns), scale, cfg=cfg)
    return model.n_groups, time.perf_counter() - started


def shuffle_smc(features: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Контроль: SMC-признаки перемешаны по строкам.

    Размерность та же, распределение каждого признака в точности то же, связь
    с рынком разрушена полностью. Единственный способ отличить «источник
    изменил структуру состояний» от «структура изменилась оттого, что
    признаков стало больше».
    """
    rng = np.random.default_rng(seed)
    control = features.copy()
    order = rng.permutation(len(control))
    for name in smc.FEATURE_CANDIDATES:
        if name in control.columns:
            control[name] = control[name].to_numpy()[order]
    return control


def sweep(symbol: str, start: str | None, sigmas: list[float]) -> None:
    print(f"\n{'=' * 78}\n=== {symbol}: устойчивость числа состояний\n{'=' * 78}")

    features = build(symbol, start, enabled=False)
    parts = windows(features)
    print(f"Признаков {features.shape[1]}, окна: "
          + ", ".join(f"{name} — {len(f)} строк" for name, f in parts))

    header = (f"\n{'sigma':>7}" + "".join(f"{name:>10}" for name, _ in parts)
              + f"{'разброс':>10} {'вердикт':>28}")
    print(header)
    print("─" * len(header))

    best: tuple[float, float] | None = None
    for sigma in sigmas:
        counts = []
        for _, part in parts:
            count, seconds = n_states(part, symbol, sigma)
            counts.append(count)
        mean = statistics.fmean(counts)
        spread = (max(counts) - min(counts)) / mean if mean else float("inf")
        sane = SANE_RANGE[0] <= mean <= SANE_RANGE[1]
        verdict = []
        if spread <= TARGET_SPREAD:
            verdict.append("устойчиво")
        else:
            verdict.append(f"разброс > {TARGET_SPREAD:.0%}")
        if not sane:
            verdict.append(f"вне {SANE_RANGE[0]}–{SANE_RANGE[1]}")
        print(f"{sigma:>7.2f}" + "".join(f"{c:>10d}" for c in counts)
              + f"{spread:>9.1%} {', '.join(verdict):>28}")
        if sane and (best is None or spread < best[1]):
            best = (sigma, spread)

    if best:
        print(f"\nЛучшее по устойчивости в разумном диапазоне: "
              f"sigma = {best[0]:.2f} (разброс {best[1]:.1%})")
    else:
        print("\nНи одно значение не дало числа состояний в разумном диапазоне — "
              "смотреть min_group_share, а не порог дробления.")


def control(symbol: str, start: str | None, sigma: float) -> None:
    """Повтор эксперимента «32 против 44 с перемешанным контролем»."""
    print(f"\n── {symbol}: размерность против информации (sigma = {sigma}) ──────")

    f_base = build(symbol, start, enabled=False)
    n_base, t_base = n_states(f_base, symbol, sigma)
    print(f"  база, {f_base.shape[1]} признаков:      {n_base:>3} состояний  ({t_base:.0f} c)")

    f_smc = build(symbol, start, enabled=True)
    n_smc, t_smc = n_states(f_smc, symbol, sigma)
    print(f"  с SMC, {f_smc.shape[1]} признаков:      {n_smc:>3} состояний  ({t_smc:.0f} c)")

    n_ctl, t_ctl = n_states(shuffle_smc(f_smc), symbol, sigma)
    print(f"  контроль (тот же шум):         {n_ctl:>3} состояний  ({t_ctl:.0f} c)")

    from_dimension = abs(n_ctl - n_base) / max(n_base, 1)
    from_source = abs(n_smc - n_ctl) / max(n_ctl, 1)
    print(f"\n  эффект размерности: {from_dimension:>5.0%}   "
          f"эффект источника сверх него: {from_source:>5.0%}")
    if from_dimension > TARGET_SPREAD:
        print("  ВЫВОД: калибровка всё ещё зависит от размерности — "
              "поднимать sigma или пересматривать критерий.")
    elif from_source < TARGET_SPREAD:
        print("  ВЫВОД: размерность больше не мешает, но и информации "
              "сверх шума источник не даёт.")
    else:
        print("  ВЫВОД: размерность обезврежена, источник даёт разделимость "
              "сверх контроля — вопрос о включении можно ставить заново.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Калибровка порога дробления")
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--start", help="Не вся история — быстрее, но грубее")
    parser.add_argument("--sigma", action="append", type=float,
                        help=f"Значения для свипа; по умолчанию {DEFAULT_SIGMAS}")
    parser.add_argument("--control", action="store_true",
                        help="Плюс контроль на перемешанных признаках")
    parser.add_argument("--control-sigma", type=float, default=None,
                        help="Порог для раздела контроля; по умолчанию из конфига")
    args = parser.parse_args()

    if args.all:
        targets = symbols.tickers(only_enabled=True)
    elif args.symbol:
        targets = args.symbol
    else:
        targets = [config.data.symbol]

    sigmas = args.sigma or list(DEFAULT_SIGMAS)
    for symbol in targets:
        sweep(symbol, args.start, sigmas)
        if args.control:
            control(symbol, args.start,
                    args.control_sigma or config.states.split_gain_sigma)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
