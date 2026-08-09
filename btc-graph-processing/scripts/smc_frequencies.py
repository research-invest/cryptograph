"""
Замер SMC-детекторов до интеграции: частоты, корреляции, стационарность.

Фаза 2 из `docs/task_smc_integration.md`. Детекторы уже написаны
(`btcproc/features/smc.py`), но никуда не подключены. Этот скрипт даёт числа,
по которым принимается решение о маршрутизации каждой концепции — в
signature-атомы, в context-атомы, в признаки или никуда.

    python3 scripts/smc_frequencies.py --symbol BTCUSDT
    python3 scripts/smc_frequencies.py --all
    python3 scripts/smc_frequencies.py --section swings   # только развёртка окна

Правила отбора, применяются механически (разделы 4 и 5.6 ТЗ):

* частота срабатывания **выше 3%** — в signature нельзя. Каждый signature-атом
  в худшем случае удваивает число блоков событий, а запас по выборке всего
  2.4-кратный: два лишних атома убивают историческую выборку насовсем;
* корреляция **выше 0.9** с существующим признаком — не нужен нигде, это
  дубликат под новым именем;
* условная вероятность **выше 0.8** относительно существующего атома — нового
  тот не несёт.

Разделы вывода:

  freq      частоты булевых детекторов и вердикт по бюджету signature
  years     частота по годам — проверка, что детектор не измеряет эпоху
  corr      максимальная корреляция с существующими атомами и признаками
  cond      условные вероятности для пар, которые заведомо пересекаются
  fvg       распределение размера разрыва в ATR по монетам (калибровка порога)
  swings    развёртка окна подтверждения свинга — открытый вопрос 1 ТЗ
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from btcproc import config, symbols  # noqa: E402
from btcproc.features import builder as feat  # noqa: E402
from btcproc.features import events as ev  # noqa: E402
from btcproc.features import smc  # noqa: E402
from btcproc.ingest import binance  # noqa: E402

SIGNATURE_MAX_SHARE = 0.03
DUPLICATE_CORRELATION = 0.9
DUPLICATE_CONDITIONAL = 0.8

# Пары «новый детектор → существующий атом», которые пересекаются по смыслу
# и потому проверяются отдельно (открытый вопрос 3 ТЗ).
CONDITIONAL_PAIRS = [
    ("sweep_high_reclaim", "breakout_1d_high"),
    ("sweep_low_reclaim", "breakdown_1d_low"),
    ("bos_up", "breakout_1d_high"),
    ("bos_down", "breakdown_1d_low"),
    ("in_premium", "at_range_high"),
    ("in_discount", "at_range_low"),
    ("structure_bullish", "trend_up_align"),
    ("structure_bearish", "trend_down_align"),
]


def load(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Бары, SMC-величины, существующие атомы и существующие признаки.

    SMC считается напрямую через `smc.build_smc`, а не через флаг: замер
    должен работать при любом значении `SMC_ENABLED`, в том числе при
    выключенном — иначе им нельзя было бы пользоваться до включения.
    """
    base = binance.load_ohlcv(symbol, config.data.base_tf, None, None)
    if base.empty:
        raise SystemExit(f"Нет баров по {symbol} — сначала ingest.")

    context = {
        tf: binance.load_ohlcv(symbol, tf, None, None)
        for tf in config.data.context_tfs
    }
    features = feat.build_features(base, context)
    atoms = ev.detect_atoms(base)
    values = smc.build_smc(base)

    # Общий индекс: признаки короче на прогрев окон.
    index = features.index
    return base.loc[index], values.loc[index], atoms.loc[index], features


def section_frequencies(values: pd.DataFrame) -> None:
    print("\n── Частоты булевых детекторов ─────────────────────────────────────")
    print(f"{'детектор':<24} {'доля баров':>11} {'роль по ТЗ':>12}  вердикт")
    print("─" * 78)
    for name in smc.BOOLEAN_COLUMNS:
        share = float(values[name].mean())
        role = "signature" if name in smc.SIGNATURE_CANDIDATES else "context"
        if role == "signature":
            verdict = ("проходит бюджет" if share <= SIGNATURE_MAX_SHARE
                       else f"СЛИШКОМ ЧАСТО (>{SIGNATURE_MAX_SHARE:.0%}) → только context")
        elif share > 0.95 or share < 0.005:
            verdict = "вырожден — почти константа"
        else:
            verdict = "ок"
        print(f"{name:<24} {share:>10.2%} {role:>12}  {verdict}")


def section_years(values: pd.DataFrame) -> None:
    """
    Частота по годам. Детектор, чья частота скачет в разы между эпохами,
    измеряет не рынок, а режим — и в кластеризацию его пускать нельзя.
    """
    print("\n── Частота по годам (стационарность) ──────────────────────────────")
    yearly = values[smc.BOOLEAN_COLUMNS].groupby(values.index.year).mean()
    years = list(yearly.index)
    header = f"{'детектор':<24}" + "".join(f"{y:>7}" for y in years) + f"{'размах':>9}"
    print(header)
    print("─" * len(header))
    for name in smc.BOOLEAN_COLUMNS:
        row = yearly[name]
        # Отношение максимума к минимуму: во сколько раз частота гуляет.
        low = max(row.min(), 1e-6)
        spread = row.max() / low
        flag = "  ⚠" if spread > 5 else ""
        print(f"{name:<24}" + "".join(f"{v:>7.1%}" for v in row)
              + f"{spread:>8.1f}×{flag}")


def _correlation(left: pd.Series, right: pd.Series) -> float:
    a = left.to_numpy(dtype=float)
    b = right.to_numpy(dtype=float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def section_correlations(values: pd.DataFrame, atoms: pd.DataFrame,
                         features: pd.DataFrame) -> None:
    """
    Максимальная корреляция каждого детектора с тем, что уже есть в системе.

    Дубликат под новым именем не добавляет информации, но добавляет стоимость:
    признак — переобучение всей модели состояний, атом — дробление блоков.
    """
    print("\n── Максимальная корреляция с существующими ────────────────────────")
    print(f"{'детектор':<28} {'ближайший существующий':<24} {'r':>7}  вердикт")
    print("─" * 78)

    # SMC теперь входит и в ATOM_FAMILY, и в вектор признаков (фаза 3).
    # Без этой чистки детектор нашёл бы ближайшим соседом самого себя с r = 1.0
    # и вся таблица превратилась бы в список дубликатов.
    own = set(smc.ALL_COLUMNS)
    existing = pd.concat([
        atoms[[c for c in atoms.columns if c not in own]].astype(float),
        features[[c for c in features.columns if c not in own]],
    ], axis=1)
    for name in smc.ALL_COLUMNS:
        series = values[name].astype(float)
        best_name, best_r = "—", 0.0
        for other in existing.columns:
            r = _correlation(series, existing[other])
            if abs(r) > abs(best_r):
                best_name, best_r = other, r
        verdict = ("ДУБЛИКАТ — не брать" if abs(best_r) > DUPLICATE_CORRELATION
                   else "самостоятелен")
        print(f"{name:<28} {best_name:<24} {best_r:>+7.3f}  {verdict}")


def section_conditionals(values: pd.DataFrame, atoms: pd.DataFrame) -> None:
    """
    P(существующий | новый) — не пересказывает ли новый детектор старый атом.

    Именно в эту сторону: если всякий раз, когда срабатывает новый, срабатывает
    и старый, — новый не несёт ничего сверх него.
    """
    print("\n── Условные вероятности с пересекающимися атомами ─────────────────")
    print(f"{'новый':<22} {'существующий':<22} {'P(стар|нов)':>12} {'n':>8}  вердикт")
    print("─" * 78)
    for new_name, old_name in CONDITIONAL_PAIRS:
        if old_name not in atoms.columns:
            continue
        mask = values[new_name].to_numpy(dtype=bool)
        n = int(mask.sum())
        if n == 0:
            print(f"{new_name:<22} {old_name:<22} {'—':>12} {0:>8}  не срабатывал")
            continue
        p = float(atoms.loc[mask, old_name].mean())
        verdict = ("пересказывает старый" if p > DUPLICATE_CONDITIONAL
                   else "несёт своё")
        print(f"{new_name:<22} {old_name:<22} {p:>12.3f} {n:>8}  {verdict}")


def section_fvg(base: pd.DataFrame, symbol: str) -> None:
    """
    Распределение размера разрыва в ATR — калибровка SMC_FVG_MIN_ATR.

    Открытый вопрос 5 ТЗ: порог 0.5 подобран на глаз, и у SOL с его
    волатильностью он может означать не то же, что у BTC.
    """
    from btcproc.features.smc import _safe_atr

    atr = pd.Series(_safe_atr(base), index=base.index)
    fvg = smc.detect_fvg(base, atr)
    sizes = pd.concat([fvg["bull_atr"].dropna(), fvg["bear_atr"].dropna()])
    if sizes.empty:
        print(f"\n{symbol}: разрывов не найдено")
        return

    quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]
    values = sizes.quantile(quantiles)
    print(f"\n{symbol}: разрывов {len(sizes)} ({len(sizes)/len(base):.2%} баров), "
          f"размер в ATR по перцентилям:")
    print("   " + "  ".join(f"p{int(q*100)}={v:.2f}" for q, v in zip(quantiles, values)))
    for threshold in (0.3, 0.5, 0.7, 1.0):
        share = float((sizes > threshold).sum()) / len(base)
        mark = " ← проходит бюджет signature" if share <= SIGNATURE_MAX_SHARE else ""
        print(f"   порог {threshold:.1f} ATR → {share:.2%} баров{mark}")


def section_swings(base: pd.DataFrame, symbol: str) -> None:
    """
    Развёртка окна подтверждения свинга — открытый вопрос 1 ТЗ.

    Базовый ТФ 15m, а SMC обычно применяют на 1h/4h. Вместо отдельного расчёта
    на старших барах меняем окно подтверждения: left=right=12 на 15-минутке
    даёт структуру примерно часового масштаба, 48 — четырёхчасового.
    """
    print(f"\n{symbol}: структура при разном окне подтверждения")
    print(f"{'left=right':>11} {'≈ТФ':>6} {'bos_up':>9} {'choch_up':>9} "
          f"{'сломов всего':>13} {'лаг, ч':>8}")
    print("─" * 62)
    per_hour = config.data.bars_per("1h")
    for width in (3, 6, 12, 24, 48):
        cfg = config.SMCConfig(swing_left=width, swing_right=width)
        values = smc.build_smc(base, cfg)
        breaks = (values["bos_up"] | values["bos_down"]
                  | values["choch_up"] | values["choch_down"])
        approx_tf = f"{width * config.data.base_minutes // 60}h" if width >= 4 else "15m"
        print(f"{width:>11} {approx_tf:>6} {values['bos_up'].mean():>8.2%} "
              f"{values['choch_up'].mean():>8.2%} {breaks.mean():>12.2%} "
              f"{width / per_hour:>8.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Замер SMC-детекторов")
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--section", action="append",
        choices=["freq", "years", "corr", "cond", "fvg", "swings"],
        help="Только эти разделы; по умолчанию все, кроме swings",
    )
    args = parser.parse_args()

    if args.all:
        targets = symbols.tickers(only_enabled=True)
    elif args.symbol:
        targets = args.symbol
    else:
        targets = [config.data.symbol]

    sections = set(args.section or ["freq", "years", "corr", "cond", "fvg"])

    for symbol in targets:
        print(f"\n{'=' * 78}\n=== {symbol}\n{'=' * 78}")
        base, values, atoms, features = load(symbol)
        print(f"Баров после прогрева: {len(values)} "
              f"({values.index[0].date()}…{values.index[-1].date()})")

        if "freq" in sections:
            section_frequencies(values)
        if "years" in sections:
            section_years(values)
        if "corr" in sections:
            section_correlations(values, atoms, features)
        if "cond" in sections:
            section_conditionals(values, atoms)
        if "fvg" in sections:
            section_fvg(base, symbol)
        if "swings" in sections:
            section_swings(base, symbol)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
