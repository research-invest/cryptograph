"""
Замер Fear & Greed Index до включения флагов: частоты, корреляции, новизна.

Методика — `docs/task_fear_greed.md`, разделы 1 и 5.1. Три раздела, которых
нет у образца (`smc_frequencies.py`), и они обязательны именно для этого
источника:

  novelty   гейт N (§1.2) — out-of-sample R² регрессии fgi_level на 32
            БАЗОВЫХ признака (v1, без SMC и без него самого). Штатная
            проверка «макс. парная корреляция > 0.9» здесь ничего не поймает:
            индекс наполовину состоит из волатильности и моментума BTC,
            парные r будут 0.6–0.7, а МНОЖЕСТВЕННАЯ R² может быть 0.85+.
  autocorr  длина автокорреляции ДНЕВНОГО ряда индекса в БАЗОВЫХ барах —
            параметр идёт дальше во все блочные бутстрапы (§1.3): наивный
            блок по горизонту исхода занижен, потому что зависимость здесь
            идёт ещё и по предиктору, а он меняется раз в сутки.

Раздел `mirror` здесь мельче, чем у SMC (всего 4 атома), но правило то же:
пары с |r| ≥ 0.5 считаются в поправку на множественные сравнения ОДНОЙ
гипотезой (`sentiment_flip_up`/`sentiment_flip_down` по построению не могут
сработать в один день одновременно — ожидаемо сильная антикорреляция).

    python3 scripts/fgi_frequencies.py --symbol BTCUSDT
    python3 scripts/fgi_frequencies.py --all
    python3 scripts/fgi_frequencies.py --section novelty --section autocorr
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
from btcproc.features import fear_greed as fg  # noqa: E402
from btcproc.ingest import bars, external  # noqa: E402

DUPLICATE_CORRELATION = 0.9
MIRROR_CORRELATION = 0.5
#: Гейт N — порог из docs/task_fear_greed.md §1.2.
NOVELTY_GATE_R2 = 0.80
#: Автокорреляция дневного ряда считается «умершей» ниже этого порога.
AUTOCORR_FLOOR = 0.2
#: Дальше скольких дней автокорреляцию не ищем — поиск обязан где-то
#: остановиться. ТЗ ожидало 5–15 дней; по факту ряд куда более персистентный
#: (реальный первый переход ниже порога — около 150 дней, см. журнал), поэтому
#: потолок взят с большим запасом, а не по первоначальной оценке.
AUTOCORR_MAX_LAG_DAYS = 400


def load(symbol: str):
    """
    Бары, дневной ряд FGI, величины FGI (напрямую, минуя флаг — как и у
    SMC-образца: замер обязан работать при любом состоянии FGI_ENABLED),
    существующие атомы и признаки (под ТЕКУЩИМИ флагами конфигурации).
    """
    base = bars.load_ohlcv(symbol, config.data.base_tf, None, None)
    if base.empty:
        raise SystemExit(f"Нет баров по {symbol} — сначала ingest.")
    daily = external.load_external_daily(external.FEAR_GREED_SERIES)
    if daily.empty:
        raise SystemExit("external_daily пуст — сначала `fetch-external`.")

    context = {tf: bars.load_ohlcv(symbol, tf, None, None) for tf in config.data.context_tfs}
    features = feat.build_features(base, context)
    atoms = ev.detect_atoms(base)
    values = fg.build_fear_greed(base, daily)

    index = features.index
    return base.loc[index], daily, values.loc[index], atoms.loc[index], features


def section_frequencies(values: pd.DataFrame) -> None:
    print("\n── Частоты атомов (все контекстные, бюджет signature не применяется) ──")
    print(f"{'детектор':<24} {'доля баров':>11}  вердикт")
    print("─" * 60)
    for name in fg.CONTEXT_CANDIDATES:
        share = float(values[name].mean())
        # Ожидание из ТЗ §4.1: extreme 10–25%, flip 3–6% ДНЕЙ (не баров — на
        # 15m это в 96 раз меньше доли баров при том же числе дней).
        verdict = "вырожден — почти константа" if share > 0.95 or share < 0.001 else "ок"
        print(f"{name:<24} {share:>10.2%}  {verdict}")


def section_years(values: pd.DataFrame) -> None:
    print("\n── Частота по годам (стационарность) ──────────────────────────────")
    yearly = values[fg.CONTEXT_CANDIDATES].groupby(values.index.year).mean()
    years = list(yearly.index)
    header = f"{'детектор':<24}" + "".join(f"{y:>7}" for y in years) + f"{'размах':>9}"
    print(header)
    print("─" * len(header))
    for name in fg.CONTEXT_CANDIDATES:
        row = yearly[name]
        low = max(row.min(), 1e-6)
        spread = row.max() / low
        flag = "  ⚠" if spread > 5 else ""
        print(f"{name:<24}" + "".join(f"{v:>7.1%}" for v in row) + f"{spread:>8.1f}×{flag}")


def _correlation(left: pd.Series, right: pd.Series) -> float:
    a, b = left.to_numpy(dtype=float), right.to_numpy(dtype=float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def section_correlations(values: pd.DataFrame, atoms: pd.DataFrame, features: pd.DataFrame) -> None:
    print("\n── Максимальная парная корреляция с существующими ─────────────────")
    print("Штатный порог 0.9 здесь ожидаемо ничего не поймает (см. novelty ниже) —")
    print("индекс наполовину состоит из тех же величин, но множественно, не парно.")
    print(f"{'детектор':<28} {'ближайший существующий':<24} {'r':>7}  вердикт")
    print("─" * 78)

    own = set(fg.ALL_COLUMNS)
    existing = pd.concat([
        atoms[[c for c in atoms.columns if c not in own]].astype(float),
        features[[c for c in features.columns if c not in own]],
    ], axis=1)
    for name in fg.ALL_COLUMNS:
        series = values[name].astype(float)
        best_name, best_r = "—", 0.0
        for other in existing.columns:
            r = _correlation(series, existing[other])
            if abs(r) > abs(best_r):
                best_name, best_r = other, r
        verdict = "ДУБЛИКАТ — не брать" if abs(best_r) > DUPLICATE_CORRELATION else "самостоятелен"
        print(f"{name:<28} {best_name:<24} {best_r:>+7.3f}  {verdict}")


def section_mirror(values: pd.DataFrame) -> None:
    print("\n── Зеркальные пары внутри набора FGI ───────────────────────────────")
    print(f"{'детектор A':<24} {'детектор B':<24} {'r':>7}  вердикт")
    print("─" * 70)
    booleans = [c for c in fg.CONTEXT_CANDIDATES if values[c].astype(float).std() > 0]
    found = False
    for i, left in enumerate(booleans):
        for right in booleans[i + 1:]:
            r = _correlation(values[left].astype(float), values[right].astype(float))
            if abs(r) < MIRROR_CORRELATION:
                continue
            found = True
            verdict = "зеркало — считать ОДНОЙ гипотезой" if r < 0 else "дубликат — считать ОДНОЙ гипотезой"
            print(f"{left:<24} {right:<24} {r:>+7.3f}  {verdict}")
    if not found:
        print(f"Пар с |r| ≥ {MIRROR_CORRELATION} нет.")


def section_novelty(values: pd.DataFrame, features: pd.DataFrame, symbol: str) -> float:
    """
    Гейт N. OOS R² регрессии fgi_level на 32 БАЗОВЫХ признака (v1), обучение
    на первых 70% истории по времени, проверка на последних 30% — тот же
    сплит, что и в measure_lift/measure_gradation, ради сопоставимости.
    """
    from btcproc.analysis.novelty import novelty_r2

    print("\n── Гейт N: новизна (out-of-sample R² на 32 базовых признака) ──────")
    # Базовые признаки — всё, что не пришло из зарегистрированных источников
    # (SMC, FGI и т.п.): ТЗ требует регрессию именно на них (v1, 32 штуки),
    # а не на весь вектор с уже включённым SMC.
    r2, _model, top, base_columns, idx, cut = novelty_r2(values["fgi_level"], features)
    n = len(idx)

    print(f"{symbol}: обучение {idx[0].date()}…{idx[cut - 1].date()} ({cut} строк), "
          f"проверка {idx[cut].date()}…{idx[-1].date()} ({n - cut} строк)")
    print(f"Out-of-sample R² = {r2:.4f} (порог гейта: < {NOVELTY_GATE_R2})")
    print("Топ-5 признаков по вкладу:")
    for name, value in top:
        print(f"   {name:<20} {value:.4f}")
    if r2 >= NOVELTY_GATE_R2:
        print("ГЕЙТ N НЕ ПРОЙДЕН: индекс пересказывает уже имеющиеся признаки, "
              "источник закрывается здесь.")
    else:
        print("Гейт N пройден — идём дальше." + (
            " R² в диапазоне 0.5–0.8: дублировать все замеры на fgi_residual."
            if r2 >= 0.5 else ""
        ))
    return r2


def section_autocorr(daily: pd.DataFrame, symbol: str) -> int:
    """
    Гейт-параметр §1.3: первый лаг (в ДНЯХ), на котором автокорреляция
    дневного ряда падает ниже AUTOCORR_FLOOR, переведённый в БАЗОВЫЕ бары.
    Идёт дальше как block_rows во все блочные бутстрапы этого источника.
    """
    from btcproc.analysis.range_lift import autocorr_block_rows

    print("\n── Длина автокорреляции дневного ряда (параметр блока бутстрапа) ──")
    bars_per_day = config.data.bars_per("1d")
    lag_days, block_rows_by_autocorr = autocorr_block_rows(
        daily["value"].astype(float), bars_per_day, AUTOCORR_FLOOR, AUTOCORR_MAX_LAG_DAYS
    )
    if lag_days == AUTOCORR_MAX_LAG_DAYS:
        print(f"⚠ автокорреляция не упала ниже {AUTOCORR_FLOOR} за {AUTOCORR_MAX_LAG_DAYS} дней — "
              f"беру потолок.")
    print(f"{symbol}: автокорреляция дневного ряда < {AUTOCORR_FLOOR} на лаге {lag_days} дней "
          f"({block_rows_by_autocorr} баров при {config.data.base_tf})")
    print("Это block_rows для --block-rows в measure_atom_lift.py и measure_fgi_range.py.")
    return block_rows_by_autocorr


def main() -> int:
    parser = argparse.ArgumentParser(description="Замер Fear & Greed Index")
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--section", action="append",
        choices=["freq", "years", "corr", "mirror", "novelty", "autocorr"],
    )
    args = parser.parse_args()

    if args.all:
        targets = symbols.tickers(only_enabled=True)
    elif args.symbol:
        targets = args.symbol
    else:
        targets = [config.data.symbol]

    sections = set(args.section or ["freq", "years", "corr", "mirror", "novelty", "autocorr"])

    for symbol in targets:
        print(f"\n{'=' * 78}\n=== {symbol}\n{'=' * 78}")
        base, daily, values, atoms, features = load(symbol)
        print(f"Баров после прогрева: {len(values)} ({values.index[0].date()}…{values.index[-1].date()})")
        print(f"Суток в external_daily: {len(daily)} ({daily.index[0].date()}…{daily.index[-1].date()})")

        if "freq" in sections:
            section_frequencies(values)
        if "years" in sections:
            section_years(values)
        if "corr" in sections:
            section_correlations(values, atoms, features)
        if "mirror" in sections:
            section_mirror(values)
        if "novelty" in sections:
            section_novelty(values, features, symbol)
        if "autocorr" in sections:
            section_autocorr(daily, symbol)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
