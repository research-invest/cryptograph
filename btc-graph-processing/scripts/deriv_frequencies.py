"""
Замер деривативных метрик Binance USD-M до включения флагов: частоты,
корреляции, новизна, зеркала, эпохи, автокорреляция, доля дыр.

Методика — docs/tz_deriv_ingest_14-08-26.md, раздел 5. По образцу
fgi_frequencies.py (сам он — по образцу smc_frequencies.py), плюс разделы,
специфичные для этого источника:

  b4        B4 ТЗ: taker_z — потенциальный дубликат taker_buy_dominance /
            taker_sell_dominance (контекст) и taker_bias (признак),
            заведённых из СПОТОВОГО потока. Рынки разные (спот/перп),
            поэтому проверка обязательна, а не гарантирован вывод;
  novelty   гейт N — out-of-sample R² регрессии КАЖДОЙ из 6 непрерывных
            величин B2 на 32 базовых признака (не парная корреляция —
            источник, наполовину пересобранный из уже имеющегося, парный
            порог проходит легко, а множественная R² — нет);
  autocorr  длина автокорреляции каждой непрерывной величины В БАРАХ
            (разрешение 5 минут, а не сутки — ожидается на порядки короче,
            чем у FGI, и мощность гейта R выше);
  missing   доля баров без метрик (src_rows пуст) по эпохам — иначе «атом не
            сработал» и «данных ещё нет» сливаются в одно число.

Эпохи те же, что у FGI (заданы ДО замера, не двигать, увидев числа):

    2018-02…2020-12   ранний рынок
    2021-01…2022-12   цикл роста и обвал (LUNA/FTX)
    2023-01…2026-08   после обвала, ETF-эпоха

Метрики начинаются позже эпох (BTC — 2020-09, остальные — 2021-12 или позже,
HYPE — 2025-05), поэтому первая эпоха у всех монет почти или полностью
пуста — это проговаривается в выводе, а не молчаливо даёт три строки, одна
из которых на нескольких месяцах данных.

HYPE — двойное смешение площадок (бары Bybit, метрики Binance USD-M) и
держится отдельной строкой в выводах, не смешивается с BTC/ETH/SOL при
подведении итога (§1.2 ТЗ).

    python3 scripts/deriv_frequencies.py --symbol BTCUSDT
    python3 scripts/deriv_frequencies.py --all
    python3 scripts/deriv_frequencies.py --section novelty --section autocorr
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
from btcproc.features import deriv  # noqa: E402
from btcproc.features import events as ev  # noqa: E402
from btcproc.ingest import bars, metrics as metrics_ingest  # noqa: E402

DUPLICATE_CORRELATION = 0.9
MIRROR_CORRELATION = 0.5
#: Гейт N — порог из §2 ТЗ.
NOVELTY_GATE_R2 = 0.80
#: Коридор частоты гейта F.
FREQ_MIN, FREQ_MAX = 0.03, 0.40
#: Автокорреляция величины считается «умершей» ниже этого порога.
AUTOCORR_FLOOR = 0.2
#: Потолок поиска в БАРАХ (5m-величины должны затухать быстро — не сутки).
AUTOCORR_MAX_LAG_BARS = 5000

EPOCHS = [
    ("2018-02..2020-12", "2018-02-01", "2020-12-31"),
    ("2021-01..2022-12", "2021-01-01", "2022-12-31"),
    ("2023-01..2026-08", "2023-01-01", None),
]


def load(symbol: str):
    """
    Бары, deriv_metrics монеты, деривативные величины (напрямую, минуя флаг —
    как у SMC/FGI: замер обязан работать при любом состоянии DERIV_ENABLED),
    существующие атомы и признаки (под ТЕКУЩИМИ флагами конфигурации).
    """
    base = bars.load_ohlcv(symbol, config.data.base_tf, None, None)
    if base.empty:
        raise SystemExit(f"Нет баров по {symbol} — сначала ingest.")
    metrics_frame = metrics_ingest.load_deriv_metrics(symbol, config.data.base_tf)
    if metrics_frame.empty:
        raise SystemExit(f"deriv_metrics пуст по {symbol} — сначала ingest-metrics.")

    context = {tf: bars.load_ohlcv(symbol, tf, None, None) for tf in config.data.context_tfs}
    features = feat.build_features(base, context, symbol=symbol)
    atoms = ev.detect_atoms(base, symbol)
    values = deriv.build_deriv(base, metrics_frame, symbol)

    index = features.index
    return base.loc[index], metrics_frame, values.loc[index], atoms.loc[index], features


def section_frequencies(values: pd.DataFrame, symbol: str) -> None:
    print("\n── Частоты атомов (все контекстные квадранты oi_vs_price) ─────────")
    print(f"{'детектор':<24} {'доля баров':>11}  вердикт")
    print("─" * 60)
    for name in deriv.CONTEXT_CANDIDATES:
        share = float(values[name].mean())
        if share > FREQ_MAX:
            verdict = f"выше коридора F ({FREQ_MAX:.0%})"
        elif share < FREQ_MIN:
            verdict = f"ниже бюджета signature ({FREQ_MIN:.0%}) — редкий"
        else:
            verdict = "в коридоре гейта F"
        print(f"{name:<24} {share:>10.2%}  {verdict}")
    total_active = values[deriv.CONTEXT_CANDIDATES].sum(axis=1)
    print(f"Барах хотя бы с одним квадрантом: {(total_active > 0).mean():.2%} "
          f"(взаимно исключающие — не должно превышать 100% минус доля NaN)")


def section_years(values: pd.DataFrame) -> None:
    print("\n── Частота атомов по годам (стационарность) ────────────────────────")
    yearly = values[deriv.CONTEXT_CANDIDATES].groupby(values.index.year).mean()
    years = list(yearly.index)
    header = f"{'детектор':<24}" + "".join(f"{y:>7}" for y in years) + f"{'размах':>9}"
    print(header)
    print("─" * len(header))
    for name in deriv.CONTEXT_CANDIDATES:
        row = yearly[name]
        low = max(row[row > 0].min() if (row > 0).any() else 1e-6, 1e-6)
        spread = row.max() / low
        flag = "  ⚠" if spread > 5 else ""
        print(f"{name:<24}" + "".join(f"{v:>7.1%}" for v in row) + f"{spread:>8.1f}×{flag}")


def section_epochs(values: pd.DataFrame, symbol: str) -> None:
    print("\n── Три фиксированные эпохи (устойчивость, не монеты — §1.1 задачи FGI) ──")
    print(f"{'эпоха':<20} {'баров':>10} {'доля метрик есть':>18}")
    print("─" * 52)
    for label, start, end in EPOCHS:
        window = values[values.index >= pd.Timestamp(start, tz="UTC")]
        if end:
            window = window[window.index <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
        has_data = window["oi_rank"].notna().mean() if len(window) else float("nan")
        note = ""
        if len(window) == 0:
            note = "  ⚠ ПУСТО — метрик в эту эпоху ещё не существует"
        elif has_data < 0.05:
            note = "  ⚠ почти нет данных — эпоха у этого источника не показательна"
        print(f"{label:<20} {len(window):>10} {has_data if len(window) else 0:>17.1%}{note}")


def _correlation(left: pd.Series, right: pd.Series) -> float:
    a, b = left.to_numpy(dtype=float), right.to_numpy(dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 20 or a[ok].std() == 0 or b[ok].std() == 0:
        return 0.0
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def section_correlations(values: pd.DataFrame, atoms: pd.DataFrame, features: pd.DataFrame) -> None:
    print("\n── Максимальная парная корреляция с существующими ─────────────────")
    print(f"{'детектор':<24} {'ближайший существующий':<24} {'r':>7}  вердикт")
    print("─" * 78)

    own = set(deriv.ALL_COLUMNS)
    existing = pd.concat([
        atoms[[c for c in atoms.columns if c not in own]].astype(float),
        features[[c for c in features.columns if c not in own]],
    ], axis=1)
    for name in deriv.ALL_COLUMNS:
        series = values[name].astype(float)
        best_name, best_r = "—", 0.0
        for other in existing.columns:
            r = _correlation(series, existing[other])
            if abs(r) > abs(best_r):
                best_name, best_r = other, r
        verdict = "ДУБЛИКАТ — не брать" if abs(best_r) > DUPLICATE_CORRELATION else "самостоятелен"
        print(f"{name:<24} {best_name:<24} {best_r:>+7.3f}  {verdict}")


def section_b4(values: pd.DataFrame, atoms: pd.DataFrame, features: pd.DataFrame) -> None:
    """
    B4 ТЗ: taker_z (перп, деривативы) против taker_buy_dominance/
    taker_sell_dominance (контекст, СПОТ) и taker_bias (признак, СПОТ).

    taker_z непрерывен, а не булев — «условная вероятность» в булевом смысле
    к нему неприменима буквально, поэтому обе стороны сравниваются
    корреляцией (естественное обобщение: булев атом как {0,1} коррелирует
    с непрерывной величиной ровно так же корректно, как булев с булевым).
    Порог тот же — |r| > 0.9 = дубликат под новым именем.
    """
    print("\n── B4: taker_z — потенциальный дубликат спотового потока тейкеров ──")
    print(f"{'пара':<45} {'r':>7}  вердикт")
    print("─" * 68)
    candidates = []
    if "taker_buy_dominance" in atoms.columns:
        candidates.append(("taker_z ~ taker_buy_dominance (атом, спот)",
                           atoms["taker_buy_dominance"].astype(float)))
    if "taker_sell_dominance" in atoms.columns:
        candidates.append(("taker_z ~ taker_sell_dominance (атом, спот)",
                           atoms["taker_sell_dominance"].astype(float)))
    if "taker_bias" in features.columns:
        candidates.append(("taker_z ~ taker_bias (признак, спот)",
                           features["taker_bias"].astype(float)))
    for label, other in candidates:
        r = _correlation(values["taker_z"], other)
        verdict = "ДУБЛИКАТ — рынки разные, но пересказывает" if abs(r) > DUPLICATE_CORRELATION else "самостоятелен от спотового потока"
        print(f"{label:<45} {r:>+7.3f}  {verdict}")
    if not candidates:
        print("Существующие спотовые величины не найдены в векторе — сверка невозможна.")


def section_mirror(values: pd.DataFrame) -> None:
    print("\n── Зеркальные пары внутри набора деривативов ───────────────────────")
    print(f"{'детектор A':<24} {'детектор B':<24} {'r':>7}  вердикт")
    print("─" * 70)
    columns = [c for c in deriv.ALL_COLUMNS if values[c].astype(float).std(skipna=True) > 0]
    found = False
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            r = _correlation(values[left].astype(float), values[right].astype(float))
            if abs(r) < MIRROR_CORRELATION:
                continue
            found = True
            verdict = "зеркало — считать ОДНОЙ гипотезой" if r < 0 else "дубликат — считать ОДНОЙ гипотезой"
            print(f"{left:<24} {right:<24} {r:>+7.3f}  {verdict}")
    if not found:
        print(f"Пар с |r| ≥ {MIRROR_CORRELATION} нет.")


def section_novelty(values: pd.DataFrame, features: pd.DataFrame, symbol: str) -> dict[str, float]:
    """Гейт N на КАЖДУЮ из 6 непрерывных величин B2 (§5 ТЗ)."""
    from btcproc.analysis.novelty import novelty_r2

    print("\n── Гейт N: новизна (out-of-sample R² на 32 базовых признака) ──────")
    results = {}
    for name in deriv.FEATURE_CANDIDATES:
        series = values[name]
        if series.notna().sum() < 200:
            print(f"{name}: меньше 200 непустых наблюдений — R² не считается")
            continue
        r2, _model, top, _cols, idx, cut = novelty_r2(series, features)
        results[name] = r2
        verdict = "ГЕЙТ N НЕ ПРОЙДЕН" if r2 >= NOVELTY_GATE_R2 else "пройден"
        print(f"{name:<16} R²={r2:.4f} (n={len(idx)}, train {cut}) — {verdict}")
        top_str = ", ".join(f"{n}={v:.3f}" for n, v in top[:3])
        print(f"    топ-3 по вкладу: {top_str}")
    return results


def section_autocorr(values: pd.DataFrame, symbol: str) -> dict[str, int]:
    """
    Длина автокорреляции КАЖДОЙ непрерывной величины в БАРАХ (§2.3 ТЗ):
    `autocorr_block_rows` работает с любым рядом, `bars_per_day=1` на уже
    барном ряде переводит лаг напрямую в бары (а не в сутки, как у FGI).
    """
    from btcproc.analysis.range_lift import autocorr_block_rows

    print("\n── Автокорреляция предиктора в БАРАХ (параметр блока бутстрапа) ───")
    result = {}
    for name in deriv.FEATURE_CANDIDATES:
        series = values[name].dropna()
        if len(series) < 200:
            print(f"{name}: меньше 200 наблюдений — пропуск")
            continue
        lag_bars, block_rows = autocorr_block_rows(
            series, bars_per_day=1, floor=AUTOCORR_FLOOR, max_lag_days=AUTOCORR_MAX_LAG_BARS
        )
        result[name] = block_rows
        flag = " ⚠ потолок" if lag_bars == AUTOCORR_MAX_LAG_BARS else ""
        print(f"{name:<16} автокорреляция < {AUTOCORR_FLOOR} на лаге {lag_bars} баров{flag}")
    return result


def section_missing(metrics_frame: pd.DataFrame, values: pd.DataFrame) -> None:
    """Доля баров без метрик по эпохам — отличает «атом не сработал» от «данных ещё нет»."""
    print("\n── Доля баров БЕЗ метрик по эпохам (src_rows пуст) ─────────────────")
    print(f"{'эпоха':<20} {'баров':>10} {'без метрик':>12}")
    print("─" * 44)
    aligned = metrics_frame.reindex(values.index)
    for label, start, end in EPOCHS:
        mask = (values.index >= pd.Timestamp(start, tz="UTC"))
        if end:
            mask &= (values.index <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))
        window = aligned.loc[mask]
        if len(window) == 0:
            print(f"{label:<20} {0:>10} {'—':>12}")
            continue
        missing_share = window["src_rows"].isna().mean()
        print(f"{label:<20} {len(window):>10} {missing_share:>11.1%}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Замер деривативных метрик Binance USD-M")
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--section", action="append",
        choices=["freq", "years", "epochs", "corr", "b4", "mirror", "novelty",
                 "autocorr", "missing"],
    )
    args = parser.parse_args()

    if args.all:
        targets = symbols.tickers(only_enabled=True)
    elif args.symbol:
        targets = args.symbol
    else:
        targets = [config.data.symbol]

    sections = set(args.section or [
        "freq", "years", "epochs", "corr", "b4", "mirror", "novelty",
        "autocorr", "missing",
    ])

    for symbol in targets:
        print(f"\n{'=' * 78}\n=== {symbol}" + (" (ДВОЙНОЕ смешение площадок — держать отдельно от BTC/ETH/SOL)"
              if symbols.get(symbol).venue != "binance_spot" else "") + f"\n{'=' * 78}")
        base, metrics_frame, values, atoms, features = load(symbol)
        print(f"Баров после прогрева: {len(values)} ({values.index[0].date()}…{values.index[-1].date()})")
        print(f"deriv_metrics: {len(metrics_frame)} строк "
              f"({metrics_frame.index[0].date()}…{metrics_frame.index[-1].date()}), "
              f"metrics_start монеты: {symbols.get(symbol).metrics_start_date()}")

        if "freq" in sections:
            section_frequencies(values, symbol)
        if "years" in sections:
            section_years(values)
        if "epochs" in sections:
            section_epochs(values, symbol)
        if "corr" in sections:
            section_correlations(values, atoms, features)
        if "b4" in sections:
            section_b4(values, atoms, features)
        if "mirror" in sections:
            section_mirror(values)
        if "novelty" in sections:
            section_novelty(values, features, symbol)
        if "autocorr" in sections:
            section_autocorr(values, symbol)
        if "missing" in sections:
            section_missing(metrics_frame, values)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
