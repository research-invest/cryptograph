"""
Валидация системы на отложенной части истории — Ш0 плана 2026-08-12.

Единственный вопрос: **реализуется ли исторический перекос на данных,
которых модель не видела.** До сих пор система целиком не проверялась ни
разу («walk-forward валидации графа нет» стоит в README с первой версии),
и все замеры отдельных атомов делались внутри механизма, про который
неизвестно, воспроизводит ли он своё основное утверждение.

    python3 scripts/validate_holdout.py --symbol BTCUSDT
    python3 scripts/validate_holdout.py --all
    python3 scripts/validate_holdout.py --all --random-state 1337   # вторая модель
    python3 scripts/validate_holdout.py --symbol BTCUSDT --model-run 31 --no-train
    python3 scripts/validate_holdout.py --symbol BTCUSDT --from-dump out/btc.csv

**Замороженный датасет.** `--end` по умолчанию 2026-08-01 (раздел 0 плана).
Расхождение «локально 51 состояние, на сервере 40» оказалось разницей в два
дня данных; пока сравнительные прогоны не ходят на одну и ту же границу,
любые две цифры несравнимы. Менять `--end` можно, но тогда и сравнивать
результат не с чем.

## Что делает

1. Обучает модель состояний на первых `--train-frac` истории
   (`kind='holdout'`, не `train`: иначе ближайший `live` подхватил бы
   модель, обученную на обрезанной истории, — `latest_completed_run` ищет
   именно `train`).
2. Размечает этой моделью ВСЮ историю до `--end` — механикой `live`, без
   переобучения.
3. Собирает кандидатов по всей размеченной истории. Накопители выборки
   идут хронологически, поэтому кандидат holdout'а видит обучающую часть
   как прошлое, и только её.
4. Оценивает каждого скорером btc-graph (профиль монеты) — ради разреза по
   `rating`.
5. Меряет на holdout: калибровка, Brier, directional accuracy, разрезы по
   rating / sample_size / research_score. Методика — `analysis/holdout.py`,
   значимость всюду блочным бутстрапом (инвариант 11).

## Три вещи, которые здесь сделаны специально

**Карты редкости считаются только по обучающей части.** `transition_rarity`
и статистика блоков в штатном прогоне считаются по ВСЕЙ истории — это
незакрытый look-ahead (Ш2.1 плана). Внутри holdout-валидации оставить его
было нельзя: кандидат 2025 года знал бы, насколько редким его переход
окажется по итогам 2026-го, и проверка перестала бы быть проверкой.
Блок, впервые встретившийся на holdout, карты не имеет — он получает
`rarity='common'` и нулевую долю. Это не изъян, а честное «на тот момент мы
о нём ничего не знали».

**Скорер вызывается напрямую, без записи.** btc-graph импортируется как
пакет (тот же приём, что в `sink/graph_sink.py`), но дальше используются
только чистые функции `score_candidate` и `get_rating`. Ни PostgreSQL, ни
Neo4j, ни Redis не трогаются: валидация ничего не публикует. Дедуп по
family_key и порог `min_quality_score` тоже не применяются — вопрос
«различает ли рейтинг» задаётся ко всей выдаче, а не к её верхушке.

**`SINK_MODE=none` выставляется до импорта конфига.** Иначе сохранение
модели снесло бы накопленный граф монеты в Neo4j (инвариант 13) — модель
обучена на обрезанной истории, и её нумерация в бою не нужна никому.

## Чего эта проверка НЕ даёт

Полного walk-forward. Два look-ahead остаются внутри обучающей части:
нормировка признаков берёт медиану и IQR по всей ей (Ш2.2), а сглаживание
разметки подтверждает смену состояния через `smoothing_bars` баров, то есть
`group_id` бара опирается на два следующих бара. Первое на holdout не
протекает вовсе (модель обучена до границы), второе даёт заглядывание на
2 бара при горизонте в 96 — но помнить о нём надо, и после Ш2 замер стоит
повторить.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ДО импорта btcproc.config: конфиг читает окружение на импорте, а sink с
# режимом direct/http внутри train'а снёс бы граф монеты в Neo4j.
os.environ["SINK_MODE"] = "none"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from btcproc import config, symbols  # noqa: E402
from btcproc.analysis import holdout as ho  # noqa: E402
from btcproc.analysis.lift import DEFAULT_N_BOOT  # noqa: E402
from btcproc.candidates import builder as cand  # noqa: E402
from btcproc.candidates.outcomes import compute_outcomes  # noqa: E402
from btcproc.db import repo, runs  # noqa: E402
from btcproc.features import builder as feat  # noqa: E402
from btcproc.features import events as ev  # noqa: E402
from btcproc.ingest import bars  # noqa: E402
from btcproc.pipeline.train import run_train  # noqa: E402
from btcproc.states import assign  # noqa: E402

#: Замороженная граница датасета (раздел 0 плана). Все сравнительные замеры
#: этого плана обязаны ходить на неё.
FROZEN_END = "2026-08-01"

#: Поля кандидата, которые нужны замеру. Остальное лежит в payload и уходит
#: в скорер целиком.
FRAME_COLUMNS = (
    "ts", "side", "long_outcome_share", "sample_size", "research_score",
    "scope", "transition_rarity",
)


# ─── Подготовка ─────────────────────────────────────────────────────────────
def resolve_split(symbol: str, start: str, end: str,
                  frac: float) -> tuple[pd.Timestamp, int]:
    """
    Граница обучения: бар, на котором заканчиваются первые `frac` истории.

    Граница берётся по числу БАРОВ, а не по календарю: история монеты
    неравномерна (у BTC ранние годы разрежены дырами), и календарная
    середина дала бы не ту долю данных, которую заявляем.
    """
    base = bars.load_ohlcv(symbol, config.data.base_tf, start, end)
    if base.empty:
        raise SystemExit(
            f"{symbol}: в БД нет баров до {end}. Сначала ingest."
        )
    # Формула живёт в analysis/holdout: контрольная модель без графа (D1)
    # обязана делить историю тем же бором, иначе её числа несравнимы с этими.
    try:
        return ho.split_bar(base.index, frac), len(base)
    except ValueError as exc:
        raise SystemExit(f"{symbol}: {exc}.") from exc


def train_prefix_model(symbol: str, start: str, split_ts: pd.Timestamp,
                       random_state: int | None) -> int:
    """
    Обучает модель на префиксе истории и возвращает run_id.

    Прогон заводится с `kind='holdout'` намеренно. `live` и все замеры ищут
    модель через `latest_completed_run('train', symbol)`; появись здесь
    `train`, ближайший регулярный прогон разметил бы рынок моделью,
    обученной на данных до 2024 года, и заметить это было бы нечем —
    кандидаты продолжали бы выпускаться.
    """
    overrides = {"random_state": random_state} if random_state is not None else None
    run_id = runs.start_run(
        "holdout",
        {"symbol": symbol, "start": start, "end": split_ts.isoformat(),
         "purpose": "валидация на отложенной части (Ш0)",
         "states_overrides": overrides},
        symbol=symbol,
    )
    print(f"[{symbol}] обучение модели на префиксе до {split_ts:%Y-%m-%d} "
          f"(прогон #{run_id}, это 30–60 минут)…")
    try:
        stats = run_train(
            run_id=run_id, symbol=symbol, start=start, end=split_ts.isoformat(),
            do_ingest=False, do_emit=False, states_overrides=overrides,
        )
    except Exception as exc:  # noqa: BLE001 — прогон обязан закрыться диагнозом
        runs.fail_run(run_id, f"{type(exc).__name__}: {exc}")
        raise
    runs.finish_run(run_id, stats)
    print(f"[{symbol}] модель обучена: {stats['states']['groups']} состояний "
          f"на {stats['bars']} барах")
    return run_id


# ─── Разметка holdout'а ─────────────────────────────────────────────────────
def build_candidates(symbol: str, model_run: int, start: str, end: str,
                     split_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Кандидаты по всей истории до `end`, размеченные моделью `model_run`.

    Возвращает таблицу: поля замера + факт (`is_up`, `valid`) + сам payload
    для скорера. Разделение на обучающую часть и holdout вызывающий код
    делает по `ts`: генерировать надо всё подряд, иначе накопители выборки
    у holdout-кандидатов оказались бы пустыми.
    """
    model = repo.load_state_model(model_run)
    if model is None:
        raise SystemExit(f"У прогона #{model_run} нет сохранённой модели состояний.")

    base = bars.load_ohlcv(symbol, config.data.base_tf, start, end)
    context = {
        tf: bars.load_ohlcv(symbol, tf, start, end)
        for tf in config.data.context_tfs
    }
    print(f"[{symbol}] признаки на {len(base)} барах…")
    features = feat.build_features(base, context)
    missing = set(model.feature_names) - set(features.columns)
    if missing:
        raise SystemExit(
            f"Набор признаков разошёлся с моделью #{model_run}: нет {sorted(missing)}."
        )
    features = features[model.feature_names]

    labels = model.predict(feat.apply_scale(features, model.scale))
    states = assign.assign_states(features.index, labels)
    events = ev.build_event_blocks(base).reindex(features.index).dropna(
        subset=["event_block_id"]
    )
    outcomes = compute_outcomes(base).reindex(features.index)

    # Карты редкости — ТОЛЬКО по обучающей части (см. шапку модуля).
    rarity_map, block_map = ho.prefix_maps(states, events, outcomes, split_ts)
    print(f"[{symbol}] карты по обучающей части: {len(rarity_map)} переходов, "
          f"{len(block_map)} блоков событий")

    snapshots = cand.build_snapshots(states, events, outcomes)
    rows, payloads = [], []
    for candidate in cand.generate(snapshots, rarity_map, block_map, symbol):
        meta = candidate["_meta"]
        rows.append({
            "ts": pd.Timestamp(meta["ts"]),
            "side": candidate["research_side"],
            "long_outcome_share": candidate["long_outcome_share"],
            "sample_size": candidate["sample_size"],
            "research_score": candidate["research_score"],
            "scope": candidate["sample_scope"],
            "transition_rarity": candidate["transition_rarity"],
        })
        payloads.append(cand.strip_meta(candidate))

    frame = pd.DataFrame(rows, columns=list(FRAME_COLUMNS))
    if frame.empty:
        raise SystemExit(f"{symbol}: кандидатов не выпущено вовсе.")
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)

    fact = outcomes[["is_up", "valid"]].reindex(frame["ts"]).reset_index(drop=True)
    frame["is_up"] = fact["is_up"].to_numpy()
    frame["valid"] = fact["valid"].fillna(False).to_numpy(dtype=bool)
    frame["payload"] = payloads
    print(f"[{symbol}] кандидатов всего: {len(frame)}, "
          f"из них на holdout: {int((frame['ts'] >= split_ts).sum())}")
    return frame


def attach_ratings(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Проставляет `quality_score` и `rating` скорером btc-graph.

    Импортируется только скорер: ни pipeline, ни его запись в БД здесь не
    нужны, а лишний импорт потянул бы pgvector и neo4j.
    """
    path = str(config.sink.btc_graph_path)
    if not os.path.isdir(os.path.join(path, "src")):
        raise SystemExit(
            f"BTC_GRAPH_PATH={path} не похож на репозиторий btc-graph. "
            "Без него разрез по rating не посчитать."
        )
    if path not in sys.path:
        sys.path.insert(0, path)

    from src.config.profiles import get_profile, is_known_symbol  # noqa: E402
    from src.models.candidate import Candidate  # noqa: E402
    from src.scorer.candidate_scorer import get_rating, score_candidate  # noqa: E402

    if not is_known_symbol(symbol):
        print(f"[{symbol}] ВНИМАНИЕ: профиля оценки нет, рейтинг посчитан "
              f"линейкой биткоина — разрез по rating недостоверен.")
    profile = get_profile(symbol)

    scores, ratings = [], []
    for payload in frame["payload"]:
        candidate = Candidate.model_validate(payload)
        total = score_candidate(candidate, profile).total
        scores.append(total)
        ratings.append(get_rating(total, profile))
    frame = frame.copy()
    frame["quality_score"] = scores
    frame["rating"] = ratings
    print(f"[{symbol}] оценено скорером btc-graph (профиль {profile.name}): "
          + ", ".join(f"{k} {v}" for k, v in
                      frame["rating"].value_counts().items()))
    return frame


# ─── Замер ──────────────────────────────────────────────────────────────────
def prepare_measurable(frame: pd.DataFrame, split_ts: pd.Timestamp,
                       part: str) -> pd.DataFrame:
    """
    Часть выборки с созревшим исходом и посчитанным попаданием.

    Невалидные строки (последние сутки истории и участки с дырами) режутся
    здесь, а не в метриках: `is_up` там None, и молчаливое приведение к False
    означало бы «система ошиблась» вместо «факта нет».
    """
    subset = frame[frame["ts"] >= split_ts] if part == "holdout" \
        else frame[frame["ts"] < split_ts]
    subset = subset[subset["valid"] & subset["is_up"].notna()].copy()
    if subset.empty:
        return subset
    subset["is_up"] = subset["is_up"].astype(bool)
    subset["hit"] = ho.hit_flags(subset["side"], subset["is_up"])
    return subset.sort_values("ts").reset_index(drop=True)


def _model_run_from_name(path: str) -> int:
    """run_id из имени выгрузки `holdout_<SYMBOL>_<run>.csv`; 0 — не разобрали."""
    stem = Path(path).stem.split("_")
    return int(stem[-1]) if stem and stem[-1].isdigit() else 0


def run_symbol(symbol: str, args: argparse.Namespace) -> ho.HoldoutReport | None:
    spec = symbols.get(symbol)
    start = spec.start_date()

    if args.from_dump:
        frame = pd.read_csv(args.from_dump)
        # format="ISO8601", а не автоопределение: у BTC в дампах Binance есть
        # горстка баров с нестандартной отметкой (2018-02-09 12:13:14.789),
        # и pandas, вывернув формат по первой строке, падает на них.
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True, format="ISO8601")
        split_ts = pd.Timestamp(args.split_ts or frame.attrs.get("split_ts"), tz="UTC") \
            if (args.split_ts or frame.attrs.get("split_ts")) else None
        if split_ts is None:
            raise SystemExit("Для --from-dump нужен --split-ts.")
        # Прогон достаётся из имени файла (`holdout_BTCUSDT_28.csv`): отчёт без
        # номера модели нечем сопоставить с прогоном в БД, а именно это и
        # требуется, когда сравниваются две модели.
        model_run = args.model_run or _model_run_from_name(args.from_dump)
    else:
        split_ts, n_bars = resolve_split(symbol, start, args.end, args.train_frac)
        print(f"\n[{symbol}] история {start}…{args.end}: {n_bars} баров, "
              f"граница обучения {split_ts:%Y-%m-%d %H:%M} "
              f"({args.train_frac:.0%})")

        model_run = args.model_run
        if not args.no_train:
            model_run = train_prefix_model(symbol, start, split_ts,
                                           args.random_state)
        if not model_run:
            raise SystemExit("--no-train требует --model-run.")

        frame = build_candidates(symbol, model_run, start, args.end, split_ts)
        frame = attach_ratings(frame, symbol)
        if args.dump:
            path = Path(args.dump)
            path.mkdir(parents=True, exist_ok=True)
            target = path / f"holdout_{symbol}_{model_run}.csv"
            frame.drop(columns=["payload"]).to_csv(target, index=False)
            print(f"[{symbol}] выборка сохранена: {target} "
                  f"(--from-dump {target} --split-ts {split_ts.isoformat()})")

    measurable = prepare_measurable(frame, split_ts, "holdout")
    if measurable.empty:
        print(f"[{symbol}] на holdout нет кандидатов с созревшим исходом.")
        return None

    in_sample = prepare_measurable(frame, split_ts, "train")
    if not in_sample.empty:
        print(f"[{symbol}] для сравнения — на ОБУЧАЮЩЕЙ части accuracy "
              f"{in_sample['hit'].mean():.4f} на {len(in_sample)} кандидатах "
              f"(эта величина не доказательство: те же данные видела модель)")

    report = ho.measure(
        measurable, symbol=symbol, model_run=model_run, split_ts=split_ts,
        horizon_minutes=config.data.horizon_minutes,
        n_candidates=int((frame["ts"] >= split_ts).sum()),
        n_boot=args.n_boot, seed=args.seed,
    )
    print()
    print(ho.format_report(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Валидация на отложенной части истории (Ш0)"
    )
    parser.add_argument("--symbol", action="append", help="монета (можно несколько)")
    parser.add_argument("--all", action="store_true", help="все активные монеты")
    parser.add_argument("--end", default=FROZEN_END,
                        help=f"граница датасета, замороженная (по умолчанию {FROZEN_END})")
    parser.add_argument("--train-frac", type=float, default=0.7,
                        help="доля истории на обучение (по умолчанию 0.7)")
    parser.add_argument("--no-train", action="store_true",
                        help="не обучать, взять готовую модель из --model-run")
    parser.add_argument("--model-run", type=int, help="run_id готовой модели")
    parser.add_argument("--random-state", type=int,
                        help="зерно кластеризации — вторая модель на тех же данных")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT,
                        help=f"реплик блочного бутстрапа (по умолчанию {DEFAULT_N_BOOT})")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dump", help="каталог для выгрузки выборки в CSV")
    parser.add_argument("--from-dump", help="считать метрики по готовой выгрузке")
    parser.add_argument("--split-ts", help="граница обучения для --from-dump")
    args = parser.parse_args()

    targets = (symbols.tickers(only_enabled=True) if args.all
               else (args.symbol or [config.data.symbol]))
    if len(targets) > 1 and (args.model_run or args.from_dump):
        raise SystemExit("--model-run и --from-dump работают с одной монетой.")

    reports = []
    for symbol in targets:
        report = run_symbol(symbols.get(symbol).ticker, args)
        if report is not None:
            reports.append(report)

    if len(reports) > 1:
        print(ho.format_verdict(reports))
    if reports:
        print("\nКрасная линия критерия: тот же вывод обязан повториться на "
              "ВТОРОЙ модели состояний — повтори прогон с --random-state.")


if __name__ == "__main__":
    main()
