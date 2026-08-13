"""
Подбор профиля оценки для новой монеты по выгрузке её кандидатов.

Зачем скрипт, а не «покрутить пороги руками». Правка ступеней на глаз даёт
профиль, про который через месяц никто не скажет, почему числа именно такие.
Здесь процедура воспроизводима: ступени предлагаются как перцентили реального
распределения монеты, а отчёт сразу показывает, что от этого изменилось.

    python3 scripts/calibrate_profile.py --symbol ETHUSDT \
        --input dumps/eth_candidates.jsonl --dry-run

    python3 scripts/calibrate_profile.py --symbol ETHUSDT \
        --input dumps/eth_candidates.jsonl --out config/symbols/ETHUSDT.yaml

Вход — JSONL: по одному кандидату в строке, в формате схемы btc-graph
(ровно то, что отдаёт генератор). Выгружать **обязательно с ORDER BY ts**:

    psql -c "\\copy (SELECT payload FROM processing.candidates
             WHERE symbol='ETHUSDT' ORDER BY ts) TO 'dumps/eth_candidates.jsonl'"

Порядок важен потому, что в схеме кандидата **нет поля времени** — `_meta.ts`
снимается перед отправкой. Скрипт не может отсортировать вход сам и доверяет
порядку строк: на нём держатся и `--tail`, и `--holdout`. Выгрузка без
ORDER BY превращает «проверку на будущем» в проверку на случайной половине,
и она перестаёт что-либо ловить.

## Почему по умолчанию калибруем по свежему окну (`--tail`)

`sample_size` кандидата — это число накопленных аналогов на момент его выпуска,
и оно монотонно растёт по мере истории. Кандидат 2018 года видит сотню случаев,
кандидат 2026-го — тысячи. Перцентили, посчитанные по всей истории, ставят
планку туда, где сидели бедные ранние годы, а применяется профиль к сегодняшним
кандидатам — и те массово оказываются «выдающимися».

На реальной выгрузке ETH это видно прямо: профиль, откалиброванный по всей
истории, даёт 10% STRONG на калибровочной части и 26% на свежей трети.
Поэтому по умолчанию берётся хвост выгрузки: калибруем на том режиме,
в котором профиль будет работать. `--tail 0` — считать по всей истории.

ВАЖНО: скрипт выдаёт ЧЕРНОВИК, а не готовый профиль. Перцентили не знают,
что sample_size = 120 статистически несостоятелен независимо от того, что это
p90 конкретной монеты. Порядок работы — правила 1–6 из
docs/step_04_multi_symbol.md:

  1. сначала ступени, потом веса — вес не чинит вырожденную ось;
  2. ступени от перцентилей выборки, а не от круглых чисел;
  3. три диагноза читаются по распределению (см. отчёт);
  4. чем меньше данных, тем больше веса статистической надёжности;
  5. два якоря: регрессия BTC и sanity-кандидат монеты;
  6. калибровка — событие: меняешь профиль → бампаешь version.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.profiles import (  # noqa: E402
    _SHARE_FIELDS as SHARE_FIELDS,
    ScoringProfile,
    _deep_merge,
    default_profile,
    profiles_dir,
)
from src.models.candidate import Candidate  # noqa: E402
from src.scorer.candidate_scorer import (  # noqa: E402
    _score_context,
    _score_directional,
    _score_rarity,
    _score_statistical,
    get_rating,
    score_candidate,
)

AXES = ("statistical", "directional", "context", "rarity")

#: Откуда брать `floor` для ступеней, которых в базовом профиле нет.
_FLOOR_TWIN = {"effective_sample_size": "sample_size"}

# Перцентили → баллы. Верхняя ступень отсекает лучшую десятину выборки,
# нижняя — медиану: ниже медианы признак перестаёт различать кандидатов.
HIGHER_BETTER_PERCENTILES = ((90, 1.0), (75, 0.7), (50, 0.4))
LOWER_BETTER_PERCENTILES = ((10, 1.0), (25, 0.7), (50, 0.3))

# Признаки, по которым предлагаются ступени, и куда они кладутся в профиле.
LADDER_FIELDS = {
    ("statistical", "valid_label_pct"): ("valid_label_pct", "higher_better", 4),
    ("statistical", "sample_size"): ("sample_size", "higher_better", 0),
    # Ступень по реализациям заводится рядом со ступенью по строкам, а не
    # вместо неё: первая применяется к кандидатам, которые поле несут, вторая
    # остаётся для выпущенных до 2026-08-13 (см. _sample_size_score в скорере).
    ("statistical", "effective_sample_size"): ("effective_sample_size", "higher_better", 0),
    ("statistical", "monthly_concentration"): ("monthly_concentration", "lower_better", 4),
    ("statistical", "repeatability_months"): ("repeatability_months", "higher_better", 0),
    ("directional", "win_rate"): ("__win_rate", "higher_better", 4),
    ("directional", "abs_outcome_skew"): ("__abs_skew", "higher_better", 4),
    ("directional", "fa_ratio"): ("__fa_ratio", "higher_better", 3),
    ("rarity", "research_score"): ("research_score", "higher_better", 4),
}

# Отправная точка по весам из правила 4. Не догма — проверяется отчётом.
WEIGHT_PRESETS = {
    "btc": {"statistical": 0.30, "directional": 0.35, "context": 0.20, "rarity": 0.15},
    "major-alt": {"statistical": 0.35, "directional": 0.30, "context": 0.20, "rarity": 0.15},
    "low-liquidity": {"statistical": 0.40, "directional": 0.25, "context": 0.20, "rarity": 0.15},
    "young": {"statistical": 0.40, "directional": 0.25, "context": 0.25, "rarity": 0.10},
}

# Ниже этого разброса ось считается вырожденной: её вес просто сдвигает
# всем кандидатам score на константу, ничего не различая (правило 3).
DEGENERATE_VARIANCE = 0.05


# ─── Загрузка выборки ─────────────────────────────────────────────────────────

def load_candidates(path: Path, symbol: str) -> list[Candidate]:
    """Читает JSONL, отбирает нужную монету, пропускает битые строки с диагнозом."""
    candidates, skipped = [], 0
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                # Выгрузка из processing.candidates может отдавать {"payload": {...}}.
                if set(raw) == {"payload"}:
                    raw = raw["payload"]
                raw.pop("_meta", None)
                candidate = Candidate(**raw)
            except Exception as exc:  # noqa: BLE001 — черновик не должен падать на строке
                skipped += 1
                if skipped <= 3:
                    print(f"  строка {line_no} пропущена: {exc}", file=sys.stderr)
                continue
            if candidate.symbol.upper() == symbol.upper():
                candidates.append(candidate)

    if skipped:
        print(f"  всего пропущено строк: {skipped}", file=sys.stderr)
    return candidates


def _feature_values(candidates: Iterable[Candidate], field: str) -> list[float]:
    """Сырые значения признака. Служебные имена — производные метрики."""
    from src.scorer.candidate_scorer import fa_ratio_for, win_rate_for

    values = []
    for c in candidates:
        if field == "__win_rate":
            values.append(win_rate_for(c))
        elif field == "__abs_skew":
            values.append(abs(c.historical_outcome_skew))
        elif field == "__fa_ratio":
            ratio = fa_ratio_for(c)
            # None у short — это отсутствие метрики, а не ноль: включать
            # её в распределение значило бы занижать все ступени.
            if ratio is not None:
                values.append(ratio)
        else:
            raw = getattr(c, field)
            # Optional-поля у старых выгрузок пусты. None — это «признака нет»,
            # а не ноль: подставив ноль, мы бы уронили все перцентили.
            if raw is None:
                continue
            values.append(float(raw))
    return values


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q / 100.0
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (pos - low)


# ─── Предложение ступеней ─────────────────────────────────────────────────────

def suggest_ladders(candidates: list[Candidate]) -> dict[str, dict[str, Any]]:
    """Ступени как перцентили распределения монеты (правило 2)."""
    base = default_profile()
    suggestion: dict[str, dict[str, Any]] = {}

    for (axis, key), (field, mode, digits) in LADDER_FIELDS.items():
        values = sorted(_feature_values(candidates, field))
        if len(values) < 20:
            # На такой выборке перцентиль — шум. Наследуем базовую ступень.
            continue

        table = (
            HIGHER_BETTER_PERCENTILES if mode == "higher_better"
            else LOWER_BETTER_PERCENTILES
        )
        thresholds = []
        for percentile, _score in table:
            value = _percentile(values, percentile)
            thresholds.append(round(value, digits) if digits else int(value))

        # Порог на границе области значений — мёртвая ступень: сравнение
        # строгое, и `value > 1.0` не сработает никогда. На реальной выгрузке
        # ETH такое вышло само: p90 доли валидных меток оказался ровно 1.0,
        # то есть у девяти кандидатов из десяти метки валидны все до одной.
        # Такую ступень надо не «поправить на 0.999», а выбросить: признак
        # просто не различает верхушку выборки.
        thresholds = [t for t in thresholds if _reachable(key, mode, t)]

        # Совпавшие перцентили на вырожденном признаке дают немонотонную
        # лесенку — она не пройдёт валидацию профиля.
        ladder = _dedupe_ladder(
            [[t, score] for t, (_p, score) in zip(thresholds, table)], mode
        )
        if len(ladder) < 2:
            continue

        current = getattr(getattr(base, axis), key)
        suggestion.setdefault(axis, {})[key] = {
            "mode": mode,
            "ladder": ladder,
            # Пол наследуем у базовой линейки. У ступени, которой в `_default`
            # нет вовсе (effective_sample_size — она опциональна и заводится
            # только в профилях монет), берём пол её строкового двойника:
            # это одна и та же ось, выраженная в других единицах.
            "floor": current.floor if current is not None
            else getattr(getattr(base, axis), _FLOOR_TWIN.get(key, key)).floor,
        }

    return suggestion


def _reachable(key: str, mode: str, threshold: float) -> bool:
    """
    Может ли значение признака вообще перешагнуть этот порог.

    Доли лежат в [0, 1], сравнение строгое: порог 1.0 у higher_better и 0.0
    у lower_better недостижимы. Профиль с такой ступенью не пройдёт валидацию
    схемы — и правильно сделает.
    """
    if key not in SHARE_FIELDS:
        return True
    if mode == "higher_better":
        return threshold < 1.0
    return threshold > 0.0


def _dedupe_ladder(ladder: list[list], mode: str) -> list[list]:
    """Оставляет строго монотонные ступени, сохраняя порядок сверху вниз."""
    result: list[list] = []
    for threshold, score in ladder:
        if not result:
            result.append([threshold, score])
            continue
        previous = result[-1][0]
        strictly_ordered = (
            threshold < previous if mode == "higher_better" else threshold > previous
        )
        if strictly_ordered:
            result.append([threshold, score])
    return result


# Доли, на которые целится черновик по рейтингам.
#
# Это не абстрактное «пусть STRONG будет редким», а **селективность эталона**:
# BTCUSDT на своих данных по базовому профилю даёт 1.0% STRONG, 37.6% MODERATE,
# 61.4% WEAK. Целиться надо туда же, и вот почему. Рейтинг сравним только
# внутри монеты — но человек-то смотрит сводный список «что сегодня сильного».
# Если у ETH STRONG выдаётся вдесятеро щедрее, он этот список зальёт, и
# внимание уедет к монете не потому, что там интереснее, а потому что у неё
# планка ниже. Одинаковая селективность делает сводную выдачу осмысленной.
#
# Переопределяется флагами --target-strong / --target-moderate.
TARGET_STRONG_SHARE = 0.01
TARGET_MODERATE_SHARE = 0.39


def build_profile(
    symbol: str,
    version: int,
    candidates: list[Candidate],
    weights_preset: str,
    keep_rating: bool = False,
    target_strong: float = TARGET_STRONG_SHARE,
    target_moderate: float = TARGET_MODERATE_SHARE,
) -> tuple[ScoringProfile, dict[str, Any]]:
    """
    Собирает профиль-черновик и его YAML-представление (только отличия).

    Границы рейтинга тоже предлагаются от распределения, и это не
    самодеятельность, а следствие способа подбора ступеней. Перцентильная
    лесенка p90/p75/p50 → 1.0/0.7/0.4 по построению даёт средний балл оси
    около 0.3: верхнюю ступень берёт десятая часть выборки. С границами BTC
    (0.75/0.55) любой такой черновик выдаёт «90% WEAK» — то есть срабатывает
    собственный диагноз правила 3, и он оказывается шумом, а не сигналом.
    Поэтому границы двигаются вместе со ступенями, а диагноз снова начинает
    означать «с этой монетой что-то не так», а не «скрипт так устроен».

    keep_rating=True — оставить границы базового профиля (когда монету
    сознательно меряют линейкой BTC).
    """
    overrides: dict[str, Any] = {
        "symbol": symbol.upper(),
        "version": version,
        "description": (
            f"Черновик калибровки по {len(candidates)} кандидатам "
            f"(scripts/calibrate_profile.py, пресет весов «{weights_preset}»)"
        ),
        "extends": "_default",
        "weights": WEIGHT_PRESETS[weights_preset],
    }
    overrides.update(suggest_ladders(candidates))

    base_dump = default_profile().model_dump(mode="json")
    merged = _deep_merge(base_dump, overrides)
    merged.pop("extends", None)
    profile = ScoringProfile(**merged)

    if not keep_rating and len(candidates) >= 20:
        totals = sorted(score_candidate(c, profile).total for c in candidates)
        strong_min = round(_percentile(totals, 100 * (1 - target_strong)), 3)
        moderate_min = round(_percentile(totals, 100 * (1 - target_moderate)), 3)
        # Границы обязаны различаться: иначе MODERATE недостижим и профиль
        # не пройдёт валидацию схемы.
        if strong_min <= moderate_min:
            strong_min = round(moderate_min + 0.05, 3)
        rating = {"strong_min": strong_min, "moderate_min": moderate_min}
        overrides["rating"] = rating
        # Порог батча = граница MODERATE: ниже неё кандидат не заслуживает
        # полного прохода pipeline с вызовом LLM.
        overrides["batch"] = {
            "min_quality_score": moderate_min,
            "fresh_bonus": profile.batch.fresh_bonus,
        }
        merged = _deep_merge(base_dump, overrides)
        merged.pop("extends", None)
        profile = ScoringProfile(**merged)

    return profile, overrides


# ─── Отчёт ────────────────────────────────────────────────────────────────────

def axis_scores(candidates: list[Candidate], profile: ScoringProfile) -> dict[str, list[float]]:
    return {
        "statistical": [_score_statistical(c, profile) for c in candidates],
        "directional": [_score_directional(c, profile) for c in candidates],
        "context": [_score_context(c, profile) for c in candidates],
        "rarity": [_score_rarity(c, profile) for c in candidates],
    }


def _stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _iqr(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    return _percentile(ordered, 25), _percentile(ordered, 75)


def report(candidates: list[Candidate], profile: ScoringProfile, title: str) -> dict:
    breakdowns = [score_candidate(c, profile) for c in candidates]
    totals = [b.total for b in breakdowns]
    ratings = Counter(get_rating(t, profile) for t in totals)
    axes = axis_scores(candidates, profile)
    weights = profile.weights.as_dict()

    print(f"\n─── {title} ({profile.name}) " + "─" * max(0, 40 - len(title)))
    total = len(candidates)
    for rating in ("STRONG", "MODERATE", "WEAK"):
        n = ratings.get(rating, 0)
        print(f"  {rating:<9} {n:>6}  {n / total:>6.1%}")

    q1, q3 = _iqr(totals)
    print(f"  медиана quality_score: {statistics.median(totals):.4f}   "
          f"IQR: {q1:.4f}…{q3:.4f}")

    print("  вклад осей (вес × среднее) и разброс:")
    for axis in AXES:
        values = axes[axis]
        mean = statistics.fmean(values)
        sd = _stdev(values)
        marker = "  ← ось вырождена" if sd < DEGENERATE_VARIANCE else ""
        print(f"    {axis:<12} вклад {weights[axis] * mean:.4f}  "
              f"среднее {mean:.3f}  σ {sd:.3f}{marker}")

    return {
        "ratings": ratings, "totals": totals, "axes": axes,
        "scored": list(zip(candidates, breakdowns)), "profile": profile,
    }


def print_diagnosis(after: dict, total: int) -> None:
    """Три диагноза, которые читаются по распределению (правило 3)."""
    print("\n─── Диагноз " + "─" * 48)
    problems = []

    weak_share = after["ratings"].get("WEAK", 0) / total
    strong_share = after["ratings"].get("STRONG", 0) / total
    median = statistics.median(after["totals"])

    if weak_share > 0.90:
        problems.append(
            f"  {weak_share:.0%} кандидатов WEAK при медиане {median:.2f} — "
            "ступени монете недостижимы.\n"
            "    Что двигать: ступени вниз (перцентили), веса не трогать."
        )
    if strong_share > 0.40:
        problems.append(
            f"  {strong_share:.0%} кандидатов STRONG — ступени слишком мягкие, "
            "score перестал различать.\n"
            "    Что двигать: ступени вверх и/или rating.strong_min вверх."
        )
    for axis in AXES:
        sd = _stdev(after["axes"][axis])
        if sd < DEGENERATE_VARIANCE:
            problems.append(
                f"  Ось {axis}: разброс σ={sd:.3f} — она константна и только "
                "сдвигает всем score.\n"
                "    Что двигать: перераспределить её вес на различающие оси."
            )

    print("\n".join(problems) if problems else
          "  Явных перекосов нет. Проверь топ-10 глазами и зафиксируй "
          "sanity-кандидата в tests/fixtures/.")


def print_top(after: dict, limit: int = 10) -> None:
    """
    Топ с разбивкой по осям. Читается глазами: если верхние кандидаты
    вытянуты одной осью, профиль ранжирует не то, что задумано.
    """
    print(f"\n─── Топ-{limit} по профилю " + "─" * 38)
    print(f"  {'candidate_id':<22} {'total':>7} {'stat':>6} {'dir':>6} "
          f"{'ctx':>6} {'rar':>6} {'baseline':>9}  rating")

    ranked = sorted(after["scored"], key=lambda pair: -pair[1].total)[:limit]
    profile = after["profile"]
    for candidate, b in ranked:
        print(f"  {candidate.candidate_id[:22]:<22} {b.total:>7.4f} "
              f"{b.statistical:>6.3f} {b.directional:>6.3f} {b.context:>6.3f} "
              f"{b.rarity:>6.3f} {b.baseline_total:>9.4f}  "
              f"{get_rating(b.total, profile)}")


def to_yaml(overrides: dict[str, Any]) -> str:
    import yaml

    header = (
        "# ЧЕРНОВИК, сгенерированный scripts/calibrate_profile.py.\n"
        "# Ступени — перцентили выборки монеты; они не знают, что маленькая\n"
        "# выборка статистически несостоятельна даже если это p90 монеты.\n"
        "# Перед вводом в работу: проверь топ-10 глазами, положи sanity-кандидата\n"
        "# в tests/fixtures/ и зафиксируй его ожидаемый рейтинг.\n"
        "# Меняешь профиль потом — бампай version (правило 6).\n\n"
    )
    return header + yaml.safe_dump(
        overrides, allow_unicode=True, sort_keys=False, default_flow_style=None
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Черновик профиля калибровки по выгрузке кандидатов монеты",
    )
    parser.add_argument("--symbol", required=True, help="Тикер, например ETHUSDT")
    parser.add_argument("--input", required=True, type=Path, help="JSONL с кандидатами")
    parser.add_argument("--out", type=Path, help="Куда записать YAML-черновик")
    parser.add_argument("--dry-run", action="store_true", help="Только отчёт, без записи")
    parser.add_argument("--version", type=int, default=1, help="version профиля")
    parser.add_argument(
        "--weights", choices=sorted(WEIGHT_PRESETS), default="major-alt",
        help="Пресет весов из правила 4 (по умолчанию major-alt)",
    )
    parser.add_argument(
        "--tail", type=float, default=0.25,
        help="Доля (0..1) или число последних кандидатов выгрузки, по которым "
             "калибруем. Профиль применяется к сегодняшним кандидатам, а "
             "sample_size растёт по истории — планка по всей истории выйдет "
             "заниженной. 0 — считать по всему файлу",
    )
    parser.add_argument(
        "--target-strong", type=float, default=TARGET_STRONG_SHARE,
        help=f"Доля STRONG, на которую целятся границы рейтинга "
             f"(по умолчанию {TARGET_STRONG_SHARE} — селективность BTCUSDT)",
    )
    parser.add_argument(
        "--target-moderate", type=float, default=TARGET_MODERATE_SHARE,
        help=f"Доля MODERATE и выше (по умолчанию {TARGET_MODERATE_SHARE})",
    )
    parser.add_argument(
        "--keep-rating", action="store_true",
        help="Не двигать границы рейтинга — мерить монету линейкой BTC",
    )
    parser.add_argument(
        "--holdout", type=float, default=0.0,
        help="Доля выборки [0..1), отложенная на проверку: калибруем по первой "
             "половине, отчёт печатаем и по второй — защита от подгонки",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Нет файла {args.input}", file=sys.stderr)
        return 2

    candidates = load_candidates(args.input, args.symbol)
    if len(candidates) < 50:
        print(
            f"Кандидатов по {args.symbol}: {len(candidates)}. Этого мало для "
            "перцентилей — прогони train по монете подольше.",
            file=sys.stderr,
        )
        if not candidates:
            return 2

    print(f"Кандидатов по {args.symbol}: {len(candidates)}")
    print(
        "  порядок строк принимается как хронологический — "
        "выгрузка обязана быть с ORDER BY ts"
    )

    # Свежее окно: профиль применяется к сегодняшним кандидатам, а не к тем,
    # что выпускались, когда история только копилась (см. шапку файла).
    if args.tail:
        window = (
            int(len(candidates) * args.tail) if args.tail < 1 else int(args.tail)
        )
        window = max(1, min(window, len(candidates)))
        if window < len(candidates):
            candidates = candidates[-window:]
            print(f"  окно калибровки: последние {window} кандидатов")

    fit = candidates
    holdout: list[Candidate] = []
    if 0.0 < args.holdout < 1.0:
        # Делим по времени выпуска, а не случайно: калибровка на будущем
        # и проверка на прошлом — это подгонка с другого конца.
        cut = int(len(candidates) * (1 - args.holdout))
        fit, holdout = candidates[:cut], candidates[cut:]
        print(f"  калибровка по {len(fit)}, проверка по {len(holdout)}")

    profile, overrides = build_profile(
        args.symbol, args.version, fit, args.weights,
        keep_rating=args.keep_rating,
        target_strong=args.target_strong,
        target_moderate=args.target_moderate,
    )

    report(fit, default_profile(), "До: базовая калибровка")
    after = report(fit, profile, "После: профиль монеты")
    print_diagnosis(after, len(fit))
    print_top(after)

    if holdout:
        report(holdout, profile, "Проверка на отложенной части")

    print("\n─── Предложенные ступени " + "─" * 35)
    print(to_yaml({k: v for k, v in overrides.items()
                   if k in ("weights", "rating", "batch", *AXES)}))

    if args.out and not args.dry_run:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(to_yaml(overrides), encoding="utf-8")
        print(f"Черновик записан: {args.out}")
        print(f"Проверить: make profiles-check  (каталог {profiles_dir()})")
    elif args.out:
        print("--dry-run: файл не записан")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
