"""
Что профиль делает с ВЫГРУЗКОЙ: распределение рейтингов и вклад осей.

    python3 scripts/profile_shift.py --symbol BTCUSDT --input dumps/btc_candidates_v2.jsonl
    python3 scripts/profile_shift.py --symbol BTCUSDT --input dumps/btc_candidates_v2.jsonl \
        --proposed config/symbols/BTCUSDT.draft.yaml

Дополняет `calibrate_profile.py`, а не повторяет его. Тот сравнивает черновик
с БАЗОВОЙ линейкой `_default` — это правильный вопрос, когда монета заводится
впервые. При ПЕРЕкалибровке вопрос другой: что на новой выборке делает профиль,
который сейчас в работе, и стоит ли его двигать. Ответ на него — здесь.

Оба сравнения идут по одной и той же выборке (одному файлу), потому что
сравнивать распределение рейтингов, посчитанное на разных наборах кандидатов,
бессмысленно: половина разницы придёт от смены набора.

`--tail` по умолчанию тот же, что у калибратора (0.25): профиль применяется к
сегодняшним кандидатам, а `sample_size` монотонно растёт по истории.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.profiles import ScoringProfile, _deep_merge, default_profile, get_profile  # noqa: E402
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


def load(path: Path, symbol: str, tail: float) -> list[Candidate]:
    items: list[Candidate] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if set(raw) == {"payload"}:
                raw = raw["payload"]
            raw.pop("_meta", None)
            candidate = Candidate(**raw)
            if candidate.symbol.upper() == symbol.upper():
                items.append(candidate)
    if tail and 0 < tail < 1 and len(items) > 1:
        items = items[-max(1, int(len(items) * tail)):]
    return items


def load_yaml_profile(path: Path) -> ScoringProfile:
    """Профиль из YAML-черновика — с тем же наследованием от `_default`."""
    import yaml

    overrides = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    overrides.pop("extends", None)
    merged = _deep_merge(default_profile().model_dump(mode="json"), overrides)
    return ScoringProfile(**merged)


def describe(candidates: list[Candidate], profile: ScoringProfile, title: str) -> dict:
    totals = [score_candidate(c, profile).total for c in candidates]
    ratings = Counter(get_rating(t, profile) for t in totals)
    axes = {
        "statistical": [_score_statistical(c, profile) for c in candidates],
        "directional": [_score_directional(c, profile) for c in candidates],
        "context": [_score_context(c, profile) for c in candidates],
        "rarity": [_score_rarity(c, profile) for c in candidates],
    }

    print(f"\n─── {title} ({profile.name}) " + "─" * max(0, 34 - len(title)))
    for rating in ("STRONG", "MODERATE", "WEAK"):
        n = ratings.get(rating, 0)
        print(f"  {rating:<9} {n:>7}  {n / len(candidates):>6.1%}")
    ordered = sorted(totals)
    q1 = ordered[len(ordered) // 4]
    q3 = ordered[3 * len(ordered) // 4]
    print(f"  медиана quality_score: {statistics.median(totals):.4f}   "
          f"IQR: {q1:.4f}…{q3:.4f}")
    print(f"  границы рейтинга: STRONG ≥ {profile.rating.strong_min}, "
          f"MODERATE ≥ {profile.rating.moderate_min}; "
          f"порог батча {profile.batch.min_quality_score}")
    for axis in AXES:
        values = axes[axis]
        sd = statistics.pstdev(values) if len(values) > 1 else 0.0
        mark = "  ← ось вырождена" if sd < 0.05 else ""
        print(f"    {axis:<12} среднее {statistics.fmean(values):.3f}  σ {sd:.3f}{mark}")
    return {"ratings": ratings, "totals": totals}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--proposed", type=Path,
                        help="YAML предлагаемого профиля (иначе только текущий)")
    parser.add_argument("--tail", type=float, default=0.25,
                        help="доля свежего окна (0 — вся выгрузка)")
    parser.add_argument(
        "--ignore-sample-scope", action="store_true",
        help="считать ось rarity по трём критериям всегда, как до 2026-08-13. "
             "Диагностика: показывает, какая часть сдвига оценок пришла именно "
             "от того, что за чужой блок событий баллы больше не начисляются",
    )
    args = parser.parse_args()

    candidates = load(args.input, args.symbol, args.tail)
    if not candidates:
        print(f"В {args.input} нет кандидатов по {args.symbol}", file=sys.stderr)
        return 2
    print(f"{args.symbol}: {len(candidates)} кандидатов "
          f"(окно {args.tail if args.tail else 'вся выгрузка'})")
    # Доля выдачи, у которой выборка не обусловлена блоком событий: ось rarity
    # у неё считается по двум критериям, и знать эту долю при калибровке нужно.
    fallback = sum(1 for c in candidates
                   if c.sample_scope is not None and c.sample_scope.value == "transition")
    print(f"  sample_scope=transition: {fallback} ({fallback / len(candidates):.1%})")

    if args.ignore_sample_scope:
        # Копия с обнулённым полем: None означает «не знаем», и ось rarity
        # снова считается по трём критериям.
        candidates = [c.model_copy(update={"sample_scope": None}) for c in candidates]
        print("  --ignore-sample-scope: ось rarity считается по трём критериям")

    describe(candidates, get_profile(args.symbol), "Действующий профиль")
    if args.proposed:
        describe(candidates, load_yaml_profile(args.proposed), "Предлагаемый профиль")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
