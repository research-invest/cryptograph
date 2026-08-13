#!/usr/bin/env python3
"""
Sanity-якорь профиля монеты — правило 5 из btc-graph/docs/step_04_multi_symbol.md.

Монету нельзя завести без якоря: на это есть падающий тест
`test_every_profile_has_a_sanity_anchor`. Смысл якоря — поймать
перекалибровку, которая выглядит безобидной правкой одной ступени, а на деле
переводит понятный кандидат через границу рейтинга.

Скрипт берёт реальную выгрузку кандидатов монеты, оценивает её ПОСТАВЛЯЕМЫМ
профилем и выбирает кандидата с нужным рейтингом, чей `quality_score` лежит
дальше всего от границ, — якорь у самой границы менял бы рейтинг от любого
шевеления и падал бы без причины.

Запускать из каталога btc-graph, ПОСЛЕ записи профиля монеты:

    python3 ../.claude/skills/add-symbol/scripts/make_sanity_anchor.py \
        --symbol SUIUSDT --input dumps/sui_candidates.jsonl --dry-run

Без --dry-run дописывает запись в tests/fixtures/sanity_candidates.json,
остальные монеты не трогает.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[4] / "btc-graph"


def _load_repo() -> None:
    """btc-graph импортируется как пакет `src` — нужен его корень в sys.path."""
    root = REPO if (REPO / "src").is_dir() else Path.cwd()
    if not (root / "src" / "config" / "profiles.py").exists():
        sys.exit(
            f"Не нашёл btc-graph рядом со скиллом (пробовал {root}). "
            "Запусти скрипт из каталога btc-graph."
        )
    sys.path.insert(0, str(root))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True, help="Тикер, например SUIUSDT")
    parser.add_argument(
        "--input", required=True, type=Path,
        help="JSONL-выгрузка кандидатов монеты (та же, что для calibrate_profile.py)",
    )
    parser.add_argument(
        "--rating", default="MODERATE", choices=["STRONG", "MODERATE", "WEAK"],
        help="Рейтинг якоря. MODERATE по умолчанию: у него есть обе границы, "
             "поэтому он ловит сдвиг калибровки в любую сторону",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Файл фикстуры (по умолчанию tests/fixtures/sanity_candidates.json)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Только отчёт, без записи")
    parser.add_argument(
        "--force", action="store_true",
        help="Перезаписать якорь монеты, если он уже есть",
    )
    args = parser.parse_args()

    _load_repo()
    from src.config.profiles import get_profile  # noqa: PLC0415
    from src.models.candidate import Candidate  # noqa: PLC0415
    from src.scorer.candidate_scorer import get_rating, score_candidate  # noqa: PLC0415

    symbol = args.symbol.strip().upper()
    fixture = args.out or (REPO / "tests" / "fixtures" / "sanity_candidates.json")

    profile = get_profile(symbol)
    if profile.symbol.upper() != symbol:
        return _fail(
            f"У {symbol} нет своего профиля — get_profile отдал {profile.name}. "
            f"Сначала откалибруй config/symbols/{symbol}.yaml, потом якорь: "
            "якорь по чужой линейке зафиксировал бы неверный рейтинг."
        )

    if not args.input.exists():
        return _fail(f"Нет файла выгрузки {args.input}")

    scored: list[tuple[float, str, Candidate]] = []
    bad = 0
    for line in args.input.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            candidate = Candidate(**json.loads(line))
        except Exception:  # noqa: BLE001 — битая строка выгрузки не должна ронять разбор
            bad += 1
            continue
        if candidate.symbol.upper() != symbol:
            bad += 1
            continue
        total = score_candidate(candidate, profile).total
        scored.append((total, get_rating(total, profile), candidate))

    if not scored:
        return _fail(
            f"В {args.input} нет ни одного кандидата {symbol}. "
            "Проверь фильтр symbol в SQL-выгрузке."
        )

    pool = [row for row in scored if row[1] == args.rating]
    if not pool:
        shares = {r: sum(1 for x in scored if x[1] == r) for _, r, _ in scored}
        return _fail(
            f"Среди {len(scored)} кандидатов нет ни одного {args.rating}. "
            f"Распределение: {shares}. Либо возьми другой --rating, либо "
            "профиль откалиброван не по этой выгрузке."
        )

    # Якорь берём максимально далеко от обеих границ рейтинга: у границы он
    # менял бы вердикт от любой правки и падал бы, ничего не диагностируя.
    lo, hi = _bounds(args.rating, profile)
    total, rating, candidate = max(pool, key=lambda row: min(row[0] - lo, hi - row[0]))

    note = (
        f"Реальный кандидат {symbol} из выгрузки {args.input.name} "
        f"({date.today().isoformat()}). quality_score {total:.4f} при границах "
        f"{profile.rating.moderate_min}/{profile.rating.strong_min}, профиль "
        f"{profile.name}. Выбран как самый удалённый от границ рейтинга "
        f"из {len(pool)} кандидатов {rating}."
    )

    print(f"Профиль: {profile.name} (fingerprint {profile.fingerprint[:12]}…)")
    print(f"Кандидатов разобрано: {len(scored)}" + (f", пропущено строк: {bad}" if bad else ""))
    print(f"Распределение: " + ", ".join(
        f"{r}={sum(1 for x in scored if x[1] == r)}" for r in ("STRONG", "MODERATE", "WEAK")
    ))
    print(f"Якорь: candidate_id={candidate.candidate_id} score={total:.4f} → {rating}")
    print(f"Заметка: {note}")

    if args.dry_run:
        print("\n--dry-run: фикстура не изменена")
        return 0

    anchors = json.loads(fixture.read_text(encoding="utf-8")) if fixture.exists() else {}
    if symbol in anchors and not args.force:
        return _fail(
            f"Якорь {symbol} уже есть в {fixture.name}. Перезапись — это смена "
            "эталона, делается осознанно: --force."
        )

    anchors[symbol] = {
        "expected_rating": rating,
        "note": note,
        "candidate": candidate.model_dump(mode="json"),
    }
    fixture.write_text(
        json.dumps(dict(sorted(anchors.items())), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nЗаписано в {fixture}")
    print("Проверка: pytest tests/test_sanity_candidates.py -q")
    return 0


def _bounds(rating: str, profile) -> tuple[float, float]:  # noqa: ANN001
    if rating == "STRONG":
        return profile.rating.strong_min, 1.0
    if rating == "MODERATE":
        return profile.rating.moderate_min, profile.rating.strong_min
    return 0.0, profile.rating.moderate_min


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
