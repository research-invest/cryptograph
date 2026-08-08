"""
Снимает эталон значений скорера в tests/fixtures/scorer_snapshot.json.

Запускается ОДИН раз — до перехода скорера на профили (шаг 4.2). Дальше файл
только читается тестом test_scorer_snapshot.py: он доказывает, что data-driven
скорер с профилем `_default` считает ровно то же, что считал хардкод.

Перезаписывать снапшот допустимо только вместе с осознанной перекалибровкой
`_default` — и тогда обновляется README (раздел «Как читать результат оценки»).

    python3 scripts/make_scorer_snapshot.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import make_snapshot_candidates  # noqa: E402
from src.scorer.candidate_scorer import score_candidate  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "scorer_snapshot.json"


def main() -> None:
    rows = []
    for candidate in make_snapshot_candidates():
        breakdown = score_candidate(candidate)
        rows.append({
            "candidate_id": candidate.candidate_id,
            "statistical": breakdown.statistical,
            "directional": breakdown.directional,
            "context": breakdown.context,
            "rarity": breakdown.rarity,
            "total": breakdown.total,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"Записано {len(rows)} строк в {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
