"""
Выгрузка кандидатов монеты в JSONL — вход для калибровки профиля оценки.

    python3 scripts/dump_candidates.py --all --out-dir ../btc-graph/dumps
    python3 scripts/dump_candidates.py --symbol BTCUSDT --end 2026-08-01

Зачем отдельный скрипт, когда есть `\\copy ... FROM processing.candidates`.
Выгрузка из таблицы отдаёт то, что там лежит, а лежит там результат прогонов
РАЗНЫХ версий кода: сборка кандидата менялась, и `payload` старых строк
несёт старые правила. Калибровать профиль на смеси — значит ставить пороги
по распределению, которого больше не существует. Здесь история проигрывается
текущим кодом целиком, одной моделью, за один проход.

Второе отличие от `\\copy`: порядок. У схемы кандидата нет поля времени
(`_meta` снимается перед отправкой), и `calibrate_profile.py` доверяет порядку
строк — на нём держатся `--tail` и `--holdout`. Генератор идёт по истории
хронологически, поэтому файл выходит упорядоченным по построению, а не по
надежде на `ORDER BY`.

**В БД ничего не пишется и никуда ничего не отправляется.** Модель состояний
берётся готовая — последнего успешного `train` монеты; переобучения здесь нет
и быть не должно (`train` перенумеровывает `group_id` и сносит граф монеты
в Neo4j).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ДО импорта btcproc.config: конфиг читает окружение на импорте, а любой
# режим sink, кроме none, означал бы отправку выгружаемых кандидатов наружу.
os.environ["SINK_MODE"] = "none"

from btcproc import config, symbols  # noqa: E402
from btcproc.analysis import replay  # noqa: E402
from btcproc.candidates import builder as cand  # noqa: E402

#: Та же замороженная граница, что у validate_holdout и measure_block_a.
FROZEN_END = "2026-08-01"

#: Короткие имена файлов — те же, что у прежних выгрузок в btc-graph/dumps.
FILE_STEMS = {
    "BTCUSDT": "btc", "ETHUSDT": "eth", "SOLUSDT": "sol", "HYPEUSDT": "hype",
}


def dump_symbol(symbol: str, end: str, out_dir: Path, suffix: str) -> Path:
    data = replay.label_history(symbol, end=end, log=print)

    stem = FILE_STEMS.get(symbol, symbol.lower())
    path = out_dir / f"{stem}_candidates{suffix}.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    scopes: dict[str, int] = {}
    with path.open("w", encoding="utf-8") as handle:
        for candidate in data.candidates():
            payload = cand.strip_meta(candidate)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
            scope = payload["sample_scope"]
            scopes[scope] = scopes.get(scope, 0) + 1

    size_mb = path.stat().st_size / 1024 / 1024
    print(f"[{symbol}] {written} кандидатов → {path} ({size_mb:.1f} МБ)")
    print(f"[{symbol}] scope: " + ", ".join(f"{k} {v}" for k, v in sorted(scopes.items())))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--end", default=FROZEN_END,
                        help=f"граница истории (по умолчанию {FROZEN_END})")
    parser.add_argument("--out-dir", type=Path,
                        default=config.sink.btc_graph_path / "dumps",
                        help="каталог для JSONL (по умолчанию dumps/ в btc-graph)")
    parser.add_argument("--suffix", default="",
                        help="суффикс имени файла, чтобы не затирать прежнюю "
                             "выгрузку (например --suffix _v2)")
    args = parser.parse_args()

    specs = symbols.resolve_many(args.symbol, args.all)
    print(f"Выгрузка {len(specs)} монет(ы) в {args.out_dir}, граница {args.end}.")
    print(f"Пороги выборки: min_sample_size={config.candidates.min_sample_size}, "
          f"min_effective_sample_size={config.candidates.min_effective_sample_size}")
    for spec in specs:
        try:
            dump_symbol(spec.ticker, args.end, args.out_dir, args.suffix)
        except replay.ReplayError as exc:
            print(f"[{spec.ticker}] пропущена: {exc}")


if __name__ == "__main__":
    main()
