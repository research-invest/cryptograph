"""
Еженедельное обслуживание базы.

Скрипт ходит по крону без присмотра, поэтому проверяем ровно то, что при
ошибке будет стоить дорого и не будет замечено: что он не лезет в базу
поперёк идущего расчёта и что VACUUM FULL зовётся по делу, а не каждую неделю
(он берёт ACCESS EXCLUSIVE и на время перезаписи блокирует таблицу целиком).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def maintenance():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "maintenance", ROOT / "scripts" / "maintenance.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quiet(monkeypatch, maintenance, **over):
    """Все обращения к БД заменены; остаётся чистая логика решений."""
    calls = {"vacuum": [], "pruned": 0}

    monkeypatch.setattr(maintenance, "report", lambda: None)
    monkeypatch.setattr(
        maintenance, "table_stats",
        lambda t: {"bytes": 1, "pretty": "1 MB",
                   "live": over.get("live", 100), "dead": over.get("dead", 0)},
    )
    monkeypatch.setattr(
        maintenance, "prune_live_states",
        lambda keep, dry: calls.__setitem__("pruned", over.get("removed", 0))
        or over.get("removed", 0),
    )
    monkeypatch.setattr(
        maintenance, "vacuum",
        lambda table, full: calls["vacuum"].append((table, full)),
    )
    # Чистка журнала доставок уведомлений — тоже поход в БД, и к решениям
    # про вакуум она отношения не имеет.
    monkeypatch.setattr(
        maintenance, "prune_deliveries",
        lambda dry: calls.__setitem__("deliveries", dry) or 0,
    )
    monkeypatch.setattr(
        maintenance.runs_repo, "active_runs", lambda: over.get("active", [])
    )
    monkeypatch.setattr(maintenance.runs_repo, "reap_if_stale", lambda run: False)
    monkeypatch.setattr(sys, "argv", ["maintenance.py", *over.get("argv", [])])
    return calls


def test_skips_while_a_run_is_active(monkeypatch, maintenance):
    """Обслуживание не должно соревноваться за диск с расчётом."""
    calls = _quiet(monkeypatch, maintenance,
                   active=[{"run_id": 5, "symbol": "BTCUSDT"}])

    assert maintenance.main() == 0
    assert calls["vacuum"] == [], "при идущем прогоне вакуум не запускается"


def test_dead_run_does_not_block_maintenance(monkeypatch, maintenance):
    """Убитый прогон висит в running навсегда — он не повод не обслуживаться."""
    calls = _quiet(monkeypatch, maintenance,
                   active=[{"run_id": 5, "symbol": "BTCUSDT"}])
    monkeypatch.setattr(maintenance.runs_repo, "reap_if_stale", lambda run: True)

    assert maintenance.main() == 0
    assert calls["vacuum"] == [("bar_states", False)]


def test_force_ignores_active_runs(monkeypatch, maintenance):
    calls = _quiet(monkeypatch, maintenance,
                   active=[{"run_id": 5, "symbol": "BTCUSDT"}], argv=["--force"])

    assert maintenance.main() == 0
    assert calls["vacuum"], "с --force обслуживание идёт несмотря на прогон"


def test_plain_vacuum_when_nothing_was_deleted(monkeypatch, maintenance):
    calls = _quiet(monkeypatch, maintenance, removed=0, dead=0)

    maintenance.main()

    assert calls["vacuum"] == [("bar_states", False)]


def test_full_vacuum_after_a_big_cleanup(monkeypatch, maintenance):
    calls = _quiet(monkeypatch, maintenance,
                   removed=maintenance.FULL_AFTER_DELETED_ROWS)

    maintenance.main()

    assert calls["vacuum"] == [("bar_states", True)]


def test_full_vacuum_when_file_is_bloated_from_before(monkeypatch, maintenance):
    """
    Раздутость, оставшаяся с прошлых прогонов, невидима для двух других
    порогов: autovacuum обнуляет счётчик мёртвых строк, а место ОС не
    возвращает. Таблица остаётся большим файлом с редкими живыми строками.
    """
    calls = _quiet(monkeypatch, maintenance, removed=0, dead=0)
    monkeypatch.setattr(
        maintenance, "table_stats",
        lambda t: {"bytes": maintenance.FULL_BYTES_PER_ROW * 1000,
                   "pretty": "1 MB", "live": 1000, "dead": 0},
    )

    maintenance.main()

    assert calls["vacuum"] == [("bar_states", True)]


def test_healthy_table_is_not_rewritten(monkeypatch, maintenance):
    """Здоровая таблица не должна еженедельно блокироваться под FULL."""
    calls = _quiet(monkeypatch, maintenance, removed=0, dead=0)
    monkeypatch.setattr(
        maintenance, "table_stats",
        lambda t: {"bytes": 200 * 1000, "pretty": "1 MB", "live": 1000, "dead": 0},
    )

    maintenance.main()

    assert calls["vacuum"] == [("bar_states", False)]


def test_bytes_per_row_handles_empty_table(maintenance):
    assert maintenance.bytes_per_row({"bytes": 8192, "live": 0}) == 0.0


def test_full_vacuum_when_bloated(monkeypatch, maintenance):
    """Раздутость накапливается и без наших удалений."""
    calls = _quiet(monkeypatch, maintenance, removed=0, live=50, dead=50)

    maintenance.main()

    assert calls["vacuum"] == [("bar_states", True)]


def test_dry_run_changes_nothing(monkeypatch, maintenance):
    calls = _quiet(monkeypatch, maintenance, removed=0, argv=["--dry-run"])

    assert maintenance.main() == 0
    assert calls["vacuum"] == []


def test_failed_vacuum_is_reported_not_swallowed(monkeypatch, maintenance):
    """Сорвавшийся вакуум обязан дать ненулевой код возврата — иначе в кроне
    он неотличим от успешного."""
    _quiet(monkeypatch, maintenance, removed=0)

    def boom(table, full):
        raise RuntimeError("could not resize shared memory segment")

    monkeypatch.setattr(maintenance, "vacuum", boom)

    assert maintenance.main() == 1


def test_dead_ratio_handles_empty_table(maintenance):
    assert maintenance.dead_ratio({"live": 0, "dead": 0}) == 0.0


def test_prune_covers_disabled_symbols(monkeypatch, maintenance):
    """
    У выключенной монеты новых прогонов нет, но накопленная разметка старых
    live никуда не делась — уборка обязана обходить весь реестр, а не только
    активные монеты.
    """
    seen: list[str] = []
    monkeypatch.setattr(
        maintenance.symbols, "tickers",
        lambda only_enabled=False: (
            ["BTCUSDT"] if only_enabled else ["BTCUSDT", "OLDUSDT"]
        ),
    )
    monkeypatch.setattr(
        maintenance, "fat_live_runs", lambda symbol, keep: seen.append(symbol) or []
    )

    maintenance.prune_live_states(4, dry_run=True)

    assert seen == ["BTCUSDT", "OLDUSDT"], "выключенная монета пропущена"
