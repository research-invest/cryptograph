"""
Размах в интерфейсе: единственная измеренно работающая величина проекта.

Регрессор размаха прошёл критерий на отложенной части (журнал генератора,
разделы 49–50) и с 2026-08-19 считается в конвейере — но до 2026-08-24 его
числа не встречались в админке ни разу. Здесь закреплено то, что при этом
легко сломать обратно.

Три правила, каждое своим тестом:

1. **наружу идёт `range_lift`, а не абсолютные квантили.** Абсолютные
   наполовину объясняются часом дня и недавней волатильностью (правило
   корневого CLAUDE.md), то есть в таблице выглядели бы содержательнее, чем
   являются;
2. **пустое значение подписано словами.** Кандидаты `train` полей размаха не
   несут по построению (инвариант 13а), у монет без калиброванной модели их
   нет тоже. Прочерк без объяснения читается как поломка;
3. **колонки берутся в SQL, а не разворачиваются из JSONB в шаблоне** —
   иначе отсутствующий ключ молча превращается в пустую строку вместо
   честного NULL.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from btcproc.admin import queries


class Row(dict):
    __getattr__ = dict.get


def _env() -> Environment:
    root = Path(__file__).resolve().parents[1] / "btcproc" / "admin" / "templates"
    return Environment(loader=FileSystemLoader(str(root)))


def _row(**overrides) -> Row:
    now = dt.datetime(2026, 8, 24, 12, 0, tzinfo=dt.timezone.utc)
    base = dict(
        candidate_id="c1", symbol="BTCUSDT", run_id=7, ts=now,
        transition_id="12->7", event_block_id="ab12", research_side="long",
        research_score=0.62, sample_size=1400, quality_score=0.71,
        rating="MODERATE", warning_flags=[], emitted_at=now, emit_error=None,
        range_lift=1.31, range_regime="expanded",
    )
    base.update(overrides)
    return Row(base)


def _render_candidates(**overrides) -> str:
    data = {"rows": [_row(**overrides)], "total": 1, "page": 1, "pages": 1}
    return _env().get_template("partials/candidates_table.html").render(
        data=data, filters={}
    )


def test_candidates_table_shows_relative_range_not_absolute():
    """×1.31 — это `range_lift`. Абсолютных квантилей в таблице быть не должно."""
    out = _render_candidates()
    assert "×1.31" in out
    assert "expanded" in out
    assert "expected_range_ratio" not in out, (
        "абсолютные квантили в таблице выглядят содержательнее, чем являются"
    )


def test_missing_range_is_explained_not_just_dashed():
    """
    Пустой размах — штатное состояние, а не ошибка.

    У кандидата из `train` полей размаха нет по построению: модель училась в
    том числе на будущем этого бара. Прочерк без подписи оператор читает как
    сломанный расчёт и идёт чинить работающее.
    """
    out = _render_candidates(range_lift=None, range_regime=None)
    assert "—" in out
    assert "train" in out, "причина пустоты обязана быть в подсказке"


def test_detail_card_keeps_absolute_quantiles_with_a_caveat():
    """
    В карточке абсолютные числа показываются — но второй карточкой и с
    оговоркой. Это единственное место, где они уместны: там есть куда написать
    предложение, а в строке таблицы — нет.
    """
    row = _row()
    row["payload"] = {
        "range_lift": 1.31, "range_regime": "expanded",
        "expected_range_ratio_p50": 1.12, "expected_range_ratio_p90": 2.34,
    }
    row["evaluation"] = None
    out = _env().get_template("candidate_detail.html").render(row=row)
    assert "×1.31" in out
    assert "1.12" in out and "2.34" in out
    assert "часом дня" in out, "абсолютные числа без оговорки вводят в заблуждение"


def test_detail_card_survives_a_candidate_without_range():
    """Кандидат без размаха не должен ронять карточку — их большинство."""
    row = _row(range_lift=None, range_regime=None)
    row["payload"] = {"range_lift": None, "range_regime": None}
    row["evaluation"] = None
    out = _env().get_template("candidate_detail.html").render(row=row)
    assert "запоминанием" in out


def test_range_columns_come_from_sql(monkeypatch):
    """
    Оба списка кандидатов обязаны тянуть размах запросом.

    Разворачивать `payload` в шаблоне нельзя: отсутствующий ключ там молча
    станет пустой строкой, и «модель ничего не сказала» будет неотличимо от
    «поле потеряли».
    """
    seen: list[str] = []

    def fake_fetch_all(sql, params=None):
        seen.append(" ".join(sql.split()))
        return []

    def fake_fetch_one(sql, params=None):
        return {"n": 0}

    monkeypatch.setattr(queries, "fetch_all", fake_fetch_all)
    monkeypatch.setattr(queries, "fetch_one", fake_fetch_one)

    queries.candidates_page()
    queries.recent_highlights()

    assert len(seen) == 2
    for sql in seen:
        assert "range_lift" in sql and "range_regime" in sql
