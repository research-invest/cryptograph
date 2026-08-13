"""
Правила уведомлений: кому слать и что именно.

Правило — это адрес получателя плюс фильтр по кандидату. Оси фильтра те же,
что в селекторе на странице кандидатов (монета, рейтинг, сторона, порог
качества, переход), плюс семейство событий — «тип события» в терминах
детекторов.

Матчинг здесь — чистая функция от кандидата и оценки, без БД и без сети:
именно её проверяют тесты, и именно она решает, уйдёт запрос или нет.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

#: Режимы полезной нагрузки. `full` — кандидат целиком (37 полей схемы) плюс
#: оценка btc-graph; `compact` — только сводка. Формат обоих описан в
#: docs/notifications.md, он же контракт для принимающей системы.
PAYLOAD_MODES = ("full", "compact")

#: Схемы, по которым разрешено слать. Не «безопасность», а защита от опечатки:
#: `localhost:9000` без схемы httpx не примет, и правило молча не работало бы.
ALLOWED_SCHEMES = ("http", "https")


@dataclass(frozen=True)
class Rule:
    """
    Одно правило. Пустой список и None в фильтрах означают «эта ось не
    ограничена» — правило без единого фильтра шлёт всё, что видит.
    """

    rule_id: int
    name: str
    url: str
    enabled: bool = True
    headers: dict[str, str] = field(default_factory=dict)
    payload_mode: str = "full"

    symbol: str | None = None
    ratings: tuple[str, ...] = ()
    directions: tuple[str, ...] = ()
    min_quality: float | None = None
    min_research_score: float | None = None
    min_sample_size: int | None = None
    transitions: tuple[str, ...] = ()
    event_families: tuple[str, ...] = ()
    require_evaluation: bool = True

    # ── Матчинг ─────────────────────────────────────────────────────────────
    def matches(self, candidate: dict, evaluation: dict | None = None) -> bool:
        """
        Подходит ли кандидат под правило.

        `evaluation` — ответ btc-graph (quality_score, rating, direction) либо
        None, если кандидат до него не дошёл: отсеян фильтром, отправка
        выключена или SINK_MODE=none.

        Ключевая тонкость — фильтры, которых без оценки просто нет. Рейтинг и
        quality_score считает btc-graph, а не мы. Правило с такими фильтрами
        на неоценённом кандидате НЕ срабатывает: «рейтинг неизвестен» это не
        «рейтинг подходит». Иначе включённый фильтр «только STRONG» слал бы
        всё подряд ровно в тех прогонах, где приёмник недоступен.
        """
        if not self.enabled:
            return False
        if self.symbol and candidate.get("symbol") != self.symbol:
            return False

        if self.require_evaluation and not evaluation:
            return False

        if self.transitions and candidate.get("transition_id") not in self.transitions:
            return False
        if self.event_families:
            family = candidate.get("primary_event_family")
            if family not in self.event_families:
                return False
        if self.min_research_score is not None:
            if _as_float(candidate.get("research_score")) < self.min_research_score:
                return False
        if self.min_sample_size is not None:
            if int(candidate.get("sample_size") or 0) < self.min_sample_size:
                return False

        # Сторона: у оценённого кандидата берём направление из оценки —
        # btc-graph считает его сам и вправе разойтись с research_side.
        if self.directions:
            side = (evaluation or {}).get("direction") or candidate.get("research_side")
            if side not in self.directions:
                return False

        if self.ratings:
            if not evaluation or evaluation.get("rating") not in self.ratings:
                return False
        if self.min_quality is not None:
            if not evaluation:
                return False
            if _as_float(evaluation.get("quality_score")) < self.min_quality:
                return False

        return True

    # ── Сериализация ────────────────────────────────────────────────────────
    @classmethod
    def from_row(cls, row: dict) -> "Rule":
        return cls(
            rule_id=int(row["rule_id"]),
            name=row["name"],
            url=row["url"],
            enabled=bool(row["enabled"]),
            headers=dict(row.get("headers") or {}),
            payload_mode=row.get("payload_mode") or "full",
            symbol=row.get("symbol"),
            ratings=tuple(row.get("ratings") or ()),
            directions=tuple(row.get("directions") or ()),
            min_quality=_none_or_float(row.get("min_quality")),
            min_research_score=_none_or_float(row.get("min_research_score")),
            min_sample_size=(
                None if row.get("min_sample_size") is None
                else int(row["min_sample_size"])
            ),
            transitions=tuple(row.get("transitions") or ()),
            event_families=tuple(row.get("event_families") or ()),
            require_evaluation=bool(row.get("require_evaluation", True)),
        )


class RuleError(ValueError):
    """Правило не сохранено: форма заполнена так, что оно не сработало бы."""


def validate(name: str, url: str, payload_mode: str) -> None:
    """
    Проверка перед сохранением.

    Смысл её в том, что неверное правило не даёт ошибки НИКОГДА: оно просто
    молча не срабатывает или роняет фоновый поток, о чём оператор узнаёт из
    журнала доставок, если догадается туда посмотреть. Дешевле не дать
    сохранить.
    """
    problems = []
    if not (name or "").strip():
        problems.append("не заполнено имя")
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in ALLOWED_SCHEMES:
        problems.append(
            f"адрес должен начинаться с http:// или https:// (получено {url!r})"
        )
    elif not parsed.netloc:
        problems.append(f"в адресе нет хоста: {url!r}")
    if payload_mode not in PAYLOAD_MODES:
        problems.append(
            f"неизвестный формат payload: {payload_mode!r}, "
            f"допустимы {', '.join(PAYLOAD_MODES)}"
        )
    if problems:
        raise RuleError("; ".join(problems))


# ── Хранение ────────────────────────────────────────────────────────────────
def list_rules(only_enabled: bool = False) -> list[Rule]:
    from btcproc.db.session import fetch_all

    sql = "SELECT * FROM notification_rules"
    if only_enabled:
        sql += " WHERE enabled"
    sql += " ORDER BY rule_id"
    return [Rule.from_row(row) for row in fetch_all(sql)]


def get_rule(rule_id: int) -> Rule | None:
    from btcproc.db.session import fetch_one

    row = fetch_one("SELECT * FROM notification_rules WHERE rule_id = %s", (rule_id,))
    return Rule.from_row(row) if row else None


#: Колонки, которые правит форма. Порядок задаёт порядок плейсхолдеров —
#: держим один список, чтобы INSERT и UPDATE не разъехались.
EDITABLE = (
    "name", "enabled", "url", "headers", "payload_mode", "symbol", "ratings",
    "directions", "min_quality", "min_research_score", "min_sample_size",
    "transitions", "event_families", "require_evaluation",
)


def save_rule(values: dict[str, Any], rule_id: int | None = None) -> int:
    """
    Создаёт правило или обновляет существующее. Возвращает rule_id.

    `values` — уже разобранные значения формы (списки списками, числа
    числами). Разбор строк живёт в админке: это её забота, а не хранилища.
    """
    import psycopg2.extras

    from btcproc.db.session import connect

    validate(values.get("name", ""), values.get("url", ""),
             values.get("payload_mode", "full"))

    payload = dict(values)
    payload["headers"] = psycopg2.extras.Json(values.get("headers") or {})
    row = [payload.get(column) for column in EDITABLE]

    with connect() as conn, conn.cursor() as cur:
        if rule_id is None:
            cur.execute(
                f"INSERT INTO notification_rules ({', '.join(EDITABLE)}) "
                f"VALUES ({', '.join(['%s'] * len(EDITABLE))}) RETURNING rule_id",
                row,
            )
        else:
            setters = ", ".join(f"{c} = %s" for c in EDITABLE)
            cur.execute(
                f"UPDATE notification_rules SET {setters}, updated_at = NOW() "
                f"WHERE rule_id = %s RETURNING rule_id",
                [*row, rule_id],
            )
        result = cur.fetchone()
    if not result:
        raise RuleError(f"Правила #{rule_id} нет")
    return int(result[0])


def delete_rule(rule_id: int) -> None:
    from btcproc.db.session import execute

    # Журнал доставок правила удаляется вместе с ним: без правила он не
    # читается (в админке нечего показать рядом), а место занимает.
    execute("DELETE FROM notification_deliveries WHERE rule_id = %s", (rule_id,))
    execute("DELETE FROM notification_rules WHERE rule_id = %s", (rule_id,))


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _none_or_float(value: Any) -> float | None:
    return None if value is None else float(value)


def clean_list(values: Sequence[str] | None) -> list[str] | None:
    """
    Список фильтра из формы. Пустой список превращается в NULL: «ни одного
    выбранного значения» и «фильтр не задан» для оператора одно и то же, а в
    матчинге пустой кортеж и None ведут себя одинаково.
    """
    cleaned = [v.strip() for v in (values or []) if v and v.strip()]
    return cleaned or None
