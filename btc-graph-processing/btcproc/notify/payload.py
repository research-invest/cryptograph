"""
Формат JSON, который уходит получателю.

Это внешний контракт: его читает чужая система, которую мы не контролируем и
не деплоим вместе с собой. Отсюда два правила.

1. **Версия в теле.** Поле `schema` меняется при несовместимой правке формата.
   Получатель обязан уметь на неё смотреть — иначе любая наша правка тихо
   ломает его разбор.
2. **Сводка наверху, подробности внутри.** Всё, ради чего уведомление
   существует (монета, время, сторона, рейтинг, оценка), лежит в корне
   объекта. Полный кандидат и полный ответ btc-graph — во вложенных блоках,
   и получателю не обязательно в них лезть.

Человеческое описание формата — docs/notifications.md.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

#: Версия формата. Растёт только при НЕСОВМЕСТИМОЙ правке: добавление нового
#: поля совместимо (получатель его игнорирует), переименование и удаление —
#: нет.
SCHEMA = "btcproc.candidate.v1"

#: Поля сводки, которые лежат в корне в обоих режимах payload. Вынесены
#: списком, чтобы docs/notifications.md и тест сверялись с одним источником.
SUMMARY_FIELDS = (
    "candidate_id", "symbol", "ts", "side", "rating", "quality_score",
    "research_score", "sample_size", "transition_id", "event_block_id",
    "primary_event_family", "horizon", "run_id",
)


def build(rule, row: dict, sent_at: datetime | None = None) -> dict:
    """
    Тело запроса по одному кандидату.

    `row` — строка таблицы candidates (payload кандидата + колонки оценки).
    Функция чистая: ни БД, ни сети, ни глобального состояния — её и проверяет
    тест на соответствие документированному формату.
    """
    candidate: dict[str, Any] = dict(row.get("payload") or {})
    evaluation = row.get("evaluation") or None

    body = {
        "schema": SCHEMA,
        "event": "candidate",
        "sent_at": _iso(sent_at or datetime.now(timezone.utc)),
        "rule": {"id": rule.rule_id, "name": rule.name},

        "candidate_id": row.get("candidate_id") or candidate.get("candidate_id"),
        "symbol": row.get("symbol") or candidate.get("symbol"),
        # Время БАРА, а не отправки: уведомление может уйти на минуты позже,
        # и путать эти два момента получателю нельзя.
        "ts": _iso(row.get("ts")),
        # Сторона из оценки, если она есть: btc-graph считает направление сам.
        "side": row.get("direction") or candidate.get("research_side"),
        "rating": row.get("rating"),
        "quality_score": _float_or_none(row.get("quality_score")),
        "research_score": _float_or_none(candidate.get("research_score")),
        "sample_size": candidate.get("sample_size"),
        "transition_id": candidate.get("transition_id"),
        "event_block_id": candidate.get("event_block_id"),
        "primary_event_family": candidate.get("primary_event_family"),
        "horizon": candidate.get("horizon"),
        "run_id": row.get("run_id"),
        "warning_flags": list(row.get("warning_flags") or []),
        # Кандидат — исследовательская идея, а не сигнал: ни входа, ни стопа,
        # ни размера позиции. Флаг стоит в теле, чтобы принимающая система не
        # могла принять его за торговую команду по недосмотру.
        "is_trading_signal": False,
    }

    if rule.payload_mode == "compact":
        return body

    # full: кандидат целиком (те же 37 полей, что уходят в btc-graph) и его
    # оценка целиком — включая summary, strengths и risks, если приёмник их
    # считал.
    body["candidate"] = candidate
    body["evaluation"] = evaluation
    return body


def example_row() -> dict:
    """
    Показательная строка кандидата — ровно та, из которой собран пример в
    docs/notifications.md.

    Живёт в коде, а не в документации, по той же причине, по какой имена
    состояний считаются, а не выписываются руками: пример в markdown протухает
    молча. Отсюда его печатает `btcproc.cli notify-example`, им же пользуется
    кнопка «проверить» в админке, когда реальных кандидатов ещё нет, и на нём
    же стоит тест формата.
    """
    return {
        "candidate_id": "3f2a1c9b8e7d6a5c4b30",
        "run_id": 1287,
        "symbol": "BTCUSDT",
        "ts": datetime(2026, 8, 11, 9, 45, tzinfo=timezone.utc),
        "quality_score": 0.71,
        "rating": "STRONG",
        "direction": "long",
        "warning_flags": ["small_sample"],
        "payload": {
            "candidate_id": "3f2a1c9b8e7d6a5c4b30",
            "symbol": "BTCUSDT",
            "configuration_hash": "9c1d4e77a0b3f215",
            "candidate_family_key": "BTCUSDT|7|42->7|b19f0c2a|long_skew",
            "research_score": 0.63,
            "previous_group_id": 42.0,
            "current_group_id": 7.0,
            "transition_id": "42->7",
            "current_group_age_bucket": "age_30_60",
            "context_status": "fresh",
            "trajectory_entropy": "low",
            "transition_rarity": "uncommon",
            "event_block_id": "b19f0c2a",
            "primary_event_family": "volatility_events",
            "event_intensity_bucket": "moderate",
            "event_rarity_bucket": "uncommon",
            "signature_atom_count": 3,
            "event_family_count": 2,
            "event_block_total_rows": 1840,
            "event_block_row_share": 0.006,
            "horizon": "24h",
            "sample_size": 412,
            "valid_label_count": 401,
            "invalid_label_count": 11,
            "valid_label_pct": 0.973,
            "repeatability_days": 288,
            "repeatability_months": 31,
            "monthly_concentration": 0.09,
            "historical_bias_context": "long_skew",
            "research_side": "long",
            "long_outcome_count": 245,
            "short_outcome_count": 156,
            "long_outcome_share": 0.611,
            "historical_outcome_skew": 0.222,
            "p70_long_favorable_pct": 1.84,
            "p80_long_adverse_pct": 0.92,
            "long_favorable_adverse_ratio_p70_p80": 2.0,
        },
        "evaluation": {
            "candidate_id": "3f2a1c9b8e7d6a5c4b30",
            "symbol": "BTCUSDT",
            "scoring_profile": "btcusdt_v1",
            "profile_fingerprint": "a41f0c93",
            "quality_score": 0.71,
            "quality_score_baseline": 0.68,
            "rating": "STRONG",
            "direction": "long",
            "win_rate": 0.611,
            "favorable_adverse_ratio": 2.0,
            "context_freshness": "fresh",
            "warning_flags": ["small_sample"],
            "strengths": ["устойчивая выборка", "редкий переход"],
            "risks": ["перекос по месяцам"],
            "summary": "Историческая аналогия с перекосом вверх на горизонте 24h.",
            "score_statistical": 0.74,
            "score_directional": 0.66,
            "score_context": 0.80,
            "score_rarity": 0.59,
        },
    }


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat()
    return str(value)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
