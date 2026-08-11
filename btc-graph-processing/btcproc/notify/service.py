"""
Связка «появился кандидат → ушёл вебхук».

Точка входа одна — `dispatch(candidate_ids)`. Она берёт кандидатов из БД по
идентификаторам, потому что источник правды один и тот же для обоих мест
вызова (после оценки в btc-graph и в конце live-прогона), и потому что тело
уведомления обязано совпадать с тем, что записано у нас: расхождение между
«что отправили» и «что сохранили» отлаживается потом неделями.

Три предохранителя, каждый закрывает свой способ выстрелить себе в ногу:

  * **окно свежести** — `train` выпускает сотни тысяч кандидатов на истории
    с 2017 года, и первый же прогон с отправкой разослал бы их все;
  * **захват доставки** — окна `live` намеренно перекрываются, и один
    кандидат попадается нескольким прогонам подряд; PK журнала (правило +
    кандидат) делает повтор невозможным;
  * **отсутствие правил** — если их нет, не делается ни одного запроса к БД.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

from btcproc import config
from btcproc.notify import payload as payload_mod
from btcproc.notify import rules as rules_mod
from btcproc.notify.sender import Job, dispatcher

logger = logging.getLogger(__name__)


def plan(rows: Sequence[dict], rules: Sequence[rules_mod.Rule]) -> list[tuple]:
    """
    Какие пары «правило → кандидат» сработали. Чистая функция: ни БД, ни сети.
    """
    matched = []
    for row in rows:
        candidate = row.get("payload") or {}
        evaluation = row.get("evaluation") or None
        for rule in rules:
            if rule.matches(candidate, evaluation):
                matched.append((rule, row))
    return matched


def dispatch(candidate_ids: Iterable[str], reason: str = "") -> dict[str, Any]:
    """
    Разослать уведомления по кандидатам с этими идентификаторами.

    Возвращает сводку для `runs.stats` — она же видна в логе прогона.
    Ошибок наружу не отдаёт: сбой рассылки не должен ронять прогон, который
    свою работу уже сделал и записал.
    """
    ids = [cid for cid in candidate_ids if cid]
    if not ids:
        return {"matched": 0}
    if not config.notify.enabled:
        return {"matched": 0, "skipped": "NOTIFY_ENABLED=false"}

    try:
        active = rules_mod.list_rules(only_enabled=True)
        if not active:
            return {"matched": 0, "skipped": "нет включённых правил"}

        rows = _load(ids)
        skipped_by_age = len(ids) - len(rows)
        matched = plan(rows, active)
        if not matched:
            return {"matched": 0, "stale": skipped_by_age}

        claimed = _claim(matched)
        queued = 0
        for rule, row in matched:
            if (rule.rule_id, row["candidate_id"]) not in claimed:
                continue  # уже отправляли по этому правилу — молча пропускаем
            body = payload_mod.build(rule, row)
            queued += dispatcher().submit(
                Job(
                    rule_id=rule.rule_id,
                    name=rule.name,
                    url=rule.url,
                    headers=rule.headers,
                    candidate_id=row["candidate_id"],
                    body=body,
                )
            )
        summary = {
            "matched": len(matched),
            "queued": queued,
            "duplicates": len(matched) - len(claimed),
            "stale": skipped_by_age,
        }
        if reason:
            summary["reason"] = reason
        return summary
    except Exception as exc:  # noqa: BLE001 — прогон важнее рассылки
        logger.exception("Рассылка уведомлений сорвалась")
        return {"matched": 0, "error": f"{type(exc).__name__}: {exc}"}


def flush(timeout: float | None = None) -> int:
    """
    Дождаться разбора очереди. Зовётся в конце прогона: потоки-отправители
    демонические и умрут вместе с крон-процессом.
    """
    if not config.notify.enabled:
        return 0
    return dispatcher().flush(timeout)


def _load(candidate_ids: Sequence[str]) -> list[dict]:
    """
    Кандидаты из БД, уже отфильтрованные по возрасту.

    Возраст режется в SQL, а не в Python, именно ради `train`: там список
    идентификаторов бывает в десятки тысяч, и запрос по индексу вернёт из них
    ноль строк, ничего не подняв в память.
    """
    from btcproc.db.session import fetch_all

    return fetch_all(
        "SELECT candidate_id, run_id, symbol, ts, payload, quality_score, rating, "
        "direction, warning_flags, evaluation FROM candidates "
        "WHERE candidate_id = ANY(%s) "
        "  AND ts >= NOW() - (%s * INTERVAL '1 minute') "
        "ORDER BY ts",
        (list(candidate_ids), config.notify.max_candidate_age_minutes),
    )


def _claim(matched: Sequence[tuple]) -> set[tuple[int, str]]:
    """
    Захватывает пары «правило + кандидат» в журнале доставок.

    Возвращает только те, которых там ещё не было: именно они и уходят в
    очередь. `ON CONFLICT DO NOTHING RETURNING` делает захват атомарным — два
    параллельных прогона (разные монеты идут одновременно) не отправят один
    вебхук дважды.
    """
    import psycopg2.extras

    from btcproc.db.session import connect

    rows = [
        (rule.rule_id, row["candidate_id"], row.get("symbol"))
        for rule, row in matched
    ]
    with connect() as conn, conn.cursor() as cur:
        result = psycopg2.extras.execute_values(
            cur,
            "INSERT INTO notification_deliveries (rule_id, candidate_id, symbol) "
            "VALUES %s ON CONFLICT (rule_id, candidate_id) DO NOTHING "
            "RETURNING rule_id, candidate_id",
            rows,
            fetch=True,
        )
    return {(int(r[0]), r[1]) for r in result}


def send_one(rule: rules_mod.Rule, row: dict) -> dict:
    """
    Разовая отправка мимо очереди и мимо журнала — кнопка «проверить» в
    админке. Ждёт ответа синхронно: оператор нажал и смотрит на результат,
    здесь ожидание и есть смысл действия.
    """
    from btcproc.notify import sender

    job = Job(
        rule_id=rule.rule_id, name=rule.name, url=rule.url, headers=rule.headers,
        candidate_id=row.get("candidate_id", "test"),
        body=payload_mod.build(rule, row),
    )
    result = sender.http_send(job)
    return {"status": result.status, "http_status": result.http_status,
            "error": result.error}
