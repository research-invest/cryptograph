"""
Celery задачи:
- evaluate_candidate   — оценка одного кандидата (с retry)
- process_stream_batch — читает Redis Stream и диспатчит evaluate_candidate
- refresh_continuous_aggregates — принудительный refresh TimescaleDB aggregates
"""
from __future__ import annotations

from celery.utils.log import get_task_logger
from pydantic import ValidationError

from src.worker.celery_app import celery_app

logger = get_task_logger(__name__)


@celery_app.task(
    name="src.worker.tasks.evaluate_candidate",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
)
def evaluate_candidate(self, raw_payload: dict, use_llm: bool = True) -> dict:
    """
    Оценивает одного кандидата и сохраняет результат в БД.

    Повторяет до 3 раз с интервалом 30 сек — но только для сбоев, которые
    могут пройти сами (сеть, недоступная БД). Невалидный payload детерминирован:
    ретраить его бессмысленно, поэтому задача сразу завершается с ошибкой.
    """
    try:
        from src.agent.pipeline import run_pipeline
        evaluation = run_pipeline(raw_payload, use_llm=use_llm, save=True)
        logger.info(
            "Evaluated candidate %s → %s (score=%.3f)",
            evaluation.candidate_id,
            evaluation.rating,
            evaluation.quality_score,
        )
        return evaluation.model_dump()
    except (ValidationError, ValueError, TypeError) as exc:
        logger.error(
            "Невалидный кандидат — задача отброшена без повтора: %s", exc
        )
        return {"error": "invalid_candidate", "detail": str(exc)}
    except Exception as exc:
        logger.warning("Evaluate failed for payload, retrying: %s", exc)
        raise self.retry(exc=exc)


@celery_app.task(name="src.worker.tasks.process_stream_batch")
def process_stream_batch(batch_size: int = 50) -> dict:
    """
    Читает до batch_size сообщений из Redis Stream candidates:stream
    и диспатчит задачу evaluate_candidate для каждого.
    Запускается по расписанию каждые 10 секунд через Celery Beat.

    Стрим общий на все монеты: символ лежит в payload, и профиль калибровки
    резолвится уже внутри pipeline.
    """
    try:
        from src.cache.redis_cache import read_pending_candidates, ack_candidate

        messages = read_pending_candidates(count=batch_size)
        if not messages:
            return {"dispatched": 0}

        dispatched = 0
        for msg in messages:
            evaluate_candidate.apply_async(
                args=[msg["payload"]],
                queue="evaluate",
            )
            ack_candidate(msg["id"])
            dispatched += 1

        logger.info("Dispatched %d candidates from stream", dispatched)
        return {"dispatched": dispatched}
    except Exception as exc:
        logger.error("Stream batch processing failed: %s", exc)
        return {"dispatched": 0, "error": str(exc)}


@celery_app.task(name="src.worker.tasks.refresh_continuous_aggregates")
def refresh_continuous_aggregates() -> dict:
    """
    Принудительно обновляет TimescaleDB continuous aggregates.
    Запускается ежечасно через Celery Beat (на случай если auto-policy не успевает).
    """
    try:
        from src.db.connection import engine
        import sqlalchemy as sa

        refreshed = []
        # refresh_continuous_aggregate нельзя выполнять внутри транзакции —
        # TimescaleDB отвергает вызов. Нужен явный AUTOCOMMIT.
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            for view in ("hourly_candidate_stats", "daily_group_stats"):
                conn.execute(
                    sa.text("CALL refresh_continuous_aggregate(:view, NULL, NULL)"),
                    {"view": view},
                )
                refreshed.append(view)

        logger.info("Refreshed continuous aggregates: %s", refreshed)
        return {"refreshed": refreshed}
    except Exception as exc:
        logger.error("Continuous aggregate refresh failed: %s", exc)
        return {"refreshed": [], "error": str(exc)}
