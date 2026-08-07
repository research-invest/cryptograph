"""
Celery application: broker и backend — Redis, воркеры обрабатывают кандидатов асинхронно.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

# Воркер и beat запускаются отдельными процессами — им тоже нужен .env
# при локальном запуске без Docker.
load_dotenv()

from celery import Celery  # noqa: E402
from celery.schedules import crontab  # noqa: E402

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "btc_graph",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["src.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "src.worker.tasks.evaluate_candidate": {"queue": "evaluate"},
        "src.worker.tasks.process_stream_batch": {"queue": "celery"},
    },
    beat_schedule={
        # Каждые 10 секунд читаем Redis Stream и раздаём задачи
        "process-stream-every-10s": {
            "task": "src.worker.tasks.process_stream_batch",
            "schedule": 10.0,
        },
        # Ежечасно принудительно обновляем continuous aggregates
        "refresh-aggregates-hourly": {
            "task": "src.worker.tasks.refresh_continuous_aggregates",
            "schedule": crontab(minute=5),
        },
    },
)
