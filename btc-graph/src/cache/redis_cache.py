"""
Redis: кэш, дедупликация по configuration_hash, pub/sub для STRONG-кандидатов.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import redis

logger = logging.getLogger(__name__)

_client: Optional[redis.Redis] = None

DEDUP_TTL = 1800         # 30 мин — TTL дедупликации по хэшу
CACHE_TTL = 1800         # 30 мин — TTL кэша последних оценок
STRONG_CHANNEL = "btc:strong_candidates"


def get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(
            os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
        )
    return _client


# --- Дедупликация ---

def is_hash_cached(configuration_hash: str) -> str | None:
    """Возвращает candidate_id если хэш уже в кэше, иначе None."""
    try:
        return get_client().get(f"candidate:hash:{configuration_hash}")
    except redis.RedisError:
        logger.warning("Redis недоступен при проверке дедупликации", exc_info=True)
        return None


def mark_hash_cached(configuration_hash: str, candidate_id: str) -> bool:
    """Помечает хэш как обработанный с TTL 30 мин. Возвращает успех записи."""
    try:
        get_client().setex(f"candidate:hash:{configuration_hash}", DEDUP_TTL, candidate_id)
        return True
    except redis.RedisError:
        logger.warning("Не удалось пометить хэш %s в Redis", configuration_hash, exc_info=True)
        return False


# --- Кэш оценок ---

def cache_evaluation(candidate_id: str, evaluation_json: str) -> bool:
    """Кладёт оценку в кэш с TTL. Возвращает успех записи."""
    try:
        get_client().setex(f"evaluation:{candidate_id}", CACHE_TTL, evaluation_json)
        return True
    except redis.RedisError:
        logger.warning("Не удалось закэшировать оценку %s", candidate_id, exc_info=True)
        return False


def get_cached_evaluation(candidate_id: str) -> str | None:
    try:
        return get_client().get(f"evaluation:{candidate_id}")
    except redis.RedisError:
        logger.warning("Redis недоступен при чтении кэша оценки %s", candidate_id, exc_info=True)
        return None


# --- Pub/Sub для STRONG кандидатов ---

def publish_strong_candidate(evaluation_dict: dict) -> bool:
    """Публикует STRONG-кандидата в Redis канал для downstream подписчиков."""
    try:
        get_client().publish(STRONG_CHANNEL, json.dumps(evaluation_dict, ensure_ascii=False))
        return True
    except redis.RedisError:
        logger.warning("Не удалось опубликовать STRONG-кандидата", exc_info=True)
        return False


def subscribe_strong_candidates():
    """
    Генератор для подписки на STRONG-кандидатов.
    Использование:
        for msg in subscribe_strong_candidates():
            data = json.loads(msg['data'])
    """
    pubsub = get_client().pubsub()
    pubsub.subscribe(STRONG_CHANNEL)
    for message in pubsub.listen():
        if message["type"] == "message":
            yield message


# --- Очередь задач через Redis Streams ---

STREAM_KEY = "btc:candidates:stream"
STREAM_GROUP = "btc:candidates:workers"
STREAM_CONSUMER = "dispatcher"


def enqueue_candidate(raw_payload: dict) -> str:
    """
    Добавляет кандидата в Redis Stream для асинхронной обработки.

    Возвращает id сообщения или "" если Redis недоступен — вызывающий обязан
    проверить результат, иначе кандидат потеряется молча.
    """
    try:
        return get_client().xadd(STREAM_KEY, {"payload": json.dumps(raw_payload)})
    except redis.RedisError:
        logger.warning("Не удалось поставить кандидата в очередь", exc_info=True)
        return ""


def _ensure_group() -> None:
    """Создаёт consumer group, если её ещё нет (идемпотентно)."""
    try:
        get_client().xgroup_create(STREAM_KEY, STREAM_GROUP, id="0", mkstream=True)
    except redis.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


def read_pending_candidates(count: int = 10, block_ms: int = 0) -> list[dict]:
    """
    Читает до count кандидатов из стрима через consumer group.

    Consumer group выдаёт каждое сообщение ровно одному читателю, поэтому
    параллельные диспетчеры не разберут один и тот же payload дважды — раньше
    чтение шло с «0» без группы (docs/audit_findings.md, №13).
    """
    try:
        _ensure_group()
        messages = get_client().xreadgroup(
            STREAM_GROUP, STREAM_CONSUMER, {STREAM_KEY: ">"},
            count=count, block=block_ms or None,
        )
        if not messages:
            return []
        result = []
        for _, entries in messages:
            for msg_id, fields in entries:
                try:
                    payload = json.loads(fields["payload"])
                except (KeyError, ValueError):
                    logger.error("Битое сообщение %s в стриме — удаляем", msg_id)
                    ack_candidate(msg_id)
                    continue
                result.append({"id": msg_id, "payload": payload})
        return result
    except redis.RedisError:
        logger.warning("Не удалось прочитать очередь кандидатов", exc_info=True)
        return []


def ack_candidate(msg_id: str) -> bool:
    """Подтверждает обработку сообщения: XACK в группе + удаление из стрима."""
    try:
        client = get_client()
        client.xack(STREAM_KEY, STREAM_GROUP, msg_id)
        client.xdel(STREAM_KEY, msg_id)
        return True
    except redis.RedisError:
        logger.warning("Не удалось подтвердить сообщение %s", msg_id, exc_info=True)
        return False
