"""
Redis: кэш, дедупликация по configuration_hash, pub/sub для STRONG-кандидатов.

Все ключи, привязанные к кандидату, живут под префиксом монеты. Причина —
дедупликация: `configuration_hash` считается генератором по конфигурации графа
и (до соответствующей правки на его стороне) символа не содержит. Совпадение
хэша между монетами в пределах TTL означало бы, что ETH получает готовую
оценку BTC. Диагностировать это по данным почти невозможно: кандидат выглядит
нормально оценённым.

Стрим приёма остаётся ОДИН на все монеты — символ уже лежит в payload,
плодить consumer-группы незачем.
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

# Канал на монету + общий канал для подписчиков, которым нужны все монеты.
STRONG_CHANNEL_ALL = "candidates:strong:all"


def strong_channel(symbol: str) -> str:
    return f"candidates:strong:{_key_symbol(symbol)}"


def _key_symbol(symbol: str | None) -> str:
    """Нормализованный символ для ключа. Пусто → UNKNOWN, а не тихо BTCUSDT."""
    return (symbol or "").strip().upper() or "UNKNOWN"


def get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(
            os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
        )
    return _client


# --- Дедупликация ---

def is_hash_cached(symbol: str, configuration_hash: str) -> str | None:
    """Возвращает candidate_id если хэш этой монеты уже в кэше, иначе None."""
    try:
        return get_client().get(f"{_key_symbol(symbol)}:candidate:hash:{configuration_hash}")
    except redis.RedisError:
        logger.warning("Redis недоступен при проверке дедупликации", exc_info=True)
        return None


def mark_hash_cached(symbol: str, configuration_hash: str, candidate_id: str) -> bool:
    """Помечает хэш как обработанный с TTL 30 мин. Возвращает успех записи."""
    try:
        get_client().setex(
            f"{_key_symbol(symbol)}:candidate:hash:{configuration_hash}",
            DEDUP_TTL, candidate_id,
        )
        return True
    except redis.RedisError:
        logger.warning("Не удалось пометить хэш %s в Redis", configuration_hash, exc_info=True)
        return False


# --- Кэш оценок ---

def cache_evaluation(symbol: str, candidate_id: str, evaluation_json: str) -> bool:
    """Кладёт оценку в кэш с TTL. Возвращает успех записи."""
    try:
        get_client().setex(
            f"{_key_symbol(symbol)}:evaluation:{candidate_id}", CACHE_TTL, evaluation_json
        )
        return True
    except redis.RedisError:
        logger.warning("Не удалось закэшировать оценку %s", candidate_id, exc_info=True)
        return False


def get_cached_evaluation(symbol: str, candidate_id: str) -> str | None:
    try:
        return get_client().get(f"{_key_symbol(symbol)}:evaluation:{candidate_id}")
    except redis.RedisError:
        logger.warning("Redis недоступен при чтении кэша оценки %s", candidate_id, exc_info=True)
        return None


# --- Pub/Sub для STRONG кандидатов ---

def publish_strong_candidate(evaluation_dict: dict) -> bool:
    """
    Публикует STRONG-кандидата в канал своей монеты и в общий канал.

    Дублирование намеренное: подписчику одной монеты незачем фильтровать чужой
    поток, а дашборду «что сегодня сильного вообще» незачем подписываться
    на каждую монету по отдельности.
    """
    payload = json.dumps(evaluation_dict, ensure_ascii=False)
    symbol = evaluation_dict.get("symbol")
    try:
        client = get_client()
        client.publish(strong_channel(symbol), payload)
        client.publish(STRONG_CHANNEL_ALL, payload)
        return True
    except redis.RedisError:
        logger.warning("Не удалось опубликовать STRONG-кандидата", exc_info=True)
        return False


def subscribe_strong_candidates(symbol: str | None = None):
    """
    Генератор для подписки на STRONG-кандидатов.

    symbol=None — все монеты (общий канал).
    Использование:
        for msg in subscribe_strong_candidates("ETHUSDT"):
            data = json.loads(msg['data'])
    """
    channel = strong_channel(symbol) if symbol else STRONG_CHANNEL_ALL
    pubsub = get_client().pubsub()
    pubsub.subscribe(channel)
    for message in pubsub.listen():
        if message["type"] == "message":
            yield message


# --- Очередь задач через Redis Streams ---
#
# Стрим общий на все монеты: символ лежит в payload, отдельные consumer-группы
# на монету только раздробили бы пул воркеров. Имена сменились вместе с
# ключами (btc:* → candidates:*) — переименование безопасно только при пустой
# очереди, проверь XLEN перед деплоем (см. docs/step_04_multi_symbol.md).

STREAM_KEY = "candidates:stream"
STREAM_GROUP = "candidates:workers"
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


# Сколько сообщение может числиться выданным, но не подтверждённым, прежде чем
# считать читателя мёртвым. Диспетчер ACK-ает сразу после apply_async, то есть
# нормальный цикл занимает миллисекунды; минута — заведомо больше него и
# заведомо меньше того, чтобы кандидат протух.
STALE_PENDING_MS = 60_000


def _parse_entries(entries) -> list[dict]:
    """Пары (id, fields) из стрима → список {"id", "payload"}."""
    result = []
    for msg_id, fields in entries:
        try:
            payload = json.loads(fields["payload"])
        except (KeyError, ValueError, TypeError):
            logger.error("Битое сообщение %s в стриме — удаляем", msg_id)
            ack_candidate(msg_id)
            continue
        result.append({"id": msg_id, "payload": payload})
    return result


def reclaim_stale_candidates(count: int = 10) -> list[dict]:
    """
    Забирает сообщения, зависшие в Pending Entries List.

    XREADGROUP с «>» отдаёт только НОВЫЕ сообщения. Всё, что группа уже выдала,
    но не получила XACK, остаётся в PEL и через «>» не придёт больше никогда.
    А выдача и подтверждение у диспетчера разнесены: он читает, ставит задачу
    в очередь Celery и только потом ACK-ает. Падение процесса между этими
    шагами — и кандидаты теряются молча, ни в одной очереди их уже нет.

    XAUTOCLAIM переназначает такие сообщения тому же консюмеру: повторный
    диспатч безопаснее потери, оценка кандидата идемпотентна (candidate_id
    детерминирован, запись идёт upsert'ом).
    """
    try:
        _ensure_group()
        response = get_client().xautoclaim(
            STREAM_KEY, STREAM_GROUP, STREAM_CONSUMER,
            min_idle_time=STALE_PENDING_MS, start_id="0-0", count=count,
        )
        # Ответ — (next_start_id, entries) в redis-py 4.x и
        # (next_start_id, entries, deleted) в 5.x/Redis 7.
        entries = response[1] if isinstance(response, (tuple, list)) and len(response) > 1 else []
        reclaimed = _parse_entries(entries)
        if reclaimed:
            logger.warning(
                "Возвращено %d зависших сообщений из PEL — предыдущий "
                "диспетчер умер между чтением и подтверждением",
                len(reclaimed),
            )
        return reclaimed
    except redis.RedisError:
        logger.warning("Не удалось забрать зависшие сообщения", exc_info=True)
        return []
    except AttributeError:
        # Очень старый клиент без xautoclaim: не повод ронять диспетчер.
        logger.warning("Клиент Redis не умеет XAUTOCLAIM — PEL не разбирается")
        return []


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
            result.extend(_parse_entries(entries))
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
