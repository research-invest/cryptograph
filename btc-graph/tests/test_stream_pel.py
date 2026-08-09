"""
Разбор Pending Entries List стрима кандидатов.

Механика Redis Streams: XREADGROUP с «>» отдаёт только новые сообщения.
Всё, что группа уже выдала, но не получила XACK, оседает в PEL, и через «>»
не придёт больше никогда. Диспетчер подтверждает сообщение ПОСЛЕ apply_async,
поэтому падение процесса между чтением и подтверждением роняло кандидатов
в PEL молча: ни в Celery, ни в стриме их уже нет.
"""
from __future__ import annotations

import json

import pytest

from src.cache import redis_cache


class FakeRedis:
    """Ровно та часть протокола, что нужна разбору PEL."""

    def __init__(self, pending=None, fresh=None, autoclaim_extra=False):
        self.pending = pending or []
        self.fresh = fresh or []
        self.acked: list[str] = []
        self.deleted: list[str] = []
        self.autoclaim_calls: list[dict] = []
        self.autoclaim_extra = autoclaim_extra

    def xgroup_create(self, *a, **kw):
        return True

    def xautoclaim(self, key, group, consumer, min_idle_time, start_id, count=None):
        self.autoclaim_calls.append(
            {"consumer": consumer, "min_idle_time": min_idle_time, "start_id": start_id}
        )
        entries = self.pending
        # redis-py 5.x / Redis 7 добавляют третий элемент — список удалённых id.
        return ("0-0", entries, []) if self.autoclaim_extra else ("0-0", entries)

    def xreadgroup(self, group, consumer, streams, count=None, block=None):
        return [(redis_cache.STREAM_KEY, self.fresh)] if self.fresh else []

    def xack(self, key, group, msg_id):
        self.acked.append(msg_id)

    def xdel(self, key, msg_id):
        self.deleted.append(msg_id)


def _entry(msg_id: str, payload: dict):
    return (msg_id, {"payload": json.dumps(payload)})


@pytest.fixture
def fake(monkeypatch):
    def _make(**kw):
        client = FakeRedis(**kw)
        monkeypatch.setattr(redis_cache, "get_client", lambda: client)
        return client

    return _make


def test_stale_pending_messages_are_reclaimed(fake):
    fake(pending=[_entry("1-1", {"candidate_id": "abc"})])

    reclaimed = redis_cache.reclaim_stale_candidates()

    assert [m["payload"]["candidate_id"] for m in reclaimed] == ["abc"]


def test_reclaim_uses_the_same_consumer_and_idle_threshold(fake):
    """
    Тот же консюмер — иначе сообщение просто переедет в чужой PEL. Порог
    простоя должен быть заметно больше нормального цикла диспетчера.
    """
    client = fake(pending=[_entry("1-1", {})])

    redis_cache.reclaim_stale_candidates()

    call = client.autoclaim_calls[0]
    assert call["consumer"] == redis_cache.STREAM_CONSUMER
    assert call["min_idle_time"] == redis_cache.STALE_PENDING_MS >= 10_000
    assert call["start_id"] == "0-0", "разбор PEL начинается с начала"


def test_empty_pel_returns_nothing(fake):
    fake(pending=[])
    assert redis_cache.reclaim_stale_candidates() == []


def test_redis_7_response_shape_is_supported(fake):
    """redis-py 5.x возвращает три элемента вместо двух."""
    fake(pending=[_entry("1-1", {"candidate_id": "abc"})], autoclaim_extra=True)

    assert len(redis_cache.reclaim_stale_candidates()) == 1


def test_broken_message_is_dropped_not_retried_forever(fake):
    """Нераспаковываемый payload обязан быть подтверждён, иначе он вечен."""
    client = fake(pending=[("1-1", {"payload": "не json"})])

    assert redis_cache.reclaim_stale_candidates() == []
    assert client.acked == ["1-1"]


def test_dispatcher_handles_pel_before_new_messages(monkeypatch):
    """
    Порядок важен: зависшие сообщения старше новых, и разбирать их надо
    первыми, иначе при стабильном потоке они не дождутся своей очереди.
    """
    pytest.importorskip("celery")  # диспетчер живёт в Celery-задаче
    from src.worker import tasks

    order = []
    monkeypatch.setattr(
        redis_cache, "reclaim_stale_candidates",
        lambda count=10: (order.append("pel"), [{"id": "1-1", "payload": {"a": 1}}])[1],
    )
    monkeypatch.setattr(
        redis_cache, "read_pending_candidates",
        lambda count=10: (order.append("new"), [{"id": "2-1", "payload": {"a": 2}}])[1],
    )
    monkeypatch.setattr(redis_cache, "ack_candidate", lambda msg_id: True)

    sent = []
    monkeypatch.setattr(
        tasks.evaluate_candidate, "apply_async",
        lambda args, queue: sent.append(args[0]),
    )

    result = tasks.process_stream_batch(batch_size=10)

    assert order == ["pel", "new"]
    assert result["dispatched"] == 2
    assert result["reclaimed"] == 1
    assert sent == [{"a": 1}, {"a": 2}]
