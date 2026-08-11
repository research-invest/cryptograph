"""
Уведомления: фильтр правила, формат тела, очередь отправки.

Ни БД, ни сети: матчинг и сборка payload — чистые функции, а транспорт и
журнал в Dispatcher инжектируются.

Главное, что здесь проверяется, — три предохранителя, каждый из которых
ломается молча:

  * правило с фильтром по рейтингу НЕ срабатывает на кандидате без оценки
    (иначе «только STRONG» слал бы всё подряд ровно тогда, когда приёмник
    недоступен);
  * формат тела содержит документированную сводку в корне (docs/notifications.md
    — контракт для чужой системы, и разъезд с ним не даёт никакой ошибки);
  * переполнение очереди отбрасывает уведомление с записью, а не копит его
    в памяти прогона.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

from btcproc.notify import service as notify_service
from btcproc.notify import payload as payload_mod
from btcproc.notify import rules as rules_mod
from btcproc.notify import sender as sender_mod


def make_rule(**kwargs) -> rules_mod.Rule:
    base = {"rule_id": 1, "name": "тест", "url": "https://example.com/hook"}
    base.update(kwargs)
    return rules_mod.Rule(**base)


CANDIDATE = payload_mod.example_row()["payload"]
EVALUATION = payload_mod.example_row()["evaluation"]


# ─── Фильтр правила ─────────────────────────────────────────────────────────
def test_empty_rule_matches_everything():
    """Правило без фильтров — это «слать всё», а не «не слать ничего»."""
    rule = make_rule()
    assert rule.matches(CANDIDATE, EVALUATION)


def test_disabled_rule_never_matches():
    assert not make_rule(enabled=False).matches(CANDIDATE, EVALUATION)


def test_symbol_filter_isolates_coins():
    assert make_rule(symbol="BTCUSDT").matches(CANDIDATE, EVALUATION)
    assert not make_rule(symbol="ETHUSDT").matches(CANDIDATE, EVALUATION)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"ratings": ("STRONG",)}, True),
        ({"ratings": ("WEAK",)}, False),
        ({"directions": ("long",)}, True),
        ({"directions": ("short",)}, False),
        ({"min_quality": 0.5}, True),
        ({"min_quality": 0.9}, False),
        ({"min_research_score": 0.6}, True),
        ({"min_research_score": 0.8}, False),
        ({"min_sample_size": 100}, True),
        ({"min_sample_size": 1000}, False),
        ({"transitions": ("42->7",)}, True),
        ({"transitions": ("13->2",)}, False),
        ({"event_families": ("volatility_events",)}, True),
        ({"event_families": ("trend_events",)}, False),
    ],
)
def test_filters(kwargs, expected):
    assert make_rule(**kwargs).matches(CANDIDATE, EVALUATION) is expected


def test_rating_filter_does_not_fire_without_evaluation():
    """
    Рейтинг и quality_score считает btc-graph. Если кандидат до него не
    доехал, «рейтинг неизвестен» — это НЕ «рейтинг подходит».

    Ловушка настоящая: неоценённые кандидаты появляются в штатных режимах
    (`--no-emit`, `SINK_MODE=none`, отсев фильтром приёмника), и правило
    «только STRONG» слало бы на них всё подряд.
    """
    rule = make_rule(ratings=("STRONG",), require_evaluation=False)
    assert not rule.matches(CANDIDATE, None)

    rule = make_rule(min_quality=0.1, require_evaluation=False)
    assert not rule.matches(CANDIDATE, None)


def test_require_evaluation_is_the_default():
    """По умолчанию шлём только тех, кого btc-graph принял и оценил."""
    assert not make_rule().matches(CANDIDATE, None)
    assert make_rule(require_evaluation=False).matches(CANDIDATE, None)


def test_direction_comes_from_evaluation_when_present():
    """
    btc-graph считает направление сам и вправе разойтись с research_side.
    Фильтр обязан смотреть на его ответ, иначе правило «только short»
    пропустит кандидата, который для приёмника long.
    """
    candidate = {**CANDIDATE, "research_side": "long"}
    evaluation = {**EVALUATION, "direction": "short"}
    assert make_rule(directions=("short",)).matches(candidate, evaluation)
    assert not make_rule(directions=("long",)).matches(candidate, evaluation)


# ─── Проверка формы правила ─────────────────────────────────────────────────
@pytest.mark.parametrize(
    "url",
    ["localhost:9000/hook", "example.com/hook", "ftp://example.com", "", "https://"],
)
def test_validate_rejects_unusable_urls(url):
    """
    Неверный адрес не даёт ошибки НИКОГДА — правило просто молча не работает.
    Поэтому отказ на сохранении.
    """
    with pytest.raises(rules_mod.RuleError):
        rules_mod.validate("имя", url, "full")


def test_validate_accepts_normal_rule():
    rules_mod.validate("имя", "https://example.com/hook", "full")
    rules_mod.validate("имя", "http://127.0.0.1:9000/hook", "compact")


def test_validate_rejects_unknown_payload_mode():
    with pytest.raises(rules_mod.RuleError):
        rules_mod.validate("имя", "https://example.com/hook", "verbose")


def test_clean_list_turns_empty_into_none():
    assert rules_mod.clean_list([]) is None
    assert rules_mod.clean_list(["", "  "]) is None
    assert rules_mod.clean_list([" STRONG ", "WEAK"]) == ["STRONG", "WEAK"]


# ─── Формат тела ────────────────────────────────────────────────────────────
def test_payload_has_documented_summary_at_the_root():
    """
    Сводка в корне — контракт для принимающей системы: она не обязана лезть
    во вложенные блоки, чтобы понять, про что уведомление.
    """
    body = payload_mod.build(make_rule(), payload_mod.example_row())
    for field in payload_mod.SUMMARY_FIELDS:
        assert field in body, f"поле {field} документировано, но пропало из тела"

    assert body["schema"] == payload_mod.SCHEMA
    assert body["symbol"] == "BTCUSDT"
    assert body["side"] == "long"
    assert body["rating"] == "STRONG"
    assert body["quality_score"] == pytest.approx(0.71)
    # Время БАРА, а не отправки, и всегда в UTC.
    assert body["ts"] == "2026-08-11T09:45:00+00:00"
    assert body["is_trading_signal"] is False


def test_full_payload_carries_whole_candidate_and_evaluation():
    body = payload_mod.build(make_rule(payload_mode="full"), payload_mod.example_row())
    assert body["candidate"]["transition_id"] == "42->7"
    assert body["evaluation"]["summary"]
    # Кандидат уходит ровно тем же набором полей, что и в btc-graph.
    assert "candidate_family_key" in body["candidate"]


def test_compact_payload_drops_the_heavy_blocks_but_keeps_the_summary():
    body = payload_mod.build(make_rule(payload_mode="compact"), payload_mod.example_row())
    assert "candidate" not in body and "evaluation" not in body
    assert body["candidate_id"] and body["rating"] and body["quality_score"]


def test_payload_is_json_serialisable():
    """
    Тело уходит через httpx(json=...). Любой datetime или numpy-число внутри
    падает уже в фоновом потоке — то есть в логе, а не в прогоне.
    """
    body = payload_mod.build(make_rule(), payload_mod.example_row())
    assert json.loads(json.dumps(body))["schema"] == payload_mod.SCHEMA


def test_naive_timestamps_are_treated_as_utc():
    row = {**payload_mod.example_row(), "ts": datetime(2026, 8, 11, 9, 45)}
    assert payload_mod.build(make_rule(), row)["ts"] == "2026-08-11T09:45:00+00:00"


# ─── Отбор кандидатов под правила ───────────────────────────────────────────
def test_plan_pairs_every_matching_rule_with_every_candidate():
    row = payload_mod.example_row()
    other = {**row, "candidate_id": "another", "rating": "WEAK",
             "evaluation": {**EVALUATION, "rating": "WEAK"}}
    strong = make_rule(rule_id=1, ratings=("STRONG",))
    everything = make_rule(rule_id=2)

    matched = notify_service.plan([row, other], [strong, everything])
    pairs = {(rule.rule_id, item["candidate_id"]) for rule, item in matched}
    assert pairs == {
        (1, row["candidate_id"]),
        (2, row["candidate_id"]),
        (2, "another"),
    }


# ─── Очередь и отправка ─────────────────────────────────────────────────────
def make_job(candidate_id: str = "c1") -> sender_mod.Job:
    return sender_mod.Job(rule_id=1, name="тест", url="https://example.com/hook",
                          headers={}, candidate_id=candidate_id, body={"ping": 1})


def test_dispatcher_sends_in_background_and_flush_waits():
    """
    Прогон не ждёт ответа — но и не теряет отправку: flush перед выходом
    процесса обязан дождаться разбора очереди.
    """
    sent, recorded = [], []
    slow = threading.Event()

    def send(job):
        slow.wait(1.0)
        sent.append(job.candidate_id)
        return sender_mod.Result("sent", http_status=200)

    d = sender_mod.Dispatcher(send=send, record=lambda job, res: recorded.append(res),
                              workers=2, queue_size=10)
    for i in range(4):
        assert d.submit(make_job(f"c{i}"))
    # Отправка ещё не случилась: submit не ждёт.
    assert sent == []

    slow.set()
    assert d.flush(timeout=5.0) == 0
    assert sorted(sent) == ["c0", "c1", "c2", "c3"]
    assert d.sent == 4 and d.failed == 0
    assert len(recorded) == 4


def test_overflow_is_dropped_with_a_record_not_silently_kept():
    """Переполнение — честная потеря с отметкой, а не рост памяти прогона."""
    block = threading.Event()
    dropped = []

    def send(job):
        block.wait(2.0)
        return sender_mod.Result("sent", http_status=200)

    def record(job, result):
        if result.status == "dropped":
            dropped.append(job.candidate_id)

    d = sender_mod.Dispatcher(send=send, record=record, workers=1, queue_size=1)
    results = [d.submit(make_job(f"c{i}")) for i in range(6)]
    block.set()
    d.flush(timeout=5.0)

    assert False in results, "переполнение обязано отражаться в ответе submit"
    assert dropped, "отброшенное уведомление обязано попасть в журнал"
    assert d.dropped == len(dropped)


def test_worker_survives_a_failing_transport():
    """
    Отказ получателя не должен ронять поток: следующие уведомления обязаны
    уйти. Иначе один недоступный адрес глушит рассылку целиком до перезапуска.
    """
    seen = []

    def send(job):
        seen.append(job.candidate_id)
        if job.candidate_id == "c0":
            raise RuntimeError("получатель недоступен")
        return sender_mod.Result("sent", http_status=200)

    errors = []
    d = sender_mod.Dispatcher(
        send=send, record=lambda job, res: errors.append(res.status),
        workers=1, queue_size=10,
    )
    d.submit(make_job("c0"))
    d.submit(make_job("c1"))
    assert d.flush(timeout=5.0) == 0
    assert seen == ["c0", "c1"]
    assert errors == ["failed", "sent"]
    assert d.failed == 1 and d.sent == 1


def test_broken_journal_does_not_break_delivery():
    """Журнал — диагностика. Его отказ не имеет права остановить отправку."""
    def record(job, result):
        raise RuntimeError("БД недоступна")

    d = sender_mod.Dispatcher(
        send=lambda job: sender_mod.Result("sent", http_status=200),
        record=record, workers=1, queue_size=10,
    )
    d.submit(make_job())
    assert d.flush(timeout=5.0) == 0
    assert d.sent == 1


# ─── Предохранитель по возрасту ─────────────────────────────────────────────
def test_age_guard_is_configured_narrow_enough_to_stop_a_backfill():
    """
    `train` выпускает сотни тысяч кандидатов на истории с 2017 года. Окно
    свежести — единственное, что стоит между первым прогоном с отправкой и
    рассылкой всей этой истории получателю.
    """
    from btcproc import config

    assert config.notify.max_candidate_age_minutes <= 24 * 60
    horizon = timedelta(minutes=config.notify.max_candidate_age_minutes)
    assert datetime.now(timezone.utc) - horizon > datetime(2020, 1, 1, tzinfo=timezone.utc)


# ─── Форма в админке ────────────────────────────────────────────────────────
#
# Проверяется ровно то, на чём этот проект уже обжигался (см. test_admin_forms):
# снятый чекбокс браузер НЕ отправляет, а пустое числовое поле приходит пустой
# строкой. Первое молча включило бы выключенное правило, второе превратило бы
# «порог не задан» в «порог 0» — оба отказа беззвучные.
fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture
def admin_client(monkeypatch):
    from btcproc import config
    from btcproc.admin import auth

    monkeypatch.setattr(
        config, "admin",
        config.AdminConfig(user="operator", password="very-long-password-42",
                           secret_key="k" * 64, ip_allowlist=[]),
    )
    from btcproc.admin import app as admin_app

    monkeypatch.setattr(auth, "current_user", lambda request: "operator")
    monkeypatch.setattr(admin_app, "init_schema", lambda: None, raising=False)
    with fastapi_testclient.TestClient(admin_app.app) as client:
        yield client


@pytest.fixture
def saved_rules(monkeypatch):
    """Перехватывает сохранение правила — БД в тесте не участвует."""
    captured = []
    monkeypatch.setattr(
        rules_mod, "save_rule",
        lambda values, rule_id=None: captured.append((values, rule_id)) or 1,
    )
    return captured


def test_form_treats_empty_thresholds_as_no_filter(admin_client, saved_rules):
    admin_client.post(
        "/notifications/save",
        data={"name": "тест", "url": "https://example.com/hook",
              "min_quality": "", "min_sample_size": "", "transitions": ""},
        follow_redirects=False,
    )
    values, rule_id = saved_rules[0]
    assert rule_id is None
    assert values["min_quality"] is None
    assert values["min_sample_size"] is None
    assert values["transitions"] is None
    assert values["ratings"] is None


def test_unchecked_boxes_turn_the_rule_off(admin_client, saved_rules):
    """Снятая галка = поля в запросе нет. Правило обязано выключиться."""
    admin_client.post(
        "/notifications/save",
        data={"name": "тест", "url": "https://example.com/hook"},
        follow_redirects=False,
    )
    values, _ = saved_rules[0]
    assert values["enabled"] is False
    assert values["require_evaluation"] is False


def test_form_collects_multiselect_and_headers(admin_client, saved_rules):
    admin_client.post(
        "/notifications/save",
        data={"name": "тест", "url": "https://example.com/hook",
              "enabled": "true", "rule_id": "7",
              "ratings": ["STRONG", "MODERATE"],
              "directions": ["long"],
              "event_families": ["trend_events"],
              "transitions": "42->7, 13->2",
              "min_quality": "0,65",
              "headers": "Authorization: Bearer xyz\nмусор без двоеточия\n"},
        follow_redirects=False,
    )
    values, rule_id = saved_rules[0]
    assert rule_id == 7
    assert values["ratings"] == ["STRONG", "MODERATE"]
    assert values["directions"] == ["long"]
    assert values["event_families"] == ["trend_events"]
    assert values["transitions"] == ["42->7", "13->2"]
    # Запятая как десятичный разделитель — то, что вводит человек.
    assert values["min_quality"] == pytest.approx(0.65)
    assert values["headers"] == {"Authorization": "Bearer xyz"}


def test_invalid_url_is_refused_by_the_route(admin_client, monkeypatch):
    """Форма не должна сохранять правило, которое молча не сработает."""
    from btcproc.admin import app as admin_app

    response = admin_client.post(
        "/notifications/save",
        data={"name": "тест", "url": "localhost:9000"},
        follow_redirects=False,
    )
    assert response.status_code == 422


def test_notifications_page_renders(admin_client, monkeypatch):
    """
    Дымовой тест шаблона: ошибка в Jinja иначе всплывает только в бою,
    пятисоткой на странице, которую открывают редко.
    """
    from btcproc.admin import app as admin_app
    from btcproc.admin import queries

    monkeypatch.setattr(rules_mod, "list_rules", lambda only_enabled=False: [
        make_rule(rule_id=3, name="строгие", symbol="BTCUSDT",
                  ratings=("STRONG",), min_quality=0.8),
    ])
    monkeypatch.setattr(rules_mod, "get_rule", lambda rule_id: None)
    monkeypatch.setattr(queries, "deliveries", lambda **kw: [])
    monkeypatch.setattr(queries, "delivery_totals", lambda: [])
    monkeypatch.setattr(queries, "transition_options", lambda run_id, limit=100: [])
    monkeypatch.setattr(admin_app, "_latest_train_id", lambda symbol=None: None)

    response = admin_client.get("/notifications")
    assert response.status_code == 200
    assert "строгие" in response.text
    assert "quality ≥ 0.80" in response.text
