# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Команды

```bash
# Стек (Docker)
make up                # поднять postgres+timescale, redis, neo4j, api, celery-worker, celery-beat, flower
make migrate           # alembic upgrade head внутри контейнера api (после ~15 сек, пока postgres пройдёт healthcheck)
make build             # ПОСЛЕ ЛЮБОЙ ПРАВКИ src/ — uvicorn в контейнере запущен БЕЗ --reload, код не подхватывается на лету
make reload            # после правки .env или docker-compose.yml
make logs / make ps / make down
make shell-api         # bash в контейнере api
make shell-pg          # psql в postgres

# Тесты — инфраструктура НЕ нужна (ни БД, ни Redis, ни Neo4j, ни ANTHROPIC_API_KEY)
make test-local        # pytest локально
make test              # pytest внутри контейнера api (с настоящими SDK)
pytest tests/test_scorer.py -v
pytest tests/test_scorer.py::test_reference_candidate_is_strong
pytest -k "family_key"
```

Локальный запуск без Docker: нужны поднятые PostgreSQL (с `vector` и `timescaledb`), Redis, Neo4j;
`uvicorn src.main:app --reload` + `celery -A src.worker.celery_app worker -Q evaluate,celery` + `celery ... beat`.
Без инфраструктуры: `export USE_DB=false USE_REDIS=false USE_GRAPH=false` — оценка работает полностью, просто ничего не сохраняется.

## Что это за система

Проект — **фильтр и интерпретатор между внешним генератором кандидатов и человеком**. Генератор
(в этот репозиторий не входит; см. соседний проект btc-graph-processing) выдаёт «кандидата» — досье
вида «такая конфигурация рынка BTC уже была 1339 раз, в 74% случаев цена шла вверх». Здесь кандидат
оценивается (`quality_score` [0..1]), объясняется, фильтруется и накапливается в хранилищах.

Словарь предметной области (`group_id`, `transition_id`, `candidate_family_key`, `context_status`,
`monthly_concentration`, p70/p80 и т.д.) — раздел 2 `README.md`; полное ТЗ — `README_agent_spec.md`.

## Архитектура

Поток одного кандидата: `parser → Candidate (Pydantic) → validator + scorer → объяснение → persist`.

**Ключевой инвариант: LLM ничего не решает.** `quality_score`, `rating`, `direction`, `win_rate`
считает детерминированный скорер; Claude получает готовые числа и только формулирует
strengths/risks/summary. Поэтому `use_llm=false` даёт полностью валидный результат без API-ключа,
а при сбое Anthropic API `llm_node` делает fallback на `src/agent/deterministic.py`.

Оркестрация — `src/agent/pipeline.py`: `run_pipeline()` (один кандидат) и `run_batch_pipeline()`
(фильтр по порогу → `select_best_per_family` → полный pipeline для выживших). Там же `_persist()`
и дедупликация через Redis.

Два пути обработки:
- **синхронный** — `POST /evaluate/json` и др. → `run_pipeline()`;
- **асинхронный** — `POST /queue/enqueue` → Redis Stream `btc:candidates:stream` → Celery Beat
  каждые 10 сек запускает `process_stream_batch` (XREADGROUP по группе `btc:candidates:workers`,
  затем XACK+XDEL) → `.delay()` на каждого → очередь **`evaluate`** → воркеры.

Хранилища (`_persist`, каждое отключается своим `USE_*`): PostgreSQL `candidates` (upsert) +
`candidate_events` (hypertable, append-only, retention 90 дней) + continuous aggregates;
Neo4j (`MarketGroup` ← `TRANSITION`); Redis (кэш оценок, dedup по `configuration_hash` TTL 30 мин,
pub/sub `btc:strong_candidates`, стрим приёма).

## Правила, которые легко нарушить

- **Все пороги оценки живут только в `src/scorer/candidate_scorer.py`** — ступени внутри
  `_score_*`, веса в `WEIGHTS`, границы рейтинга в `RATING_STRONG_MIN`/`RATING_MODERATE_MIN`.
  Никогда не сравнивай `quality_score` с числом напрямую — зови `get_rating()`. Раньше пороги были
  продублированы в `llm_node` и `/score/quick`, это чинили (audit #10).
- **`win_rate_for()` и `fa_ratio_for()` — единственные места перевода `long_outcome_share` и p70/p80
  в метрики кандидата.** Для `short` F/A ratio неприменим и равен `None` (симметричных short-полей
  в данных нет), ось `directional` считается по двум критериям вместо трёх. Не «инвертируй» ratio.
- **`USE_DB` / `USE_REDIS` / `USE_GRAPH` читаются на момент импорта `src/agent/pipeline.py`** —
  на лету не меняются, тесты правят их через monkeypatch модульных `_USE_*`.
- **Никакого `except: pass` вокруг хранилищ.** Контракт: недоступное хранилище не ломает оценку,
  но обязательно логируется (`logger.exception`) и попадает в статус, который возвращает `_persist()`.
  То же в `graph_repo` и `redis_cache`.
- **HTTP-коды: только через `_fail(exc, context)` в `src/api/routes.py`** — `ValidationError` /
  `ValueError` / `TypeError` → 422, всё остальное → 500 с логом traceback.
- **Ленивые импорты — намеренные.** БД/граф/Redis импортируются внутри функций (`routes.py`,
  `pipeline._persist`), чтобы API поднимался при лежащем Postgres; `llm_node` (а с ним `anthropic`)
  импортируется только при `use_llm=True`.
- **Кэш отдаётся только при совпадении `candidate_id`.** `configuration_hash` описывает конфигурацию
  рынка, а не кандидата — под одним хэшем могут идти разные `candidate_id`. `save` и `use_cache` —
  независимые параметры `run_pipeline()`.
- **Порядок признаков в `src/db/embedding.py` фиксирован.** Изменение порядка или `VECTOR_DIM` (=32)
  требует миграции с пересчётом существующих строк из `raw_payload` — см. `alembic/versions/003_shrink_embedding_dim.py`.
- **Celery-воркер обязан слушать `-Q evaluate,celery`.** `task_routes` шлёт `evaluate_candidate` в
  очередь `evaluate`; без `-Q` задачи молча копятся в Redis (audit #20).
- **Валидационные ошибки в `evaluate_candidate` не ретраятся** — они детерминированы, задача
  завершается результатом `{"error": "invalid_candidate", ...}`.
- **`refresh_continuous_aggregates` работает только вне транзакции** — соединение с
  `execution_options(isolation_level="AUTOCOMMIT")`, а не `get_session()`.
- **Любая правка порогов делает старые сохранённые `quality_score` несравнимыми с новыми** —
  пересчёта записей нет. Это стоит проговаривать в ответе пользователю.
- `validator` только помечает (`warning_flags`), никогда не бросает исключений.

## Тесты

`tests/conftest.py` уводит `DATABASE_URL` на in-memory sqlite (до первого импорта `src.db.*`,
который создаёт engine прямо на импорте) и подменяет заглушками отсутствующие SDK
(`anthropic`, `neo4j`, `redis`, `pgvector`) — заглушка ставится только если реального пакета нет.
Фикстуры: `reference_payload` / `reference_candidate` / `make_candidate(**overrides)` — эталонный
кандидат из ТЗ. Числа из раздела «Как читать результат оценки» в README зафиксированы тестом
`test_reference_candidate_is_strong`: правишь пороги — обновляй README.

Непокрыто (нужен живой стек): репозитории PostgreSQL, Cypher-запросы, SQL к continuous aggregates,
HTTP-роуты, Celery-задачи.

## Документация

- `README.md` — исчерпывающая (50 КБ): словарь понятий, формула score, API-справочник, сценарии, отладка.
- `README_agent_spec.md` — исходное ТЗ, все 37 полей кандидата.
- `docs/audit_findings.md` — 20 разобранных замечаний аудита с прогонами на живом стеке; читай перед
  тем, как «чинить» что-то в скорере, дедупликации или Cypher (замечание #5 отозвано как ошибочное —
  инкрементальное среднее в Neo4j `SET` корректно).
- `docs/step_01..03_*.md` — история построения слоёв.

## Известные ограничения

API без аутентификации и rate-limit (`use_llm=true` тратит токены Anthropic по анонимному запросу) —
допустимо только локально. Таблица `market_events` создана миграцией, но кодом не используется.
Кандидат — исследовательская идея, а не торговый сигнал: нет entry timing, стопов, размера позиции.
