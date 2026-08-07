# Шаг 3: Celery Worker + TimescaleDB Continuous Aggregates

## Что добавлено

Асинхронная обработка кандидатов через Celery и статистические агрегаты в реальном времени через TimescaleDB.

---

## Новые файлы

```
btc-graph/
├── Dockerfile                                  ← образ для API и Celery воркеров
└── src/
    ├── worker/
    │   ├── celery_app.py                       ← Celery application + Beat расписание
    │   └── tasks.py                            ← задачи: evaluate, stream batch, refresh
    └── db/
        └── stats_repo.py                       ← запросы к continuous aggregates
```

**Обновлённые файлы:**
- `docker-compose.yml` — добавлены `celery-worker`, `celery-beat`, `flower`, `api`
- `requirements.txt` — добавлены `celery`, `flower`
- `src/db/orm_models.py` — добавлена `CandidateEventRecord` (hypertable)
- `src/db/candidate_repo.py` — добавлен `log_event()`
- `src/agent/pipeline.py` — `_persist()` теперь пишет в `candidate_events`
- `src/api/routes.py` — добавлены `/stats/*` эндпоинты
- `alembic/versions/002_candidate_events_aggregates.py` — вторая миграция

---

## Архитектура Celery

```
  HTTP /queue/enqueue          Celery Beat (каждые 10 сек)
        │                               │
        ▼                               ▼
  Redis Stream                  process_stream_batch task
  btc:candidates:stream    ──►         │
                                       │ xread + ack
                                       ▼
                               evaluate_candidate.delay()
                                       │
                                 [Celery Queue: evaluate]
                                       │
                               ┌───────┴───────┐
                               ▼               ▼
                           Worker 1        Worker 2   ...  (concurrency=4)
                               │
                               ▼
                         run_pipeline()
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
                PostgreSQL  Neo4j      Redis
```

### Retry-логика

`evaluate_candidate` — `max_retries=3`, `retry_delay=30s`. При сбое LLM API или БД задача повторяется автоматически.

---

## Celery задачи

| Задача | Очередь | Расписание | Описание |
|--------|---------|-----------|----------|
| `evaluate_candidate(payload)` | `evaluate` | по требованию | Полный pipeline для одного кандидата |
| `process_stream_batch()` | `celery` | каждые 10 сек | Читает Redis Stream → диспатчит `evaluate_candidate` |
| `refresh_continuous_aggregates()` | `celery` | каждый час (:05) | Принудительный refresh TimescaleDB views |

---

## TimescaleDB: candidate_events

Новый hypertable `candidate_events` — лог всех событий оценки. Каждый вызов `run_pipeline()` создаёт строку.

| Колонка | Тип | Описание |
|---------|-----|----------|
| `event_time` | TIMESTAMPTZ (partition key) | Время оценки |
| `candidate_id` | TEXT | ID кандидата |
| `quality_score` | FLOAT | Итоговый score |
| `rating` | TEXT | STRONG / MODERATE / WEAK |
| `direction` | TEXT | long / short |
| `win_rate` | FLOAT | Win rate |
| `fa_ratio` | FLOAT | Favorable/adverse ratio |
| `current_group_id` | FLOAT | Текущая группа |
| `transition_id` | TEXT | Переход |
| `context_status` | TEXT | fresh / stale |

**Retention policy:** сырые события хранятся 90 дней, затем удаляются автоматически.

---

## Continuous Aggregates

### `hourly_candidate_stats`

Почасовое агрегирование по `direction` + `rating`.

```sql
SELECT
    time_bucket('1 hour', event_time) AS bucket,
    direction,
    rating,
    COUNT(*)          AS candidate_count,
    AVG(quality_score) AS avg_quality_score,
    AVG(win_rate)      AS avg_win_rate,
    AVG(fa_ratio)      AS avg_fa_ratio,
    MIN(quality_score),
    MAX(quality_score)
FROM candidate_events
GROUP BY bucket, direction, rating
```

- **Refresh policy:** каждый час, покрывает последние 3 часа
- **Принудительный refresh:** задача Celery `refresh_continuous_aggregates` (каждый час в :05)

### `daily_group_stats`

Дневное агрегирование по `current_group_id` + `direction`.

```sql
SELECT
    time_bucket('1 day', event_time) AS bucket,
    current_group_id,
    direction,
    COUNT(*)           AS event_count,
    AVG(quality_score) AS avg_quality_score,
    AVG(win_rate)      AS avg_win_rate,
    SUM(CASE WHEN rating = 'STRONG' THEN 1 ELSE 0 END) AS strong_count,
    SUM(CASE WHEN rating = 'WEAK'   THEN 1 ELSE 0 END) AS weak_count
FROM candidate_events
GROUP BY bucket, current_group_id, direction
```

- **Refresh policy:** каждый день, покрывает последние 7 дней

---

## Новые API эндпоинты

| Метод | Путь | Параметры | Описание |
|-------|------|-----------|----------|
| `GET` | `/stats/hourly` | `?hours=24` | Почасовая статистика из aggregate |
| `GET` | `/stats/groups` | `?days=7&group_id=1.0` | Дневная статистика по группам |
| `GET` | `/stats/ratings` | `?days=7` | Распределение STRONG/MODERATE/WEAK |
| `GET` | `/stats/events` | `?limit=50` | Последние события из hypertable |
| `POST` | `/queue/enqueue` | body: кандидат JSON | Поставить в Redis Stream |

---

## Запуск шага 3

```bash
# 1. Пересобрать образы (добавлены новые сервисы)
docker-compose build

# 2. Поднять все сервисы
docker-compose up -d

# 3. Применить новую миграцию
alembic upgrade head

# 4. Проверить что воркер запустился
docker-compose logs celery-worker

# 5. Проверить Flower (мониторинг Celery)
open http://localhost:5555
```

### Локальный запуск без Docker

```bash
# Воркер
celery -A src.worker.celery_app worker --loglevel=info --concurrency=4 -Q evaluate,celery

# Beat (scheduler)
celery -A src.worker.celery_app beat --loglevel=info

# Flower
celery -A src.worker.celery_app flower --port=5555
```

### Отправить кандидата в очередь

```bash
curl -X POST http://localhost:8000/queue/enqueue \
  -H "Content-Type: application/json" \
  -d '{"candidate": { ... }, "use_llm": true}'
```

Задача попадает в Redis Stream → через ≤10 сек подхватывается Beat → диспатчится воркеру.

---

## Сервисы Docker Compose (итого)

| Сервис | Образ / Build | Порт | Роль |
|--------|---------------|------|------|
| `postgres` | `timescaledb-ha:pg16-latest` | 5432 | PostgreSQL + TimescaleDB + pgvector |
| `redis` | `redis:7-alpine` | 6379 | Broker + Backend + Stream + Pub/Sub |
| `neo4j` | `neo4j:5` | 7474/7687 | Граф состояний рынка |
| `api` | `./Dockerfile` | 8000 | FastAPI HTTP API |
| `celery-worker` | `./Dockerfile` | — | Асинхронная оценка кандидатов |
| `celery-beat` | `./Dockerfile` | — | Планировщик задач |
| `flower` | `./Dockerfile` | 5555 | Мониторинг Celery |

---

## Что не реализовано (запланировано далее)

- [ ] Prometheus метрики (latency оценки, queue depth, error rate)
- [ ] Grafana дашборд по непрерывным агрегатам
- [ ] structlog → JSON логирование во всех слоях
- [ ] Тесты: unit для scorer/validator, integration для pipeline + БД
