# Шаг 2: Persistence Layer — PostgreSQL, Redis, Neo4j, TimescaleDB

## Что добавлено

Полный слой хранения данных: реляционная БД, кэш, граф состояний и временные ряды.
Pipeline обновлён — теперь каждый кандидат автоматически сохраняется после оценки.

---

## Новые файлы

```
btc-graph/
├── docker-compose.yml              ← запуск всех сервисов
├── alembic.ini                     ← конфигурация миграций
├── alembic/
│   ├── env.py                      ← настройка alembic для SQLAlchemy
│   └── versions/
│       └── 001_initial_schema.py   ← первая миграция (таблицы + расширения)
├── scripts/
│   └── init_db.sql                 ← CREATE EXTENSION (pgvector, timescaledb)
└── src/
    ├── db/
    │   ├── connection.py           ← SQLAlchemy engine + get_session()
    │   ├── orm_models.py           ← ORM: CandidateRecord, MarketEventRecord
    │   ├── candidate_repo.py       ← CRUD: save, get, find_similar, get_strong
    │   ├── graph_repo.py           ← Neo4j: upsert_from_candidate, find_transitions
    │   └── embedding.py            ← feature vector 384-dim для pgvector
    └── cache/
        └── redis_cache.py          ← dedup, cache, pub/sub, Redis Stream
```

---

## Инфраструктура (docker-compose.yml)

| Сервис | Образ | Порт | Роль |
|--------|-------|------|------|
| `postgres` | `timescale/timescaledb-ha:pg16-latest` | 5432 | Основное хранилище + TimescaleDB + pgvector |
| `redis` | `redis:7-alpine` | 6379 | Кэш + очередь + pub/sub |
| `neo4j` | `neo4j:5` | 7474/7687 | Граф состояний рынка |

Образ `timescaledb-ha` уже включает оба расширения — устанавливать вручную не нужно.

---

## PostgreSQL: таблицы

### `candidates`

Хранит все оценённые кандидаты с результатами анализа.

| Колонка | Тип | Описание |
|---------|-----|----------|
| `candidate_id` | TEXT PK | Уникальный ID кандидата |
| `configuration_hash` | TEXT | Для дедупликации |
| `family_key` | TEXT | `candidate_family_key` для группировки |
| `quality_score` | FLOAT | Итоговая оценка |
| `rating` | TEXT | STRONG / MODERATE / WEAK |
| `direction` | TEXT | long / short |
| `warning_flags` | TEXT[] | Массив флагов |
| `raw_payload` | JSONB | Полные данные кандидата |
| `embedding` | vector(384) | Признаковый вектор для ANN-поиска |
| `evaluated_at` | TIMESTAMPTZ | Время оценки |

**Индексы:** `family_key`, `transition_id`, `(rating, direction)`, `evaluated_at DESC`, `configuration_hash`, HNSW на `embedding`.

### `market_events` (TimescaleDB hypertable)

Временной ряд рыночных событий, партиционированный по `ts`.

| Колонка | Тип | Описание |
|---------|-----|----------|
| `ts` | TIMESTAMPTZ | Временная метка (ключ партиции) |
| `symbol` | TEXT | Тикер (BTCUSDT) |
| `event_family` | TEXT | Семейство события |
| `event_block_id` | TEXT | ID блока событий |
| `group_id` | FLOAT | ID состояния рынка |
| `payload` | JSONB | Произвольные данные события |

---

## Neo4j: граф состояний рынка

### Узлы `MarketGroup`

```cypher
(:MarketGroup {group_id: 1.0, label: "group_1", dominant_bias: "long_skew"})
```

### Рёбра `TRANSITION`

```cypher
(src:MarketGroup)-[:TRANSITION {
  transition_id: "42->1",
  rarity: "common",
  count: 12,
  avg_horizon_return: 0.74,
  avg_quality_score: 0.73
}]->(dst:MarketGroup)
```

### Ключевые запросы (`graph_repo.py`)

| Функция | Описание |
|---------|----------|
| `upsert_from_candidate()` | Обновляет граф после каждой оценки |
| `find_transitions_to_group(to_id, rarity)` | Запрос из ТЗ: переходы в группу с фильтром по редкости |
| `get_group_info(group_id)` | Атрибуты узла группы |

---

## Redis: кэш и очередь

| Ключ / канал | TTL | Назначение |
|---|---|---|
| `candidate:hash:{hash}` | 30 мин | Дедупликация по `configuration_hash` |
| `evaluation:{id}` | 30 мин | Кэш результата оценки |
| `btc:strong_candidates` | pub/sub | Уведомления о STRONG-кандидатах |
| `btc:candidates:stream` | Redis Stream | Очередь на асинхронную обработку |

**Логика дедупликации в pipeline:**
1. При получении кандидата — проверяем `configuration_hash` в Redis
2. Если хэш уже есть — возвращаем кэшированный результат без пересчёта
3. После оценки — сохраняем хэш и результат в Redis

---

## pgvector: поиск похожих кандидатов

Embedding генерируется в `src/db/embedding.py` из 18 нормализованных признаков:

| Группа | Признаки |
|--------|----------|
| Числовые | `research_score`, `valid_label_pct`, `long_outcome_share`, `historical_outcome_skew`, `fa_ratio`, `monthly_concentration`, `repeatability_months`, `event_block_row_share`, `p70/p80` |
| Категориальные | `context_status`, `trajectory_entropy`, `transition_rarity`, `event_rarity_bucket`, `event_intensity_bucket`, `current_group_age_bucket`, `research_side`, `historical_bias_context` |

Вектор размером 384 (первые 18 значений — признаки, остальные — нули).
Индекс HNSW обеспечивает быстрый приближённый поиск.

---

## Обновлённый pipeline

```
parse_candidate()
    │
    ▼
check Redis dedup (configuration_hash)  ← если hit → вернуть кэш
    │
    ▼
validate_candidate() → warning_flags
    │
    ▼
score_candidate() → quality_score
    │
    ▼
evaluate_with_llm() → strengths/risks/summary
    │
    ├──► PostgreSQL: candidates (save_evaluation)
    ├──► Neo4j: MarketGroup + TRANSITION (upsert_from_candidate)
    └──► Redis: cache + dedup mark + pub/sub if STRONG
```

Управление через переменные окружения:

| Переменная | По умолчанию | Описание |
|---|---|---|
| `USE_DB` | `true` | Сохранять в PostgreSQL |
| `USE_REDIS` | `true` | Использовать Redis |
| `USE_GRAPH` | `true` | Обновлять граф Neo4j |

---

## Новые API эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/candidates/{id}` | Получить оценку из PostgreSQL |
| `GET` | `/candidates/strong/{direction}` | Последние STRONG по направлению |
| `POST` | `/candidates/similar` | Похожие кандидаты через pgvector |
| `GET` | `/graph/group/{group_id}` | Узел графа MarketGroup |
| `GET` | `/graph/transitions/to/{group_id}` | Переходы в группу (Cypher query) |
| `POST` | `/queue/enqueue` | Поставить кандидата в Redis Stream |

---

## Запуск шага 2

```bash
# 1. Поднять все сервисы
docker-compose up -d

# 2. Дождаться готовности PostgreSQL (обычно 15-20 сек)
docker-compose ps

# 3. Настроить переменные окружения
cp .env.example .env
# Вписать ANTHROPIC_API_KEY

# 4. Применить миграции
alembic upgrade head

# 5. Запустить API
uvicorn src.main:app --reload
```

Neo4j Browser доступен по адресу `http://localhost:7474` (логин: neo4j / btc_neo4j_pass).

---

## Что не реализовано в этом шаге (запланировано далее)

- [ ] Celery воркер для асинхронной обработки Redis Stream
- [ ] Continuous aggregates TimescaleDB для быстрой статистики
- [ ] Prometheus метрики (latency, quality_score distribution)
- [ ] Grafana дашборд
- [ ] structlog → JSON логирование
