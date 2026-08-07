# Техническое задание: Агент анализа рыночных кандидатов BTC

## Обзор системы

Система предназначена для **детерминирования фазы рынка BTC на несколько месяцев вперёд** (рост / падение / боковик) с вероятностным диапазоном цены. В её основе лежит граф рыночных состояний, узлы которого формируются адаптивно — без заранее заданного числа кластеров.

Агент получает на вход **торговых кандидатов** — структурированные снимки текущей конфигурации рынка с историческими метриками — и должен их интерпретировать, оценивать, фильтровать и предлагать actionable-выводы.

---

## Ключевые концепции

### 1. Граф состояний рынка

Рынок в каждый момент находится в одном из **динамически сформированных состояний (group_id)**. Состояния — это узлы графа, построенного из дискретных наблюдений расширенного пространства признаков:

- динамика цены BTC
- коррелирующие активы и индексы
- внешние факторы (макро, ончейн, деривативы и т.д.)

**Алгоритм адаптивной гранулярности:**
- Если внутри группы точек структура **неоднородна** → группа делится на более мелкие узлы.
- Если соседние узлы **практически идентичны** по структуре → они объединяются.
- Итог: граф сам находит баланс между «слишком общим» и «слишком дробным» без ручной настройки числа кластеров.

### 2. Кандидат (Candidate)

Кандидат — это конкретная **историческая конфигурация рынка**, которая сейчас воспроизводится. Он содержит:
- идентификацию текущего состояния и перехода
- блок событий, сопровождавших состояние
- историческую выборку аналогичных ситуаций
- профиль вероятного исхода (long/short bias + распределения движений)

Кандидат — это **исследовательская идея**, а не готовая торговая стратегия.

---

## Структура данных кандидата

### Блок Identity

| Поле | Тип | Описание |
|---|---|---|
| `candidate_id` | string | Уникальный ID кандидата (hex) |
| `symbol` | string | Тикер (всегда `BTCUSDT`) |
| `configuration_hash` | string | Хэш конфигурации рынка |
| `candidate_family_key` | string | Семейный ключ: `{group}\|{transition}\|{event_block}\|{skew}` |
| `research_score` | float [0..1] | Исследовательский рейтинг (сила кандидата по истории, **не прибыль**) |

### Блок State / Trajectory Context

| Поле | Тип | Описание |
|---|---|---|
| `previous_group_id` | float | Предыдущее состояние рынка |
| `current_group_id` | float | Текущее состояние рынка |
| `transition_id` | string | Переход: `{prev}->{cur}` |
| `current_group_age_bucket` | enum | Возраст текущего состояния: `age_lt_30`, `age_30_60`, `age_60_120`, `age_gt_120` |
| `context_status` | enum | Свежесть контекста: `fresh`, `stale` |
| `trajectory_entropy` | enum | Хаотичность траектории: `low`, `medium`, `high` |
| `transition_rarity` | enum | Редкость перехода: `rare`, `uncommon`, `common` |

### Блок Event Context

| Поле | Тип | Описание |
|---|---|---|
| `event_block_id` | string | ID блока событий |
| `primary_event_family` | string | Основная категория событий |
| `event_intensity_bucket` | enum | Плотность событий: `sparse`, `moderate`, `dense` |
| `event_rarity_bucket` | enum | Редкость блока: `common`, `uncommon`, `rare` |
| `signature_atom_count` | int | Число отдельных событий в блоке |
| `event_family_count` | int | Число семейств событий |
| `event_block_total_rows` | int | Сколько раз этот event_block встречался в истории |
| `event_block_row_share` | float | Доля от всей истории (0..1) |

### Блок Historical Sample

| Поле | Тип | Описание |
|---|---|---|
| `horizon` | string | Горизонт оценки (например, `24h`) |
| `sample_size` | int | Всего найдено аналогичных ситуаций |
| `valid_label_count` | int | Ситуаций с полными данными |
| `invalid_label_count` | int | Неполных/плохих данных |
| `valid_label_pct` | float | Доля валидных меток |
| `repeatability_days` | int | Число разных дней с такой конфигурацией |
| `repeatability_months` | int | Число разных месяцев с такой конфигурацией |
| `monthly_concentration` | float | Доля случаев, приходящихся на один месяц (0..1) |

### Блок Outcome Profile

| Поле | Тип | Описание |
|---|---|---|
| `historical_bias_context` | enum | `long_skew`, `short_skew`, `neutral` |
| `research_side` | enum | `long`, `short` |
| `long_outcome_count` | int | Число случаев с движением вверх |
| `short_outcome_count` | int | Число случаев с движением вниз |
| `long_outcome_share` | float | Доля LONG исходов (0..1) |
| `historical_outcome_skew` | float | Сила перекоса [-1..1] |

### Блок Favorable / Adverse Distributions

| Поле | Тип | Описание |
|---|---|---|
| `p70_long_favorable_pct` | float | Движение вверх в 70% «хороших» случаев, % |
| `p80_long_adverse_pct` | float | Движение против в 80% «плохих» случаев, % |
| `long_favorable_adverse_ratio_p70_p80` | float | Соотношение потенциала к риску |

---

## Задачи агента

### Задача 1: Парсинг и валидация кандидата

**Вход:** raw-текст кандидата (формат из примера в файле)  
**Выход:** структурированный объект со всеми полями

Агент должен:
- Извлечь все поля из неструктурированного текста
- Провалидировать типы и диапазоны значений
- Сформировать JSON-объект кандидата
- Пометить поля с аномальными значениями (`warning_flags`)

### Задача 2: Оценка качества кандидата

Агент оценивает кандидата по нескольким осям и выдаёт итоговый `quality_score` [0..1]:

#### Критерии оценки

**A. Статистическая надёжность выборки**
- `valid_label_pct` > 0.80 → хорошо
- `sample_size` > 500 → достаточная выборка
- `monthly_concentration` < 0.15 → нет сезонного смещения
- `repeatability_months` > 12 → долгосрочная повторяемость

**B. Сила directional asymmetry**
- `long_outcome_share` > 0.65 (или < 0.35 для short) → значимый перекос
- `historical_outcome_skew` > 0.30 → сильный перекос
- `long_favorable_adverse_ratio_p70_p80` > 3.0 → хорошее соотношение

**C. Контекстуальная свежесть**
- `context_status == "fresh"` → предпочтительно
- `current_group_age_bucket` не `age_gt_120` → состояние не устарело
- `trajectory_entropy == "low"` → предсказуемая траектория

**D. Редкость и специфичность**
- `event_rarity_bucket` в `uncommon` или `rare` → специфичный сигнал
- `transition_rarity` в `uncommon` или `rare` → нечастый переход
- `research_score` > 0.85 → высокий исследовательский рейтинг

### Задача 3: Генерация аналитического резюме

**Формат вывода:**

```
КАНДИДАТ: {candidate_id}
Горизонт: {horizon} | Направление: {research_side} | Win rate: {win_rate}%

КЛЮЧЕВЫЕ МЕТРИКИ:
- Выборка: {valid_label_count} валидных из {sample_size} ({valid_label_pct:.0%})
- Win rate: {long_outcome_share:.1%} в сторону {research_side}
- Потенциал / риск: +{p70_favorable}% / -{p80_adverse}% (ratio: {ratio:.1f}x)
- Повторяемость: {repeatability_months} мес, {repeatability_days} дней

КОНТЕКСТ:
- Переход: {transition_id} ({transition_rarity})
- Состояние: group {current_group_id}, возраст {age_bucket}, статус {context_status}
- События: {event_block_id}, интенсивность {intensity}, семей {event_family_count}

ВЫВОД: {STRONG/MODERATE/WEAK} {LONG/SHORT} кандидат
Сильные стороны: [список из 3-5 пунктов]
Риски: [список из 2-3 пунктов]
```

### Задача 4: Фильтрация и сравнение кандидатов

При получении **списка кандидатов** агент должен:

1. Рассчитать `quality_score` для каждого
2. Отфильтровать кандидатов ниже порога (по умолчанию `quality_score < 0.6`)
3. Сгруппировать по `candidate_family_key`
4. Внутри группы выбрать наиболее свежий и сильный
5. Вернуть ранжированный список с обоснованием

### Задача 5: Детектирование конфликтов

Агент должен обнаруживать **противоречивые сигналы**:

- Несколько кандидатов с одинаковым `transition_id` но разным `research_side`
- Кандидаты с `context_status == "stale"` и `age_gt_120` одновременно
- Очень высокий `research_score` при низком `valid_label_pct` (< 0.70) → ложная уверенность
- Высокая `monthly_concentration` (> 0.30) → возможный сезонный артефакт

---

## Правила интерпретации (Business Logic)

```
research_score — это НЕ вероятность прибыли.
Это оценка силы исторической аналитики кандидата.

Кандидат — это ИДЕЯ, а не сигнал на вход.
Для торговли требуется дополнительный фильтр на:
  - entry timing
  - risk management
  - portfolio context
```

### Пороговые значения по умолчанию

| Параметр | Слабый | Средний | Сильный |
|---|---|---|---|
| `research_score` | < 0.70 | 0.70–0.85 | > 0.85 |
| `long_outcome_share` | < 0.60 | 0.60–0.70 | > 0.70 |
| `historical_outcome_skew` | < 0.20 | 0.20–0.40 | > 0.40 |
| `favorable_adverse_ratio` | < 2.0 | 2.0–4.0 | > 4.0 |
| `valid_label_pct` | < 0.75 | 0.75–0.85 | > 0.85 |
| `repeatability_months` | < 6 | 6–15 | > 15 |

---

## Пример кандидата (Reference Case)

Из входного файла — **эталонный сильный кандидат**:

```
candidate_id:        245be5fb0908d59f6e89
symbol:              BTCUSDT
research_score:      0.957  ← очень высокий
transition:          42 → 1 (common)
context_status:      stale   ← слабый сигнал по свежести
horizon:             24h
sample_size:         1339  (valid: 1151, 86%)
win_rate (long):     74.5%
outcome_skew:        +0.489 (сильный long перекос)
favorable/adverse:   +3.18% / -0.73% = 4.33x ratio
repeatability:       21 дней, 19 месяцев
monthly_conc:        9.99%  ← хорошее распределение

Вывод: STRONG LONG кандидат несмотря на stale-контекст.
Риск: состояние устарело (>120 мин), может быть неактуальным.
```

---

## Входной формат агента

Агент принимает кандидатов в трёх форматах:

### 1. Raw text (как в файле)
Свободный текст с полями `ключ: значение`. Агент парсит самостоятельно.

### 2. JSON объект
```json
{
  "candidate_id": "245be5fb0908d59f6e89",
  "symbol": "BTCUSDT",
  "configuration_hash": "0f8928cb2fc1547b",
  "candidate_family_key": "1.0|42->1|event_block_098200|long_skew",
  "research_score": 0.9571800918456002,
  "previous_group_id": 42.0,
  "current_group_id": 1.0,
  "transition_id": "42->1",
  "current_group_age_bucket": "age_gt_120",
  "context_status": "stale",
  "trajectory_entropy": "medium",
  "transition_rarity": "common",
  "event_block_id": "event_block_098200",
  "primary_event_family": "zone_context_events",
  "event_intensity_bucket": "dense",
  "event_rarity_bucket": "uncommon",
  "signature_atom_count": 6,
  "event_family_count": 2,
  "event_block_total_rows": 23444,
  "event_block_row_share": 0.0067561570250315,
  "horizon": "24h",
  "sample_size": 1339,
  "valid_label_count": 1151,
  "invalid_label_count": 188,
  "valid_label_pct": 0.859596713965646,
  "repeatability_days": 21,
  "repeatability_months": 19,
  "monthly_concentration": 0.0999131190269331,
  "historical_bias_context": "long_skew",
  "research_side": "long",
  "long_outcome_count": 857,
  "short_outcome_count": 294,
  "long_outcome_share": 0.7445699391833188,
  "historical_outcome_skew": 0.4891398783666377,
  "p70_long_favorable_pct": 3.175384334258424,
  "p80_long_adverse_pct": 0.732807017851166,
  "long_favorable_adverse_ratio_p70_p80": 4.333179476414578
}
```

### 3. Список кандидатов (JSON array)
Массив объектов формата выше.

---

## Выходной формат агента

```json
{
  "candidate_id": "245be5fb0908d59f6e89",
  "quality_score": 0.87,
  "rating": "STRONG",
  "direction": "long",
  "win_rate": 0.7446,
  "favorable_adverse_ratio": 4.33,
  "context_freshness": "stale",
  "warning_flags": ["context_status_stale", "age_gt_120", "transition_rarity_common"],
  "strengths": [
    "Очень высокий research_score (0.957)",
    "74.5% long outcomes при выборке 1151",
    "F/A ratio 4.33x — отличное соотношение",
    "Повторяемость в 19 месяцах без концентрации"
  ],
  "risks": [
    "Контекст устарел (stale, age > 120 мин)",
    "Переход 42→1 часто встречается (common)"
  ],
  "summary": "Сильный long кандидат с высоким историческим перекосом. Основной риск — устаревший контекст состояния."
}
```

---

## Ограничения и важные оговорки

1. **research_score ≠ прибыль** — это исследовательская метрика.
2. **Кандидат ≠ торговый сигнал** — нужны дополнительные фильтры входа.
3. **`stale` контекст** — состояние могло смениться, данные требуют обновления.
4. **`common` transition** — частые переходы менее информативны.
5. **Горизонт 24h** — оценка релевантна только для краткосрочных движений.
6. **Система обучена только на BTCUSDT** — не применять к другим активам.

---

## Глоссарий

| Термин | Определение |
|---|---|
| **group_id** | Узел графа состояний рынка |
| **transition** | Переход между двумя состояниями графа |
| **event_block** | Набор рыночных событий, сопровождавших состояние |
| **candidate** | Конфигурация рынка с историческим профилем исходов |
| **research_score** | Синтетическая оценка силы кандидата по истории |
| **directional asymmetry** | Статистический перекос исходов в одну сторону |
| **favorable/adverse ratio** | Соотношение потенциального движения «за» к движению «против» |
| **long_skew** | Исторически данная конфигурация чаще приводила к росту |
| **context_status: stale** | Текущее состояние рынка устарело, данные могут быть неактуальны |
| **monthly_concentration** | Доля случаев в одном месяце — индикатор сезонного смещения |

---

## Технический стек

### Язык и рантайм

**Python 3.11+** — основной язык реализации агента.

Обоснование:
- нативная поддержка всех ML/data-библиотек, нужных для работы с графами и кластерами
- экосистема для работы с финансовыми данными (pandas, numpy, scipy)
- зрелые фреймворки для построения агентов (LangChain, LangGraph, Anthropic SDK)
- простая интеграция со всеми целевыми БД

Альтернатива для высоконагруженного пайплайна (> 10k кандидатов/мин): **Go** для transport/ingestion-слоя + Python для логики агента.

---

### Фреймворк агента

| Вариант | Когда выбирать |
|---|---|
| **Anthropic Claude API + LangGraph** | Рекомендуется. Stateful граф агента, поддержка multi-step reasoning, нативные tool calls |
| **LangChain AgentExecutor** | Если нужна быстрая интеграция с готовыми инструментами (поиск, SQL, API) |
| **Кастомный цикл на Anthropic SDK** | Если логика простая и не нужен оверхед фреймворков |

**Рекомендуемая архитектура агента:**

```
Incoming candidate (raw text / JSON)
        │
        ▼
  [Parser node]  ──► structured Candidate object
        │
        ▼
  [Validator node]  ──► warning_flags, schema check
        │
        ▼
  [Scorer node]  ──► quality_score по 4 осям
        │
        ▼
  [LLM Reasoning node]  ──► Claude: strengths, risks, summary
        │
        ▼
  [Output node]  ──► JSON response + сохранение в БД
```

---

### Базы данных

#### 1. Основное хранилище кандидатов — **PostgreSQL 16**

Для хранения всех кандидатов, результатов оценки и истории.

```sql
CREATE TABLE candidates (
    candidate_id        TEXT PRIMARY KEY,
    symbol              TEXT NOT NULL,
    configuration_hash  TEXT,
    family_key          TEXT,
    research_score      FLOAT,
    transition_id       TEXT,
    context_status      TEXT,
    horizon             TEXT,
    sample_size         INT,
    valid_label_pct     FLOAT,
    long_outcome_share  FLOAT,
    outcome_skew        FLOAT,
    fa_ratio            FLOAT,
    quality_score       FLOAT,
    rating              TEXT,
    direction           TEXT,
    warning_flags       TEXT[],
    raw_payload         JSONB,
    evaluated_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX ON candidates (family_key);
CREATE INDEX ON candidates (transition_id);
CREATE INDEX ON candidates (rating, direction);
CREATE INDEX ON candidates (evaluated_at DESC);
```

Почему PostgreSQL:
- `JSONB` для хранения полного raw_payload без жёсткой схемы
- массивы (`TEXT[]`) для `warning_flags`
- полнотекстовый поиск по `family_key` и `event_block_id`
- надёжные транзакции, легко масштабируется read-репликами

---

#### 2. Граф состояний рынка — **Neo4j** (или **Apache AGE** поверх Postgres)

Граф состояний — это буквально граф: узлы (group_id) и рёбра (transitions). Хранить его в реляционной БД неудобно.

```cypher
// Узел состояния
CREATE (g:MarketGroup {
  group_id: 1,
  label: "group_1",
  sample_count: 15000,
  dominant_bias: "long_skew"
})

// Переход между состояниями
CREATE (g42:MarketGroup {group_id: 42})-[:TRANSITION {
  transition_id: "42->1",
  rarity: "common",
  count: 1200,
  avg_horizon_return: 0.031
}]->(g1:MarketGroup {group_id: 1})
```

Запрос «найди все переходы, ведущие в группу 1 с long_skew и rarity=uncommon»:

```cypher
MATCH (src)-[t:TRANSITION]->(dst:MarketGroup {group_id: 1})
WHERE t.rarity IN ["uncommon", "rare"]
  AND dst.dominant_bias = "long_skew"
RETURN src.group_id, t.transition_id, t.count
ORDER BY t.count DESC
```

Альтернатива без отдельного сервиса: **Apache AGE** — расширение для PostgreSQL, добавляющее граф поверх существующей БД.

---

#### 3. Временны́е ряды и исторические данные — **TimescaleDB**

Расширение поверх PostgreSQL для хранения OHLCV и event-потока.

```sql
CREATE TABLE market_events (
    ts              TIMESTAMPTZ NOT NULL,
    symbol          TEXT NOT NULL,
    event_family    TEXT,
    event_block_id  TEXT,
    group_id        FLOAT,
    payload         JSONB
);

SELECT create_hypertable('market_events', 'ts');
CREATE INDEX ON market_events (symbol, ts DESC);
CREATE INDEX ON market_events (event_block_id);
```

Почему TimescaleDB, а не InfluxDB:
- работает как обычный Postgres (тот же драйвер, тот же SQL)
- поддерживает JOIN с таблицей `candidates`
- непрерывные агрегаты (continuous aggregates) для быстрого пересчёта статистик

---

#### 4. Кэш и очередь задач — **Redis 7**

- кэш последних N кандидатов на горизонте 24h (TTL = 30 мин)
- дедупликация по `configuration_hash` — не оценивать одно и то же дважды
- очередь входящих кандидатов через Redis Streams
- pub/sub для уведомлений о STRONG-кандидатах

```python
# Дедупликация перед оценкой
def is_already_evaluated(config_hash: str) -> bool:
    return redis.exists(f"candidate:hash:{config_hash}")

def mark_evaluated(config_hash: str, candidate_id: str):
    redis.setex(f"candidate:hash:{config_hash}", 1800, candidate_id)
```

---

#### 5. Векторный поиск похожих кандидатов — **pgvector** (расширение PostgreSQL)

Для поиска исторически похожих кандидатов по embedding'у конфигурации.

```sql
CREATE EXTENSION vector;

ALTER TABLE candidates ADD COLUMN embedding vector(384);

-- Поиск 10 ближайших кандидатов к текущему
SELECT candidate_id, rating, direction, quality_score,
       embedding <=> $1 AS distance
FROM candidates
ORDER BY embedding <=> $1
LIMIT 10;
```

Embedding формируется из числовых признаков кандидата (research_score, outcome_share, fa_ratio и т.д.) + категориальных (transition_rarity, context_status, event_rarity_bucket).

---

### Итоговая схема стека

```
┌─────────────────────────────────────────────────────┐
│                   Агент (Python)                    │
│         LangGraph + Anthropic Claude API            │
└──────┬──────────┬──────────┬───────────┬────────────┘
       │          │          │           │
       ▼          ▼          ▼           ▼
  PostgreSQL   Neo4j /   TimescaleDB   Redis
  + pgvector   Apache       (OHLCV,    (кэш,
  (кандидаты,  AGE          события)   очередь,
   embeddings) (граф                   pub/sub)
               состояний)
```

---

### Инфраструктура и деплой

| Компонент | Рекомендация |
|---|---|
| Контейнеризация | Docker Compose (dev) / Kubernetes (prod) |
| Оркестрация задач | **Celery** + Redis broker для асинхронной оценки кандидатов |
| API слой | **FastAPI** — async, автодокументация, pydantic-валидация входных данных |
| Мониторинг | **Prometheus + Grafana** — метрики качества кандидатов, latency агента |
| Логирование | **structlog** → JSON-логи → Loki / CloudWatch |
| Миграции БД | **Alembic** для PostgreSQL схемы |

---

### Минимальная конфигурация для старта (MVP)

Если не нужен граф и векторный поиск на первом этапе:

```
PostgreSQL 16  ← всё: кандидаты, события, JSONB-payload
Redis 7        ← кэш + очередь
Python 3.11    ← агент на чистом Anthropic SDK без LangGraph
FastAPI        ← HTTP endpoint для приёма кандидатов
```

Этого достаточно для обработки до ~1000 кандидатов в сутки с полным логированием и дедупликацией. Neo4j и TimescaleDB добавляются по мере роста нагрузки и усложнения запросов.
