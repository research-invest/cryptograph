# Шаг 1: MVP структура проекта

## Что сделано

Создан базовый скелет агента анализа рыночных кандидатов BTC согласно ТЗ (`README_agent_spec.md`).
Реализован полный pipeline: парсинг → валидация → оценка → LLM-резюме → вывод.

---

## Структура файлов

```
btc-graph/
├── requirements.txt
├── .env.example
├── docs/
│   └── step_01_mvp_structure.md   ← этот файл
└── src/
    ├── __init__.py
    ├── main.py                     ← точка входа (uvicorn)
    ├── models/
    │   └── candidate.py            ← Pydantic-модели
    ├── parser/
    │   └── candidate_parser.py     ← парсинг входных данных
    ├── validator/
    │   └── candidate_validator.py  ← warning_flags
    ├── scorer/
    │   └── candidate_scorer.py     ← quality_score по 4 осям
    ├── agent/
    │   ├── llm_node.py             ← Claude API reasoning node
    │   ├── pipeline.py             ← главный pipeline
    │   └── report_formatter.py     ← текстовый отчёт
    ├── filters/
    │   └── candidate_filter.py     ← фильтрация и детекция конфликтов
    └── api/
        └── routes.py               ← FastAPI эндпоинты
```

---

## Описание модулей

### `src/models/candidate.py`

Pydantic v2 модели:

- **`Candidate`** — полная структура кандидата по ТЗ. Все поля типизированы, диапазоны значений заданы через `Field(ge=..., le=...)`. Enum-поля: `AgeБucket`, `ContextStatus`, `TrajectoryEntropy`, `TransitionRarity`, `EventIntensityBucket`, `EventRarityBucket`, `HistoricalBiasContext`, `ResearchSide`.
- **`CandidateEvaluation`** — выходной объект оценки: `quality_score`, `rating`, `direction`, `win_rate`, `warning_flags`, `strengths`, `risks`, `summary`, разбивка по осям.

---

### `src/parser/candidate_parser.py`

Поддерживает три входных формата из ТЗ:

| Формат | Функция |
|--------|---------|
| Raw text (`ключ: значение`) | `parse_candidate(str)` |
| JSON объект | `parse_candidate(dict \| str)` |
| JSON array | `parse_candidates(list \| str)` |

Особенности:
- Автоматически определяет формат по первому символу (`{` → JSON, иначе → raw text).
- Убирает trailing-комментарии вида `← ...` из raw text.
- Кастует значения к целевым типам по `_FIELD_MAP`.

---

### `src/validator/candidate_validator.py`

Функция `validate_candidate(candidate) → list[str]` формирует `warning_flags`:

| Флаг | Условие |
|------|---------|
| `context_status_stale` | `context_status == "stale"` |
| `age_gt_120` | `current_group_age_bucket == "age_gt_120"` |
| `stale_and_aged_combined` | оба условия выше одновременно |
| `transition_rarity_common` | `transition_rarity == "common"` |
| `low_valid_label_pct` | `valid_label_pct < 0.70` |
| `false_confidence_high_score_low_validity` | `research_score > 0.85` и `valid_label_pct < 0.70` |
| `high_monthly_concentration_seasonal_risk` | `monthly_concentration > 0.30` |
| `very_small_sample_size` | `sample_size < 100` |
| `low_repeatability_months` | `repeatability_months < 3` |
| `symbol_not_btcusdt` | `symbol != "BTCUSDT"` |

---

### `src/scorer/candidate_scorer.py`

Вычисляет `quality_score` [0..1] по **4 осям** с весами:

| Ось | Вес | Критерии |
|-----|-----|----------|
| **A. Statistical** | 30% | `valid_label_pct`, `sample_size`, `monthly_concentration`, `repeatability_months` |
| **B. Directional** | 35% | win rate, `historical_outcome_skew`, `long_favorable_adverse_ratio_p70_p80` |
| **C. Context** | 20% | `context_status`, `current_group_age_bucket`, `trajectory_entropy` |
| **D. Rarity** | 15% | `event_rarity_bucket`, `transition_rarity`, `research_score` |

Функция `get_rating(score)` → `STRONG` / `MODERATE` / `WEAK`:
- `>= 0.75` → STRONG
- `>= 0.55` → MODERATE
- `< 0.55` → WEAK

---

### `src/agent/llm_node.py`

Вызывает Claude API (по умолчанию `claude-haiku-4-5-20251001`) для генерации:
- `strengths` — список 3–5 сильных сторон
- `risks` — список 2–3 рисков
- `summary` — одно итоговое предложение

Возвращает заполненный `CandidateEvaluation`. При ошибке парсинга JSON-ответа — graceful fallback.

---

### `src/agent/pipeline.py`

Главный pipeline. Два режима:

```
run_pipeline(raw, use_llm=True)        ← один кандидат
run_batch_pipeline(raw, use_llm=True)  ← список кандидатов
```

Порядок шагов:
```
parse_candidate → validate_candidate → score_candidate → evaluate_with_llm → CandidateEvaluation
```

Режим `use_llm=False` — детерминированная генерация strengths/risks/summary без вызова API (для тестов и dev).

---

### `src/agent/report_formatter.py`

Функция `format_report(candidate, evaluation) → str` — текстовый отчёт строго по формату ТЗ:

```
КАНДИДАТ: {candidate_id}
Горизонт: ... | Направление: ... | Win rate: ...%

КЛЮЧЕВЫЕ МЕТРИКИ:
...
КОНТЕКСТ:
...
ОЦЕНКА:
...
ВЫВОД: {STRONG/MODERATE/WEAK} {LONG/SHORT} кандидат
Сильные стороны: ...
Риски: ...
```

---

### `src/filters/candidate_filter.py`

**Задача 4 — фильтрация и сравнение:**
- `filter_candidates(candidates, min_quality_score=0.60)` — убирает слабых, сортирует по убыванию score.
- `select_best_per_family(scored)` — дедупликация по `candidate_family_key`, предпочтение `fresh` над `stale`.

**Задача 5 — детектирование конфликтов:**
- `detect_conflicts(candidates)` — возвращает список `ConflictReport`.

| Тип конфликта | Условие |
|---------------|---------|
| `contradictory_direction` | Одинаковый `transition_id`, разный `research_side` |
| `stale_and_old_context` | `context_status == stale` + `age_gt_120` |
| `false_confidence` | `research_score > 0.85` и `valid_label_pct < 0.70` |
| `seasonal_concentration` | `monthly_concentration > 0.30` |

---

### `src/api/routes.py`

FastAPI приложение. Эндпоинты:

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/health` | Проверка доступности |
| `POST` | `/evaluate/raw` | Оценка из raw text |
| `POST` | `/evaluate/json` | Оценка из JSON объекта |
| `POST` | `/evaluate/batch` | Оценка списка + фильтрация |
| `POST` | `/conflicts` | Детектирование конфликтов |
| `POST` | `/score/quick` | Быстрый score без LLM |

---

## Технический стек (MVP)

| Компонент | Версия | Роль |
|-----------|--------|------|
| Python | 3.11+ | основной язык |
| Pydantic | v2.7+ | валидация моделей |
| FastAPI | 0.115+ | HTTP API |
| Uvicorn | 0.30+ | ASGI сервер |
| Anthropic SDK | 0.40+ | LLM reasoning node |

---

## Запуск

```bash
# Установка зависимостей
pip install -r requirements.txt

# Настройка API ключа
cp .env.example .env
# вписать ANTHROPIC_API_KEY=sk-ant-...

# Запуск сервера
uvicorn src.main:app --reload

# Документация API
open http://localhost:8000/docs
```

---

## Что не реализовано в этом шаге (запланировано далее)

- [ ] PostgreSQL — хранение кандидатов и результатов оценки
- [ ] Redis — кэш, дедупликация по `configuration_hash`, очередь
- [ ] Neo4j / Apache AGE — граф состояний рынка
- [ ] TimescaleDB — временные ряды OHLCV и event-поток
- [ ] pgvector — векторный поиск похожих кандидатов
- [ ] Alembic — миграции БД
- [ ] Docker Compose — контейнеризация
- [ ] Celery — асинхронная очередь задач
- [ ] Prometheus + Grafana — мониторинг
