# BTC Market Candidate Agent

Агент анализа рыночных кандидатов BTC: принимает снимок текущей конфигурации рынка с историческими метриками и отвечает на вопрос **«насколько этой исторической аналогии можно доверять и в какую сторону она смещена»**.

> Полное ТЗ, из которого вырос проект — [`README_agent_spec.md`](README_agent_spec.md).
> Найденные при аудите проблемы кода — [`docs/audit_findings.md`](docs/audit_findings.md).

---

## Оглавление

1. [В чём смысл проекта](#1-в-чём-смысл-проекта)
2. [Ключевые понятия](#2-ключевые-понятия)
3. [Как устроена система](#3-как-устроена-система)
4. [Что делает каждый модуль](#4-что-делает-каждый-модуль)
5. [Установка и первый запуск](#5-установка-и-первый-запуск)
6. [Как с этим работать: сценарии](#6-как-с-этим-работать-сценарии)
7. [Справочник API](#7-справочник-api)
8. [Как читать результат оценки](#8-как-читать-результат-оценки)
9. [Формат входных данных](#9-формат-входных-данных)
10. [Хранилища и что в них лежит](#10-хранилища-и-что-в-них-лежит)
11. [Конфигурация](#11-конфигурация)
12. [Отладка и типичные проблемы](#12-отладка-и-типичные-проблемы)
13. [Тесты и состояние кода](#13-тесты-и-известные-проблемы-кода)
14. [Ограничения системы](#14-ограничения-системы)

---

## 1. В чём смысл проекта

Внешняя система (в этот репозиторий она **не входит**) строит граф состояний рынка BTC: каждый момент времени относится к одному из динамически найденных состояний — `group_id`. Когда рынок переходит из состояния в состояние, эта система находит в истории все похожие случаи и выдаёт **кандидата** — досье вида «такая конфигурация уже была 1339 раз, в 74% случаев за следующие 24 часа цена шла вверх».

Проблема: таких кандидатов много, они разного качества, и часть из них — статистический мусор (выборка мала, все случаи пришлись на один месяц, контекст устарел, направления противоречат друг другу).

**Этот проект — фильтр и интерпретатор между генератором кандидатов и человеком.** Он делает четыре вещи:

| Что | Зачем |
|---|---|
| **Оценивает** — считает `quality_score` [0..1] по 4 осям | Отделить надёжную статистику от шума одним числом |
| **Объясняет** — генерирует strengths / risks / summary (детерминированно или через Claude) | Чтобы решение было читаемым, а не «доверься числу» |
| **Фильтрует** — отбрасывает слабых, схлопывает дубликаты по `candidate_family_key`, находит конфликты | На вход часто приходит пачка кандидатов об одном и том же |
| **Накапливает** — пишет в PostgreSQL, граф в Neo4j, временные ряды в TimescaleDB | Чтобы можно было спросить «а что было раньше в похожих ситуациях» |

**Главное ограничение, встроенное в саму систему:**
кандидат — это **исследовательская идея, а не торговый сигнал**. `research_score` — это не вероятность прибыли, а оценка силы исторической аналогии. Для реальной торговли поверх нужны entry timing, risk management и портфельный контекст, которых здесь нет.

---

## 2. Ключевые понятия

Без них код читается как набор случайных полей.

| Термин | Что означает |
|---|---|
| **`group_id`** | Узел графа состояний рынка. Число вроде `1.0` или `42.0`. Состояния находятся адаптивно: неоднородная группа дробится, почти одинаковые сливаются |
| **`transition_id`** | Переход между состояниями, строка `"42->1"`. Это «событие», вокруг которого строится кандидат |
| **`transition_rarity`** | `rare` / `uncommon` / `common`. Редкий переход информативнее — частый происходит постоянно и мало что говорит |
| **`event_block_id`** | Блок рыночных событий, сопровождавших состояние (макро, ончейн, деривативы) |
| **`context_status`** | `fresh` / `stale`. `stale` = снимок устарел, рынок мог уже уйти из этого состояния |
| **`current_group_age_bucket`** | Сколько времени рынок сидит в текущем состоянии: `age_lt_30`, `age_30_60`, `age_60_120`, `age_gt_120` (минуты). Старое состояние ближе к смене фазы |
| **`trajectory_entropy`** | Хаотичность пути в это состояние: `low` / `medium` / `high` |
| **`sample_size` / `valid_label_count`** | Сколько похожих ситуаций нашлось в истории и у скольких из них есть полные данные об исходе |
| **`repeatability_days` / `repeatability_months`** | В скольких разных днях/месяцах встречалась конфигурация. Защита от «все 1000 случаев за одну неделю» |
| **`monthly_concentration`** | Доля случаев, попавших в один месяц. > 0.30 → вероятный сезонный артефакт |
| **`long_outcome_share`** | Доля случаев, где движение было вверх. Это и есть «win rate» для long-кандидата |
| **`historical_outcome_skew`** | Сила перекоса [-1..1] |
| **`p70_long_favorable_pct` / `p80_long_adverse_pct`** | Движение «за» в 70% хороших случаев и «против» в 80% плохих, в процентах. Их отношение — `long_favorable_adverse_ratio_p70_p80`, потенциал к риску |
| **`candidate_family_key`** | Ключ семьи `{group}\|{transition}\|{event_block}\|{skew}`. Кандидаты одной семьи — вариации одной и той же идеи, из них нужно оставить одного |
| **`configuration_hash`** | Хэш конфигурации рынка. Используется для дедупликации в Redis |

---

## 3. Как устроена система

### Поток одного кандидата

```
        raw text / JSON / dict
                 │
                 ▼
        ┌──────────────────┐
        │  parser          │  вытащить поля, привести типы
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │  Candidate       │  Pydantic-модель: типы и диапазоны проверены
        └────────┬─────────┘
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
┌───────────┐        ┌──────────────┐
│ validator │        │  scorer      │  4 оси → quality_score
│ → flags   │        │  → breakdown │
└─────┬─────┘        └──────┬───────┘
      └──────────┬──────────┘
                 ▼
        ┌──────────────────────────────┐
        │  use_llm=true  → llm_node    │  Claude пишет strengths/risks/summary
        │  use_llm=false → детермин.   │  те же тексты, но по if-правилам
        └────────────┬─────────────────┘
                     ▼
            CandidateEvaluation
                     │
         save=true   ▼
        ┌────────────────────────────────────────┐
        │ PostgreSQL  candidates (upsert)        │
        │             candidate_events (append)  │
        │ Neo4j       MarketGroup + TRANSITION   │
        │ Redis       кэш + dedup + pub/sub      │
        └────────────────────────────────────────┘
```

Ключевая деталь: **LLM ничего не решает.** `quality_score`, `rating`, `direction`, `win_rate` считает детерминированный скорер. Claude получает уже готовые числа и только формулирует объяснение. Поэтому `use_llm=false` даёт полностью валидный результат — без API-ключа, мгновенно и бесплатно.

### Синхронный и асинхронный пути

```
СИНХРОННО                          АСИНХРОННО
POST /evaluate/json                POST /queue/enqueue
      │                                  │
      │                                  ▼
      │                          Redis Stream «btc:candidates:stream»
      │                                  │
      │                          Celery Beat: каждые 10 сек
      │                                  ▼
      │                          process_stream_batch
      │                                  │  .delay() на каждого
      │                                  ▼
      │                          очередь «evaluate» → воркеры (concurrency=4)
      │                                  │
      └──────────► run_pipeline() ◄──────┘
```

Синхронный путь — интерактивная работа, сразу видишь ответ. Асинхронный — пачки кандидатов и фоновая обработка; результат забирается потом через `GET /candidates/{id}` или подпиской на Redis-канал `btc:strong_candidates`.

### Формула quality_score

```
quality_score = 0.30·statistical + 0.35·directional + 0.20·context + 0.15·rarity
```

Каждая ось — среднее нескольких ступенчатых оценок в [0..1]:

| Ось | Вес | Из чего складывается | 1.0 при |
|---|---|---|---|
| **statistical** | 30% | `valid_label_pct`, `sample_size`, `monthly_concentration`, `repeatability_months` | > 0.85, > 1000, < 0.10, > 15 |
| **directional** | 35% | win rate, \|`outcome_skew`\|, F/A ratio (только long) | > 0.70, > 0.40, > 4.0 |
| **context** | 20% | `context_status`, `age_bucket`, `trajectory_entropy` | fresh, `age_lt_30`, low |
| **rarity** | 15% | `event_rarity_bucket`, `transition_rarity`, `research_score` | rare, rare, > 0.85 |

Итоговый рейтинг: `STRONG` ≥ 0.75, `MODERATE` ≥ 0.55, иначе `WEAK`.

Смысл весов: **directional важнее всего** (без перекоса кандидат бесполезен), затем **статистическая надёжность** (перекос на 50 случаях ничего не значит), контекст и редкость — модификаторы.

Возраст состояния внутри оси `context` оценивается градуированно:
`age_lt_30` → 1.0, `age_30_60` → 0.75, `age_60_120` → 0.5, `age_gt_120` → 0.0.

---

## 4. Что делает каждый модуль

```
src/
├── models/candidate.py       Candidate (37 полей) + CandidateEvaluation + 8 enum.
│                             Единственный источник правды по схеме данных.
├── parser/candidate_parser.py
│                             parse_candidate()  — dict / JSON-строка / raw text
│                             parse_candidates() — JSON array / list[dict]
│                             Raw text: строки «ключ: значение», незнакомые ключи
│                             игнорируются, хвосты вида «← комментарий» срезаются.
├── validator/candidate_validator.py
│                             validate_candidate() → list[str] warning_flags.
│                             Не бросает исключений, только помечает.
├── scorer/candidate_scorer.py
│                             score_candidate() → ScoreBreakdown(4 оси + total)
│                             get_rating(score) → STRONG / MODERATE / WEAK
│                             Здесь живут ВСЕ пороги. Хочешь перекалибровать —
│                             только этот файл.
├── filters/candidate_filter.py
│                             filter_candidates()      — отсечь ниже порога, ранжировать
│                             select_best_per_family() — по одному на family_key
│                             detect_conflicts()       — 4 типа противоречий
├── agent/
│   ├── pipeline.py           run_pipeline() / run_batch_pipeline() — оркестрация.
│   │                         Здесь же дедупликация через Redis и _persist().
│   ├── llm_node.py           Вызов Claude, парсинг JSON из ответа, сборка
│   │                         CandidateEvaluation.
│   └── report_formatter.py   Человекочитаемый текстовый отчёт (формат из ТЗ).
├── db/
│   ├── connection.py         engine + get_session() (контекстный менеджер)
│   ├── orm_models.py         candidates, candidate_events, market_events
│   ├── candidate_repo.py     upsert оценки, лог события, поиск похожих, STRONG-выборка
│   ├── embedding.py          18 признаков → вектор 32 (остальное нули) для pgvector
│   ├── graph_repo.py         Neo4j: upsert узлов/рёбер, запросы по переходам
│   └── stats_repo.py         SQL к continuous aggregates TimescaleDB
├── cache/redis_cache.py      dedup по configuration_hash (TTL 30 мин), кэш оценок,
│                             pub/sub STRONG, Redis Stream как очередь приёма
├── worker/
│   ├── celery_app.py         конфиг Celery + расписание Beat
│   └── tasks.py              evaluate_candidate, process_stream_batch,
│                             refresh_continuous_aggregates
└── api/routes.py             Все HTTP-эндпоинты. Импорты БД сделаны ленивыми внутри
                              функций — API поднимается даже если Postgres лежит.
```

---

## 5. Установка и первый запуск

### Требования

- Docker + Docker Compose
- `make` (на macOS входит в Xcode CLT: `xcode-select --install`)
- Ключ Anthropic API — **только если нужен `use_llm=true`**; без него всё работает с `use_llm=false`

### Запуск через Docker (рекомендуется)

```bash
# 1. Создать .env
make setup
# открой .env и впиши реальный ANTHROPIC_API_KEY (или оставь заглушку,
# если собираешься работать только с use_llm=false)

# 2. Поднять postgres, redis, neo4j, api, celery-worker, celery-beat, flower
make up

# 3. Подождать ~15 сек, пока postgres пройдёт healthcheck, и накатить схему
make migrate

# 4. Проверить
make ps
curl http://localhost:8000/health
```

`docker-compose.yml` перекрывает адреса из `.env` внутренними DNS-именами
(`postgres`, `redis`, `neo4j`), так что localhost-значения в `.env` нужны только
для локального запуска без Docker — трогать их не надо.

### Сервисы после запуска

| URL | Что |
|---|---|
| http://localhost:8000/docs | **Swagger UI — основная точка входа**, все запросы можно делать прямо отсюда |
| http://localhost:8000/health | Проверка живости |
| http://localhost:7474 | Neo4j Browser (`neo4j` / `btc_neo4j_pass`) |
| http://localhost:5555 | Flower — очереди и статус Celery-задач |

### Локальный запуск без Docker

Нужны уже поднятые PostgreSQL (с `vector` и `timescaledb`), Redis и Neo4j.

```bash
pip install -r requirements.txt
cp .env.example .env

# .env читается автоматически: load_dotenv() вызывается в src/main.py,
# src/worker/celery_app.py и alembic/env.py — запускай команды из корня проекта.

alembic upgrade head
uvicorn src.main:app --reload

# в отдельных терминалах:
celery -A src.worker.celery_app worker --loglevel=info -Q evaluate,celery
celery -A src.worker.celery_app beat --loglevel=info
```

> Если Postgres/Redis/Neo4j под рукой нет — выключи их:
> `export USE_DB=false USE_REDIS=false USE_GRAPH=false`. Оценка кандидатов будет
> работать полностью, просто ничего не сохранится.

### Команды Makefile

```bash
make up            # поднять всё
make down          # остановить и удалить контейнеры
make ps            # статус
make logs          # логи всех сервисов (Ctrl+C для выхода)

make reload        # после правки .env или docker-compose.yml
make build         # после правки кода в src/ или Dockerfile
make migrate       # после добавления новой миграции Alembic

make shell-api     # bash внутри контейнера api
make shell-pg      # psql внутри postgres
```

Важно: образ собирается с `COPY . .`, а uvicorn запускается **без** `--reload`.
Правка файлов в `src/` не подхватывается на лету — нужен `make build`.

---

## 6. Как с этим работать: сценарии

### Сценарий A — «просто оценить одного кандидата»

Самый частый случай. LLM не нужна, в БД писать не нужно:

```bash
curl -s -X POST http://localhost:8000/score/quick \
  -H "Content-Type: application/json" \
  -d '{"candidate": { ...поля кандидата... }}' | jq
```

Ответ — `quality_score`, `rating`, разбивка по осям и `warning_flags`. Быстро, без побочных эффектов.

### Сценарий B — «оценить и сохранить»

```bash
curl -s -X POST http://localhost:8000/evaluate/json \
  -H "Content-Type: application/json" \
  -d '{"use_llm": false, "save": true, "candidate": { ... }}' | jq
```

Добавляет к результату `strengths` / `risks` / `summary` и кладёт запись в
PostgreSQL, Neo4j и Redis. Поставь `"use_llm": true`, чтобы объяснение писал Claude
(медленнее, тратит токены, требует валидный `ANTHROPIC_API_KEY`).

### Сценарий C — «пришла пачка кандидатов, оставь достойных»

```bash
curl -s -X POST http://localhost:8000/evaluate/batch \
  -H "Content-Type: application/json" \
  -d '{"use_llm": false, "save": true, "min_quality_score": 0.6,
       "candidates": [ {...}, {...}, {...} ]}' | jq
```

Что происходит внутри:
1. считается `quality_score` для каждого;
2. отбрасываются те, кто ниже `min_quality_score`;
3. в каждой `candidate_family_key` остаётся один — по наибольшему `quality_score`, причём `fresh` получает бонус `+0.05` (сильный `stale` не проигрывает слабому `fresh`);
4. выжившие проходят полный pipeline и сохраняются.

Ответ отсортирован по убыванию качества.

### Сценарий D — «не противоречат ли кандидаты друг другу»

```bash
curl -s -X POST http://localhost:8000/conflicts \
  -H "Content-Type: application/json" \
  -d '{"candidates": [ {...}, {...} ]}' | jq
```

Ищет четыре типа противоречий:

| Тип | Когда срабатывает |
|---|---|
| `contradictory_direction` | Один `transition_id`, но разные `research_side` — система сама себе противоречит |
| `stale_and_old_context` | `stale` + `age_gt_120` — снимок вдвойне протух |
| `false_confidence` | `research_score` > 0.85 при `valid_label_pct` < 0.70 — уверенность на дырявых данных |
| `seasonal_concentration` | `monthly_concentration` > 0.30 — паттерн может быть сезонным артефактом |

Запускай это **до** того, как принимать решение по батчу.

### Сценарий E — «фоновая обработка потока»

```bash
# положить в очередь (возвращается сразу)
curl -s -X POST http://localhost:8000/queue/enqueue \
  -H "Content-Type: application/json" -d '{"candidate": { ... }}'

# наблюдать за обработкой
open http://localhost:5555          # Flower
docker compose logs -f celery-worker

# забрать результат
curl -s http://localhost:8000/candidates/245be5fb0908d59f6e89 | jq
```

Подписаться на STRONG-кандидатов в реальном времени:

```bash
docker compose exec redis redis-cli SUBSCRIBE btc:strong_candidates
```

### Сценарий F — «что было раньше в похожих ситуациях»

```bash
# ближайшие исторические кандидаты по вектору признаков (pgvector, cosine)
curl -s -X POST "http://localhost:8000/candidates/similar?limit=10" \
  -H "Content-Type: application/json" -d '{"candidate": { ... }}' | jq

# все переходы, ведущие в состояние 1.0, только редкие
curl -s "http://localhost:8000/graph/transitions/to/1.0?rarity=rare,uncommon" | jq

# статистика оценок за неделю
curl -s "http://localhost:8000/stats/ratings?days=7" | jq
```

Neo4j можно спрашивать и напрямую в браузере (http://localhost:7474):

```cypher
// самые частые переходы в состояние 1.0
MATCH (src:MarketGroup)-[t:TRANSITION]->(dst:MarketGroup {group_id: 1.0})
RETURN src.group_id, t.transition_id, t.rarity, t.count, t.avg_quality_score
ORDER BY t.count DESC LIMIT 20;

// весь граф
MATCH p=()-[:TRANSITION]->() RETURN p LIMIT 100;
```

### Сценарий G — «перекалибровать оценку под себя»

Все пороги — в одном файле `src/scorer/candidate_scorer.py`:

- ступени внутри `_score_statistical` / `_score_directional` / `_score_context` / `_score_rarity`;
- веса осей — словарь `WEIGHTS`;
- границы рейтингов — `get_rating()`.

После правки: `make build`. Учти, что `quality_score` уже сохранённых записей
пересчитан не будет — старые и новые оценки станут несравнимы.

---

## 7. Справочник API

### Оценка

| Метод | Путь | Тело | Описание |
|---|---|---|---|
| `POST` | `/evaluate/raw` | `{raw, use_llm, save}` | Кандидат из свободного текста |
| `POST` | `/evaluate/json` | `{candidate, use_llm, save}` | Кандидат из JSON-объекта |
| `POST` | `/evaluate/batch` | `{candidates[], use_llm, save, min_quality_score}` | Пачка: фильтр + дедуп по family_key |
| `POST` | `/score/quick` | `{candidate}` | Только score, без LLM и без записи |
| `POST` | `/conflicts` | `{candidates[]}` | Поиск противоречий |

Значения по умолчанию: `use_llm=true`, `save=true`, `min_quality_score=0.60`.
Для повседневной работы обычно нужен `use_llm=false`.

### Хранилище

| Метод | Путь | Описание |
|---|---|---|
| `GET` | `/candidates/{id}` | Сохранённая оценка по `candidate_id` |
| `GET` | `/candidates/strong/{direction}?limit=20` | Последние STRONG (`long` / `short`) |
| `POST` | `/candidates/similar?limit=10` | Похожие через pgvector (cosine) |

### Граф

| Метод | Путь | Описание |
|---|---|---|
| `GET` | `/graph/group/{group_id}` | Атрибуты узла MarketGroup |
| `GET` | `/graph/transitions/to/{group_id}?rarity=uncommon,rare` | Входящие переходы с фильтром по редкости |

### Статистика (TimescaleDB)

| Метод | Путь | Описание |
|---|---|---|
| `GET` | `/stats/hourly?hours=24` | Почасовые агрегаты по direction × rating |
| `GET` | `/stats/groups?days=7&group_id=1.0` | Дневные агрегаты по состояниям |
| `GET` | `/stats/ratings?days=7` | Распределение STRONG / MODERATE / WEAK |
| `GET` | `/stats/events?limit=50` | Сырые последние события из hypertable |

> `/stats/hourly`, `/stats/groups` и `/stats/ratings` читают **continuous aggregates**,
> а не сырую таблицу. У них политика обновления с `end_offset` в 1 час (для дневных — 1 день),
> поэтому **самые свежие события там появляются с задержкой**. Если нужны данные «прямо сейчас» —
> используй `/stats/events`, он читает hypertable напрямую.

### Очередь и служебное

| Метод | Путь | Описание |
|---|---|---|
| `POST` | `/queue/enqueue` | Положить кандидата в Redis Stream (422 на битых данных, 503 если Redis недоступен) |
| `GET` | `/health` | Живость API |

---

## 8. Как читать результат оценки

Реальный ответ на эталонного кандидата из ТЗ (`use_llm=false`):

```json
{
  "candidate_id": "245be5fb0908d59f6e89",
  "quality_score": 0.7783,
  "rating": "STRONG",
  "direction": "long",
  "win_rate": 0.7446,
  "favorable_adverse_ratio": 4.33,
  "context_freshness": "stale",
  "warning_flags": [
    "context_status_stale",
    "age_gt_120",
    "stale_and_aged_combined",
    "transition_rarity_common"
  ],
  "strengths": [
    "Очень высокий research_score (0.957)",
    "74.5% win rate при выборке 1151",
    "F/A ratio 4.33x — отличное соотношение",
    "Повторяемость в 19 месяцах — устойчивый паттерн",
    "Низкая сезонная концентрация (10.0%)"
  ],
  "risks": [
    "Контекст устарел (stale) — состояние может быть неактуальным",
    "Состояние группы > 120 мин — возможна смена рыночной фазы",
    "Переход 42->1 часто встречается (common) — менее специфичный сигнал"
  ],
  "summary": "STRONG LONG кандидат: quality_score=0.778, win_rate=74.5%, F/A ratio=4.33x.",
  "score_statistical": 1.0,
  "score_directional": 1.0,
  "score_context": 0.1667,
  "score_rarity": 0.6333
}
```

Как это читать:

- **Смотри на разбивку, а не только на total.** Здесь `statistical=1.0` и `directional=1.0` — статистика идеальна. Но `context=0.1667` — контекст почти на нуле: снимок устаревший и состояние висит больше 120 минут. То есть «историческая закономерность сильная, но применима ли она прямо сейчас — большой вопрос».
- **`rating` — это агрегат, а `warning_flags` — конкретика.** STRONG с четырьмя флагами требует куда большей осторожности, чем STRONG без флагов.
- **`win_rate` считается от `research_side`:** для `long` это `long_outcome_share`, для `short` — `1 − long_outcome_share`.
- **`favorable_adverse_ratio` есть только у long-кандидатов.** Поля `p70_long_favorable_pct` / `p80_long_adverse_pct` описывают движение вверх, поэтому для short он равен `null`, не участвует в оценке, и в `risks` появляется пояснение.
- **`context_freshness: stale` — не приговор, а срок годности.** Такой кандидат стоит перегенерировать из свежих данных, прежде чем на него опираться.

Полный текстовый отчёт (формат из ТЗ) можно получить из Python:

```python
from src.agent.pipeline import run_pipeline
from src.agent.report_formatter import format_report

evaluation = run_pipeline(candidate_dict, use_llm=False, save=False, print_report=True)
```

---

## 9. Формат входных данных

Кандидат — 37 полей, из них обязательных большинство. Полное описание блоков —
в [`README_agent_spec.md`](README_agent_spec.md); краткая карта:

| Блок | Поля |
|---|---|
| Identity | `candidate_id`, `symbol`, `configuration_hash`, `candidate_family_key`, `research_score` |
| State / Trajectory | `previous_group_id`, `current_group_id`, `transition_id`, `current_group_age_bucket`, `context_status`, `trajectory_entropy`, `transition_rarity` |
| Event Context | `event_block_id`, `primary_event_family`, `event_intensity_bucket`, `event_rarity_bucket`, `signature_atom_count`, `event_family_count`, `event_block_total_rows`, `event_block_row_share` |
| Historical Sample | `horizon`, `sample_size`, `valid_label_count`, `invalid_label_count`, `valid_label_pct`, `repeatability_days`, `repeatability_months`, `monthly_concentration` |
| Outcome Profile | `historical_bias_context`, `research_side`, `long_outcome_count`, `short_outcome_count`, `long_outcome_share`, `historical_outcome_skew` |
| Favorable / Adverse | `p70_long_favorable_pct`, `p80_long_adverse_pct`, `long_favorable_adverse_ratio_p70_p80` |

Опциональны: `configuration_hash`, `candidate_family_key`, `previous_group_id`,
`primary_event_family`. Всё остальное обязательно — иначе Pydantic вернёт 422 с
перечнем недостающих полей.

Принимаются три формата:

**1. JSON-объект** — основной путь (`/evaluate/json`, `/score/quick`).

**2. JSON array** — `/evaluate/batch`, `/conflicts`.

**3. Raw text** — `/evaluate/raw`, строки `ключ: значение` или `ключ = значение`:

```
candidate_id:        245be5fb0908d59f6e89
symbol:              BTCUSDT
research_score:      0.957   ← очень высокий
context_status:      stale
sample_size:         1339
```

Парсер срезает хвосты после `←`, игнорирует пустые строки, строки с `#` и любые
незнакомые ключи. **Молча** — если поле названо с опечаткой, оно просто не попадёт
в объект, и ты увидишь ошибку валидации «поле обязательно», а не «опечатка в имени».

Пример полного запроса:

```bash
curl -X POST http://localhost:8000/evaluate/json \
  -H "Content-Type: application/json" \
  -d '{
    "use_llm": false,
    "save": true,
    "candidate": {
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
  }'
```

---

## 10. Хранилища и что в них лежит

| Хранилище | Что в нём | Зачем |
|---|---|---|
| **PostgreSQL `candidates`** | Последняя оценка каждого `candidate_id` (upsert по PK) + `raw_payload` в JSONB + embedding | Текущее состояние: «что мы думаем про этого кандидата» |
| **PostgreSQL `candidate_events`** | Hypertable, append-only: каждая оценка — новая строка. Retention 90 дней | История: «как менялись оценки во времени» |
| **`hourly_candidate_stats`, `daily_group_stats`** | Continuous aggregates поверх `candidate_events` | Быстрая аналитика без сканирования сырых данных |
| **PostgreSQL `market_events`** | Hypertable под сырые рыночные события | Создана миграцией, но **кодом не используется** — задел на будущее |
| **pgvector `candidates.embedding`** | Вектор 32 (значимы первые 18 признаков, остальное — запас), HNSW-индекс, cosine | Поиск исторических аналогов текущего кандидата |
| **Neo4j** | Узлы `MarketGroup {group_id}`, рёбра `TRANSITION {transition_id, rarity, count, avg_horizon_return, avg_quality_score}` | Топология рынка: какие переходы куда ведут и насколько они хороши |
| **Redis** | `candidate:hash:*` (dedup, TTL 30 мин), `evaluation:*` (кэш оценок, TTL 30 мин), канал `btc:strong_candidates`, стрим `btc:candidates:stream` | Скорость, защита от повторной обработки, приём потока |

Дедупликация работает так: при `save=true` и наличии `configuration_hash` pipeline
проверяет Redis — если такая конфигурация уже оценивалась в последние 30 минут,
возвращается закэшированная оценка **без пересчёта и без вызова LLM**.

---

## 11. Конфигурация

| Переменная | По умолчанию | Описание |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Ключ Claude API. Нужен только при `use_llm=true` |
| `DATABASE_URL` | `postgresql://btc_user:btc_pass@localhost:5432/btc_graph` | PostgreSQL |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis: брокер, кэш, стрим |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j |
| `NEO4J_USER` / `NEO4J_PASSWORD` | `neo4j` / `btc_neo4j_pass` | Доступ к Neo4j |
| `USE_DB` | `true` | Писать в PostgreSQL |
| `USE_REDIS` | `true` | Использовать Redis (кэш, dedup, pub/sub) |
| `USE_GRAPH` | `true` | Обновлять граф Neo4j |

`USE_*` читаются **на момент импорта** `src/agent/pipeline.py` — менять их на лету
нельзя, нужен перезапуск процесса.

Модель LLM задаётся аргументом `llm_model` у `run_pipeline()`, по умолчанию
`claude-haiku-4-5-20251001`. Через HTTP API она не настраивается.

---

## 12. Отладка и типичные проблемы

| Симптом | Причина и что делать |
|---|---|
| `422` с длинным списком полей | В кандидате не хватает обязательных полей или значение enum написано неверно. Смотри `detail` в ответе |
| `/evaluate` возвращает 422 на, казалось бы, валидных данных | Роуты `/evaluate/*` заворачивают **любое** исключение в 422, включая внутренние ошибки. Смотри `docker compose logs api` |
| Оценка вернулась мгновенно и без изменений | Сработала дедупликация по `configuration_hash` в Redis (только для того же `candidate_id`). Передай `use_cache=false`, обнови хэш или подожди 30 мин |
| Ответ приходит, но в БД пусто | Сбой записи теперь пишется в лог: `docker compose logs api \| grep -i "не удалось"`. Проверь `make ps`, `make shell-pg`, `make migrate` |
| `/stats/*` возвращает пустой список | Continuous aggregates обновляются с задержкой, либо ещё не было событий. Проверь `/stats/events` |
| `make migrate` падает | Postgres ещё не готов. Подожди ~15 сек после `make up` и повтори |
| Правки в `src/` не применяются | Uvicorn в контейнере без `--reload`. Нужен `make build` |
| Ошибка про `ANTHROPIC_API_KEY` | Либо впиши реальный ключ в `.env` + `make reload`, либо работай с `use_llm=false` |
| Локальный запуск не видит `.env` | `load_dotenv()` вызывается в `src/main.py`, `src/worker/celery_app.py` и `alembic/env.py`. Если переменная всё равно не видна — проверь, что `.env` лежит в корне проекта и процесс запущен оттуда же; уже выставленные в окружении значения `load_dotenv` не перезаписывает |

Полезные команды:

```bash
docker compose logs -f api             # логи API
docker compose logs -f celery-worker   # логи воркера
make shell-pg                          # psql

# внутри psql:
\dt
SELECT candidate_id, rating, quality_score, evaluated_at FROM candidates ORDER BY evaluated_at DESC LIMIT 10;
SELECT count(*) FROM candidate_events;

# Redis:
docker compose exec redis redis-cli KEYS 'candidate:hash:*'
docker compose exec redis redis-cli XLEN btc:candidates:stream
```

---

## 13. Тесты и известные проблемы кода

### Тесты

215 тестов на чистую логику: парсер, валидатор, скорер, фильтры, pipeline, LLM-узел.
**Ни БД, ни Redis, ни Neo4j, ни ключа Anthropic не требуется** — внешние SDK
подменяются заглушками в `tests/conftest.py`, а `DATABASE_URL` уводится на
in-memory sqlite, так что тесты гарантированно не ходят в боевую базу.

```bash
make test          # внутри контейнера api
make test-local    # локально, нужен только pytest
pytest tests/test_scorer.py -v    # отдельный файл
```

| Файл | Что покрывает |
|---|---|
| `tests/test_scorer.py` | Каждая ступень каждой из 4 осей, веса, границы рейтингов, границы [0..1] |
| `tests/test_validator.py` | Каждый `warning_flag` и его пороговое значение |
| `tests/test_parser.py` | dict / JSON / raw text / array, опечатки, битые данные, выход за диапазоны |
| `tests/test_filters.py` | Фильтрация по порогу, дедуп по `family_key`, все 4 типа конфликтов |
| `tests/test_pipeline.py` | Детерминированная оценка, батч, дедупликация через Redis, деградация `_persist` |
| `tests/test_llm_node.py` | Разбор ответа Claude, извлечение текстового блока, fallback при сбое API |

Числа из раздела «Как читать результат оценки» зафиксированы тестом
`test_reference_candidate_is_strong` — если правишь пороги в скорере, тест упадёт
и напомнит обновить README.

### Исправлено

Все замечания аудита разобраны. Подробности с прогонами на живом стеке —
в [`docs/audit_findings.md`](docs/audit_findings.md).

| № | Проблема | Что сделано |
|---|---|---|
| 1 | Недостижимая ветка: `age_60_120` оценивался как `age_lt_30` | Карта `_AGE_SCORE` с градацией 1.0 / 0.75 / 0.5 / 0.0 |
| 2 | ORM-объекты читались после закрытия сессии → 500 | `expire_on_commit=False` + сбор словарей внутри сессии |
| 3 | `refresh_continuous_aggregates` падал в транзакции | Вызов на соединении с `isolation_level="AUTOCOMMIT"` |
| 4 | `_persist()` глушил все исключения | Логирование + статус по каждому хранилищу; то же в `graph_repo` и `redis_cache` |
| 6 | Дедупликация отдавала чужую оценку | Кэш только при совпадении `candidate_id`; отдельный параметр `use_cache` |
| 7 | Локальный запуск не читал `.env` | `load_dotenv()` в `main`, `celery_app`, `alembic/env` |
| 8 | Short-кандидатам начислялись баллы за long-метрику | `fa_ratio_for()` → `None` для short, ось `directional` по 2 критериям |
| 9 | Слабый `fresh` вытеснял сильный `stale` | Свежесть стала бонусом `FRESH_BONUS = 0.05` к score |
| 10 | Пороги рейтинга в 4 местах | Единственный `get_rating()` + константы |
| 11 | Вызов Claude без обработки ошибок | Таймаут, ретраи, fallback на детерминированную оценку |
| 12 | Внутренние ошибки маскировались под 422 | `_fail()`: вход → 422, остальное → 500 с логом |
| 13 | Redis Stream без consumer groups | `XREADGROUP` + `XACK`; `/queue/enqueue` больше не врёт про успех |
| 14 | Ретраи невалидных данных | `ValidationError` завершает задачу без повтора |
| 15 | Кириллическая «Б» в `AgeБucket` | Переименован в `AgeBucket` |
| 17 | Вектор 384 при 18 признаках | `VECTOR_DIM = 32` + миграция 003 с пересчётом из `raw_payload` |
| 19 | `.env` мог уехать в git, порты наружу | `.gitignore`, `.dockerignore`, порты на `127.0.0.1`, чистый `alembic.ini` |
| 20 | **Найдено при проверке:** воркер в Docker не слушал очередь `evaluate` | Добавлен `-Q evaluate,celery` — асинхронный путь не работал вовсе |

**Замечание №5 отозвано:** инкрементальное среднее в Cypher было корректно
изначально — Neo4j вычисляет правые части `SET` от состояния до предложения.
Проверено на живом Neo4j 5.

**Что осталось:**

- API без аутентификации и rate-limit — обязательно до выхода за пределы локальной машины;
- симметричных short-метрик (`p70_short_favorable_pct`) в данных нет — нужно от генератора кандидатов;
- интеграционных тестов нет: репозитории, Cypher и Celery проверены только ручными прогонами.

### Изменения, влияющие на результаты

Правки №1, №8 и №9 меняют выдачу — оценки, сохранённые до них, несравнимы с новыми:

| Что | Было | Стало |
|---|---|---|
| `age_30_60` / `age_60_120` | тот же балл, что у `age_lt_30` | −0.017 / −0.033 к `quality_score` |
| Short-кандидаты | F/A ratio давал баллы по оси 35% | Ось считается по 2 критериям, `favorable_adverse_ratio: null` |
| Выбор внутри `family_key` | любой `fresh` бил любой `stale` | score + бонус 0.05 за свежесть |

---

## 14. Ограничения системы

Не технические баги, а свойства подхода — их важно понимать до использования:

- **Только `BTCUSDT`.** Пороги и веса откалиброваны под него; на других активах результат ничего не значит.
- **`research_score` ≠ вероятность прибыли.** Это оценка силы исторической аналогии.
- **Кандидат ≠ торговый сигнал.** Нет entry timing, нет стопов, нет размера позиции, нет портфельного контекста.
- **Горизонт `24h`.** Оценка релевантна только для краткосрочных движений, несмотря на то что исходная концепция говорит о фазах рынка на месяцы вперёд.
- **`stale` контекст** означает, что состояние могло смениться — кандидата надо перегенерировать.
- **`common` transition** — частые переходы менее информативны, даже при отличной статистике.
- **Прошлое ≠ будущее.** Все метрики описывают историю; структурный сдвиг рынка обнуляет их ценность, и система об этом сама не узнает.

---

## Документация по шагам разработки

| Документ | Содержание |
|---|---|
| [`README_agent_spec.md`](README_agent_spec.md) | Исходное ТЗ: концепция, все поля, задачи агента, обоснование стека |
| [`docs/step_01_mvp_structure.md`](docs/step_01_mvp_structure.md) | Модели, парсер, валидатор, scorer, LLM node, FastAPI |
| [`docs/step_02_persistence_layer.md`](docs/step_02_persistence_layer.md) | PostgreSQL, pgvector, Neo4j, Redis, Alembic |
| [`docs/step_03_celery_timescale.md`](docs/step_03_celery_timescale.md) | Celery Worker + Beat, TimescaleDB continuous aggregates |
| [`docs/audit_findings.md`](docs/audit_findings.md) | Разбор найденных ошибок логики и инфраструктуры |
