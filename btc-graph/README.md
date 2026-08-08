# Crypto Market Candidate Agent

Агент анализа рыночных кандидатов: принимает снимок текущей конфигурации рынка с историческими метриками и отвечает на вопрос **«насколько этой исторической аналогии можно доверять и в какую сторону она смещена»**.

Система мультимонетная: калибровка оценки задаётся **профилем на символ** (`config/symbols/*.yaml`), а не зашита в код. Подробности — раздел [15](#15-мультимонетность-и-профили).

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
15. [Мультимонетность и профили](#15-мультимонетность-и-профили)

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
      │                          Redis Stream «candidates:stream»
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

Синхронный путь — интерактивная работа, сразу видишь ответ. Асинхронный — пачки кандидатов и фоновая обработка; результат забирается потом через `GET /candidates/{symbol}/{id}` или подпиской на Redis-канал `candidates:strong:{symbol}` (либо `candidates:strong:all` для всех монет сразу).

### Формула quality_score

```
quality_score = w_stat·statistical + w_dir·directional + w_ctx·context + w_rar·rarity
```

**Веса и все ступени берутся из профиля монеты** (`config/symbols/<SYMBOL>.yaml`), в коде их нет. Ниже — базовый профиль `_default.yaml`, он же калибровка под BTCUSDT:

| Ось | Вес | Из чего складывается | 1.0 при |
|---|---|---|---|
| **statistical** | 30% | `valid_label_pct`, `sample_size`, `monthly_concentration`, `repeatability_months` | > 0.85, > 1000, < 0.10, > 15 |
| **directional** | 35% | win rate, \|`outcome_skew`\|, F/A ratio (только long) | > 0.70, > 0.40, > 4.0 |
| **context** | 20% | `context_status`, `age_bucket`, `trajectory_entropy` | fresh, `age_lt_30`, low |
| **rarity** | 15% | `event_rarity_bucket`, `transition_rarity`, `research_score` | rare, rare, > 0.85 |

Итоговый рейтинг по базовому профилю: `STRONG` ≥ 0.75, `MODERATE` ≥ 0.55, иначе `WEAK`. У монеты со своим профилем границы свои — **никогда не сравнивай `quality_score` с числом напрямую**, зови `get_rating(score, profile)`.

Рядом с профильным считается **`quality_score_baseline`** — тот же кандидат по базовой калибровке. Профильный score ранжирует кандидатов внутри монеты и между монетами не сравним; baseline существует ровно для вопроса «какая монета сегодня интереснее».

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
├── config/profiles.py        ScoringProfile (Pydantic) + ленивый загрузчик
│                             config/symbols/*.yaml со слиянием от _default.
│                             ЗДЕСЬ ЖИВУТ ВСЕ ПОРОГИ — точнее, в YAML рядом.
│                             Реестр профилей = whitelist известных монет.
├── scorer/candidate_scorer.py
│                             score_candidate(c, profile) → ScoreBreakdown
│                             get_rating(score, profile) → STRONG / MODERATE / WEAK
│                             Порогов не содержит: только форма формулы —
│                             какие признаки в какую ось и как складываются оси.
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
│   ├── graph_repo.py         Neo4j: upsert узлов/рёбер по ключу (symbol, group_id)
│   └── stats_repo.py         SQL к continuous aggregates TimescaleDB
├── cache/redis_cache.py      dedup по (symbol, configuration_hash), кэш оценок,
│                             pub/sub STRONG на монету + общий канал,
│                             Redis Stream как общая очередь приёма
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

make reload        # после правки .env, docker-compose.yml или профиля монеты
make build         # после правки кода в src/ или Dockerfile
make migrate       # после добавления новой миграции Alembic

make profiles-check  # проверить YAML-профили: схема, лесенки, веса
make migrate-graph   # разово: symbol узлам Neo4j + constraint (symbol, group_id)

make shell-api     # bash внутри контейнера api
make shell-pg      # psql внутри postgres
```

Профили (`config/`) смонтированы volume'ом: правка порогов требует `make reload`,
а не `make build`.

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
curl -s http://localhost:8000/candidates/BTCUSDT/245be5fb0908d59f6e89 | jq
```

Подписаться на STRONG-кандидатов в реальном времени:

```bash
# одна монета
docker compose exec redis redis-cli SUBSCRIBE candidates:strong:BTCUSDT
# все монеты сразу
docker compose exec redis redis-cli SUBSCRIBE candidates:strong:all
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
// самые частые переходы в состояние 1.0 КОНКРЕТНОЙ монеты
// symbol обязателен: «группа 1.0» есть в графе каждой монеты
MATCH (src:MarketGroup {symbol: 'BTCUSDT'})
      -[t:TRANSITION]->
      (dst:MarketGroup {symbol: 'BTCUSDT', group_id: 1.0})
RETURN src.group_id, t.transition_id, t.rarity, t.count, t.avg_quality_score
ORDER BY t.count DESC LIMIT 20;

// граф одной монеты
MATCH p=(:MarketGroup {symbol: 'BTCUSDT'})-[:TRANSITION]->() RETURN p LIMIT 100;

// проверка изоляции: сколько узлов у каждой монеты
MATCH (g:MarketGroup) RETURN g.symbol, count(g) ORDER BY g.symbol;
```

### Сценарий G — «перекалибровать оценку под себя»

Все пороги — в YAML-профиле монеты, в коде их нет:

```bash
$EDITOR config/symbols/BTCUSDT.yaml   # ступени, веса, границы рейтинга, пороги валидатора
make profiles-check                   # схема, монотонность лесенок, сумма весов
make reload                           # конфиг смонтирован volume'ом — build не нужен
```

Что где лежит в профиле:

| Что меняешь | Ключ профиля |
|---|---|
| ступени оси | `statistical.*`, `directional.*`, `context.*`, `rarity.*` |
| веса осей | `weights` |
| границы STRONG / MODERATE | `rating.strong_min`, `rating.moderate_min` |
| порог попадания в батч и надбавку за свежесть | `batch` |
| пороги warning-флагов | `validator` |
| подсказку про рынок для LLM | `llm.market_hint` |

Три правила, без которых перекалибровка превращается в подкрутку на глаз:

1. **Бампни `version`.** Каждая запись помечается меткой профиля (`BTCUSDT@2`) и отпечатком его содержимого. Пересчёта сохранённых оценок нет и не будет: сравнение идёт внутри одного `scoring_profile`. Правка порогов без бампа версии всё равно видна — по изменившемуся `profile_fingerprint`.
2. **Правка `_default.yaml` двигает линейку всем монетам сразу** и ломает снапшот-тест скорера (`tests/test_scorer_snapshot.py`) — намеренно. Если перекалибровка осознанна: `python3 scripts/make_scorer_snapshot.py` и обнови раздел 8 этого README.
3. **Профиль, переводящий sanity-кандидата монеты через границу рейтинга, — сломанный профиль.** Якоря лежат в `tests/fixtures/sanity_candidates.json`.

Подбор профиля для новой монеты — не правка руками, а процедура: раздел [15](#15-мультимонетность-и-профили).

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

Значения по умолчанию: `use_llm=true`, `save=true`. `min_quality_score` по умолчанию **не задан** — порог берётся из профиля каждой монеты; явное число применяется ко всему батчу и перекрывает профили. Для повседневной работы обычно нужен `use_llm=false`.

Символ в этих ручках не спрашивается — он берётся из самого кандидата. В ответ добавлены `symbol`, `scoring_profile`, `profile_fingerprint` и `quality_score_baseline`.

### Хранилище

| Метод | Путь | Описание |
|---|---|---|
| `GET` | `/candidates/{symbol}/{id}` | Сохранённая оценка по паре (монета, `candidate_id`) |
| `GET` | `/candidates/strong/{direction}?symbol=BTCUSDT&limit=20` | Последние STRONG (`long` / `short`). `symbol` **обязателен**; `symbol=all` — смешанная выдача |
| `POST` | `/candidates/similar?limit=10&cross_symbol=false` | Похожие через pgvector (cosine). По умолчанию в пределах монеты кандидата; `cross_symbol=true` — по всем монетам |

Вектор pgvector намеренно symbol-agnostic: `cross_symbol=true` — рабочий сценарий «а у BTC такая конфигурация уже была?». Изоляция сделана фильтром в SQL, а не свойством вектора.

### Граф

| Метод | Путь | Описание |
|---|---|---|
| `GET` | `/graph/group/{group_id}?symbol=BTCUSDT` | Атрибуты узла MarketGroup. `symbol` **обязателен** |
| `GET` | `/graph/transitions/to/{group_id}?symbol=BTCUSDT&rarity=uncommon,rare` | Входящие переходы с фильтром по редкости |
| `GET` | `/graph/symbols` | Монеты, представленные в графе — проверка изоляции |

`symbol` обязателен потому, что `group_id` и `transition_id` — идентификаторы **внутри** графа одной монеты: «группа 7» есть у каждой и означает у каждой свой рынок.

### Статистика (TimescaleDB)

| Метод | Путь | Описание |
|---|---|---|
| `GET` | `/stats/hourly?hours=24&symbol=` | Почасовые агрегаты по symbol × профиль × direction × rating |
| `GET` | `/stats/groups?days=7&symbol=BTCUSDT&group_id=1.0` | Дневные агрегаты по состояниям. Фильтр по `group_id` **требует** `symbol` |
| `GET` | `/stats/ratings?days=7&symbol=` | Распределение STRONG / MODERATE / WEAK |
| `GET` | `/stats/events?limit=50&symbol=` | Сырые последние события из hypertable |
| `GET` | `/stats/symbols?days=30` | Монеты, по которым были оценки |

`symbol` здесь опционален; в ответе он есть всегда. `scoring_profile` входит в `GROUP BY` агрегатов — средние не склеивают оценки, посчитанные разными калибровками.

> `/stats/hourly`, `/stats/groups` и `/stats/ratings` читают **continuous aggregates**,
> а не сырую таблицу. У них политика обновления с `end_offset` в 1 час (для дневных — 1 день),
> поэтому **самые свежие события там появляются с задержкой**. Если нужны данные «прямо сейчас» —
> используй `/stats/events`, он читает hypertable напрямую.

### Очередь и служебное

| Метод | Путь | Описание |
|---|---|---|
| `POST` | `/queue/enqueue?symbol=` | Положить кандидата в Redis Stream (422 на битых данных, 503 если Redis недоступен). `?symbol=` проставляет монету payload'у, у которого её нет |
| `GET` | `/config/symbols` | Реестр профилей: версии, отпечатки, ошибки загрузки |
| `GET` | `/config/symbols/{symbol}` | Эффективный профиль после слияния с `_default` — то, чем реально считается |
| `POST` | `/config/reload` | Перечитать профили без рестарта. Выключено по умолчанию (`ENABLE_CONFIG_RELOAD`) |
| `GET` | `/health` | Живость API |

---

## 8. Как читать результат оценки

Реальный ответ на эталонного кандидата из ТЗ (`use_llm=false`):

```json
{
  "candidate_id": "245be5fb0908d59f6e89",
  "symbol": "BTCUSDT",
  "scoring_profile": "BTCUSDT@2",
  "profile_fingerprint": "a0907c2f5435",
  "quality_score": 0.6133,
  "quality_score_baseline": 0.7783,
  "rating": "MODERATE",
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
  "summary": "MODERATE LONG кандидат по BTCUSDT: quality_score=0.613, win_rate=74.5%, F/A ratio=4.33x.",
  "score_statistical": 0.45,
  "score_directional": 1.0,
  "score_context": 0.1667,
  "score_rarity": 0.6333
}
```

> **Почему `quality_score` (0.613) и `quality_score_baseline` (0.778) разошлись.** Это тот самый эталонный кандидат из ТЗ, и по **базовой линейке** он по-прежнему даёт 0.7783 STRONG — эти числа зафиксированы снапшот-тестом и не меняются. Но у BTCUSDT с версии 2 **своя калибровка** по реальной выгрузке, и по ней тот же кандидат — MODERATE.
>
> Разница целиком в оси `statistical`: 1.0 против 0.45. Выборка кандидата (1339 случаев) и сезонная концентрация (10.0%) были выдающимися для калибровки шага 1, но на девятилетней истории типичный кандидат накопил кратно больше — p50 концентрации на свежих данных 6.3%. Базовая линейка перестала различать верхушку, калибровка это исправила.
>
> Мораль: **`baseline` — это «как бы оценил эталонный измеритель», `quality_score` — «насколько кандидат хорош среди своих».** Первое сравнимо между монетами, второе точнее внутри монеты.

Как это читать:

- **Смотри на разбивку, а не только на total.** Здесь `directional=1.0` — перекос идеален. Но `context=0.1667` — контекст почти на нуле: снимок устаревший и состояние висит больше 120 минут. То есть «историческая закономерность заметная, но применима ли она прямо сейчас — большой вопрос».
- **`rating` — это агрегат, а `warning_flags` — конкретика.** STRONG с четырьмя флагами требует куда большей осторожности, чем STRONG без флагов.
- **`win_rate` считается от `research_side`:** для `long` это `long_outcome_share`, для `short` — `1 − long_outcome_share`.
- **`favorable_adverse_ratio` есть только у long-кандидатов.** Поля `p70_long_favorable_pct` / `p80_long_adverse_pct` описывают движение вверх, поэтому для short он равен `null`, не участвует в оценке, и в `risks` появляется пояснение.
- **`context_freshness: stale` — не приговор, а срок годности.** Такой кандидат стоит перегенерировать из свежих данных, прежде чем на него опираться.
- **`quality_score` сравним только внутри монеты.** Он посчитан профилем `scoring_profile`, у которого свои ступени и свои границы рейтинга. Чтобы сравнить кандидатов разных монет, бери `quality_score_baseline` — он всегда посчитан базовой калибровкой. У BTCUSDT эти два числа совпадают: его профиль и есть базовый.
- **`profile_fingerprint` отвечает на вопрос «той же ли линейкой это меряли».** Если у двух записей одинаковый `scoring_profile`, но разные отпечатки — между ними правили пороги без бампа версии, и сравнивать их нельзя.

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
| **Redis** | `{symbol}:candidate:hash:*` (dedup, TTL 30 мин), `{symbol}:evaluation:*` (кэш оценок, TTL 30 мин), каналы `candidates:strong:{symbol}` и `candidates:strong:all`, стрим `candidates:stream` | Скорость, защита от повторной обработки, приём потока |

Дедупликация работает так: при `save=true` и наличии `configuration_hash` pipeline
проверяет Redis — если такая конфигурация **этой же монеты** уже оценивалась
в последние 30 минут, возвращается закэшированная оценка **без пересчёта
и без вызова LLM**.

Символ в ключе дедупликации обязателен: `configuration_hash` описывает
конфигурацию графа, и совпадение между монетами возможно. Без префикса ETH
получил бы готовую оценку BTC — и выглядел бы при этом нормально оценённым
кандидатом, что почти невозможно заметить по данным.

Стрим приёма, наоборот, **общий на все монеты**: символ лежит в payload,
и дробить пул воркеров по монетам незачем.

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
| `PROFILES_DIR` | `config/symbols` | Каталог профилей калибровки. В контейнерах смонтирован volume'ом — правка не требует `make build` |
| `ENABLE_CONFIG_RELOAD` | `false` | Разрешить `POST /config/reload`. Ручка меняет поведение скоринга, а API без аутентификации — включать только локально |

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
SELECT symbol, candidate_id, rating, quality_score, scoring_profile, evaluated_at
  FROM candidates ORDER BY evaluated_at DESC LIMIT 10;
SELECT symbol, count(*) FROM candidates GROUP BY symbol;
SELECT count(*) FROM candidate_events;

# Redis:
docker compose exec redis redis-cli KEYS 'BTCUSDT:candidate:hash:*'
docker compose exec redis redis-cli XLEN candidates:stream

# Профили:
make profiles-check
curl -s localhost:8000/config/symbols | python3 -m json.tool
curl -s 'localhost:8000/config/symbols/ETHUSDT' | python3 -m json.tool  # эффективный профиль

# Изоляция графа по монетам:
curl -s localhost:8000/graph/symbols
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

- **Профильный `quality_score` между монетами не сравним.** У каждой монеты своя линейка — в этом весь смысл профилей. Для межмонетного сравнения есть `quality_score_baseline`, и только он.
- **Монета без своего профиля меряется линейкой BTC.** Она получает флаг `unknown_symbol_profile`, но не отвергается. На инструменте с короткой историей это почти гарантированно означает «всё WEAK» — и это свойство линейки, а не кандидатов.
- **`research_score` ≠ вероятность прибыли.** Это оценка силы исторической аналогии.
- **Кандидат ≠ торговый сигнал.** Нет entry timing, нет стопов, нет размера позиции, нет портфельного контекста.
- **Горизонт `24h`.** Оценка релевантна только для краткосрочных движений, несмотря на то что исходная концепция говорит о фазах рынка на месяцы вперёд.
- **`stale` контекст** означает, что состояние могло смениться — кандидата надо перегенерировать.
- **`common` transition** — частые переходы менее информативны, даже при отличной статистике.
- **Прошлое ≠ будущее.** Все метрики описывают историю; структурный сдвиг рынка обнуляет их ценность, и система об этом сама не узнает.


---

## 15. Мультимонетность и профили

### Зачем профили вообще нужны

Дело не в том, что где-то было написано `BTCUSDT` — это как раз мелочь. Проблема в порогах: `sample_size > 1000`, `repeatability_months > 15`, `monthly_concentration < 0.10` — характеристики рынка с десятилетней историей и высокой ликвидностью. Монета с двумя годами торгов и в 50 раз меньшим объёмом их не наберёт никогда: **все её кандидаты уедут в `WEAK`, и система для неё просто замолчит** — не потому, что кандидатов нет, а потому что линейка чужая.

Профиль — способ дать каждой монете свою линейку, не размножая код.

### Где что лежит

```
config/symbols/
├── _default.yaml      ← неподвижная общая линейка: baseline и монеты без профиля
├── BTCUSDT.yaml       ← своя калибровка с версии 2 (было пустое наследование)
├── ETHUSDT.yaml       ← откалиброван по 9628 свежим кандидатам
├── SOLUSDT.yaml       ← откалиброван по 5710 свежим кандидатам
└── <SYMBOL>.yaml      ← только отличия, остальное наследуется
```

Профиль читается **лениво, при первом обращении**, и битый YAML одной монеты не роняет API: файл пропускается с логом, монета считается базовой калибровкой. Единственное исключение — сам `_default`: без него считать нечем.

**Реестр профилей и есть whitelist монет.** Отдельного списка символов в проекте нет: монета, у которой нет своего YAML, оценивается базовой калибровкой и получает `warning_flags: ["unknown_symbol_profile"]`.

Схема профиля валидируется при загрузке (Pydantic), и это не формальность:

| Проверка | Что ловит |
|---|---|
| сумма `weights` = 1.0 ± 1e-6 | score, который перестал лежать в [0, 1] |
| пороги `ladder` строго монотонны | **недостижимую ступень** — машинная защита от повторения audit #1 |
| верхняя ступень доли достижима | порог `1.0` у `higher_better` при строгом сравнении не сработает никогда. Ровно такую мёртвую ступень предложил калибратор на реальных данных ETH: p90 доли валидных меток оказался равен 1.0 |
| enum-карты покрывают все значения | `KeyError` на живом кандидате вместо «значения по умолчанию» |
| `strong_min > moderate_min` | рейтинг MODERATE, в который нельзя попасть |

### Семантика `ladder`

Список `[порог, балл]` проверяется сверху вниз, сравнение **строгое**: `higher_better` → `value > порог`, `lower_better` → `value < порог`. Ни одна ступень не сработала → `floor`. Ровно поэтому `valid_label_pct = 0.85` даёт 0.7, а не 1.0.

При наследовании список `ladder` заменяется **целиком** (частично унаследованной лесенки не бывает — её нельзя прочитать по одному файлу), а `mode` и `floor` рядом с ней наследуются. Enum-карты, наоборот, сливаются по ключам: подвинуть один бакет можно, не переписывая все четыре.

### Как завести новую монету

```bash
# 1. Выгрузить кандидатов монеты из генератора. ORDER BY ts ОБЯЗАТЕЛЕН:
#    в схеме кандидата нет поля времени, и скрипт считает порядок строк
#    хронологическим — на нём держатся --tail и --holdout.
docker compose exec -T postgres psql -U btc_user -d btc_graph -tA -c \
  "SELECT payload::text FROM processing.candidates \
   WHERE symbol='ETHUSDT' ORDER BY ts;" > dumps/eth_candidates.jsonl

# 2. Черновик профиля от перцентилей её распределения
python3 scripts/calibrate_profile.py --symbol ETHUSDT \
    --input dumps/eth_candidates.jsonl --holdout 0.3 --dry-run

# 3. Устроил отчёт — записать
python3 scripts/calibrate_profile.py --symbol ETHUSDT \
    --input dumps/eth_candidates.jsonl --weights major-alt \
    --out config/symbols/ETHUSDT.yaml

# 4. Проверить схему и положить sanity-якорь
make profiles-check
$EDITOR tests/fixtures/sanity_candidates.json   # иначе упадёт test_sanity_candidates
make test-local

# 5. Подхватить без пересборки
make reload
```

Скрипт выдаёт **черновик**, а не готовый профиль: перцентили не знают, что `sample_size = 120` статистически несостоятелен независимо от того, что это p90 конкретной монеты.

#### Два умолчания, которые важно понимать

**`--tail 0.25` — калибруем по свежей четверти выгрузки, а не по всей истории.** `sample_size` кандидата это число накопленных аналогов на момент выпуска, и оно монотонно растёт: кандидат 2018 года видит сотню случаев, кандидат 2026-го — тысячи. Перцентили по всей истории ставят планку туда, где сидели бедные ранние годы, а применяется профиль к сегодняшним кандидатам. На реальной выгрузке ETH разница видна прямо: калибровка по всей истории дала 10% STRONG на калибровочной части и **26% на свежей трети**. `--tail 0` — считать по всему файлу.

**`--target-strong 0.01` — целимся в селективность эталона.** BTCUSDT по базовому профилю даёт 1.0% STRONG / 37.6% MODERATE / 61.4% WEAK. Рейтинг сравним только внутри монеты, но человек смотрит сводный список «что сегодня сильного»: если у одной монеты STRONG выдаётся вдесятеро щедрее, она этот список зальёт, и внимание уедет к ней не потому, что там интереснее, а потому что планка ниже.

#### Что получилось на реальных данных

| | BTC | ETH | SOL |
|---|---|---|---|
| баров истории | 314 177 | 314 164 | 209 986 |
| состояний в графе | 43 | 29 | 32 |
| профиль | `btc`, ступени вверх | `major-alt`, ступени вверх | `low-liquidity`, ступени вверх |
| `statistical` до калибровки | 0.822 (σ 0.143) | 0.827 (σ 0.121) | 0.736 (σ 0.137) |
| `statistical` после | 0.496 (σ 0.224) | 0.495 (σ 0.222) | 0.496 (σ 0.223) |
| STRONG на калибровочной части | 1.0% | 1.0% | 1.2% |
| STRONG на отложенной трети | 0.2% | 1.9% | 2.9% |

Главный вывод неожиданный: **линейка `_default` оказалась не слишком строгой, а слишком мягкой — и для альткоинов, и для самого биткоина.** Ось `statistical` насыщалась у всех трёх монет: почти все кандидаты упирались в потолок, и ось переставала различать. Причина в накоплении — ступень `sample_size > 1000` ставилась под эталонный кандидат ТЗ с выборкой 1339, а за девять лет истории типичная выборка выросла кратно. Калибровка вернула оси разрешающую способность везде.

Отсюда смена роли `_default`: он больше не «калибровка под BTC», а **неподвижная общая мерка** для `quality_score_baseline`. Его не калибруют — иначе обесценятся все накопленные baseline разом.

Проверять фактом, а не ожиданием «альткоину будет тесно»: направление, в которое надо двигать ступени, заранее неизвестно.

### Шесть правил калибровки

1. **Сначала ступени, потом веса.** Вес перераспределяет вклад оси, но ничего не чинит, если ось вырождена. У монеты с медианным `sample_size = 180` ось `statistical` даст ~0.25 при любом весе.
2. **Ступени — от перцентилей выборки монеты, не от круглых чисел.** `higher_better` → p90/p75/p50 → 1.0/0.7/0.4; `lower_better` → p10/p25/p50.
3. **Три диагноза читаются по распределению** (их печатает `calibrate_profile.py`):

   | Симптом | Диагноз | Что двигать |
   |---|---|---|
   | >90% `WEAK`, медиана ~0.35 | ступени взяты с BTC, монете недостижимы | ступени вниз, веса не трогать |
   | >40% `STRONG` | ступени слишком мягкие, score перестал различать | ступени вверх и/или `strong_min` вверх |
   | разброс оси σ < 0.05 | ось константна, её вес просто сдвигает всем score | перераспределить вес на различающие оси |

4. **Веса отражают, чему на этой монете можно верить.** Отправная точка (пресеты `--weights`):

   | Тип монеты | statistical | directional | context | rarity | Почему |
   |---|---|---|---|---|---|
   | `btc` (эталон) | 0.30 | 0.35 | 0.20 | 0.15 | текущая калибровка |
   | `major-alt` (ETH, SOL) | 0.35 | 0.30 | 0.20 | 0.15 | выборки короче → надёжность важнее перекоса |
   | `low-liquidity` | 0.40 | 0.25 | 0.20 | 0.15 | directional шумит: перекос на 200 случаях чаще артефакт |
   | `young` (< 1 года) | 0.40 | 0.25 | 0.25 | 0.10 | rarity недостоверна — «редкость» на коротком окне не редкость |

   Логика: **чем меньше данных, тем больше веса статистической надёжности и меньше — directional и rarity**, потому что именно они на малых выборках дают ложную уверенность.

5. **Два обязательных якоря.** Регрессия базовой линейки: `tests/test_scorer_snapshot.py` сверяет 200 кандидатов с эталоном, снятым до перехода на профили, — он сторожит `_default`, а не какую-то монету. Sanity-кандидат монеты: `tests/fixtures/sanity_candidates.json`, по одному реальному кандидату на монету с зафиксированным рейтингом. Профиль без якоря завести нельзя — `test_every_profile_has_a_sanity_anchor` не даст.
6. **Калибровка — событие, а не правка.** Меняешь профиль → бампаешь `version`. Старые записи остаются со старым `scoring_profile`, новые пишутся с новым; пересчёта нет, сравнение идёт внутри одного профиля. `profile_fingerprint` ловит правку порогов без бампа версии.

Защита от подгонки под прошлое: `--holdout 0.3` калибрует по первым 70% выгрузки (по времени, не случайно) и печатает распределение на отложенных 30%. Расходятся доли рейтингов — профиль подогнан.

### Изоляция данных между монетами

Это самая тихая часть шага: ошибки здесь не падают, а портят данные.

| Где | Ключ | Что было бы без символа |
|---|---|---|
| Neo4j | `MarketGroup {symbol, group_id}`, `symbol` на ребре | «Группа 1.0» BTC и ETH схлопнулись бы в один узел, `avg_quality_score` усреднился по разным рынкам |
| Redis dedup | `{symbol}:candidate:hash:*` | ETH получил бы готовую оценку BTC при совпадении `configuration_hash` |
| PostgreSQL | PK `(symbol, candidate_id)`, `(event_time, symbol, candidate_id)` | Перезапись строк при не-глобально-уникальных id |
| Continuous aggregates | `symbol` + `scoring_profile` в `GROUP BY` | Средние по всем монетам и всем калибровкам сразу |
| `select_best_per_family` | `(symbol, family_key)` | Кандидат одной монеты вытеснял бы кандидата другой из «семьи» |
| `detect_conflicts` | `(symbol, transition_id)` | BTC-long и ETH-short на «42->1» дали бы ложный `contradictory_direction` |

Единственное, что намеренно **не** изолировано, — вектор pgvector: он symbol-agnostic, чтобы можно было спросить «а у BTC такая конфигурация была?». Изоляция там делается фильтром в SQL (`?cross_symbol=`).

### Разовая миграция перед первой не-BTC монетой

```bash
make migrate         # включая 004_multi_symbol — ПРОЧИТАЙ шапку миграции
make migrate-graph   # symbol узлам Neo4j + constraint (symbol, group_id)
```

⚠️ **`004_multi_symbol` пересоздаёт оба continuous aggregate** — `ALTER` для них TimescaleDB не поддерживает. Пересборка идёт из `candidate_events` с retention 90 дней, поэтому **агрегаты старше 90 дней будут потеряны безвозвратно**. Если история важна, сними дамп до миграции (команды — в шапке файла миграции).

⚠️ **Redis-ключи и стрим переименованы** (`btc:*` → `candidates:*`). Ключи не мигрируем: TTL 30 минут, протухнут сами. Стрим переименовывать безопасно только при пустой очереди — проверь `XLEN btc:candidates:stream` перед деплоем.

`make migrate-graph` прогнать **до** первого кандидата не по BTCUSDT: constraint не создастся на узлах без `symbol`, и это защита, а не помеха.

---

## Документация по шагам разработки

| Документ | Содержание |
|---|---|
| [`README_agent_spec.md`](README_agent_spec.md) | Исходное ТЗ: концепция, все поля, задачи агента, обоснование стека |
| [`docs/step_01_mvp_structure.md`](docs/step_01_mvp_structure.md) | Модели, парсер, валидатор, scorer, LLM node, FastAPI |
| [`docs/step_02_persistence_layer.md`](docs/step_02_persistence_layer.md) | PostgreSQL, pgvector, Neo4j, Redis, Alembic |
| [`docs/step_03_celery_timescale.md`](docs/step_03_celery_timescale.md) | Celery Worker + Beat, TimescaleDB continuous aggregates |
| [`docs/step_04_multi_symbol.md`](docs/step_04_multi_symbol.md) | Мультимонетность: профили оценки на символ, изоляция данных по монетам |
| [`docs/audit_findings.md`](docs/audit_findings.md) | Разбор найденных ошибок логики и инфраструктуры |
