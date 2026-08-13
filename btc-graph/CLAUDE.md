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

# Профили калибровки на монету
make profiles-check    # схема YAML, монотонность лесенок, сумма весов, полнота enum-карт
make migrate-graph     # РАЗОВО: symbol узлам Neo4j + constraint (symbol, group_id)
```

Локальный запуск без Docker: нужны поднятые PostgreSQL (с `vector` и `timescaledb`), Redis, Neo4j;
`uvicorn src.main:app --reload` + `celery -A src.worker.celery_app worker -Q evaluate,celery` + `celery ... beat`.
Без инфраструктуры: `export USE_DB=false USE_REDIS=false USE_GRAPH=false` — оценка работает полностью, просто ничего не сохраняется.

## Что это за система

Проект — **фильтр и интерпретатор между внешним генератором кандидатов и человеком**. Генератор
(соседний каталог того же репозитория, `../btc-graph-processing`) выдаёт «кандидата» — досье
вида «такая конфигурация рынка уже была 1339 раз, в 74% случаев цена шла вверх». Здесь кандидат
оценивается (`quality_score` [0..1]), объясняется, фильтруется и накапливается в хранилищах.

Система мультимонетная. Калибровка оценки — **профиль на символ** (`config/symbols/*.yaml`), и это
не косметика: пороги вида `sample_size > 1000` откалиброваны под рынок с десятилетней историей,
и для монеты помоложе та же линейка означает «всё WEAK». Реестр профилей заодно работает whitelist'ом
монет — отдельного списка символов нет.

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
- **асинхронный** — `POST /queue/enqueue` → Redis Stream `candidates:stream` → Celery Beat
  каждые 10 сек запускает `process_stream_batch` (XREADGROUP по группе `candidates:workers`,
  затем XACK+XDEL) → `.delay()` на каждого → очередь **`evaluate`** → воркеры.

Хранилища (`_persist`, каждое отключается своим `USE_*`): PostgreSQL `candidates` (upsert) +
`candidate_events` (hypertable, append-only, retention 90 дней) + continuous aggregates;
Neo4j (`MarketGroup {symbol, group_id}` ← `TRANSITION`); Redis (кэш оценок, dedup по
`(symbol, configuration_hash)` TTL 30 мин, pub/sub `candidates:strong:{symbol}` + `candidates:strong:all`,
общий стрим приёма `candidates:stream`).

## Правила, которые легко нарушить

- **Порогов в коде нет — они в `config/symbols/*.yaml`.** `candidate_scorer.py` описывает только
  форму формулы: какие признаки в какую ось и как оси складываются. Ступени, веса, границы рейтинга,
  пороги валидатора и порог батча приходят из `ScoringProfile`. Никогда не сравнивай `quality_score`
  с числом напрямую — зови `get_rating(score, profile)`: у монет разные границы. `WEIGHTS`,
  `RATING_STRONG_MIN`, `_AGE_SCORE`, `FRESH_BONUS` остались как **deprecated** read-only проекции
  `_default` через модульный `__getattr__` (их импортируют тесты).
- **Профиль резолвится один раз на кандидата, в начале pipeline, и прокидывается во все узлы.**
  Иначе `POST /config/reload` посреди батча даст оценку, у которой score посчитан одной версией
  калибровки, а rating и тексты — другой.
- **`quality_score` сравним только внутри монеты.** Рядом всегда считается `quality_score_baseline`
  (тот же кандидат по `_default`) — единственное межмонетно сравнимое число. Rating считается
  **только** от профильного; двух рейтингов не заводим.
- **`_default.yaml` не калибруют.** С версии BTCUSDT@2 он перестал быть «калибровкой под BTC»
  и стал неподвижной общей меркой: по нему считается `quality_score_baseline` у всех монет
  и оцениваются монеты без своего профиля. Его правка обесценивает все накопленные baseline
  разом и рушит сравнимость задним числом. Заморожен снапшот-тестом на 200 кандидатах. Если
  правка осознанна: бампнуть version, `python3 scripts/make_scorer_snapshot.py`, обновить копию
  в `tests/fixtures/profiles/` и README, раздел 8.
- **Тесты механики считают по ЗАМОРОЖЕННЫМ профилям** из `tests/fixtures/profiles` — там
  BTCUSDT равен `_default`. Скорер без явного профиля резолвит его по символу кандидата,
  а эталонный кандидат из ТЗ — BTCUSDT; пока боевой профиль был пустым наследованием, тесты
  ступеней проходили по совпадению, и первая же калибровка BTC уронила 66 случаев, не имевших
  к ней отношения. Боевые профили отдаёт фикстура `shipped_profiles` — ею пользуются снапшот,
  sanity-якоря и проверки реестра, то есть ровно те тесты, которым нужна поставляемая правда.
- **Монету нельзя завести без sanity-якоря** (`tests/fixtures/sanity_candidates.json`) — на это
  есть отдельный падающий тест. Профиль, который нечем проверить, проверить некому.
- **Профиль калибруется по СВЕЖЕМУ окну выгрузки, а не по всей истории** (`--tail`, по умолчанию
  последняя четверть). `sample_size` кандидата монотонно растёт по мере накопления истории, поэтому
  перцентили по всей выгрузке ставят планку туда, где сидели бедные ранние годы. На данных ETH это
  дало 10% STRONG на калибровочной части против 26% на свежей трети. Выгрузка обязана идти
  `ORDER BY ts`: в схеме кандидата поля времени нет, и скрипт доверяет порядку строк.
- **Границы рейтинга целятся в селективность эталона** (`--target-strong`, по умолчанию 0.01 —
  столько STRONG даёт BTC). Рейтинг сравним только внутри монеты, но сводный список «что сегодня
  сильного» общий: монета со щедрой планкой залила бы его.
- **У доли верхняя ступень не может быть 1.0.** Сравнение строгое, значение доли больше единицы
  не бывает — ступень мертва. Монотонность такое не ловит, поэтому есть отдельная проверка
  `_validate_reachable` для полей из `_SHARE_FIELDS`. Не гипотетика: калибратор предложил такую
  ступень на реальных данных ETH.
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
- **Изоляция по монете обязательна везде, где ключ принадлежит графу одной монеты.** `group_id`,
  `transition_id`, `candidate_family_key`, `configuration_hash` осмысленны только внутри инструмента:
  генератор обучает модель состояний на каждую монету отдельно. Ключ Neo4j — `(symbol, group_id)`,
  Redis — `{symbol}:candidate:hash:*` и `{symbol}:evaluation:*`, PK Postgres — `(symbol, candidate_id)`,
  группировки фильтров — `(symbol, ключ)`, `GROUP BY` агрегатов — с `symbol` и `scoring_profile`.
  Ошибка здесь не падает, а **молча портит данные**: узлы монет схлопываются, ETH получает оценку BTC
  по совпавшему хэшу, средние считаются по двум рынкам сразу.
- **Исключение — вектор pgvector: он symbol-agnostic намеренно.** Это фича («а у BTC такая
  конфигурация была?»), изоляция делается фильтром в SQL (`?cross_symbol=`), а не вектором.
- **Стрим приёма один на все монеты** (`candidates:stream`): символ лежит в payload, дробить пул
  воркеров незачем.
- **Порядок признаков в `src/db/embedding.py` фиксирован.** Изменение порядка или `VECTOR_DIM` (=32)
  требует миграции с пересчётом существующих строк из `raw_payload` — см. `alembic/versions/003_shrink_embedding_dim.py`.
- **Celery-воркер обязан слушать `-Q evaluate,celery`.** `task_routes` шлёт `evaluate_candidate` в
  очередь `evaluate`; без `-Q` задачи молча копятся в Redis (audit #20).
- **Валидационные ошибки в `evaluate_candidate` не ретраятся** — они детерминированы, задача
  завершается результатом `{"error": "invalid_candidate", ...}`.
- **`refresh_continuous_aggregates` работает только вне транзакции** — соединение с
  `execution_options(isolation_level="AUTOCOMMIT")`, а не `get_session()`.
- **Любая правка порогов делает старые сохранённые `quality_score` несравнимыми с новыми** —
  пересчёта записей нет. Поэтому каждая запись помечена `scoring_profile` («ETHUSDT@3») и
  `profile_fingerprint` (sha256 содержимого): правка порогов без бампа `version` всё равно видна.
  Сравнение корректно только внутри одного `scoring_profile` **и** одного отпечатка. Это стоит
  проговаривать в ответе пользователю.
- `validator` только помечает (`warning_flags`), никогда не бросает исключений.

## Тесты

`tests/conftest.py` уводит `DATABASE_URL` на in-memory sqlite (до первого импорта `src.db.*`,
который создаёт engine прямо на импорте) и подменяет заглушками отсутствующие SDK
(`anthropic`, `neo4j`, `redis`, `pgvector`) — заглушка ставится только если реального пакета нет.
Фикстуры: `reference_payload` / `reference_candidate` / `make_candidate(**overrides)` — эталонный
кандидат из ТЗ; `eth_profile` — профиль монеты с короткой историей, собранный в памяти (тесты не
должны зависеть от того, заведена ли ETH в боевом реестре); `snapshot_candidates` — 200 кандидатов
снапшот-теста. Числа из раздела «Как читать результат оценки» в README зафиксированы тестом
`test_reference_candidate_is_strong`: правишь пороги — обновляй README.

Три теста мультимонетности стоит знать по именам: `test_scorer_snapshot.py` (рефакторинг не сдвинул
числа BTC), `test_sanity_candidates.py` (монету нельзя завести без якоря), и
`test_dedup_does_not_leak_between_symbols` в `test_pipeline.py` (совпавший `configuration_hash` не
отдаёт ETH оценку BTC — самый тихий из возможных багов).

Непокрыто (нужен живой стек): репозитории PostgreSQL, Cypher-запросы, SQL к continuous aggregates,
HTTP-роуты, Celery-задачи.

## Документация

- `README.md` — исчерпывающая (50 КБ): словарь понятий, формула score, API-справочник, сценарии, отладка.
- `README_agent_spec.md` — исходное ТЗ, 37 исходных полей кандидата (плюс
  `effective_sample_size` и `sample_scope`, заведённые 2026-08-13, — см. README, раздел 9).
- `docs/audit_findings.md` — 20 разобранных замечаний аудита с прогонами на живом стеке; читай перед
  тем, как «чинить» что-то в скорере, дедупликации или Cypher (замечание #5 отозвано как ошибочное —
  инкрементальное среднее в Neo4j `SET` корректно).
- `docs/development_log.md` — журнал решений с 2026-08-11: что менялось, почему
  именно так и что было отвергнуто. Изменения приёмника чаще всего приходят
  парой к изменениям генератора, поэтому записи ссылаются на его журнал.
- `docs/step_01..03_*.md` — история построения слоёв.
- `docs/step_04_multi_symbol.md` — мультимонетность: дизайн профилей, инвентаризация мест с BTC,
  процедура калибровки новой монеты.

## Известные ограничения

API без аутентификации и rate-limit (`use_llm=true` тратит токены Anthropic по анонимному запросу) —
допустимо только локально. По той же причине `POST /config/reload` закрыт за `ENABLE_CONFIG_RELOAD=false`:
ручка меняет поведение скоринга на лету. Таблица `market_events` создана миграцией, но кодом не
используется. Кандидат — исследовательская идея, а не торговый сигнал: нет entry timing, стопов,
размера позиции.

Миграция `004_multi_symbol` пересоздаёт оба continuous aggregate (ALTER для них TimescaleDB не
поддерживает) — **агрегаты старше retention 90 дней теряются безвозвратно**. Дамп снимать до запуска,
команды в шапке файла миграции.
