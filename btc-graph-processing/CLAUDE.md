# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Язык проекта — русский: docstrings, комментарии, сообщения CLI и UI пишутся по-русски.
Подробная документация — `README.md` (архитектура и обоснования), `docs/operator_guide.md`
(эксплуатация), `docs/development_log.md` (история решений).

## Команды

```bash
make setup                  # .env из .env.example
make install                # зависимости в активный интерпретатор
make init-db                # схема processing в БД btc_graph
make train                  # полный прогон истории (30–60 мин)
make train-fast             # пересчёт по данным из БД, без отправки
make live                   # инкрементальный прогон — штатное обновление
make status                 # покрытие истории, отставание, очередь, приёмник
make admin                  # админка на http://127.0.0.1:8100 (--reload)
make test                   # pytest -q
```

Всё это обёртки над `python3 -m btcproc.cli <команда>`. Интерпретатор задаётся
через `make train PY=.venv/bin/python`.

Быстрая проверка без ожидания полного прогона:

```bash
python3 -m btcproc.cli ingest --start 2024-06-01
python3 -m btcproc.cli train --no-ingest --no-emit --start 2024-06-01
```

### Тесты

```bash
python3 -m pytest tests/test_states.py                              # один файл
python3 -m pytest tests/test_candidates.py::test_outcomes_match_prices -q   # один тест
python3 -m pytest -k lookahead
```

Тесты не ходят ни в БД, ни в сеть — данные генерирует `make_bars()` из
`tests/conftest.py` (синтетические свечи с чередующимися режимами, чтобы
кластеризации было что находить). Фикстуры `bars` / `context` / `features` —
session-scoped.

Прогонять стоит в обоих окружениях, если их несколько (`python3` и
`.venv/bin/python`): версии pandas/numpy расходятся, и часть багов
(read-only массив из `to_numpy()` на pandas 3, `NaN` вместо `None`)
воспроизводится только на свежих библиотеках.

## Внешние зависимости

БД своей нет: PostgreSQL, Redis и Neo4j берутся из соседнего проекта
**btc-graph** (`../btc-graph`, `BTC_GRAPH_PATH`). Его стек должен быть поднят
(`cd ../btc-graph && make up && make migrate`) до любой команды, работающей с БД.
`docker-compose.yml` поднимает только админку и подключается к внешней сети
`btc-graph_default`.

Ловушка окружения: `pgvector` и `neo4j` импортирует не наш код, а btc-graph, и
делает это лениво уже внутри сохранения. При расхождении venv кандидаты
считаются, отправка рапортует об успехе, а записи нет — в логе
`ModuleNotFoundError` посреди прогона. `btcproc.cli status` печатает путь к
интерпретатору и проверяет наличие этих пакетов.

## Архитектура

Конвейер (`btcproc/pipeline/train.py` — стадии в этом порядке, каждая пишет
результат в БД под общим `run_id`):

```
бары → признаки → события → исходы → состояния → граф → кандидаты → btc-graph
```

| Слой | Модуль | Суть |
|---|---|---|
| Загрузка | `ingest/binance.py` | месячные дампы data.binance.vision + REST-хвост, агрегация старших ТФ из базового |
| Признаки | `features/builder.py`, `indicators.py` | 32 стационарных признака; старшие ТФ входят со сдвигом на бар |
| События | `features/events.py` | 29 атомов в 7 семействах → `event_block_id` (хэш битовой маски) |
| Состояния | `states/clustering.py` | адаптивная гранулярность: дробление по gap-критерию + слияние по d-prime; результат — `StateModel` |
| Разметка | `states/assign.py` | `group_id` по барам, сглаживание, возраст, переходы, энтропия траектории |
| Граф | `states/graph.py` | статистика узлов и рёбер, `transition_rarity` |
| Исходы | `candidates/outcomes.py` | `ret` / `MFE` / `MAE` на горизонте 24h, валидность метки |
| Кандидаты | `candidates/builder.py` | снимки конфигураций → 37 полей схемы btc-graph |
| Отправка | `sink/graph_sink.py` | `direct` / `http` / `none` |

`train` и `live` разделены принципиально: `train` переобучает модель состояний
с нуля (номера `group_id` нового прогона несопоставимы со старыми, накопленный в
Neo4j граф после этого смешивает две нумерации), `live` загружает сохранённую
модель последнего успешного train (`repo.load_state_model`) и только размечает
ей свежие бары. Регулярное обновление — всегда `live`.

Точки продолжения `live` берутся из данных, а не из календаря: бары — от
`max(ts)` в `ohlcv`, кандидаты — от последнего выпущенного
(`pipeline/live.py:resolve_cutoff`). Перекрытие окон безопасно: `candidate_id`
детерминирован, запись идёт upsert'ом, `emitted_at` не сбрасывается.

Доставка в btc-graph в режиме `direct` — импорт его пакета: каталог добавляется
в `sys.path`, `DATABASE_URL` / `REDIS_URL` / `USE_DB` / `USE_REDIS` / `USE_GRAPH`
выставляются **до** импорта (его pipeline читает флаги на уровне модуля), дальше
вызывается `run_batch_pipeline(save=True)`. Кандидаты дублируются в
`processing.candidates` намеренно — btc-graph принимает только прошедших фильтр,
а для разбора нужен полный список.

Хранение: `db/session.py` — psycopg2 напрямую (`connect` выставляет `search_path`,
`bulk_upsert` — `INSERT ... ON CONFLICT` пачками), ORM нет из-за bulk-вставок
сотен тысяч строк. `db/runs.py` работает в autocommit — прогресс и лог прогона
должны быть видны админке до завершения транзакции.

Админка (`admin/app.py`) запускает прогоны через `BackgroundTasks` в том же
процессе; `queries.py` — только чтение, `auth.py` — сессии, rate-limit, allowlist.

## Инварианты, которые легко сломать

Полное обоснование — README, раздел 3. Кратко, чего нельзя делать вслепую:

1. **Пакет называется `btcproc`, а не `src`.** Пакет btc-graph — `src`, и его
   ленивые импорты `from src.db...` резолвились бы в наш код, тихо ломая режим
   `direct`.
2. **Слияние состояний считается через d-prime** (разброс вдоль оси между
   центроидами), а не через радиус группы: радиус растёт как √d, при 32 признаках
   критерий «центроиды ближе суммы радиусов» схлопывает граф до 3–4 состояний
   вместо 43.
3. **В `event_block_id` входят только `SIGNATURE_ATOMS`** (пробои, всплески,
   кроссы, развороты), а не `CONTEXT_ATOMS` (тренд, сессии, доминирование
   тейкеров). Фон активен почти всегда и даёт комбинаторный взрыв блоков, после
   которого историческая выборка по блоку не набирается никогда. Проверка после
   правок: `runs.stats.candidates.scopes` — доля `transition+event_block` не
   должна быть близка к нулю.
4. **Признаки не заглядывают вперёд.** Старшие ТФ сдвигаются на бар,
   индикаторы считаются только по прошлому, выборка кандидата собирается из
   случаев раньше `t` и только с созревшими исходами. Закреплено тестами
   `test_features_do_not_look_ahead` и
   `test_candidate_sample_uses_only_matured_past`.
5. **Схема кандидата совместима с btc-graph по букве.** Каждый кандидат обязан
   пройти его pydantic-модель `Candidate` без правок —
   `test_generated_candidates_match_btc_graph_schema`.

## Конвенции

* Окружение читается **только** в `btcproc/config.py` — модули берут настройки
  из готовых dataclass'ов (`config.data`, `config.states`, `config.candidates`,
  `config.sink`, `config.admin`). Новый параметр добавляется туда же, с
  комментарием о смысле значения по умолчанию.
* Тяжёлые импорты (`pandas`, pipeline, uvicorn, httpx-приёмник) делаются лениво
  внутри команд CLI — старт `--help` и `status` не должен тянуть весь стек.
* Служебные поля кандидата живут в `_meta` и снимаются `strip_meta` перед
  отправкой.
* Цвета состояний для фронтенда генерируются на сервере в hex:
  lightweight-charts падает на `hsl(...)` с «Cannot parse color».
* `except Exception` в прогонах и в диагностике приёмника — намеренный, чтобы
  прогон завершался с диагнозом, а не трейсбеком; помечается `# noqa: BLE001`.

## Куда смотреть при типичных правках

| Хочу | Файл |
|---|---|
| Поменять пороги состояний, выборки, горизонт | `btcproc/config.py` |
| Добавить признак | `btcproc/features/builder.py` |
| Добавить тип события | `btcproc/features/events.py` (`ATOM_FAMILY` + `detect_atoms`) |
| Изменить правила сборки кандидата | `btcproc/candidates/builder.py` |
| Поменять способ доставки в btc-graph | `btcproc/sink/graph_sink.py` |
| Добавить таблицу | `btcproc/db/schema.sql` (идемпотентно) + `btcproc/db/repo.py` |
| Добавить страницу в админку | `btcproc/admin/app.py` + `templates/` |
