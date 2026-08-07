# Аудит кода: найденные ошибки и замечания

Разбор проекта по состоянию на 2026-08-07.

**Замечания №1–4 исправлены** (см. пометки ✅ ниже), остальные описывают текущее
состояние кода. Одновременно добавлен набор из 193 тестов в `tests/` — он покрывает
скорер, валидатор, парсер, фильтры и pipeline и не требует ни БД, ни внешних сервисов.

Часть выводов проверена запуском кода (помечено «проверено»), часть — чтением
(помечено «по коду»): для них нужна поднятая инфраструктура, чтобы подтвердить
экспериментально.

Легенда: 🔴 ломает работу — 🟠 портит результаты — 🟡 качество и сопровождение.

---

## ✅ 🔴 1. Недостижимая ветка в `_score_context` — возраст состояния оценивается неверно

**Исправлено.** `src/scorer/candidate_scorer.py` — было:

```python
if c.current_group_age_bucket != AgeБucket.age_gt_120:
    score += 1.0
elif c.current_group_age_bucket == AgeБucket.age_60_120:   # недостижимо
    score += 0.5
else:
    score += 0.0
```

Если bucket равен `age_60_120`, первое условие (`!= age_gt_120`) уже истинно, и
ветка `elif` не выполняется никогда. Вместо задуманной градации получается
бинарная оценка.

**Проверено:**

| bucket | фактический `score_context` | ожидаемый по замыслу |
|---|---|---|
| `age_lt_30` | 0.5 | 0.5 |
| `age_30_60` | 0.5 | ~0.33–0.5 |
| `age_60_120` | **0.5** | **0.3333** |
| `age_gt_120` | 0.1667 | 0.1667 |

(Замеры при `context_status=stale`, `trajectory_entropy=medium`.)

Кандидат с состоянием возрастом 60–120 минут получает столько же, сколько
абсолютно свежий. Ось `context` весит 20%, так что расхождение в итоговом
`quality_score` — до 0.033, а это разница между MODERATE и STRONG на границе.

**Как исправлено** — ступенчатая логика заменена на явную карту:

```python
_AGE_SCORE = {
    AgeБucket.age_lt_30:  1.0,
    AgeБucket.age_30_60:  0.75,
    AgeБucket.age_60_120: 0.5,
    AgeБucket.age_gt_120: 0.0,
}
score += _AGE_SCORE[c.current_group_age_bucket]
```

**Влияние на оценки.** Правка меняет `quality_score` для двух бакетов
(ось `context` весит 20%, вклад бакета — треть оси):

| bucket | было | стало | Δ quality_score |
|---|---|---|---|
| `age_lt_30` | 1.0 | 1.0 | — |
| `age_30_60` | 1.0 | 0.75 | −0.017 |
| `age_60_120` | 1.0 | 0.5 | −0.033 |
| `age_gt_120` | 0.0 | 0.0 | — |

Оценки, сохранённые до правки, несравнимы с новыми. Эталонный кандидат из ТЗ
(`age_gt_120`) не затронут: `quality_score` остался 0.7783, рейтинг STRONG.

**Регрессия закрыта тестами** `test_context_age_bucket_is_graduated` и
`test_context_age_buckets_are_all_distinct` (`tests/test_scorer.py`). Проверено:
на старом коде оба падают с `{'age_30_60': 1.0, 'age_60_120': 1.0, 'age_lt_30': 1.0}`.

---

## ✅ 🔴 2. ORM-объекты читаются после закрытия сессии → 500 на трёх эндпоинтах

**Исправлено.** `src/api/routes.py:137-148`, `161-172`, `186-197` + `src/db/connection.py`

```python
with get_session() as session:
    record = candidate_repo.get_by_id(session, candidate_id)
# сессия уже закрыта
return {"candidate_id": record.candidate_id, ...}   # ← DetachedInstanceError
```

`SessionLocal` создан без `expire_on_commit=False`, то есть с умолчанием `True`.
`get_session()` на выходе делает `commit()` (это помечает все атрибуты как
expired) и затем `close()` (объект становится detached). Любое обращение к полю
после этого требует повторной загрузки, которая невозможна без живой сессии.

Затронуты: `GET /candidates/{id}`, `GET /candidates/strong/{direction}`,
`POST /candidates/similar`. **По коду** — механизм однозначный, но подтверждения
на живой БД я не делал.

**Как исправлено** — выбран минимальный вариант, не трогающий роуты:

```python
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False,
                            expire_on_commit=False)
```

Проверить на живой БД я не мог (нет psycopg2/Postgres в среде аудита), но механизм
однозначен: без `expire_on_commit=True` атрибуты не истекают, и detached-объект
читается штатно.

**Что осталось в этих роутах:** `HTTPException(404)` выбрасывается **вне** блока
`with` и в `/candidates/{id}` перехватывается собственным `except HTTPException: raise`.
Работает, но держится на одной строчке. Более чистый вариант — собирать словарь
внутри `with`:

```python
with get_session() as session:
    record = candidate_repo.get_by_id(session, candidate_id)
    if record is None:
        raise HTTPException(404, "Кандидат не найден")
    result = {"candidate_id": record.candidate_id, ...}
return result
```

---

## ✅ 🔴 3. `refresh_continuous_aggregates` не может выполниться — вызов внутри транзакции

**Исправлено.** `src/worker/tasks.py` — было:

```python
with get_session() as session:
    for view in ("hourly_candidate_stats", "daily_group_stats"):
        session.execute(sa.text(f"CALL refresh_continuous_aggregate('{view}', NULL, NULL)"))
```

TimescaleDB запрещает `refresh_continuous_aggregate` внутри транзакционного блока
(«cannot run inside a transaction block»). SQLAlchemy Session открывает транзакцию
неявно при первом `execute`. Задача, запускаемая Beat ежечасно, будет падать
**всегда** — и падать тихо, потому что исключение перехватывается и превращается
в `{"refreshed": [], "error": ...}` с логом уровня ERROR.

Практическое следствие: обновление aggregates держится только на политиках
TimescaleDB (`add_continuous_aggregate_policy`), а страховочный механизм не работает.

**Как исправлено:**

```python
from src.db.connection import engine

with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
    for view in ("hourly_candidate_stats", "daily_group_stats"):
        conn.execute(
            sa.text("CALL refresh_continuous_aggregate(:view, NULL, NULL)"),
            {"view": view},
        )
```

Заодно ушла f-string в SQL — имя вью теперь bind-параметр (значения и раньше были
захардкожены, так что инъекции не было, но привычка плохая).

Синтаксис цепочки проверен на sqlite; поведение `AUTOCOMMIT` на реальном
TimescaleDB не проверялось — нужен прогон задачи на поднятом стеке.

---

## ✅ 🔴 4. `_persist()` глушит все исключения — данные теряются молча

**Исправлено.** `src/agent/pipeline.py` — было:

```python
if _USE_DB:
    try:
        ...
    except Exception:
        pass          # ← ни лога, ни метрики, ни признака в ответе
```

Так во всех трёх блоках: PostgreSQL, Neo4j, Redis. Задумка понятна — падение
хранилища не должно ломать оценку. Но реализация делает отладку невозможной:
API отвечает 200 с полноценной оценкой, а в БД ничего нет, и узнать об этом
можно только по факту пустых выборок.

**Как исправлено:**

1. Каждый блок логирует сбой через `logger.exception(...)` и заполняет статус.
2. `_persist()` возвращает `{"db": bool|None, "graph": ..., "redis": ...}`,
   где `None` означает «хранилище отключено через `USE_*`».
3. При частичном сохранении пишется сводный `logger.warning` со всем статусом.
4. Тот же анти-паттерн убран в соседних слоях, иначе статус врал бы:
   - `src/db/graph_repo.py` — `upsert_from_candidate()` возвращает `bool` и логирует
     `ServiceUnavailable`; `find_transitions_to_group()` и `get_group_info()` логируют
     недоступность Neo4j (контракт возврата `[]` / `None` сохранён);
   - `src/cache/redis_cache.py` — `cache_evaluation()`, `mark_hash_cached()` и
     `publish_strong_candidate()` возвращают `bool` и логируют `RedisError`;
     функции чтения логируют и по-прежнему возвращают `None`.

Поведение для клиента не изменилось: недоступное хранилище всё так же не ломает
оценку, API возвращает 200. Изменилось то, что сбой теперь виден в логах:

```bash
docker compose logs api | grep -i "не удалось"
```

**Покрыто тестами** (`tests/test_pipeline.py`): `test_persist_logs_and_reports_db_failure`,
`test_persist_reports_graph_failure`, `test_persist_success_path`,
`test_persist_reports_disabled_storages`.

**Что осталось:** статус никуда не отдаётся наружу — `run_pipeline()` его
игнорирует. Чтобы деградация была видна вызывающему, статус стоит добавить в
ответ API отдельным полем (это меняет схему ответа, поэтому в рамках правки
не делалось).

---

## 🟠 5. Неверное инкрементальное среднее в Cypher

`src/db/graph_repo.py:91-95`

```cypher
ON MATCH SET
    t.count = t.count + 1,
    t.avg_horizon_return = (t.avg_horizon_return * t.count + $win_rate) / (t.count + 1),
    ...
```

В Cypher элементы одного `SET` применяются последовательно, и последующие видят
уже обновлённые значения. К моменту вычисления среднего `t.count` равен `n+1`,
а не `n`. Фактически считается

```
(avg·(n+1) + x) / (n+2)     вместо     (avg·n + x) / (n+1)
```

Ошибка небольшая на каждом шаге, но накапливается и систематически занижает вес
нового наблюдения. Поля `avg_horizon_return` и `avg_quality_score` в графе
смещены — их нельзя использовать для сравнения переходов между собой.

**Исправление** — считать до инкремента:

```cypher
MERGE (src)-[t:TRANSITION {transition_id: $tid}]->(dst)
ON CREATE SET t.rarity = $rarity, t.count = 1,
              t.avg_horizon_return = $win_rate, t.avg_quality_score = $qs
ON MATCH SET  t.avg_horizon_return = (t.avg_horizon_return * t.count + $win_rate) / (t.count + 1),
              t.avg_quality_score  = (t.avg_quality_score  * t.count + $qs)       / (t.count + 1),
              t.count = t.count + 1,
              t.rarity = $rarity
```

Надёжнее — хранить `sum` и `count` и делить при чтении.

Там же: `g.sample_count` выставляется в 1 при `ON CREATE` и никогда не растёт
(`src/db/graph_repo.py:59-60`) — поле бессмысленно.

---

## 🟠 6. Дедупликация возвращает оценку чужого кандидата

`src/agent/pipeline.py:36-43`

```python
cached_id = redis_cache.is_hash_cached(candidate.configuration_hash)
if cached_id:
    cached_json = redis_cache.get_cached_evaluation(cached_id)
    if cached_json:
        return CandidateEvaluation(**json.loads(cached_json))
```

`configuration_hash` — хэш конфигурации рынка, а не кандидата. Два разных
`candidate_id` могут иметь одинаковый хэш. Тогда на запрос по кандидату B
вернётся сохранённая оценка кандидата A — с полем `candidate_id: A`. Клиент
получает ответ, который не соответствует его запросу, и запись по B не создаётся.

Дополнительно: дедупликация включается только при `save=true` — два ортогональных
поведения связаны одним флагом. Нельзя «посчитать, но не сохранять, с кэшем» и
нельзя «сохранить, но обязательно пересчитать».

**Исправление:** возвращать кэш только при совпадении `candidate_id`, либо
подменять `candidate_id` в ответе и добавлять поле вида
`deduplicated_from: "<исходный id>"`, чтобы это было явно. И развести `save` и
`use_cache` на отдельные параметры.

---

## 🟠 7. Локальный запуск не читает `.env`

`python-dotenv` есть в `requirements.txt:6`, но `load_dotenv()` не вызывается
нигде (проверено grep'ом по `src/` и `alembic/`). При этом README описывает
сценарий локального запуска без Docker.

Итог: `uvicorn src.main:app` подхватит дефолты из кода, а `ANTHROPIC_API_KEY`
останется пустым — `use_llm=true` упадёт. В Docker проблемы нет, там переменные
приходят через `env_file` и `environment`.

**Исправление** — в `src/main.py` и `src/worker/celery_app.py`, до остальных импортов:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 🟠 8. Для short-кандидатов используются long-ориентированные метрики без инверсии

`src/scorer/candidate_scorer.py:95-103`, `src/agent/pipeline.py:187`, `src/agent/llm_node.py:139`

`win_rate` для short корректно инвертируется (`1 − long_outcome_share`), а вот
`long_favorable_adverse_ratio_p70_p80` берётся как есть — и в скорере, и в поле
`favorable_adverse_ratio` итоговой оценки, и в промпте для LLM.

Метрика по определению описывает движение вверх: `p70_long_favorable_pct` —
потенциал long, `p80_long_adverse_pct` — риск long. Для short-кандидата
благоприятное и неблагоприятное меняются местами, поэтому высокое значение
ratio у short-кандидата означает ровно обратное задуманному — и всё равно
добавляет ему баллов по оси `directional` (вес 35%).

ТЗ этот случай не покрывает: в источнике данных просто нет short-версий p70/p80.

**Исправление** — как минимум перестать начислять баллы за этот ratio, когда
`research_side == short`, и не показывать его в выводе для short. Правильное
решение — запросить у генератора кандидатов симметричные поля
(`p70_short_favorable_pct`, `p80_short_adverse_pct`).

---

## 🟠 9. `select_best_per_family`: свежесть всегда важнее качества

`src/filters/candidate_filter.py:53-59`

```python
group.sort(key=lambda x: (
    0 if x[0].context_status == ContextStatus.fresh else 1,   # первичный ключ
    -x[1],                                                     # вторичный
))
```

Сортировка лексикографическая, поэтому любой `fresh` кандидат бьёт любой `stale`.
Кандидат с `quality_score = 0.61` и `fresh` вытеснит кандидата с `0.95` и `stale`.

ТЗ говорит «выбрать наиболее свежий и сильный» — формулировка допускает оба
прочтения, но текущее делает `quality_score` внутри семьи почти неважным.

**Исправление** — сделать свежесть штрафом, а не абсолютным приоритетом:

```python
FRESH_BONUS = 0.05
group.sort(key=lambda x: -(x[1] + (FRESH_BONUS if x[0].context_status == ContextStatus.fresh else 0)))
```

---

## 🟠 10. Пороги рейтинга продублированы в четырёх местах

Одна и та же логика «≥0.75 → STRONG, ≥0.55 → MODERATE, иначе WEAK» написана в:

- `src/scorer/candidate_scorer.py:193` — `get_rating()`, канонический вариант;
- `src/agent/llm_node.py:59` — внутри `_build_prompt()`;
- `src/agent/llm_node.py:130` — при сборке результата;
- `src/api/routes.py:92` — в `/score/quick`.

Изменение порогов в скорере не затронет остальные три места, и `/score/quick`
начнёт отвечать не тем, что `/evaluate/json`.

**Исправление:** везде вызывать `get_rating()`.

---

## 🟠 11. Вызов Claude без обработки ошибок, таймаута и ретраев

`src/agent/llm_node.py:105-112`

`client.messages.create()` вызывается напрямую. Любая ошибка API (нет ключа,
rate limit, таймаут, overloaded) поднимается вверх и на HTTP-уровне превращается
в **422** (`routes.py:58-59` ловит `Exception` и отдаёт 422) — то есть клиент
получает «неверные данные» вместо «сервис недоступен».

Дополнительно `response.content[0].text` (строка 112) предполагает, что первый
блок ответа — текстовый. Это верно для текущей конфигурации, но сломается, если
включить extended thinking или tool use.

**Исправление:** обернуть вызов в try/except по `anthropic.APIError`, при сбое
делать fallback на `_deterministic_evaluation()` (тексты будут хуже, но оценка
корректна — числа-то считает не LLM), передать `timeout=` и `max_retries=` в
конструктор клиента, а текст искать как первый блок с `type == "text"`.

---

## 🟠 12. Роуты `/evaluate/*` маскируют внутренние ошибки под 422

`src/api/routes.py:56-59, 65-68, 74-82, 105-106, 125-126`

```python
except Exception as e:
    raise HTTPException(status_code=422, detail=str(e))
```

422 означает «данные не прошли валидацию». Сюда же попадают ошибки соединения с
БД, сбои Anthropic и любые баги в коде. Отладка по коду ответа становится
невозможной, и клиент не понимает, надо ли повторять запрос.

**Исправление:** ловить `pydantic.ValidationError` / `ValueError` → 422, всё
остальное → 500 с логированием traceback.

---

## 🟡 13. Redis Stream используется без consumer groups

`src/cache/redis_cache.py:101-121`, `src/worker/tasks.py:56-66`

```python
messages = get_client().xread({STREAM_KEY: "0"}, count=count, block=block_ms)
...
evaluate_candidate.apply_async(...)
ack_candidate(msg["id"])          # xdel
```

Чтение всегда с начала стрима (`"0"`), «подтверждение» — это `XDEL`. Схема
работает, пока `process_stream_batch` выполняется строго в одном экземпляре.
При двух Beat-инстансах или наложении запусков (задача каждые 10 секунд) один и
тот же кандидат может быть прочитан и задиспатчен дважды — до того, как первый
успеет сделать `XDEL`.

**Исправление:** `XGROUP CREATE` + `XREADGROUP` + `XACK` — Redis Streams
спроектированы именно под это.

Там же `enqueue_candidate` при ошибке Redis возвращает `""`, а
`POST /queue/enqueue` всё равно отвечает `{"queued": true, "msg_id": ""}` —
ложноположительный ответ (`routes.py:291-292`).

---

## 🟡 14. `evaluate_candidate` ретраит невалидные данные

`src/worker/tasks.py:41-43`

Любое исключение приводит к `self.retry()`. Для сетевых сбоев это правильно, но
кандидат с отсутствующим обязательным полем будет трижды перепроверен с
интервалом 30 секунд, прежде чем окончательно упасть. Валидационные ошибки
детерминированы — ретраить их бессмысленно.

**Исправление:** ловить `pydantic.ValidationError` отдельно, логировать и
возвращать результат-ошибку без retry.

---

## 🟡 15. Кириллическая буква в имени класса `AgeБucket`

`src/models/candidate.py:8` — в `AgeБucket` вторая буква не латинская `B`, а
кириллическая `Б` (U+0411). Код работает (Python допускает Unicode в
идентификаторах), но имя невозможно набрать вслепую, оно не находится
grep'ом по `AgeBucket`, и импортируется в шести файлах:
`scorer/candidate_scorer.py`, `validator/candidate_validator.py`,
`filters/candidate_filter.py`, `db/embedding.py`, `agent/pipeline.py`.

**Исправление:** переименовать в `AgeBucket` во всех местах.

---

## 🟡 16. Мёртвый код и неиспользуемые импорты

| Место | Что |
|---|---|
| `src/db/orm_models.py:59` + миграция 001 | Таблица `market_events` и модель `MarketEventRecord` создаются, но кодом не читаются и не пишутся |
| `src/agent/llm_node.py:13` | `validate_candidate` импортирован, не используется |
| `src/filters/candidate_filter.py:11` | `validate_candidate` импортирован, не используется |
| `src/worker/tasks.py:9,11` | `logging` и `shared_task` импортированы, не используются |
| `src/db/embedding.py:9` | `HistoricalBiasContext` импортирован, используется только `.value` через словарь |
| `src/db/candidate_repo.py:62` | `is_hash_evaluated()` не вызывается ниоткуда — дедупликация делается через Redis |
| `src/cache/redis_cache.py:73` | `subscribe_strong_candidates()` не используется в проекте (это API для внешних потребителей — ок, но стоит отметить) |

---

## 🟡 17. Вектор 384 при 18 значащих признаках

`src/db/embedding.py:11,56-58` — заполняются первые 18 позиций, остальные 366 —
нули. Косинусное расстояние считается корректно (нули не влияют), но:

- HNSW-индекс строится по 384 измерениям вместо 18 — лишняя память и время;
- размерность 384 выглядит как «под sentence-transformers», хотя эмбеддинг здесь
  чисто табличный;
- расширение набора признаков в будущем сместит семантику существующих векторов —
  комментарий об этом в коде есть, но пересчёта не предусмотрено.

**Исправление:** `Vector(18)` (или 32 с запасом) + миграция с пересчётом.
Отдельно стоит подумать о нормализации признаков — сейчас часть в [0..1],
часть обрезана по `min(x/10, 1.0)`, и вес признаков в косинусной метрике
получается произвольным.

---

## ✅ 🟡 18. Тестов нет

**Исправлено.** Добавлен пакет `tests/` — 193 теста, `pytest` в `requirements.txt`,
конфиг в `pytest.ini`, цели `make test` / `make test-local`.

| Файл | Тестов | Что покрывает |
|---|---|---|
| `tests/test_scorer.py` | 94 | Каждая ступень каждой из 4 осей, инверсия win rate для short, веса, границы рейтингов, границы [0..1], эталонный кандидат |
| `tests/test_validator.py` | 39 | Каждый `warning_flag`, его пороговое значение и отсутствие ложных срабатываний |
| `tests/test_parser.py` | 23 | dict / JSON-строка / raw text / array, разделители `:` и `=`, хвосты `←`, комментарии, опечатки в ключах, выход за диапазоны |
| `tests/test_filters.py` | 22 | Фильтрация по порогу, ранжирование, дедуп по `family_key`, все 4 типа конфликтов и их границы |
| `tests/test_pipeline.py` | 15 | Детерминированная оценка, батч, отсутствие вызова LLM при `use_llm=False`, деградация `_persist` |

Тесты не требуют ни PostgreSQL, ни Redis, ни Neo4j, ни ключа Anthropic:
в `tests/conftest.py` отсутствующие SDK подменяются заглушками (только если
реального пакета нет), а `DATABASE_URL` принудительно уводится на in-memory
sqlite — тесты физически не могут обратиться к боевой БД.

**Что осталось непокрытым:** слои, требующие живой инфраструктуры — репозитории
PostgreSQL, Cypher-запросы Neo4j, SQL к continuous aggregates, HTTP-роуты
(`TestClient` + фикстура БД), Celery-задачи. Для них нужны интеграционные тесты
с поднятым стеком (`testcontainers` или `docker compose` в CI).

---

## 🟡 19. Инфраструктурные мелочи

| Что | Где | Почему важно |
|---|---|---|
| `.env` с реальным API-ключом лежит в проекте, `.gitignore` отсутствует | корень | При первом же `git init && git add .` ключ уедет в историю |
| Порты Postgres (5432), Redis (6379), Neo4j (7474/7687) опубликованы на хост | `docker-compose.yml` | Redis и Neo4j — с дефолтными/слабыми паролями. Для локальной машины терпимо, наружу выставлять нельзя |
| API без аутентификации и rate-limit | `src/api/routes.py` | `/evaluate` с `use_llm=true` тратит токены Anthropic по анонимному запросу |
| `celery-beat` не ждёт готовности Postgres (`depends_on` только на redis) | `docker-compose.yml` | Первый запуск `refresh_continuous_aggregates` может уйти в ошибку |
| `version: "3.9"` в compose | `docker-compose.yml:1` | Устаревшее поле, современный Docker Compose выдаёт warning |
| `api` собирается из того же образа, что и воркеры, с `COPY . .` | `Dockerfile` | В образ попадает и `.env`, и `.idea/`, `.git` — нет `.dockerignore` |
| `alembic.ini` содержит логин/пароль БД в открытом виде | `alembic.ini:4` | Переопределяется через `DATABASE_URL` в `env.py`, так что строку можно убрать |
| `src/agent/pipeline.py` импортирует `llm_node` на верхнем уровне | `pipeline.py:14` | Пакет `anthropic` обязателен даже при `use_llm=false` — мешает лёгким окружениям и тестам |
| README указывал образ `timescaledb-ha:pg16-latest` | было в README | Фактически используется `timescale/timescaledb-ha:pg16` — исправлено |

---

## Расхождения README с фактическим поведением (исправлены)

Прежняя версия README содержала пример ответа, не совпадающий с тем, что
возвращает код. Проверено запуском на эталонном кандидате из ТЗ:

| Поле | Было в README | Фактически |
|---|---|---|
| `quality_score` | 0.734 | **0.7783** |
| `score_statistical` | 0.875 | **1.0** |
| `score_directional` | 0.933 | **1.0** |
| `score_context` | 0.167 | 0.1667 ✓ |
| `score_rarity` | 0.633 | 0.6333 ✓ |
| `warning_flags` | 3 флага | **4 флага** (добавляется `stale_and_aged_combined`) |
| `strengths` | 4 пункта | **5 пунктов** |
| `risks` | 2 пункта | **3 пункта** |
| `summary` | литературный текст | шаблонная строка с числами |

В ТЗ (`README_agent_spec.md`) в блоке «Выходной формат» указан `quality_score: 0.87`
— это тоже иллюстрация, а не результат текущей реализации.

---

## Статус и что делать дальше

**Сделано:** №1 (скорер), №2 (сессия SQLAlchemy), №3 (AUTOCOMMIT для refresh),
№4 (логирование вместо `except: pass`), №18 (193 теста).

**Дальше, по соотношению «цена ошибки / стоимость исправления»:**

1. **№7** — `load_dotenv()`. Две строки, чинит весь локальный сценарий из README.
2. **№10** — свести рейтинг к единственному `get_rating()`: сейчас пороги живут в
   четырёх местах, и правка скорера разъедет `/score/quick` с `/evaluate/json`.
3. **№11, №12** — обработка ошибок Anthropic и корректные HTTP-коды: без этого
   любой сбой внешнего API выглядит как «клиент прислал плохие данные».
4. **№5** — среднее в Cypher: портит аналитику по графу, но не основной путь.
5. **№15, №16** — переименование `AgeБucket` и вычистка мёртвого кода; дёшево,
   но затрагивает шесть файлов, поэтому лучше отдельным коммитом.
6. **№6, №8, №9** — требуют продуктового решения, а не только правки кода:
   что считать дубликатом, как оценивать short-кандидатов, что важнее внутри семьи.
7. **№13, №19** — consumer groups для Redis Stream и инфраструктурная гигиена
   (`.gitignore`, `.dockerignore`, закрытые наружу порты) — перед любым выходом
   за пределы локальной машины.

**Проверка после правок:**

```bash
make test-local     # 193 теста, ~0.7 сек, инфраструктура не нужна
```
