# Аудит кода: найденные ошибки и замечания

Разбор проекта по состоянию на 2026-08-07.

**Все замечания разобраны:** №1–4, 6–15, 17, 18 и найденное позже №20 исправлены
(пометка ✅), №16 исправлено частично и осознанно, №5 отозвано как ошибочное (❌),
№19 закрыто по инфраструктурной части. Добавлен набор из 215 тестов в `tests/`.

Фиксы проверены на поднятом стеке (PostgreSQL + TimescaleDB, Redis, Neo4j,
Celery) — результаты прогонов приведены внутри пунктов.

Легенда: ✅ исправлено — ❌ замечание отозвано — 🔴 ломало работу —
🟠 портило результаты — 🟡 качество и сопровождение.

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
`POST /candidates/similar`.

**Как исправлено** — выбран минимальный вариант, не трогающий роуты:

```python
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False,
                            expire_on_commit=False)
```

Дополнительно словари теперь собираются **внутри** блока `with`, так что роуты
не зависят от настройки `expire_on_commit`, а `HTTPException(404)` больше не
выбрасывается из-за границы сессии.

**Проверено на живой БД** после пересборки контейнеров:

| Эндпоинт | HTTP |
|---|---|
| `GET /candidates/245be5fb0908d59f6e89` | 200 |
| `GET /candidates/strong/long` | 200 |
| `POST /candidates/similar` | 200 |

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

**Проверено на живом TimescaleDB** — предсказание подтвердилось буквально:

```
новый код (AUTOCOMMIT):  {'refreshed': ['hourly_candidate_stats', 'daily_group_stats']}
старый код (в транзакции): ОШИБКА: (psycopg2.errors.ActiveSqlTransaction)
                           refresh_continuous_aggregate() cannot run inside a transaction block
```

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

## ❌ 5. Инкрементальное среднее в Cypher — ЗАМЕЧАНИЕ ОШИБОЧНО

**Отозвано после проверки на живом Neo4j 5. Код был корректен, изменений не требовалось.**

Исходное замечание утверждало, что в

```cypher
ON MATCH SET
    t.count = t.count + 1,
    t.avg_horizon_return = (t.avg_horizon_return * t.count + $win_rate) / (t.count + 1),
```

элементы `SET` применяются последовательно, и к моменту вычисления среднего
`t.count` уже равен `n+1`, из-за чего среднее «уезжает».

**Это неверно.** Neo4j вычисляет правые части всех элементов одного `SET`
относительно состояния **до** предложения, поэтому `t.count` в формуле — ещё
старый `n`, и `(avg·n + x)/(n+1)` даёт точное среднее.

Проверка на поднятом Neo4j 5, последовательность значений `[1.0, 0.0, 0.0]`
(истинное среднее 0.3333):

| Формула | count | avg |
|---|---|---|
| исходная (из репозитория) | 3 | **0.3333** ✓ |
| «исправленная» через sum/count | 4 (на данных [1,0,0.5,0.5]) | 0.5 ✓ |

Обе дают верный результат; исходная проще, поэтому она и оставлена. В коде
добавлен только комментарий с результатом проверки, чтобы следующий читатель
не пришёл к тому же ошибочному выводу.

**Что в этом замечании оказалось верным:** `g.sample_count` выставлялся в 1 при
`ON CREATE` и никогда не рос. Исправлено:

```cypher
ON MATCH SET g.sample_count = coalesce(g.sample_count, 0) + 1, ...
```

---

## ✅ 🟠 6. Дедупликация возвращает оценку чужого кандидата

**Исправлено.** Кэш отдаётся только при совпадении `candidate_id`; чужая запись
логируется и игнорируется, кандидат считается заново. Дедупликация развязана с
сохранением: у `run_pipeline()` появился отдельный параметр `use_cache`
(по умолчанию `True`), так что «посчитать без записи, но с кэшем» и «сохранить,
но обязательно пересчитать» теперь выражаются независимо. Функция вынесена в
`_cached_evaluation()`; повреждённый JSON в кэше больше не роняет запрос.
Покрыто тестами `test_cache_never_returns_another_candidates_evaluation`,
`test_use_cache_false_forces_recompute`, `test_corrupted_cache_is_ignored`.

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

## ✅ 🟠 7. Локальный запуск не читает `.env`

**Исправлено.** `load_dotenv()` вызывается в `src/main.py`, `src/worker/celery_app.py`
и `alembic/env.py` — до импортов, читающих окружение. В Docker переменные уже
выставлены через `env_file`/`environment`, и `load_dotenv` их не перезаписывает.

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

## ✅ 🟠 8. Для short-кандидатов используются long-ориентированные метрики без инверсии

**Исправлено** (решение согласовано с владельцем проекта: метрику для short не
учитывать). Появились две функции в скорере — единственные места, где
`long_outcome_share` и p70/p80 переводятся в метрики кандидата:

```python
win_rate_for(c)   # long → long_outcome_share, short → 1 - long_outcome_share
fa_ratio_for(c)   # long → ratio, short → None
```

Для short ось `directional` усредняется по двум критериям (win rate + skew)
вместо трёх, `favorable_adverse_ratio` в ответе API равен `null`, в промпт LLM
уходит явное «неприменимо к short-кандидату», а в `risks` добавляется пояснение.
Проверено на живом API: short-кандидат возвращает `fa_ratio None`.

Альтернатива «инвертировать как 1/ratio» отклонена: p70 и p80 — разные
перцентили, простая инверсия их не симметрирует. Правильное решение — запросить
у генератора кандидатов поля `p70_short_favorable_pct` / `p80_short_adverse_pct`.

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

## ✅ 🟠 9. `select_best_per_family`: свежесть всегда важнее качества

**Исправлено** (решение согласовано: score с бонусом за свежесть). Свежесть стала
надбавкой `FRESH_BONUS = 0.05` к `quality_score`, а не первичным ключом
сортировки. Сильный `stale` больше не проигрывает слабому `fresh`, но при
сопоставимых оценках побеждает свежий.

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

## ✅ 🟠 10. Пороги рейтинга продублированы в четырёх местах

**Исправлено.** Пороги вынесены в константы `RATING_STRONG_MIN` /
`RATING_MODERATE_MIN` рядом с `get_rating()`, и все три дубля (`_build_prompt`,
сборка результата в `llm_node`, `/score/quick`) заменены вызовом `get_rating()`.

Одна и та же логика «≥0.75 → STRONG, ≥0.55 → MODERATE, иначе WEAK» написана в:

- `src/scorer/candidate_scorer.py:193` — `get_rating()`, канонический вариант;
- `src/agent/llm_node.py:59` — внутри `_build_prompt()`;
- `src/agent/llm_node.py:130` — при сборке результата;
- `src/api/routes.py:92` — в `/score/quick`.

Изменение порогов в скорере не затронет остальные три места, и `/score/quick`
начнёт отвечать не тем, что `/evaluate/json`.

**Исправление:** везде вызывать `get_rating()`.

---

## ✅ 🟠 11. Вызов Claude без обработки ошибок, таймаута и ретраев

**Исправлено.** Клиент создаётся с `timeout=30s` и `max_retries=2`. Ошибки
`anthropic.APIError` и любые непредвиденные исключения перехватываются: оценка
не теряется, возвращается детерминированный вариант с пометкой в `risks`
(числа-то считает скорер, а не LLM). Текст берётся первым блоком с
`type == "text"` — extended thinking и tool use больше не ломают разбор.
Покрыто `tests/test_llm_node.py` (11 тестов).

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

## ✅ 🟠 12. Роуты `/evaluate/*` маскируют внутренние ошибки под 422

**Исправлено.** Появился хелпер `_fail(exc, context)`: `ValidationError`,
`ValueError`, `TypeError` → 422, всё остальное → 500 с `logger.exception`.
Применён во всех роутах, включая запросы к БД, графу и статистике, где раньше
стоял безусловный 500 без логирования. Проверено на живом API: битый кандидат
даёт 422.

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

## ✅ 🟡 13. Redis Stream используется без consumer groups

**Исправлено.** Чтение переведено на `XREADGROUP` с группой
`btc:candidates:workers` (создаётся идемпотентно через `XGROUP CREATE ... MKSTREAM`),
подтверждение — `XACK` + `XDEL`. Каждое сообщение выдаётся ровно одному читателю.

Проверено на живом Redis: до правки повторное чтение возвращало те же три
сообщения (`['c0','c1','c2']`), после — пустой список.

Заодно `POST /queue/enqueue` больше не отвечает `queued: true` при недоступном
Redis: пустой `msg_id` даёт 503, а сам кандидат теперь разбирается до постановки
в очередь, чтобы невалидные данные отбрасывались с 422, а не оседали в стриме.

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

## ✅ 🟡 14. `evaluate_candidate` ретраит невалидные данные

**Исправлено.** `ValidationError` / `ValueError` / `TypeError` логируются и
завершают задачу результатом `{"error": "invalid_candidate", ...}` без повтора.
Ретраи остались только для сбоев, которые могут пройти сами.

`src/worker/tasks.py:41-43`

Любое исключение приводит к `self.retry()`. Для сетевых сбоев это правильно, но
кандидат с отсутствующим обязательным полем будет трижды перепроверен с
интервалом 30 секунд, прежде чем окончательно упасть. Валидационные ошибки
детерминированы — ретраить их бессмысленно.

**Исправление:** ловить `pydantic.ValidationError` отдельно, логировать и
возвращать результат-ошибку без retry.

---

## ✅ 🟡 15. Кириллическая буква в имени класса `AgeБucket`

**Исправлено.** Переименован в `AgeBucket` во всех шести файлах.

`src/models/candidate.py:8` — в `AgeБucket` вторая буква не латинская `B`, а
кириллическая `Б` (U+0411). Код работает (Python допускает Unicode в
идентификаторах), но имя невозможно набрать вслепую, оно не находится
grep'ом по `AgeBucket`, и импортируется в шести файлах:
`scorer/candidate_scorer.py`, `validator/candidate_validator.py`,
`filters/candidate_filter.py`, `db/embedding.py`, `agent/pipeline.py`.

**Исправление:** переименовать в `AgeBucket` во всех местах.

---

## 🟡 16. Мёртвый код и неиспользуемые импорты — частично исправлено

**Убраны** неиспользуемые импорты: `validate_candidate` в `llm_node` и
`candidate_filter`, `logging` и `shared_task` в `worker/tasks`,
`HistoricalBiasContext` в `embedding`, `contextmanager` в `graph_repo`.

**Оставлено намеренно:** таблица `market_events` с моделью `MarketEventRecord`
(создана миграцией 001, накачена в БД — снос требует отдельной миграции и
решения, нужен ли задел) и `candidate_repo.is_hash_evaluated()` (рабочий метод
репозитория, может понадобиться при отказе от Redis-дедупликации).

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

## ✅ 🟡 17. Вектор 384 при 18 значащих признаках

**Исправлено.** `VECTOR_DIM` уменьшен с 384 до 32 (18 признаков + запас, чтобы
добавление новых не требовало миграции). Добавлена миграция
`003_shrink_embedding_dim.py`: дропает HNSW-индекс, меняет тип колонки,
**пересчитывает embedding существующих строк из `raw_payload`** и пересоздаёт
индекс. `orm_models` берёт размерность из `VECTOR_DIM`, так что рассинхрон
между моделью и схемой больше невозможен.

Проверено на живой БД: тип стал `vector(32)`, значение пересчитано
(`vector_dims = 32`), `POST /candidates/similar` отвечает 200.

**Что осталось:** нормализация признаков по-прежнему неоднородна — часть в
[0..1], часть обрезана через `min(x/10, 1.0)`, поэтому вклад признаков в
косинусную метрику неявно взвешен. Это вопрос качества поиска, не корректности.

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

**Исправлено.** Добавлен пакет `tests/` — 215 тестов, `pytest` в `requirements.txt`,
конфиг в `pytest.ini`, цели `make test` / `make test-local`.

| Файл | Тестов | Что покрывает |
|---|---|---|
| `tests/test_scorer.py` | 98 | Каждая ступень каждой из 4 осей, инверсия win rate и неприменимость F/A для short, веса, границы рейтингов, эталонный кандидат |
| `tests/test_validator.py` | 39 | Каждый `warning_flag`, его пороговое значение и отсутствие ложных срабатываний |
| `tests/test_parser.py` | 23 | dict / JSON-строка / raw text / array, разделители `:` и `=`, хвосты `←`, комментарии, опечатки в ключах, выход за диапазоны |
| `tests/test_filters.py` | 24 | Фильтрация по порогу, ранжирование, дедуп по `family_key` с `FRESH_BONUS`, все 4 типа конфликтов |
| `tests/test_pipeline.py` | 20 | Детерминированная оценка, батч, дедупликация через Redis, деградация `_persist` |
| `tests/test_llm_node.py` | 11 | Извлечение текста из ответа, разбор JSON, fallback при сбое Claude API |

Тесты не требуют ни PostgreSQL, ни Redis, ни Neo4j, ни ключа Anthropic:
в `tests/conftest.py` отсутствующие SDK подменяются заглушками (только если
реального пакета нет), а `DATABASE_URL` принудительно уводится на in-memory
sqlite — тесты физически не могут обратиться к боевой БД.

**Что осталось непокрытым:** слои, требующие живой инфраструктуры — репозитории
PostgreSQL, Cypher-запросы Neo4j, SQL к continuous aggregates, HTTP-роуты
(`TestClient` + фикстура БД), Celery-задачи. Для них нужны интеграционные тесты
с поднятым стеком (`testcontainers` или `docker compose` в CI).

---

## ✅ 🟡 19. Инфраструктурные мелочи

**Исправлено:**

| Что | Как |
|---|---|
| `.env` с реальным ключом мог уехать в git | Добавлен `.gitignore` (`.env`, `.idea/`, `__pycache__`, `.DS_Store`, кэши) |
| Секреты и мусор попадали в образ | Добавлен `.dockerignore` (`.env`, `.git/`, `.idea/`, кэши) |
| Postgres/Redis/Neo4j слушали все интерфейсы | Порты привязаны к `127.0.0.1` — с хоста доступ есть, из сети нет |
| `version: "3.9"` вызывал warning | Удалён |
| `alembic.ini` содержал логин/пароль | Очищен; URL берётся из `DATABASE_URL` в `env.py` (с тем же дефолтом, что в `connection.py`) |
| `celery-beat` не ждал Postgres | `depends_on` дополнен `postgres: service_healthy` |
| `pipeline` тянул `anthropic` даже при `use_llm=False` | Импорт `llm_node` стал ленивым; детерминированная оценка вынесена в `src/agent/deterministic.py` |
| README указывал несуществующий тег образа | Исправлено на `timescale/timescaledb-ha:pg16` |

**Осталось:** API по-прежнему без аутентификации и rate-limit — `POST /evaluate`
с `use_llm=true` тратит токены Anthropic по анонимному запросу. Для локальной
машины приемлемо; перед любым выходом наружу нужен хотя бы API-ключ и лимит.

---

## ✅ 🔴 20. Celery-воркер в Docker не слушал очередь `evaluate` (найдено при проверке)

Этого замечания в первом аудите **не было** — оно всплыло при прогоне
асинхронного пути на живом стеке.

`docker-compose.yml` запускал воркер как

```
celery -A src.worker.celery_app worker --loglevel=info --concurrency=4
```

без `-Q`, то есть воркер слушал только очередь по умолчанию — `celery`. При этом
`celery_app.conf.task_routes` направляет `evaluate_candidate` в очередь
`evaluate`. Результат: `process_stream_batch` исправно разбирал стрим и
диспатчил задачи, а те **навсегда оседали в Redis необработанными**. Весь
асинхронный путь (`POST /queue/enqueue`) в Docker не работал.

Локальный запуск из README был корректен — там `-Q evaluate,celery` указан,
поэтому расхождение и не бросалось в глаза.

**Диагностика на живом стеке:**

```
LLEN evaluate            → 2          (задачи копятся)
inspect active_queues    → только 'celery'
```

**Исправлено** — очереди указаны явно:

```yaml
command: celery -A src.worker.celery_app worker --loglevel=info --concurrency=4 -Q evaluate,celery
```

После перезапуска накопленные задачи разобрались сразу:

```
LLEN evaluate → 0
Evaluated candidate short_test_001 → STRONG (score=0.983)
```

---

## Итог

**Исправлено:** №1 (скорер), №2 (сессия SQLAlchemy), №3 (AUTOCOMMIT для refresh),
№4 (логирование вместо `except: pass`), №6 (дедупликация), №7 (`load_dotenv`),
№8 (short-метрики), №9 (`FRESH_BONUS`), №10 (единый `get_rating`), №11 (ошибки LLM),
№12 (HTTP-коды), №13 (consumer groups), №14 (retry), №15 (`AgeBucket`),
№17 (размерность вектора + миграция 003), №18 (тесты), №19 (инфраструктура),
№20 (очередь Celery в Docker).

**Отозвано:** №5 — инкрементальное среднее в Cypher было корректно с самого начала.

**Исправлено частично и осознанно:** №16 — убраны неиспользуемые импорты, но
`market_events` и `is_hash_evaluated()` оставлены как задел.

### Что осталось на будущее

1. **Аутентификация и rate-limit для API** (из №19) — обязательно до выхода за
   пределы локальной машины: `/evaluate` с `use_llm=true` тратит токены анонимно.
2. **Симметричные short-метрики** (из №8) — запросить у генератора кандидатов
   `p70_short_favorable_pct` / `p80_short_adverse_pct`. Сейчас ось `directional`
   для short опирается на два критерия вместо трёх.
3. **Нормализация признаков embedding** (из №17) — вклад признаков в косинусную
   метрику сейчас неявно взвешен из-за разных схем нормализации.
4. **Интеграционные тесты** (из №18) — репозитории, Cypher, SQL к aggregates,
   HTTP-роуты и Celery-задачи покрыты только ручными прогонами на живом стеке.
5. **Снос `market_events`** (из №16), если задел не понадобится — отдельной миграцией.
6. **Статус сохранения в ответе API** (из №4) — `_persist()` возвращает статус
   по каждому хранилищу, но наружу он не отдаётся; добавление поля меняет схему
   ответа, поэтому отложено.

### Проверка после правок

```bash
make test-local          # 215 тестов, ~1 сек, инфраструктура не нужна
make test                # то же внутри контейнера, с настоящими SDK

# полный прогон на живом стеке
make up && make migrate
curl -s localhost:8000/health
```
