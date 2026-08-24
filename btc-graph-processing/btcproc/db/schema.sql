-- Схема processing: всё, что считает генератор кандидатов.
-- Таблицы btc-graph (public.candidates, public.candidate_events) не трогаются:
-- в них пишет только его собственный pipeline.
--
-- Скрипт идемпотентный, гоняется при каждом старте (make init-db).
-- {schema} подставляется из PG_SCHEMA кодом в src/db/session.py.

CREATE SCHEMA IF NOT EXISTS {schema};
SET search_path TO {schema}, public;

-- ─── Сырые бары ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol        TEXT             NOT NULL,
    tf            TEXT             NOT NULL,
    ts            TIMESTAMPTZ      NOT NULL,
    open          DOUBLE PRECISION NOT NULL,
    high          DOUBLE PRECISION NOT NULL,
    low           DOUBLE PRECISION NOT NULL,
    close         DOUBLE PRECISION NOT NULL,
    volume        DOUBLE PRECISION NOT NULL,
    quote_volume  DOUBLE PRECISION,
    trades        INTEGER,
    taker_buy_base DOUBLE PRECISION,
    PRIMARY KEY (symbol, tf, ts)
);

CREATE INDEX IF NOT EXISTS ohlcv_tf_ts_idx ON ohlcv (tf, ts DESC);

-- ─── Признаки базового ТФ ───────────────────────────────────────────────────
-- Имена признаков лежат в feature_sets.names — так набор можно менять,
-- не переписывая схему.
CREATE TABLE IF NOT EXISTS feature_sets (
    version     TEXT PRIMARY KEY,
    names       TEXT[] NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    params      JSONB
);

CREATE TABLE IF NOT EXISTS features (
    symbol   TEXT               NOT NULL,
    ts       TIMESTAMPTZ        NOT NULL,
    version  TEXT               NOT NULL,
    values   DOUBLE PRECISION[] NOT NULL,
    PRIMARY KEY (symbol, ts, version)
);

-- ─── События ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS bar_events (
    symbol          TEXT        NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    event_block_id  TEXT        NOT NULL,
    atoms           TEXT[]      NOT NULL,
    families        TEXT[]      NOT NULL,
    atom_count      INTEGER     NOT NULL,
    family_count    INTEGER     NOT NULL,
    intensity       TEXT        NOT NULL,
    primary_family  TEXT,
    PRIMARY KEY (symbol, ts)
);

CREATE INDEX IF NOT EXISTS bar_events_block_idx ON bar_events (event_block_id);

CREATE TABLE IF NOT EXISTS event_blocks (
    run_id          BIGINT      NOT NULL,
    event_block_id  TEXT        NOT NULL,
    total_rows      INTEGER     NOT NULL,
    row_share       DOUBLE PRECISION NOT NULL,
    rarity          TEXT        NOT NULL,
    intensity       TEXT        NOT NULL,
    atom_count      INTEGER     NOT NULL,
    family_count    INTEGER     NOT NULL,
    primary_family  TEXT,
    families        TEXT[],
    PRIMARY KEY (run_id, event_block_id)
);

-- ─── Состояния и граф ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS state_models (
    run_id      BIGINT PRIMARY KEY,
    version     TEXT NOT NULL,
    n_groups    INTEGER NOT NULL,
    feature_ver TEXT NOT NULL,
    params      JSONB,
    artifact    BYTEA,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS market_groups (
    run_id        BIGINT           NOT NULL,
    group_id      DOUBLE PRECISION NOT NULL,
    size          INTEGER          NOT NULL,
    share         DOUBLE PRECISION NOT NULL,
    dominant_bias TEXT,
    up_share      DOUBLE PRECISION,
    avg_ret_pct   DOUBLE PRECISION,
    avg_vol_pct   DOUBLE PRECISION,
    centroid      DOUBLE PRECISION[],
    top_features  JSONB,
    PRIMARY KEY (run_id, group_id)
);

CREATE TABLE IF NOT EXISTS bar_states (
    symbol       TEXT             NOT NULL,
    ts           TIMESTAMPTZ      NOT NULL,
    run_id       BIGINT           NOT NULL,
    group_id     DOUBLE PRECISION NOT NULL,
    prev_group_id DOUBLE PRECISION,
    state_seq    BIGINT           NOT NULL,
    age_minutes  INTEGER          NOT NULL,
    age_bucket   TEXT             NOT NULL,
    entropy      TEXT             NOT NULL,
    is_transition BOOLEAN         NOT NULL,
    transition_id TEXT,
    PRIMARY KEY (symbol, ts, run_id)
);

CREATE INDEX IF NOT EXISTS bar_states_transition_idx
    ON bar_states (run_id, transition_id) WHERE is_transition;
CREATE INDEX IF NOT EXISTS bar_states_group_idx ON bar_states (run_id, group_id);

CREATE TABLE IF NOT EXISTS transitions (
    run_id        BIGINT NOT NULL,
    transition_id TEXT   NOT NULL,
    prev_group_id DOUBLE PRECISION,
    cur_group_id  DOUBLE PRECISION NOT NULL,
    count         INTEGER NOT NULL,
    share         DOUBLE PRECISION NOT NULL,
    rarity        TEXT   NOT NULL,
    avg_horizon_return DOUBLE PRECISION,
    up_share      DOUBLE PRECISION,
    PRIMARY KEY (run_id, transition_id)
);

-- ─── Исходы ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS outcomes (
    symbol   TEXT             NOT NULL,
    ts       TIMESTAMPTZ      NOT NULL,
    horizon  TEXT             NOT NULL,
    ret_pct  DOUBLE PRECISION,
    mfe_pct  DOUBLE PRECISION,
    mae_pct  DOUBLE PRECISION,
    is_up    BOOLEAN,
    -- Размах на том же окне: см. шапку candidates/outcomes.py.
    range_pct   DOUBLE PRECISION,
    rv_fwd      DOUBLE PRECISION,
    range_ratio DOUBLE PRECISION,
    valid    BOOLEAN          NOT NULL,
    PRIMARY KEY (symbol, ts, horizon)
);

-- ─── Кандидаты ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id   TEXT PRIMARY KEY,
    run_id         BIGINT      NOT NULL,
    symbol         TEXT        NOT NULL,
    ts             TIMESTAMPTZ NOT NULL,
    transition_id  TEXT        NOT NULL,
    event_block_id TEXT        NOT NULL,
    family_key     TEXT,
    research_side  TEXT        NOT NULL,
    research_score DOUBLE PRECISION NOT NULL,
    sample_size    INTEGER     NOT NULL,
    payload        JSONB       NOT NULL,
    -- Заполняется после отправки в btc-graph.
    quality_score  DOUBLE PRECISION,
    rating         TEXT,
    direction      TEXT,
    warning_flags  TEXT[],
    evaluation     JSONB,
    emitted_at     TIMESTAMPTZ,
    emit_error     TEXT,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS candidates_run_ts_idx ON candidates (run_id, ts DESC);
CREATE INDEX IF NOT EXISTS candidates_rating_idx ON candidates (rating, direction);
CREATE INDEX IF NOT EXISTS candidates_pending_idx ON candidates (emitted_at)
    WHERE emitted_at IS NULL;
CREATE INDEX IF NOT EXISTS candidates_transition_idx ON candidates (transition_id);

-- ─── Прогоны ────────────────────────────────────────────────────────────────
-- Прогон всегда про ОДНУ монету: модель состояний обучается на каждую
-- отдельно. Из этого следует, что state_models, market_groups, transitions
-- и event_blocks помонетны автоматически — run_id уже однозначно задаёт
-- монету, и колонка symbol в них не нужна. Заодно это делает невозможным
-- состояние «в одном прогоне смешаны две монеты».
CREATE TABLE IF NOT EXISTS runs (
    run_id      BIGSERIAL PRIMARY KEY,
    kind        TEXT        NOT NULL,
    symbol      TEXT,
    status      TEXT        NOT NULL DEFAULT 'running',
    stage       TEXT,
    progress    DOUBLE PRECISION NOT NULL DEFAULT 0,
    params      JSONB,
    stats       JSONB,
    error       TEXT,
    log         TEXT        NOT NULL DEFAULT '',
    started_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS runs_started_idx ON runs (started_at DESC);

-- ─── Мультимонетность ───────────────────────────────────────────────────────
-- Скрипт идемпотентен и гоняется при каждом init-db, поэтому доводка старых
-- баз делается здесь же, а не отдельным механизмом миграций.

-- Колонка, а не params->>'symbol': по ней идут выборки «последняя модель
-- монеты» и фильтр списка прогонов, и индекс по выражению из JSONB тут лишний.
ALTER TABLE runs ADD COLUMN IF NOT EXISTS symbol TEXT;

-- UPDATE безопасен в идемпотентном скрипте: он проставляет значение только
-- там, где колонка пустая, то есть ровно один раз — на прогонах, сделанных
-- до появления мультимонетности. Все они были по BTCUSDT.
UPDATE runs SET symbol = COALESCE(params->>'symbol', 'BTCUSDT') WHERE symbol IS NULL;

CREATE INDEX IF NOT EXISTS runs_symbol_started_idx ON runs (symbol, started_at DESC);

-- last_candidate_ts и страница кандидатов всегда фильтруют по монете.
CREATE INDEX IF NOT EXISTS candidates_symbol_ts_idx ON candidates (symbol, ts DESC);

-- Покрытие истории в status и на дашборде считается по (symbol, tf).
CREATE INDEX IF NOT EXISTS ohlcv_symbol_tf_ts_idx ON ohlcv (symbol, tf, ts DESC);

-- ─── Имена состояний ────────────────────────────────────────────────────────
-- Человекочитаемая подпись состояния, выведенная из его отклонений от
-- среднего рынка (btcproc/states/naming.py). Хранится, а не считается на
-- лету, чтобы имя было тем же самым во всех местах и в старых прогонах —
-- словарь формулировок со временем меняется, а подпись прогона меняться
-- не должна.
ALTER TABLE market_groups ADD COLUMN IF NOT EXISTS name TEXT;

-- ─── Контекстные атомы ──────────────────────────────────────────────────────
-- Фоновые атомы (тренд, сессии, доминирование тейкеров) в event_block_id не
-- входят и до этой колонки нигде не сохранялись — считались и выбрасывались.
-- Без них нельзя измерить, даёт ли фон лифт: сопоставлять кандидата не с чем.
--
-- Колонка nullable без DEFAULT намеренно: NULL здесь честно означает «бар
-- размечен прогоном, который контекст ещё не писал», и это отличимо от '{}'
-- («контекст посчитан, активных атомов нет»). Задним числом не заполняется —
-- значения появятся на барах по мере train/live.
ALTER TABLE bar_events ADD COLUMN IF NOT EXISTS context_atoms TEXT[];

-- ─── Версия набора атомов ───────────────────────────────────────────────────
-- event_block_id — хэш битовой маски, и смысл блока задаёт не только маска,
-- но и то, каким НАБОРОМ детекторов она посчитана. Строки двух прогонов с
-- разным набором внешне неотличимы: те же колонки, те же идентификаторы
-- блоков, правдоподобные значения. Колонка делает состав явным.
--
-- В первичный ключ версия НЕ входит намеренно. PK остаётся (symbol, ts): при
-- смене набора атомов строка должна перезаписаться, а не удвоиться. Хранить
-- две разметки одного бара незачем — актуальна всегда последняя, а разбор
-- «чем размечено» даёт как раз эта колонка.
--
-- NULL здесь честно означает «размечено прогоном до появления колонки» и
-- задним числом не заполняется: чем именно размечены те строки, из данных
-- уже не восстановить.
ALTER TABLE bar_events ADD COLUMN IF NOT EXISTS version TEXT;
ALTER TABLE event_blocks ADD COLUMN IF NOT EXISTS version TEXT;

-- ─── Heartbeat прогона ──────────────────────────────────────────────────────
-- Статус `running` снимает только сам процесс (finish_run / fail_run). Если
-- процесс убит — OOM killer на расчёте признаков, ребут, kill -9, — строка
-- остаётся `running` НАВСЕГДА: механизма протухания у неё нет.
--
-- Дальше отказ становится тихим: крон стоит с --skip-if-busy, поэтому каждый
-- следующий live этой монеты молча пропускается (это штатное поведение skip,
-- оно не даёт ни ошибки, ни ненулевого кода возврата), а в админке мёртвый
-- прогон навсегда занимает слот ADMIN_MAX_CONCURRENT_RUNS. Обновление данных
-- по монете просто прекращается, и заметно это только по растущему отставанию.
--
-- Де-факто heartbeat уже был: update_run пишет progress и лог на каждой
-- стадии. Не хватало отметки времени этой записи — её и заводим.
ALTER TABLE runs ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

CREATE INDEX IF NOT EXISTS runs_running_idx ON runs (status, updated_at DESC)
    WHERE status = 'running';

-- ─── Размах как предмет разметки (2026-08-19) ───────────────────────────────
-- Задача D `crypto-graph/docs/tz_range_horizons_19-08-26.md`, блок C аудита
-- §C1. До этого `outcomes` описывал только НАПРАВЛЕНИЕ и путь (ret/mfe/mae) —
-- а замер раздела 47 показал, что предсказуемо здесь как раз не направление.
-- Три величины размаха считает `candidates/outcomes.py`, их смысл и
-- предупреждение про знаменатель ATR14 — в шапке того модуля.
--
-- Колонки добавляются к СУЩЕСТВУЮЩЕЙ таблице, а не заводится новая: у
-- `outcomes` в первичном ключе уже есть `horizon`, и разные горизонты живут
-- строками рядом. Старые строки не переписываются — ближайший прогон
-- заполнит новые колонки штатным upsert'ом, а до него они NULL, что честно
-- означает «не считалось».
ALTER TABLE outcomes ADD COLUMN IF NOT EXISTS range_pct   DOUBLE PRECISION;
ALTER TABLE outcomes ADD COLUMN IF NOT EXISTS rv_fwd      DOUBLE PRECISION;
ALTER TABLE outcomes ADD COLUMN IF NOT EXISTS range_ratio DOUBLE PRECISION;

-- ─── Модель размаха (2026-08-19) ────────────────────────────────────────────
-- Квантильный регрессор `analysis/range_forecast.py`, раздел 48 журнала.
-- Устроена по образцу `state_models`: одна модель на прогон обучения, живёт
-- во всех последующих `live` (связь через `runs.params.model_run_id`, тот же
-- `runs.model_run_scope`).
--
-- `calibrated` — прошла ли модель гейт из трёх условий на отложенной части.
-- Хранятся ОБЕ: непрошедшая нужна для разбора в админке, но её числа в
-- кандидата не попадают. Решение не выбрасывать её принято ровно потому, что
-- «модели нет» и «модель есть, но не годится» — разные диагнозы, а по
-- отсутствию строки их не отличить.
--
-- `artifact` — joblib-пикл (внутри обученные бустинги, в JSON они не
-- сериализуются). Версия sklearn лежит внутри артефакта: пиклы между
-- версиями библиотеки несовместимы, и загрузчик обязан это проверять.
CREATE TABLE IF NOT EXISTS range_models (
    run_id        BIGINT PRIMARY KEY,
    version       TEXT NOT NULL,
    horizon       TEXT NOT NULL,
    normalization TEXT NOT NULL,
    calibrated    BOOLEAN NOT NULL,
    train_end     TIMESTAMPTZ,
    metrics       JSONB,
    artifact      BYTEA,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- ─── Уведомления (вебхуки) ──────────────────────────────────────────────────
-- Правило = «куда слать» + «что именно слать» (фильтр по кандидату, теми же
-- осями, что и селектор на странице кандидатов).
--
-- Правила лежат в БД, а не в .env и не в коде: их правит оператор из админки,
-- и перезапуск процесса ради нового адреса или порога — неприемлемая цена.
-- Механика доставки (таймауты, число потоков, окно свежести) наоборот живёт
-- в окружении: это свойство установки, а не бизнес-настройка.
--
-- Пустой массив и NULL в фильтрах означают «не фильтруем по этой оси». Это
-- сделано намеренно вместо отдельного флага «фильтр включён»: правило без
-- ограничений должно создаваться пустой формой, а не набором галок.
CREATE TABLE IF NOT EXISTS notification_rules (
    rule_id            BIGSERIAL PRIMARY KEY,
    name               TEXT        NOT NULL,
    enabled            BOOLEAN     NOT NULL DEFAULT TRUE,
    url                TEXT        NOT NULL,
    -- Заголовки запроса: авторизация получателя (Authorization, X-Api-Key).
    -- JSONB, а не отдельные колонки, — набор зависит от чужой системы.
    headers            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    -- full   — весь кандидат (37 полей схемы) + оценка btc-graph;
    -- compact — только сводка (монета, время, сторона, рейтинг, оценка).
    payload_mode       TEXT        NOT NULL DEFAULT 'full',

    -- ── Фильтр ──────────────────────────────────────────────────────────────
    symbol             TEXT,               -- NULL = любая монета
    ratings            TEXT[],             -- STRONG / MODERATE / WEAK
    directions         TEXT[],             -- long / short
    min_quality        DOUBLE PRECISION,
    min_research_score DOUBLE PRECISION,
    min_sample_size    INTEGER,
    transitions        TEXT[],             -- конкретные переходы, «42->1»
    event_families     TEXT[],             -- primary_event_family
    -- Слать только тех, кого btc-graph принял и оценил. По умолчанию да:
    -- без оценки у кандидата нет ни rating, ни quality_score, то есть
    -- большая часть фильтров к нему неприменима в принципе.
    require_evaluation BOOLEAN     NOT NULL DEFAULT TRUE,

    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Журнал доставок. У него две роли, и вторая важнее первой:
--   1. диагностика — «почему не пришло» иначе не разобрать вообще никак,
--      отправка идёт в фоновом потоке и ответ получателя нигде не виден;
--   2. защита от повторов — PK (rule_id, candidate_id). Окна live намеренно
--      перекрываются, кандидат детерминирован по id и попадается нескольким
--      прогонам подряд; без этого ключа один и тот же кандидат уходил бы
--      получателю столько раз, сколько прогонов его увидели.
CREATE TABLE IF NOT EXISTS notification_deliveries (
    rule_id      BIGINT      NOT NULL,
    candidate_id TEXT        NOT NULL,
    symbol       TEXT,
    -- queued (взят в работу) → sent | failed | dropped
    status       TEXT        NOT NULL DEFAULT 'queued',
    http_status  INTEGER,
    error        TEXT,
    attempts     INTEGER     NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sent_at      TIMESTAMPTZ,
    PRIMARY KEY (rule_id, candidate_id)
);

CREATE INDEX IF NOT EXISTS notification_deliveries_recent_idx
    ON notification_deliveries (created_at DESC);

-- ─── Внешние дневные ряды ───────────────────────────────────────────────────
-- Общая таблица для ЛЮБОГО внешнего дневного ряда (Fear & Greed, фандинг,
-- открытый интерес…), а не именная под конкретный источник: второй источник
-- не должен требовать третьей таблицы. `series` различает их внутри одной
-- таблицы (`'fear_greed'`, позже `'funding'` и т.п.).
--
-- Ряд общерыночный: он один на все монеты и джойнится к каждой отдельно
-- (docs/task_fear_greed.md, раздел 3.1) — своей колонки symbol здесь нет.
--
-- `meta` хранит `value_classification` и прочее сырьё ответа поставщика —
-- пригодится при разборе, не тратя отдельную колонку на каждое поле API.
--
-- `fetched_at` — когда мы увидели ЭТО значение (может обновляться при
-- повторной загрузке, см. docs/task_fear_greed.md, раздел 3.3): поставщик
-- иногда пересчитывает историю задним числом, и это должно быть видно.
CREATE TABLE IF NOT EXISTS external_daily (
    series     TEXT             NOT NULL,   -- 'fear_greed'
    day        DATE             NOT NULL,   -- сутки UTC, к которым относится значение
    value      DOUBLE PRECISION NOT NULL,
    meta       JSONB,
    fetched_at TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    PRIMARY KEY (series, day)
);

-- ─── Деривативные метрики Binance USD-M ────────────────────────────────────
-- docs/tz_deriv_ingest_14-08-26.md, §1.3. Сетка баров базового ТФ, помонетная
-- (в отличие от external_daily — это не общерыночный ряд).
--
-- NULL в колонке = данных нет (не ноль и не ffill): в архиве встречаются дни
-- с живым ОИ и полностью пустыми ratio-колонками (§0.9, 2022-01-01/15). Дырка
-- по КОЛОНКАМ, а не по дням целиком — заполнять чем бы то ни было запрещено:
-- позавчерашний ОИ, выданный за сегодняшний, не виден ни по одной метрике.
--
-- tf в ключе — по образцу ohlcv, чтобы смена BASE_TIMEFRAME не перезаписала
-- чужие строки молча.
CREATE TABLE IF NOT EXISTS deriv_metrics (
    symbol      TEXT             NOT NULL,
    tf          TEXT             NOT NULL,
    ts          TIMESTAMPTZ      NOT NULL,
    oi          DOUBLE PRECISION,   -- sum_open_interest, last (уровень, не поток)
    oi_value    DOUBLE PRECISION,   -- sum_open_interest_value, last (содержит цену внутри)
    ls_top_acc  DOUBLE PRECISION,   -- count_toptrader_long_short_ratio, last
    ls_top_pos  DOUBLE PRECISION,   -- sum_toptrader_long_short_ratio, last
    ls_global   DOUBLE PRECISION,   -- count_long_short_ratio, last
    taker_ratio DOUBLE PRECISION,   -- sum_taker_long_short_vol_ratio, mean (поток за интервал)
    -- Сколько УНИКАЛЬНЫХ 5m-строк (после дедупа по create_time) легло в окно
    -- [T, T+15m). 3 = полный бар, 1–2 = частичный, строки нет вовсе = метрик
    -- за этот бар нет — тот самый `valid`-механизм по образцу outcomes.py,
    -- только по колонкам, а не по барам целиком.
    src_rows    SMALLINT NOT NULL,
    PRIMARY KEY (symbol, tf, ts)
);

CREATE INDEX IF NOT EXISTS deriv_metrics_tf_ts_idx ON deriv_metrics (tf, ts DESC);

-- ─── Стакан: снимки глубины ────────────────────────────────────────────────
-- ТЗ crypto-graph/docs/tz_wave_a_24-08-26.md, задача E. Единственная таблица
-- проекта, которая заводится ДО того, как решено, зачем она нужна, — и это
-- сознательно.
--
-- Причина в календаре, а не в гипотезе. Стакан доступен только вперёд:
-- исторической глубины не отдаёт ни одна биржа ни за какие деньги. Значит
-- решение «полезен ли он» можно принять только через год накопления, а
-- решение «накапливать ли» — только сегодня. Год ожидания не сокращается
-- ничем.
--
-- Отсюда два правила, которые нельзя нарушать, пока не появится отдельное ТЗ:
-- никаких признаков, атомов и гейтов из этих данных НЕ заводится, и ни
-- `train`, ни `live` эту таблицу не читают. Наполняет её отдельная команда
-- `ingest-depth` — по тому же принципу, что `fetch-external` и
-- `ingest-metrics`: сетевой поход в чужой API не должен ронять расчёт.
--
-- `levels` — сырые уровни как их отдал поставщик, {"b": [[цена, объём], …],
-- "a": [...]}. Сырьё, а не агрегаты, потому что неизвестно, какие агрегаты
-- понадобятся: посчитать дисбаланс из уровней можно всегда, восстановить
-- уровни из дисбаланса — никогда.
--
-- Денормализованные колонки рядом — ровно те три, без которых нельзя даже
-- посмотреть, живой ли ряд, не разбирая JSONB на каждой строке.
CREATE TABLE IF NOT EXISTS depth_snapshots (
    symbol     TEXT             NOT NULL,
    ts         TIMESTAMPTZ      NOT NULL,   -- момент снимка, усечён до секунды
    venue      TEXT             NOT NULL,
    best_bid   DOUBLE PRECISION NOT NULL,
    best_ask   DOUBLE PRECISION NOT NULL,
    -- Суммарный объём по каждой стороне в пределах снятой глубины. Метрика
    -- «дисбаланс» из них считается тривиально, а хранить производную от двух
    -- чисел смысла нет.
    bid_volume DOUBLE PRECISION NOT NULL,
    ask_volume DOUBLE PRECISION NOT NULL,
    levels     JSONB            NOT NULL,
    PRIMARY KEY (symbol, ts)
);

CREATE INDEX IF NOT EXISTS depth_snapshots_ts_idx ON depth_snapshots (ts DESC);

-- ─── Фон состояний: посчитанный агрегат ─────────────────────────────────────
-- Насколько часто контекстный атом встречается внутри состояния по сравнению
-- со всей размеченной историей. Считался на лету при каждом открытии узла
-- графа: join bar_states × bar_events по всей истории монеты плюс unnest
-- массивов — секунды на прогретом кэше, десятки секунд на холодном, и
-- параллельные воркеры на каждый запрос. Пять кликов подряд укладывали базу.
--
-- Хранится, а не кэшируется в памяти, потому что величина неизменна: фон
-- считается по разметке train-прогона, а она после его завершения не
-- меняется. Пороги отбора (доля, лифт, верхушка) здесь НЕ применяются —
-- лежат сырые доли: что показывать, решает btc-graph-приёмник этой таблицы,
-- то есть админка, и менять пороги без пересчёта агрегата она обязана уметь.
CREATE TABLE IF NOT EXISTS state_context (
    run_id   BIGINT           NOT NULL,
    group_id DOUBLE PRECISION NOT NULL,
    atom     TEXT             NOT NULL,
    share    DOUBLE PRECISION NOT NULL,
    lift     DOUBLE PRECISION,
    PRIMARY KEY (run_id, group_id, atom)
);

-- Отметка «фон этого прогона посчитан». Без неё пустой результат неотличим
-- от непосчитанного: у прогона может честно не быть ни одного контекстного
-- атома (вся история размечена до появления колонки `bar_events.context_atoms`),
-- и тогда каждое открытие узла заново гоняло бы тот самый тяжёлый запрос.
CREATE TABLE IF NOT EXISTS state_context_runs (
    run_id      BIGINT PRIMARY KEY,
    bars        INTEGER     NOT NULL,
    atom_rows   INTEGER     NOT NULL,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ─── Корреляция symbol ↔ run_id ─────────────────────────────────────────────
-- run_id однозначно задаёт монету (один прогон = одна монета), но планировщик
-- считает условия независимыми и множит селективности: на `symbol = X AND
-- run_id = N` он ожидал 23 строки вместо 80 тысяч и выбирал nested loop с
-- отдельным index scan на каждый бар — 320 тыс. обращений к буферам там, где
-- хватает 14 тыс. Статистика зависимостей чинит оценку.
CREATE STATISTICS IF NOT EXISTS bar_states_symbol_run_stx (dependencies, ndistinct)
    ON symbol, run_id FROM bar_states;
