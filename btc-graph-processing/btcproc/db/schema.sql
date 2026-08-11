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
