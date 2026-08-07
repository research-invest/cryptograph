-- Инициализация расширений PostgreSQL
-- Выполняется автоматически при первом запуске контейнера

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
