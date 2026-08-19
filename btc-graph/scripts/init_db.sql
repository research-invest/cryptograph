-- Инициализация расширений PostgreSQL
-- Выполняется автоматически при первом запуске контейнера

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Статистика по запросам за период — то, чего не даёт pg_stat_activity:
-- она показывает только идущее прямо сейчас. Библиотека грузится из
-- shared_preload_libraries (см. command сервиса postgres в docker-compose.yml);
-- здесь создаётся сама вьюха. На уже существующей базе этот файл не
-- выполняется — там расширение создаёт миграция.
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
