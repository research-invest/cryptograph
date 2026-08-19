"""pg_stat_statements: статистика по запросам за период.

Revision ID: 005
Revises: 004
Create Date: 2026-08-19

Зачем. `pg_stat_activity` показывает только то, что база делает ПРЯМО СЕЙЧАС,
и на вопрос «что её грузило ночью» не отвечает в принципе: закончившийся
запрос из неё исчезает. Разбор потребления памяти контейнера (журнал btcproc,
раздел про страницу PostgreSQL) упёрся ровно в это — увидеть виновника
задним числом было нечем.

Расширение работает только когда библиотека загружена из
`shared_preload_libraries`, а это параметр времени старта. Он выставлен в
`command` сервиса postgres в `docker-compose.yml`; здесь создаётся сама
вьюха. Порядок обязателен: без рестарта с новым `command` CREATE EXTENSION
падает с «pg_stat_statements must be loaded via shared_preload_libraries».
Деплой этот порядок соблюдает — контейнеры пересоздаются до миграций
(`01_deploy.sh`, шаг 8).

Миграция намеренно НЕ падает, если библиотека не загружена: расширение —
диагностика, а не схема, и ронять из-за неё выкатку нечем оправдать.
Страница «PostgreSQL» в админке умеет показать его отсутствие как состояние
и подсказать команду.

На новых развёртываниях то же самое делает `scripts/init_db.sql`; здесь —
для баз, созданных до этой правки.
"""
from __future__ import annotations

import logging

from alembic import op

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")


def upgrade() -> None:
    # Своя транзакция: неудача CREATE EXTENSION иначе отравила бы общую, и
    # следующие миграции упали бы на ровном месте.
    connection = op.get_bind()
    try:
        with connection.begin_nested():
            connection.exec_driver_sql(
                "CREATE EXTENSION IF NOT EXISTS pg_stat_statements"
            )
    except Exception as exc:            # noqa: BLE001 — см. докстринг
        logger.warning(
            "pg_stat_statements не создан (%s). Библиотека грузится только из "
            "shared_preload_libraries: проверь command сервиса postgres в "
            "docker-compose.yml и пересоздай контейнер "
            "(docker compose up -d postgres), затем повтори миграцию.",
            str(exc).strip().splitlines()[0] if str(exc).strip() else exc,
        )


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS pg_stat_statements")
