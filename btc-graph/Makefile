.PHONY: help setup up down reload restart build logs migrate ps shell-api shell-pg \
        test test-local profiles-check migrate-graph

help:
	@echo ""
	@echo "  Первый запуск:"
	@echo "    make setup      — создать .env из .env.example (впиши ANTHROPIC_API_KEY)"
	@echo "    make up         — поднять все сервисы"
	@echo "    make migrate    — применить миграции БД"
	@echo ""
	@echo "  Ежедневное использование:"
	@echo "    make ps         — статус контейнеров"
	@echo "    make logs       — логи всех сервисов (Ctrl+C для выхода)"
	@echo "    make down       — остановить и удалить контейнеры"
	@echo ""
	@echo "  После изменения .env или docker-compose.yml:"
	@echo "    make reload     — пересоздать контейнеры с новым конфигом"
	@echo ""
	@echo "  После изменения кода (Dockerfile/src):"
	@echo "    make build      — пересобрать образы и перезапустить"
	@echo ""
	@echo "  Тесты (БД и сервисы не нужны):"
	@echo "    make test       — прогнать тесты внутри контейнера api"
	@echo "    make test-local — прогнать тесты локально (pytest)"
	@echo ""
	@echo "  Профили калибровки (config/symbols/*.yaml):"
	@echo "    make profiles-check — проверить все YAML-профили на валидность"
	@echo "    make reload         — подхватить правку профиля (volume, без build)"
	@echo ""
	@echo "  Мультимонетность (разово, перед первой не-BTC монетой):"
	@echo "    make migrate        — включая 004_multi_symbol (ЧИТАЙ шапку миграции!)"
	@echo "    make migrate-graph  — проставить symbol узлам Neo4j + constraint"
	@echo ""
	@echo "  Отладка:"
	@echo "    make shell-api  — bash внутри контейнера api"
	@echo "    make shell-pg   — psql внутри контейнера postgres"

setup:
	cp .env.example .env
	@echo "Создан .env — впиши ANTHROPIC_API_KEY"

up:
	docker compose up -d

down:
	docker compose down

# Пересоздаёт контейнеры — подхватывает изменения в .env и docker-compose.yml
reload:
	docker compose up -d --force-recreate

# Пересобирает образы из Dockerfile и перезапускает
build:
	docker compose up -d --build

restart:
	docker compose restart

logs:
	docker compose logs -f

migrate:
	docker compose exec api alembic upgrade head

# Переводит накопленный граф на ключ (symbol, group_id). Прогнать ОДИН раз,
# до первого кандидата не по BTCUSDT — иначе узлы монет схлопнутся молча.
migrate-graph:
	docker compose exec -T neo4j cypher-shell -u neo4j -p btc_neo4j_pass \
		< scripts/migrate_graph_symbol.cypher

# Проверка профилей без поднятого стека: схема, монотонность лесенок,
# сумма весов, полнота enum-карт.
profiles-check:
	python3 -m src.config.profiles --check

# Тесты проверяют чистую логику — ни БД, ни Redis, ни Neo4j не требуются
test:
	docker compose exec api pytest

test-local:
	pytest

ps:
	docker compose ps

shell-api:
	docker compose exec api bash

shell-pg:
	docker compose exec postgres psql -U btc_user -d btc_graph
