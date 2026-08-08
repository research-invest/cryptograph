.PHONY: help setup install init-db ingest train train-fast live status admin test docker-up docker-down docker-logs

PY ?= python3

help:
	@echo "setup      — создать .env из шаблона"
	@echo "install    — поставить зависимости"
	@echo "init-db    — создать схему processing в БД btc-graph"
	@echo "ingest     — скачать историю Binance"
	@echo "train      — полный прогон истории с отправкой в btc-graph"
	@echo "train-fast — прогон по уже загруженным данным, без отправки"
	@echo "live       — инкрементальный прогон по свежим барам"
	@echo "status     — что лежит в базе и доступен ли btc-graph"
	@echo "admin      — запустить админку"
	@echo "test       — тесты"
	@echo "docker-up  — поднять админку в docker (нужен запущенный btc-graph)"

setup:
	@test -f .env || cp .env.example .env
	@echo "Готово. Заполни ADMIN_PASSWORD и ADMIN_SECRET_KEY в .env:"
	@echo "  openssl rand -hex 32"

install:
	$(PY) -m pip install -r requirements.txt

init-db:
	$(PY) -m btcproc.cli init-db

ingest:
	$(PY) -m btcproc.cli ingest

train:
	$(PY) -m btcproc.cli train

train-fast:
	$(PY) -m btcproc.cli train --no-ingest --no-emit

live:
	$(PY) -m btcproc.cli live

status:
	$(PY) -m btcproc.cli status

admin:
	$(PY) -m btcproc.cli admin --reload

test:
	$(PY) -m pytest -q

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f admin
