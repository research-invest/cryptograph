.PHONY: help setup install init-db ingest ingest-all train train-all train-fast \
        live live-all status admin test docker-up docker-down docker-logs

PY ?= python3

help:
	@echo "setup      — создать .env из шаблона"
	@echo "install    — поставить зависимости"
	@echo "init-db    — создать схему processing в БД btc-graph"
	@echo "ingest     — скачать историю Binance (монета из .env)"
	@echo "ingest-all — то же по всем активным монетам реестра"
	@echo "train      — полный прогон истории с отправкой в btc-graph"
	@echo "train-all  — полный прогон по всем активным монетам, последовательно"
	@echo "train-fast — прогон по уже загруженным данным, без отправки"
	@echo "live       — инкрементальный прогон по свежим барам"
	@echo "live-all   — инкрементальный прогон по всем активным монетам"
	@echo "status     — таблица по монетам: покрытие, отставание, очередь"
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

ingest-all:
	$(PY) -m btcproc.cli ingest --all

train:
	$(PY) -m btcproc.cli train

# Монеты идут последовательно, каждая своим run_id. Падение одной не
# останавливает остальные, но exit code будет ненулевым — иначе cron
# рапортовал бы об успехе при упавшей монете.
train-all:
	$(PY) -m btcproc.cli train --all

train-fast:
	$(PY) -m btcproc.cli train --no-ingest --no-emit

live:
	$(PY) -m btcproc.cli live

live-all:
	$(PY) -m btcproc.cli live --all

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
