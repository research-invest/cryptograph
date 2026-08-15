.PHONY: help setup install init-db ingest ingest-all ingest-metrics train train-all train-fast \
        live live-all status admin hostmon hostmon-once hostmon-test-telegram \
        maintenance validate-holdout test docker-up docker-down docker-logs

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
	@echo "hostmon    — монитор нагрузки хоста (страница «Сервер» в админке)"
	@echo "hostmon-once — один замер и выход, для проверки"
	@echo "hostmon-test-telegram — проверить канал уведомлений о нагрузке"
	@echo "maintenance— недельная уборка: разметка старых live + вакуум"
	@echo "validate-holdout — проверка системы на отложенной части истории (часы)"
	@echo "test       — тесты"
	@echo "docker-up  — поднять админку в docker (нужен запущенный btc-graph)"
	@echo "ingest-metrics — деривативные метрики Binance USD-M в deriv_metrics"

setup:
	@test -f .env || cp .env.example .env
	@echo "Готово. Заполни ADMIN_PASSWORD и ADMIN_SECRET_KEY в .env:"
	@echo "  openssl rand -hex 32"

install:
	$(PY) -m pip install -r requirements.txt

init-db:
	$(PY) -m btcproc.cli init-db

admin:
	$(PY) -m btcproc.cli admin --reload

# Монитор ресурсов. На боевом контуре это systemd-сервис btcproc-hostmon;
# здесь цель нужна для локальной разработки страницы «Сервер».
hostmon:
	$(PY) -m btcproc.cli hostmon

hostmon-once:
	$(PY) -m btcproc.cli hostmon --once

hostmon-test-telegram:
	$(PY) -m btcproc.cli hostmon --test-telegram

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

# Стоит в кроне раз в неделю; руками полезен --dry-run, чтобы сперва посмотреть.
maintenance:
	$(PY) scripts/maintenance.py

# Внешние дневные ряды (Fear & Greed и т.п.) в external_daily. Сетевой поход
# в чужой API — сюда, а не в train/live (docs/task_fear_greed.md, раздел 8).
# В кроне — раз в сутки, отдельной записью.
fetch-external:
	$(PY) -m btcproc.cli fetch-external

# Деривативные метрики Binance USD-M (ОИ, long/short ratio, тейкеры) в
# deriv_metrics — тот же принцип, что fetch-external: отдельно от train/live,
# файл суток появляется только на следующие сутки (docs/tz_deriv_ingest_14-08-26.md, §0.7).
ingest-metrics:
	$(PY) -m btcproc.cli ingest-metrics --all

# Обучает модель на первых 70% истории и меряет на остальных 30%. Часы
# машинного времени на три монеты, поэтому в кроне ему делать нечего:
# запускается руками после изменений, меняющих геометрию кластеров.
validate-holdout:
	$(PY) scripts/validate_holdout.py --all

# Контроль к предыдущей цели (D1): те же признаки и то же разбиение, но
# бустинг вместо графа состояний. Отвечает на вопрос «потолок в графе или в
# признаках»; критерий заявлен в шапке btcproc/analysis/control.py.
control-model:
	$(PY) scripts/control_model.py --all --importance

test:
	$(PY) -m pytest -q

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f admin
