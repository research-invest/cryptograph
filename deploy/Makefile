.PHONY: help deploy deploy-fast deploy-full configure status

help:
	@echo ""
	@echo "  Ежедневное использование:"
	@echo "    make deploy       — выкатить свежий код (./01_deploy.sh --code-only)"
	@echo "    make deploy-fast  — то же, но без тестов на сервере (--code-only --no-test)"
	@echo ""
	@echo "  Первый запуск (чистая VPS):"
	@echo "    make deploy-full  — пакеты, docker, venv, образы, код (./01_deploy.sh)"
	@echo "    make configure    — .env, стек, миграции, первый прогон, кроны (./02_configure.sh)"
	@echo ""
	@echo "  Отладка:"
	@echo "    make status       — статус монет на сервере (btcproc status по ssh)"
	@echo ""
	@echo "  Хост переопределяется переменными окружения (см. шапку 01_deploy.sh):"
	@echo "    VPS_HOST, VPS_USER, REMOTE_DIR"

# Ежедневная выкатка: код, зависимости (если requirements менялись), тесты,
# пересборка запущенных контейнеров, миграции, рестарт админки. Подготовку
# машины и первичную раздачу каталога не трогает — см. 01_deploy.sh.
deploy:
	./01_deploy.sh --code-only

# Без прогона тестов на сервере — самая долгая часть --code-only. Годится,
# когда код уже проверен локально и нужна быстрая выкатка.
deploy-fast:
	./01_deploy.sh --code-only --no-test

# Полный деплой на чистую Ubuntu-VPS: пакеты, UTC, swap, docker, venv, образы.
# Идемпотентен — повторный запуск догоняет состояние, не ломая сделанное.
deploy-full:
	./01_deploy.sh

# Конфигурация и первый запуск: .env, поднятие стека, миграции, кроны.
configure:
	./02_configure.sh

status:
	ssh $${VPS_USER:-vps}@$${VPS_HOST:-162.248.225.60} \
		'$${REMOTE_DIR:-/opt/crypto-graph}/bin/btcproc status'
