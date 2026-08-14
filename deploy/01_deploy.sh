#!/usr/bin/env bash
#
# Разворачивание crypto-graph на чистой Ubuntu-VPS.
#
# Запускается ЛОКАЛЬНО (с машины, где лежат оба проекта) и всё делает по ssh.
# Ничего не конфигурирует и не запускает: после него на сервере лежит код,
# стоит docker, собран venv и собраны образы — но .env нет и стек не поднят.
# Конфигурация, первый прогон и кроны — во втором скрипте, 02_configure.sh.
#
# Скрипт идемпотентен: повторный запуск догоняет состояние, не ломая уже
# сделанное. Именно поэтому он же служит и «выкатить свежий код»:
#     ./01_deploy.sh --code-only
#
# В режиме --code-only делается только то, что относится к самому коду:
# rsync обоих проектов, доустановка зависимостей (если requirements менялись),
# тесты, пересборка запущенных контейнеров, миграции схемы и перезапуск
# админки. Подготовка машины (пакеты, UTC, swap, docker, ufw, образы с нуля)
# и первичная раздача каталога не трогаются.
#
# Требования на локальной машине: ssh-ключ уже на сервере (ssh-copy-id),
# rsync. Требования на сервере: sudo без пароля.
#
set -euo pipefail

# ─── Параметры ───────────────────────────────────────────────────────────────
VPS_HOST="${VPS_HOST:-162.248.225.60}"
VPS_USER="${VPS_USER:-vps}"
REMOTE_DIR="${REMOTE_DIR:-/opt/crypto-graph}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(dirname "$SCRIPT_DIR")"

SSH_TARGET="$VPS_USER@$VPS_HOST"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30)

# ─── Вывод ───────────────────────────────────────────────────────────────────
step() { printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }
info() { printf '  %s\n' "$*"; }
die()  { printf '\n\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ─── Аргументы ───────────────────────────────────────────────────────────────
# Разбор строгий, а не «$1 == --code-only»: опечатка вроде --codeonly раньше
# молча означала полный деплой со всей подготовкой машины.
CODE_ONLY=0
NO_TEST=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --code-only) CODE_ONLY=1 ;;
        # Только сам прогон pytest на сервере — не страховка перед миграцией
        # и рестартом, а самая долгая по времени часть --code-only. Годится
        # для быстрой выкатки, когда код уже проверен локально.
        --no-test)   NO_TEST=1 ;;
        -h|--help)
            printf 'Использование: %s [--code-only] [--no-test]\n' "${0##*/}"
            printf '  без флага     полный деплой на чистую машину\n'
            printf '  --code-only   только код: rsync, зависимости, тесты, миграции, рестарт\n'
            printf '  --no-test     пропустить прогон тестов на сервере (шаг 7)\n'
            exit 0 ;;
        *) die "Неизвестный аргумент: $1 (есть только --code-only и --no-test)" ;;
    esac
    shift
done

# Каждый шаг — отдельное ssh-соединение. Это не только устойчивее к обрыву,
# но и обязательное условие: членство в группе docker применяется при входе,
# поэтому команды docker должны идти уже в новой сессии после usermod.
#
# Скрипт уходит аргументом bash -c, а не через stdin, и stdin ssh закрыт:
# иначе первая же команда, читающая stdin (`docker compose exec -T`, psql,
# apt в интерактивном режиме), сожрала бы остаток скрипта, а шаг при этом
# завершился бы с кодом 0, не выполнив ни строки после неё.
# set -euo пишется внутрь скрипта, а не во флаги bash: запущенный по ssh bash
# успевает прочитать /etc/bash.bashrc, а тот на Ubuntu начинается с проверки
# $PS1 и под -u падает с «unbound variable» ещё до первой нашей строки.
remote() {
    local b64
    b64=$(printf 'set -euo pipefail\n%s' "$(cat)" | base64 | tr -d '\n')
    ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
        "bash -c \"\$(printf '%s' '$b64' | base64 -d)\"" </dev/null
}

# ─── 0. Проверки перед тем, как что-то менять ────────────────────────────────
step "Проверяю окружение"

[[ -d "$LOCAL_DIR/btc-graph/src" ]] \
    || die "Не вижу $LOCAL_DIR/btc-graph/src — запусти скрипт из каталога crypto-graph/deploy"
[[ -d "$LOCAL_DIR/btc-graph-processing/btcproc" ]] \
    || die "Не вижу $LOCAL_DIR/btc-graph-processing/btcproc"
command -v rsync >/dev/null || die "Нужен rsync на локальной машине"

ssh "${SSH_OPTS[@]}" "$SSH_TARGET" 'true' \
    || die "Нет ssh-доступа к $SSH_TARGET. Сначала: ssh-copy-id $SSH_TARGET"

if [[ $CODE_ONLY -eq 0 ]]; then
remote <<'EOF'
command -v sudo >/dev/null || { echo "На сервере нет sudo" >&2; exit 1; }
sudo -n true 2>/dev/null || { echo "sudo требует пароль — скрипт так работать не сможет" >&2; exit 1; }
. /etc/os-release
[[ "$ID" == "ubuntu" || "$ID" == "debian" ]] || { echo "Скрипт рассчитан на Ubuntu/Debian, а тут $PRETTY_NAME" >&2; exit 1; }
echo "  сервер: $PRETTY_NAME, ядер $(nproc), память $(free -h | awk '/^Mem:/{print $2}'), свободно $(df -h --output=avail / | tail -1 | tr -d ' ')"
EOF
else
# Пригодность машины уже доказана предыдущим полным деплоем — проверять ОС и
# sudo незачем. Убеждаемся только, что деплоить есть куда: без venv это первый
# запуск, и код без окружения выкатывать бессмысленно.
REMOTE_DIR="$REMOTE_DIR" remote <<EOF
[[ -d "$REMOTE_DIR/venv" ]] \
    || { echo "На сервере нет $REMOTE_DIR/venv — сначала полный деплой: ./01_deploy.sh" >&2; exit 1; }
echo "  режим --code-only: только код, зависимости, тесты и миграции"
EOF
fi

info "локальный источник: $LOCAL_DIR"
info "цель: $SSH_TARGET:$REMOTE_DIR"

# ─── 1. Системные пакеты ─────────────────────────────────────────────────────
if [[ $CODE_ONLY -eq 0 ]]; then
step "Ставлю системные пакеты"
remote <<'EOF'
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
# python3-venv — обязателен: на Ubuntu 24.04 системный pip заблокирован
# (PEP 668), ставить зависимости можно только в виртуальное окружение.
# libpq-dev/gcc — страховка на случай сборки psycopg2 из исходников.
sudo apt-get install -y -qq \
    python3-venv python3-dev python3-pip \
    make git curl rsync ca-certificates \
    libpq-dev gcc \
    ufw logrotate cron >/dev/null
sudo systemctl enable --now cron >/dev/null 2>&1 || true
echo "  пакеты на месте"
EOF

# ─── 1.4. Часовой пояс ───────────────────────────────────────────────────────
# Явно, а не «Ubuntu и так ставит UTC»: у части хостеров образ идёт с местной
# зоной, и тогда расписание уезжает молча. Цена расхождения высокая — Binance
# отдаёт бары в UTC, и все метки в базе тоже UTC. В локальной зоне «отставание
# баров» в status и время в логах разошлись бы с реальностью на несколько
# часов, а cron-строки означали бы не то время, что написано.
step "Ставлю часовой пояс UTC"
remote <<'EOF'
current=$(timedatectl show -p Timezone --value 2>/dev/null || echo "?")
if [[ "$current" == "UTC" || "$current" == "Etc/UTC" ]]; then
    echo "  уже $current"
else
    sudo timedatectl set-timezone UTC
    # cron читает зону при старте и после смены продолжил бы считать по старой.
    sudo systemctl restart cron 2>/dev/null || true
    echo "  было $current → стало UTC (cron перезапущен)"
fi
date -u '+  сейчас: %Y-%m-%d %H:%M UTC'
EOF

# ─── 1.5. Swap ───────────────────────────────────────────────────────────────
# У VPS обычно нет swap вовсе, и на 8 ГБ это ловушка: расчёт признаков держит
# в pandas всю историю разом, а рядом уже сидят postgres и neo4j. Пик упирается
# в потолок, и ядро убивает прогон — в логе btcproc при этом НИЧЕГО, прогон
# просто обрывается на середине этапа, а запись в processing.runs навсегда
# остаётся в статусе running и блокирует следующий cron-овый live.
# Диагноз виден только в journalctl: «killed by the OOM killer».
step "Настраиваю swap"
remote <<'EOF'
if swapon --show=NAME --noheadings | grep -q .; then
    echo "  swap уже есть: $(free -h | awk '/^Swap:/{print $2}')"
else
    # Размер по объёму памяти: страховка от пика, а не второй уровень памяти.
    size_gb=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) / 1024 / 1024 ))
    (( size_gb < 2 )) && size_gb=2
    sudo fallocate -l "${size_gb}G" /swapfile 2>/dev/null \
        || sudo dd if=/dev/zero of=/swapfile bs=1M count=$(( size_gb * 1024 )) status=none
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile >/dev/null
    sudo swapon /swapfile
    grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
    echo "  создан /swapfile на ${size_gb} ГБ"
fi
# Низкая swappiness: swap нужен как подушка под пик, а не как повседневная
# память — иначе кластеризация уедет на диск и прогон растянется на часы.
sudo sysctl -qw vm.swappiness=10
grep -q '^vm.swappiness' /etc/sysctl.conf \
    || echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf >/dev/null
free -h | sed 's/^/  /'
EOF

# ─── 2. Docker ───────────────────────────────────────────────────────────────
step "Ставлю Docker"
DOCKER_USER="$VPS_USER" remote <<EOF
if command -v docker >/dev/null; then
    echo "  docker уже стоит: \$(docker --version)"
else
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sudo sh /tmp/get-docker.sh >/dev/null
    rm -f /tmp/get-docker.sh
    echo "  поставлен: \$(docker --version)"
fi
sudo systemctl enable --now docker >/dev/null
# Без этого каждая команда docker требовала бы sudo, и cron от имени $VPS_USER
# не смог бы дёрнуть стек.
if ! id -nG "$VPS_USER" | tr ' ' '\n' | grep -qx docker; then
    sudo usermod -aG docker "$VPS_USER"
    echo "  $VPS_USER добавлен в группу docker"
fi
docker compose version >/dev/null 2>&1 || sudo apt-get install -y -qq docker-compose-plugin >/dev/null
EOF
fi

# ─── 3. Каталог и код ────────────────────────────────────────────────────────
# Раздача каталога нужна ровно один раз: дальше он уже принадлежит $VPS_USER,
# а rsync и так пишет от его имени. В --code-only шаг пропускается — иначе
# каждая выкатка кода дёргала бы sudo chown -R по всему дереву.
if [[ $CODE_ONLY -eq 0 ]]; then
step "Готовлю каталог $REMOTE_DIR"
REMOTE_DIR="$REMOTE_DIR" VPS_USER="$VPS_USER" remote <<EOF
sudo mkdir -p "$REMOTE_DIR/logs"
sudo chown -R "$VPS_USER":"$VPS_USER" "$REMOTE_DIR"
EOF
fi

step "Копирую код (rsync)"
# .env не переносим намеренно: секреты сервера генерирует 02_configure.sh,
# и перезаписывать их выкаткой кода нельзя.
# dumps/ — локальные выгрузки кандидатов, серверу не нужны (123 МБ).
# data/binance переносим: это кэш месячных дампов, он экономит первому
# ingest десятки минут и трафик.
# data/bybit — НЕ переносим. Там тиковые архивы, из которых бары считаются
# на месте, и весит это на порядок больше: 386 МБ на одну монету против 43 МБ
# всего кэша Binance, плюс ~6 МБ за каждый новый месяц каждой монеты. На
# домашнем канале это час выкатки вместо минуты, причём при КАЖДОЙ выкатке.
# Серверу дешевле скачать их самому: public.bybit.com ему доступен (в отличие
# от api.bybit.com, который отвечает 403 — см. журнал btcproc, 27.8).
RSYNC_EXCLUDES=(
    --exclude '.git/'
    --exclude '.env'
    --exclude '.env.local'
    --exclude '__pycache__/'
    --exclude '*.pyc'
    --exclude '.pytest_cache/'
    --exclude '.DS_Store'
    --exclude '.idea/'
    --exclude 'venv/'
    --exclude '.venv/'
    --exclude 'dumps/'
    --exclude 'data/bybit/'
    --exclude 'docker-compose.override.yml'
)
for proj in btc-graph btc-graph-processing; do
    info "→ $proj"
    rsync -az --delete "${RSYNC_EXCLUDES[@]}" \
        -e "ssh ${SSH_OPTS[*]}" \
        "$LOCAL_DIR/$proj/" "$SSH_TARGET:$REMOTE_DIR/$proj/"
done
rsync -az -e "ssh ${SSH_OPTS[*]}" "$SCRIPT_DIR/" "$SSH_TARGET:$REMOTE_DIR/deploy/"

# ─── 4. Порты наружу ─────────────────────────────────────────────────────────
# API btc-graph не имеет аутентификации вообще, а flower показывает очередь
# задач. С 2026-08-09 в самом compose оба привязаны к 127.0.0.1, поэтому шаг
# обычно ничего не меняет и печатает «уже закрыт». Оставлен намеренно: он
# страхует раскатку старого чекаута, где биндинг ещё 0.0.0.0, и проверяет
# факт закрытия перед стартом стека.
#
# Правим сам docker-compose.yml, а не override: списки ports в compose при
# слиянии СКЛЕИВАЮТСЯ, поэтому override добавил бы второй биндинг, а не
# заменил первый. И ufw тут не помощник — docker пишет свои правила в обход
# его цепочек.
#
# В --code-only шаг не нужен: локальный compose давно закрыт на 127.0.0.1,
# rsync привозит именно его, а факт закрытия проверяется одной строкой перед
# перезапуском стека — там, где он и важен.
if [[ $CODE_ONLY -eq 0 ]]; then
step "Закрываю API и flower на localhost"
REMOTE_DIR="$REMOTE_DIR" remote <<EOF
python3 - <<'PY'
import pathlib, re
p = pathlib.Path("$REMOTE_DIR/btc-graph/docker-compose.yml")
s = p.read_text(encoding="utf-8")
before = s
for port in ("8000", "5555"):
    s = re.sub(rf'- "(?:0\.0\.0\.0:)?{port}:{port}"', f'- "127.0.0.1:{port}:{port}"', s)
if s != before:
    p.write_text(s, encoding="utf-8")
    print("  compose поправлен: 8000 и 5555 теперь только на 127.0.0.1")
else:
    print("  compose уже закрыт на 127.0.0.1")
PY
grep -n '127.0.0.1:\(8000\|5555\)' "$REMOTE_DIR/btc-graph/docker-compose.yml" >/dev/null \
    || { echo "Не удалось закрыть порты — проверь docker-compose.yml вручную" >&2; exit 1; }
EOF
fi

# ─── 5. Общий venv ───────────────────────────────────────────────────────────
# Один интерпретатор на оба проекта — это не экономия, а требование режима
# SINK_MODE=direct: btcproc импортирует пакет btc-graph как библиотеку, и
# pgvector с neo4j должны лежать там же, откуда запущен btcproc. Разъезд
# интерпретаторов проявляется молча: кандидаты считаются, отправка рапортует
# об успехе, записей в базе нет.
#
# В --code-only pip не гоняется вхолостую: рядом с venv лежит отпечаток обоих
# requirements.txt, и пока он совпадает — ставить нечего. Отпечаток берётся
# уже после rsync, поэтому правка зависимостей подхватывается автоматически.
step "Собираю общий venv"
CODE_ONLY="$CODE_ONLY" REMOTE_DIR="$REMOTE_DIR" remote <<EOF
cd "$REMOTE_DIR"
[[ -d venv ]] || python3 -m venv venv

req_hash=\$(cat btc-graph/requirements.txt btc-graph-processing/requirements.txt | md5sum | awk '{print \$1}')
stamp="venv/.requirements.md5"
if [[ "$CODE_ONLY" == "1" && -f "\$stamp" && "\$(cat "\$stamp")" == "\$req_hash" ]]; then
    echo "  зависимости не менялись — pip пропущен"
    exit 0
fi

./venv/bin/pip install --quiet --upgrade pip setuptools wheel
echo "  ставлю зависимости btc-graph…"
./venv/bin/pip install --quiet -r btc-graph/requirements.txt
echo "  ставлю зависимости btc-graph-processing…"
./venv/bin/pip install --quiet -r btc-graph-processing/requirements.txt
echo "  python: \$(./venv/bin/python -V)"
./venv/bin/python - <<'PY'
import importlib.util as u
# pgvector и neo4j проверяются не для порядка: их импортирует чужой код
# btc-graph, лениво и уже внутри сохранения. Без них кандидаты считаются,
# sink рапортует об успехе, а записей в базе нет — молча.
need = ("pgvector", "neo4j", "anthropic", "sklearn", "pandas", "sqlalchemy", "typer")
missing = [m for m in need if u.find_spec(m) is None]
if missing:
    raise SystemExit("В venv не хватает: " + ", ".join(missing))
print("  ключевые пакеты на месте: " + ", ".join(need))
PY
# Отпечаток пишется последним: если pip или проверка упали, следующий запуск
# честно повторит установку, а не решит, что всё уже стоит.
echo "\$req_hash" > "\$stamp"
EOF

# ─── 6. Образы ───────────────────────────────────────────────────────────────
if [[ $CODE_ONLY -eq 0 ]]; then
step "Собираю docker-образы (это долго, качается ~2 ГБ)"
REMOTE_DIR="$REMOTE_DIR" remote <<EOF
cd "$REMOTE_DIR/btc-graph"
docker compose build --quiet
docker compose pull --quiet postgres redis neo4j 2>/dev/null || docker compose pull postgres redis neo4j
echo "  образы готовы"
EOF
fi

# ─── 7. Тесты: код на сервере живой? ─────────────────────────────────────────
# Оба набора тестов работают на чистой логике и инфраструктуры не требуют,
# поэтому их можно прогнать до поднятия стека — и это лучшая доступная
# проверка, что перенос ничего не порвал. --no-test пропускает именно этот
# шаг: миграции и рестарт ниже всё равно выполняются.
if [[ $NO_TEST -eq 1 ]]; then
    step "Пропускаю тесты (--no-test)"
else
step "Прогоняю тесты на сервере"
REMOTE_DIR="$REMOTE_DIR" remote <<EOF
cd "$REMOTE_DIR/btc-graph" && "$REMOTE_DIR/venv/bin/python" -m pytest -q 2>&1 | tail -5
cd "$REMOTE_DIR/btc-graph-processing" && "$REMOTE_DIR/venv/bin/python" -m pytest -q 2>&1 | tail -5
EOF
fi

# ─── 8. Перезапуск того, что уже работает ────────────────────────────────────
# Выкатка кода без этого шага обманчива: файлы на диске новые, а процессы —
# старые. Контейнеры btc-graph копируют src внутрь образа при сборке, поэтому
# нужна именно пересборка. Прогоны перезапускать не нужно — они короткоживущие
# и следующий cron-овый live возьмёт новый код сам.
#
# Порядок важен: сначала контейнеры, потом миграции, потом админка. Alembic
# запускается ВНУТРИ контейнера api — на старом образе он не увидит новых
# ревизий; админка же должна стартовать на уже мигрированной схеме.
step "Перезапускаю стек btc-graph"
STACK_RUNNING=0
# Код 3 = «стек не поднят», это не ошибка. Отсюда set +e вокруг вызова:
# при set -e скрипт умер бы раньше, чем мы прочитали $?.
set +e
REMOTE_DIR="$REMOTE_DIR" remote <<EOF
cd "$REMOTE_DIR/btc-graph"
if ! docker compose ps --status running --quiet 2>/dev/null | grep -q .; then
    echo "  стек btc-graph не запущен — пропускаю (поднимает 02_configure.sh)"
    exit 3
fi
# Дешёвая страховка вместо отдельного шага правки compose: rsync только что
# привёз файл с локальной машины, и API без аутентификации не должен уехать
# наружу незамеченным.
grep -q '127.0.0.1:8000:8000' docker-compose.yml \
    || { echo "API в docker-compose.yml не привязан к 127.0.0.1 — проверь файл" >&2; exit 1; }

set +e
docker compose up -d --build 2>&1 | tail -5
rc=\${PIPESTATUS[0]}
set -e
[[ \$rc -eq 0 ]] || { echo "docker compose up --build упал (код \$rc)" >&2; exit 1; }
echo "  контейнеры btc-graph пересобраны и перезапущены"
EOF
rc=$?
set -e
if [[ $rc -eq 3 ]]; then
    STACK_RUNNING=0
elif [[ $rc -eq 0 ]]; then
    STACK_RUNNING=1
else
    die "Не удалось перезапустить стек btc-graph"
fi

# ─── 9. Миграции ─────────────────────────────────────────────────────────────
# Схема догоняется здесь же: выкатка кода, который ждёт новых колонок, без
# миграции даёт падения уже в рантайме. Логика — та же, что в 02_configure.sh
# (там же и объяснение, зачем глушатся фоновые воркеры TimescaleDB): миграция
# 004 пересоздаёт continuous aggregates и дерётся с их же политикой обновления,
# оставляя alembic_version позади фактического DDL.
#
# Миграции графа Neo4j сюда не вынесены намеренно: они разовые, привязаны к
# заведению новой монеты и живут в 02_configure.sh.
if [[ $STACK_RUNNING -eq 1 ]]; then
step "Накатываю миграции"
REMOTE_DIR="$REMOTE_DIR" remote <<EOF
cd "$REMOTE_DIR/btc-graph"

wait_pg() {
    for _ in \$(seq 1 60); do
        [[ "\$(docker inspect --format '{{.State.Health.Status}}' btc_postgres 2>/dev/null || echo none)" == healthy ]] && return 0
        sleep 5
    done
    echo "postgres не вернулся в healthy" >&2
    return 1
}
psql_c() { docker compose exec -T postgres psql -U btc_user -d btc_graph -q -c "\$1"; }

wait_pg
current=\$(docker compose exec -T postgres psql -U btc_user -d btc_graph -tAc \
    "SELECT version_num FROM alembic_version" 2>/dev/null | tr -d '[:space:]' || true)
head_rev=\$(docker compose exec -T api alembic heads 2>/dev/null | awk '{print \$1; exit}' | tr -d '[:space:]' || true)

if [[ -n "\$current" && -n "\$head_rev" && "\$current" == "\$head_rev" ]]; then
    # Ничего не применяем — и базу не трогаем вовсе: обход стоит двух
    # рестартов postgres, платить их на каждой выкатке кода незачем.
    echo "  PostgreSQL: схема уже на \$current — новых миграций нет"
else
    echo "  PostgreSQL: \${current:-нет версии} → \${head_rev:-?}"
    # max_background_workers читается только при старте — отсюда рестарт, а не reload.
    psql_c "ALTER SYSTEM SET timescaledb.max_background_workers = 0;"
    docker compose restart postgres >/dev/null
    wait_pg

    set +e
    docker compose exec -T api alembic upgrade head 2>&1 | tail -15
    rc=\${PIPESTATUS[0]}
    set -e

    # Воркеры возвращаются в любом случае: база без них хуже упавшей миграции —
    # политики обновления агрегатов молча перестали бы работать.
    psql_c "ALTER SYSTEM RESET timescaledb.max_background_workers;"
    docker compose restart postgres >/dev/null
    wait_pg
    docker compose restart api celery-worker celery-beat >/dev/null
    [[ \$rc -eq 0 ]] || { echo "alembic upgrade head упал (код \$rc)" >&2; exit \$rc; }
    echo "  версия схемы: \$(docker compose exec -T postgres psql -U btc_user -d btc_graph -tAc 'SELECT version_num FROM alembic_version')"
fi

# Схема processing btcproc: своих alembic-ревизий у неё нет, init-db создаёт
# недостающие таблицы и идемпотентен.
if [[ -x "$REMOTE_DIR/bin/btcproc" ]]; then
    "$REMOTE_DIR/bin/btcproc" init-db 2>&1 | tail -5
else
    echo "  processing: bin/btcproc ещё нет — сделает 02_configure.sh"
fi
EOF
fi

step "Перезапускаю админку"
remote <<'EOF'
if systemctl list-unit-files btcproc-admin.service >/dev/null 2>&1 \
   && systemctl is-enabled --quiet btcproc-admin 2>/dev/null; then
    sudo systemctl restart btcproc-admin
    sleep 4
    if systemctl is-active --quiet btcproc-admin; then
        echo "  админка перезапущена"
    else
        echo "  админка не поднялась после перезапуска:" >&2
        sudo journalctl -u btcproc-admin -n 20 --no-pager >&2
        exit 1
    fi
else
    echo "  админка ещё не установлена — это сделает 02_configure.sh"
fi
EOF

printf '\n\033[1;32m✓ Код развёрнут\033[0m\n'
if [[ $CODE_ONLY -eq 1 ]]; then
cat <<TXT

  Код на сервере обновлён$( ((NO_TEST)) && echo ", тесты пропущены (--no-test)" || echo ", тесты прошли")$( ((STACK_RUNNING)) && echo ", схема догнана" || echo " (стек не запущен — миграции пропущены)").
  Что НЕ трогалось: пакеты, часовой пояс, swap, docker, ufw, .env, кроны.

  Проверить состояние:  ssh $SSH_TARGET '$REMOTE_DIR/bin/btcproc status'
TXT
else
cat <<TXT

  Что на сервере:
    $REMOTE_DIR/btc-graph               приёмник (docker-стек)
    $REMOTE_DIR/btc-graph-processing    генератор (запускается из venv)
    $REMOTE_DIR/venv                    общий интерпретатор обоих проектов
    $REMOTE_DIR/logs                    логи прогонов

  Стек ещё не поднят и .env не создан — это делает второй скрипт:

    ./02_configure.sh

  Выкатить только свежий код позже:  ./01_deploy.sh --code-only
TXT
fi
