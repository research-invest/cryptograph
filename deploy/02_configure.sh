#!/usr/bin/env bash
#
# Конфигурация, первый прогон и расписание.
#
# Запускается ЛОКАЛЬНО, после 01_deploy.sh. Делает по порядку:
#   1. .env обоих проектов (пароли админки генерируются здесь и печатаются в конце);
#   2. поднимает docker-стек btc-graph и ждёт, пока он станет здоровым;
#   3. миграции PostgreSQL и Neo4j, схема processing для btcproc;
#   4. firewall: наружу только ssh и http(s);
#   5. админка btcproc как systemd-сервис;
#   5b. монитор ресурсов хоста (btcproc-hostmon) — тоже systemd, на хосте;
#   6. nginx перед админкой + сертификат Let's Encrypt;
#   7. ПЕРВЫЙ ПРОГОН — train по BTCUSDT, под systemd, переживает обрыв ssh;
#   8. и только после успешного прогона — кроны.
#
# Порядок «сначала прогон, потом кроны» не косметический: cron-овый live
# отказался бы стартовать поверх идущего train той же монеты, а без обученной
# модели состояний ему всё равно не с чем работать.
#
# Скрипт идемпотентен. Существующий .env не перезаписывается — для этого
# нужен --force-env (пароли админки при этом сменятся).
#
# Точечные режимы: --nginx-only (только прокси), --hostmon-only (только монитор
# ресурсов: его настройки в .env и systemd-сервис). Оба ничего больше не трогают
# и нужны на уже развёрнутой машине.
#
set -euo pipefail

# ─── Параметры ───────────────────────────────────────────────────────────────
VPS_HOST="${VPS_HOST:-162.248.225.60}"
VPS_USER="${VPS_USER:-vps}"
REMOTE_DIR="${REMOTE_DIR:-/opt/crypto-graph}"

# ПОЛНЫЙ список монет контура через пробел — не «что доучить».
# Порядок значим: BTCUSDT первым — на нём откалиброваны пороги, и с ним
# сверяется граф остальных монет.
#
# Переменная работает сразу на два шага, и в этом легко ошибиться:
#   * шаг 8 обучает из неё только НЕобученные монеты;
#   * шаг 9 собирает из неё ВСЁ расписание заново.
# Поэтому «добавляю монету» — это TRAIN_SYMBOLS со всеми монетами, включая
# старые: с одной новой в списке остальные тихо выпадут из live и train.
# Шаг 4а сверяет переменную с обученными монетами в базе и не даёт этому
# случиться молча.
#
# Повторный запуск скрипта уже обученную монету НЕ переобучает: train
# переобучает модель состояний с нуля, и group_id нового прогона несопоставим
# со старым — накопленный в Neo4j граф после этого смешал бы две нумерации.
# Переобучение — отдельная осознанная операция, для неё есть --retrain.
TRAIN_SYMBOLS="${TRAIN_SYMBOLS:-BTCUSDT ETHUSDT SOLUSDT}"
# Монета по умолчанию для одиночных команд (SYMBOL в .env) — первая из списка.
TRAIN_SYMBOL="${TRAIN_SYMBOL:-${TRAIN_SYMBOLS%% *}}"
# Как часто cron гоняет live. Прогон занимает около двух минут.
LIVE_CRON="${LIVE_CRON:-*/30 * * * *}"
# Переобучение модели состояний. Раз в неделю, а не чаще, по трём причинам:
#   * прогон занимает десятки минут НА МОНЕТУ и упирается в CPU;
#   * каждый train перенумеровывает group_id — накопленный в Neo4j граф
#     (ключ там (symbol, group_id) без измерения модели) смешивает нумерации;
#   * рынок за неделю не меняется настолько, чтобы модель успела устареть.
# Время 03:20 UTC — воскресенье, между запусками live (:00 и :30), чтобы старт
# не попадал на идущий инкрементальный прогон.
TRAIN_CRON="${TRAIN_CRON:-20 3 * * 0}"
# Уборка базы: разметка старых live-прогонов + вакуум. Раз в неделю и
# СРЕДИНОЙ недели, а не в воскресенье: там train, который идёт часами на три
# монеты, а обслуживание при идущем прогоне само себя откладывает — стоя
# рядом с train, оно пропускалось бы каждый раз. 04:40 — между запусками
# live (:00 и :30), в самый тихий час.
MAINT_CRON="${MAINT_CRON:-40 4 * * 3}"
# Внешние дневные ряды (Fear & Greed) в external_daily. Отдельным кроном, а не
# внутри live, потому что это единственный поход в ЧУЖОЙ API: его недоступность
# не должна ронять регулярный прогон. Раз в сутки достаточно — ряд дневной.
# 05:10 UTC: alternative.me обновляет индекс около полуночи UTC, а к этому часу
# live уже отработал и train по воскресеньям ещё не начался.
FETCH_CRON="${FETCH_CRON:-10 5 * * *}"
# Деривативные метрики Binance USD-M (ОИ, long/short ratio, тейкеры) в
# deriv_metrics — тот же принцип, что FETCH_CRON: файл суток появляется
# только на следующие сутки (docs/tz_deriv_ingest_14-08-26.md, §0.7), поэтому
# сетевой поход отдельно от train/live. 05:25 — после FETCH_CRON, тем же
# спокойным часом.
DERIV_CRON="${DERIV_CRON:-25 5 * * *}"
# Откуда добирается свежий хвост баров. Дефолт здесь не совпадает с дефолтом
# btcproc намеренно: VPS часто стоит в юрисдикции, откуда api.binance.com
# отвечает 451, и обнаруживается это только падением прогона на последнем
# месяце. data-api.binance.vision — публичный эндпоинт Binance с тем же
# /api/v3/klines, работает и оттуда, и из России.
BINANCE_REST_URL="${BINANCE_REST_URL:-https://data-api.binance.vision/api/v3/klines}"

# ─── nginx перед админкой ────────────────────────────────────────────────────
# Домен должен уже резолвиться в адрес этого сервера: без этого не пройдёт
# http-01 challenge Let's Encrypt. Пустой ADMIN_DOMAIN = не ставить nginx
# вовсе, тогда админка как раньше смотрит наружу сама на :8100.
ADMIN_DOMAIN="${ADMIN_DOMAIN:-crypto-graph.selll.ru}"
# Сертификат. TLS_EMAIL нужен Let's Encrypt для писем об истечении.
ENABLE_TLS="${ENABLE_TLS:-1}"
TLS_EMAIL="${TLS_EMAIL:-starpers.com@gmail.com}"

# Две связанные настройки .env: за прокси админка слушает только localhost и
# доверяет X-Forwarded-For, без прокси — наоборот. Значения выводятся отсюда,
# чтобы они не могли разъехаться (доверять заголовку у открытого наружу порта
# означает отдать allowlist и lockout в руки клиента).
if [[ -n "$ADMIN_DOMAIN" ]]; then
    ADMIN_HOST_VALUE=127.0.0.1
    ADMIN_TRUST_PROXY_VALUE=true
else
    ADMIN_HOST_VALUE=0.0.0.0
    ADMIN_TRUST_PROXY_VALUE=false
fi

SSH_TARGET="$VPS_USER@$VPS_HOST"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30)

FORCE_ENV=0
SKIP_TRAIN=0
RESET_DB=0
RETRAIN=0
NGINX_ONLY=0
HOSTMON_ONLY=0
DROP_SYMBOLS=0
for arg in "$@"; do
    case "$arg" in
        --force-env)    FORCE_ENV=1 ;;
        --skip-train)   SKIP_TRAIN=1 ;;
        --reset-db)     RESET_DB=1 ;;
        --retrain)      RETRAIN=1 ;;
        --nginx-only)   NGINX_ONLY=1 ;;
        # Только монитор ресурсов: досыпать его настройки в .env и поставить
        # сервис. Нужен на уже развёрнутой машине — гонять весь скрипт ради
        # одного юнита значило бы трогать стек, миграции и расписание.
        --hostmon-only) HOSTMON_ONLY=1 ;;
        # Подтверждение, что обученная монета выводится из расписания
        # осознанно. Без него шаг 4а не даст ей молча выпасть из крона.
        --drop-symbols) DROP_SYMBOLS=1 ;;
        *) echo "Неизвестный аргумент: $arg (есть --force-env, --skip-train, --reset-db, --retrain, --nginx-only, --hostmon-only, --drop-symbols)" >&2; exit 1 ;;
    esac
done

if [[ $NGINX_ONLY -eq 1 && -z "$ADMIN_DOMAIN" ]]; then
    echo "--nginx-only без ADMIN_DOMAIN бессмыслен: нечего проксировать" >&2
    exit 1
fi

step() { printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }
info() { printf '  %s\n' "$*"; }
warn() { printf '\033[1;33m  ! %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# Скрипт уходит на сервер аргументом bash -c, а не через stdin, и stdin ssh
# закрыт. Иначе первая же команда, читающая stdin — а `docker compose exec -T`
# и psql его читают — сожрала бы весь остаток скрипта. Проявляется это
# отвратительно: шаг завершается с кодом 0, не выполнив ни строки после неё.
# set -euo пишется внутрь скрипта, а не во флаги bash: запущенный по ssh bash
# успевает прочитать /etc/bash.bashrc, а тот на Ubuntu начинается с проверки
# $PS1 и под -u падает с «unbound variable» ещё до первой нашей строки.
remote() {
    local b64
    b64=$(printf 'set -euo pipefail\n%s' "$(cat)" | base64 | tr -d '\n')
    ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
        "bash -c \"\$(printf '%s' '$b64' | base64 -d)\"" </dev/null
}

# ─── nginx перед админкой ────────────────────────────────────────────────────
# Отдельной функцией, потому что вызывается из двух мест: из полного прогона
# (после того, как админка поднялась) и из --nginx-only, когда всё остальное
# уже развёрнуто и трогать его не надо.
#
# Порядок внутри жёсткий и не переставляется:
#   nginx → проверка, что прокси реально отвечает → сертификат →
#   и только потом админка убирается на 127.0.0.1 и 8100 закрывается в ufw.
# Наоборот было бы «сначала отрезать старый вход, потом выяснить, что новый
# не работает»: доступ к админке остался бы только через ssh-туннель.
setup_nginx() {
    if [[ -z "$ADMIN_DOMAIN" ]]; then
        warn "ADMIN_DOMAIN пуст — nginx не ставлю, админка смотрит наружу сама"
        return 0
    fi

    step "Ставлю nginx перед админкой ($ADMIN_DOMAIN)"

    # DNS проверяется с сервера, а не локально: у Let's Encrypt тот же вопрос —
    # «куда ведёт имя», и расхождение локального резолвера с публичным здесь
    # ничего не значит. Ошибка не фатальна: без TLS прокси работает и так,
    # падает только выпуск сертификата.
    ADMIN_DOMAIN="$ADMIN_DOMAIN" remote <<EOF
resolved=\$(getent ahostsv4 "$ADMIN_DOMAIN" | awk '{print \$1; exit}')
myip=\$(curl -fsS --max-time 10 https://api.ipify.org 2>/dev/null || echo "")
echo "  $ADMIN_DOMAIN → \${resolved:-не резолвится}, внешний адрес сервера: \${myip:-неизвестен}"
if [[ -n "\$resolved" && -n "\$myip" && "\$resolved" != "\$myip" ]]; then
    echo "  ! домен ведёт не на этот сервер — сертификат не выпустится" >&2
fi
EOF

    ADMIN_DOMAIN="$ADMIN_DOMAIN" remote <<EOF
export DEBIAN_FRONTEND=noninteractive
if ! command -v nginx >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq nginx >/dev/null
    echo "  nginx установлен"
else
    echo "  nginx уже стоит"
fi

# Дефолтный сайт слушает :80 как default_server и перехватывал бы всё, что
# приходит не по нашему имени; свой catch-all ниже занимает то же место.
sudo rm -f /etc/nginx/sites-enabled/default

sudo tee /etc/nginx/sites-available/crypto-graph >/dev/null <<'NGINX'
# Админка btcproc за обратным прокси. Единственный вход снаружи.
#
# X-Forwarded-For здесь ПЕРЕЗАПИСЫВАЕТСЯ (\$remote_addr), а не дополняется
# (\$proxy_add_x_forwarded_for). Разница принципиальная: админка при
# ADMIN_TRUST_PROXY=true берёт из заголовка ПЕРВЫЙ адрес, и при склейке им
# оказался бы заголовок, присланный самим клиентом — то есть любой адрес из
# ADMIN_IP_ALLOWLIST открывал бы дверь, а новый фейковый адрес на каждую
# попытку обнулял бы счётчик неудачных входов. Цепочка прокси здесь одна,
# терять в ней нечего.
server {
    listen 80;
    listen [::]:80;
    server_name ADMIN_DOMAIN_PLACEHOLDER;

    access_log /var/log/nginx/crypto-graph.access.log;
    error_log  /var/log/nginx/crypto-graph.error.log;

    # Страницы админки ходят в PostgreSQL и Neo4j за агрегатами; минута с
    # запасом, чтобы 504 не прилетал раньше, чем ответит база.
    proxy_read_timeout 120s;
    client_max_body_size 4m;

    location / {
        proxy_pass http://127.0.0.1:8100;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$remote_addr;
        # По нему uvicorn ставит request.url.scheme=https, а админка — флаг
        # secure на сессионной cookie. Без него cookie уходит без secure.
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Connection "";
    }
}

# Всё, что пришло не по имени (сканеры по IP), обрывается без ответа —
# админка не должна отвечать на переборы адресного пространства.
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;
    return 444;
}
NGINX
sudo sed -i "s|ADMIN_DOMAIN_PLACEHOLDER|$ADMIN_DOMAIN|" /etc/nginx/sites-available/crypto-graph
sudo ln -sf /etc/nginx/sites-available/crypto-graph /etc/nginx/sites-enabled/crypto-graph
sudo nginx -t 2>&1 | sed 's/^/  /'
sudo systemctl enable --now nginx >/dev/null
sudo systemctl reload nginx

sudo ufw allow 80/tcp  >/dev/null
sudo ufw allow 443/tcp >/dev/null

# Проверка до сертификата: если прокси не работает, дальше идти незачем.
if curl -fsS -o /dev/null --max-time 10 -H "Host: $ADMIN_DOMAIN" http://127.0.0.1/health; then
    echo "  прокси отвечает: http://$ADMIN_DOMAIN/health"
else
    echo "nginx не проксирует на админку — смотри /var/log/nginx/crypto-graph.error.log" >&2
    exit 1
fi
EOF

    if [[ "$ENABLE_TLS" == "1" ]]; then
        ADMIN_DOMAIN="$ADMIN_DOMAIN" TLS_EMAIL="$TLS_EMAIL" remote <<EOF
export DEBIAN_FRONTEND=noninteractive
command -v certbot >/dev/null 2>&1 || sudo apt-get install -y -qq certbot python3-certbot-nginx >/dev/null
# Уже выпущенный сертификат не перевыпускаем: у Let's Encrypt лимит
# 5 одинаковых наборов имён в неделю, а скрипт задуман идемпотентным.
# Продлением занимается таймер certbot.timer из пакета.
if sudo test -d "/etc/letsencrypt/live/$ADMIN_DOMAIN"; then
    echo "  сертификат уже есть, перевыпуск не нужен"
    sudo certbot --nginx -d "$ADMIN_DOMAIN" --redirect --keep-until-expiring \
        --non-interactive --agree-tos -m "$TLS_EMAIL" 2>&1 | tail -3 | sed 's/^/  /'
else
    sudo certbot --nginx -d "$ADMIN_DOMAIN" --redirect \
        --non-interactive --agree-tos -m "$TLS_EMAIL" 2>&1 | tail -8 | sed 's/^/  /'
fi
systemctl is-enabled certbot.timer >/dev/null 2>&1 \
    && echo "  автопродление: certbot.timer активен" \
    || echo "  ! certbot.timer выключен — сертификат не продлится сам" >&2
if curl -fsS -o /dev/null --max-time 15 "https://$ADMIN_DOMAIN/health"; then
    echo "  https://$ADMIN_DOMAIN/health отвечает"
else
    echo "  ! https не ответил — проверь вывод certbot выше" >&2
fi
EOF
    else
        info "TLS выключен (ENABLE_TLS=0) — пароль админки ходит открытым текстом"
    fi

    # Только теперь отрезаем прямой вход. Рестарт админки убивает прогоны,
    # запущенные ИЗ НЕЁ (они идут BackgroundTasks в её процессе), поэтому при
    # идущем прогоне шаг откладывается: nginx уже работает, а 8100 просто
    # остаётся открытым до следующего запуска.
    step "Убираю админку за прокси (127.0.0.1) и закрываю 8100"
    REMOTE_DIR="$REMOTE_DIR" remote <<EOF
cd "$REMOTE_DIR/btc-graph"
running=\$(docker compose exec -T postgres psql -U btc_user -d btc_graph -tAc \
    "select count(*) from processing.runs where status='running'" 2>/dev/null \
    | tr -d '[:space:]' || echo 0)
if [[ "\${running:-0}" != "0" ]]; then
    echo "  ! идёт прогон (\$running) — админку не перезапускаю, 8100 оставляю открытым" >&2
    echo "  ! повтори ./02_configure.sh --nginx-only после его завершения" >&2
    exit 0
fi

env_file="$REMOTE_DIR/btc-graph-processing/.env"
# sed по существующей строке, а не перезапись .env целиком: .env содержит
# сгенерированный пароль админки, и трогать его этим шагом нельзя.
if grep -q '^ADMIN_HOST=' "\$env_file"; then
    sed -i 's|^ADMIN_HOST=.*|ADMIN_HOST=127.0.0.1|' "\$env_file"
else
    echo 'ADMIN_HOST=127.0.0.1' >> "\$env_file"
fi
if grep -q '^ADMIN_TRUST_PROXY=' "\$env_file"; then
    sed -i 's|^ADMIN_TRUST_PROXY=.*|ADMIN_TRUST_PROXY=true|' "\$env_file"
else
    echo 'ADMIN_TRUST_PROXY=true' >> "\$env_file"
fi

sudo systemctl restart btcproc-admin
sleep 5
if ! systemctl is-active --quiet btcproc-admin; then
    echo "админка не поднялась после смены ADMIN_HOST:" >&2
    sudo journalctl -u btcproc-admin -n 25 --no-pager >&2
    exit 1
fi
curl -fsS -o /dev/null --max-time 10 http://127.0.0.1:8100/health \
    || { echo "админка не отвечает на 127.0.0.1:8100" >&2; exit 1; }

sudo ufw delete allow 8100/tcp >/dev/null 2>&1 || true
sudo ufw status | sed 's/^/  /'
echo "  админка слушает только 127.0.0.1, снаружи — только через nginx"
EOF
}

# ─── Монитор ресурсов: настройки и сервис ────────────────────────────────────
# Отдельными функциями, потому что их вызывает и обычный прогон, и
# --hostmon-only на уже развёрнутой машине.
hostmon_env() {
    step "Досыпаю настройки монитора в .env"
    REMOTE_DIR="$REMOTE_DIR" remote <<EOF
cd "$REMOTE_DIR"
# Существующий .env не перезаписываем (там пароли и правки оператора), поэтому
# настройки добавляются по одной. Проверка по имени переменной, а не по секции:
# так следующая новая настройка не потребует ничего трогать.
add_env() {
    local name="\$1" value="\$2"
    grep -qE "^\${name}=" btc-graph-processing/.env || {
        printf '%s=%s\n' "\$name" "\$value" >> btc-graph-processing/.env
        echo "  .env: добавлен \$name"
    }
}
add_env HOSTMON_DB "$REMOTE_DIR/logs/hostmon.sqlite"
add_env HOSTMON_INTERVAL_SECONDS 60
add_env HOSTMON_KEEP_DAYS 30
add_env HOSTMON_MOUNTS /
add_env HOSTMON_DOCKER true
add_env HOSTMON_ALERTS_ENABLED true
add_env TELEGRAM_BOT_TOKEN ""
add_env TELEGRAM_CHAT_ID ""
add_env HOSTMON_ALERT_COOLDOWN_MINUTES 5
add_env HOSTMON_ALERT_HYSTERESIS 5
add_env HOSTMON_ALERT_SUSTAIN_TICKS 5
add_env HOSTMON_ALERT_DISK_PCT 90
add_env HOSTMON_ALERT_DISK_CRITICAL_PCT 96
add_env HOSTMON_ALERT_MEM_PCT 90
add_env HOSTMON_ALERT_SWAP_PCT 60
add_env HOSTMON_ALERT_CPU_PCT 90
add_env HOSTMON_ALERT_LOAD_PER_CORE 2
echo "  настройки монитора на месте"
EOF
}

hostmon_service() {
    step "Ставлю монитор ресурсов как systemd-сервис"
    # Отдельный сервис, а не крон и не поток админки. Крон дёргал бы импорт
    # btcproc (pandas, sklearn — секунды и сотня мегабайт) шестьдесят раз в час;
    # поток админки встаёт в одну очередь с прогонами и пропускал бы такты ровно
    # под нагрузкой, то есть когда монитор и нужен. На хосте, а не в docker:
    # внутри контейнера psutil видит контейнер, а следить надо за машиной.
    REMOTE_DIR="$REMOTE_DIR" VPS_USER="$VPS_USER" remote <<EOF
sudo tee /etc/systemd/system/btcproc-hostmon.service >/dev/null <<UNIT
[Unit]
Description=btcproc host monitor (crypto-graph)
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
User=$VPS_USER
Group=$VPS_USER
WorkingDirectory=$REMOTE_DIR/btc-graph-processing
ExecStart=$REMOTE_DIR/venv/bin/python -m btcproc.cli hostmon
Restart=always
RestartSec=15
# Монитор обязан быть дешевле того, что мерит, и не имеет права участвовать в
# том самом OOM, о котором предупреждает: лимит делает это гарантией, а не
# надеждой.
MemoryMax=192M
Nice=5
StandardOutput=append:$REMOTE_DIR/logs/hostmon.log
StandardError=append:$REMOTE_DIR/logs/hostmon.log

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now btcproc-hostmon >/dev/null
sudo systemctl restart btcproc-hostmon
sleep 4
if systemctl is-active --quiet btcproc-hostmon; then
    echo "  монитор запущен, замеры в $REMOTE_DIR/logs/hostmon.sqlite"
else
    # Не роняем настройку контура из-за монитора: без него работает всё
    # остальное, просто страница «Сервер» останется пустой.
    echo "  монитор не стартовал — смотри logs/hostmon.log" >&2
    sudo journalctl -u btcproc-hostmon -n 20 --no-pager >&2
fi
if grep -qE '^TELEGRAM_BOT_TOKEN=.+' "$REMOTE_DIR/btc-graph-processing/.env"; then
    echo "  канал уведомлений настроен (проверка: bin/btcproc hostmon --test-telegram)"
else
    echo "  ВНИМАНИЕ: TELEGRAM_BOT_TOKEN пуст — о заполненном диске никто не узнает"
fi
EOF
}

# Только монитор: настройки в .env плюс сервис. Всё остальное на уже
# развёрнутой машине не трогается — ни стек, ни миграции, ни расписание.
if [[ $HOSTMON_ONLY -eq 1 ]]; then
    hostmon_env
    hostmon_service
    printf '\n\033[1;32m✓ монитор ресурсов настроен\033[0m\n'
    echo "  Страница: админка → «Сервер»"
    echo "  Логи:     $REMOTE_DIR/logs/hostmon.log"
    exit 0
fi

if [[ $NGINX_ONLY -eq 1 ]]; then
    setup_nginx
    printf '\n\033[1;32m✓ nginx настроен\033[0m\n'
    if [[ "$ENABLE_TLS" == "1" ]]; then
        echo "  Админка: https://$ADMIN_DOMAIN/login"
    else
        echo "  Админка: http://$ADMIN_DOMAIN/login"
    fi
    exit 0
fi

step "Проверяю, что 01_deploy.sh уже отработал"
REMOTE_DIR="$REMOTE_DIR" remote <<EOF || die "Сначала прогони ./01_deploy.sh"
[[ -x "$REMOTE_DIR/venv/bin/python" ]] || { echo "нет venv" >&2; exit 1; }
[[ -d "$REMOTE_DIR/btc-graph/src" ]] || { echo "нет кода btc-graph" >&2; exit 1; }
[[ -d "$REMOTE_DIR/btc-graph-processing/btcproc" ]] || { echo "нет кода btcproc" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "docker недоступен от имени $VPS_USER" >&2; exit 1; }
echo "  всё на месте"
EOF

# ─── 1. .env ─────────────────────────────────────────────────────────────────
step "Создаю .env"
REMOTE_DIR="$REMOTE_DIR" FORCE_ENV="$FORCE_ENV" TRAIN_SYMBOL="$TRAIN_SYMBOL" remote <<EOF
cd "$REMOTE_DIR"

# ── btc-graph ──
# DATABASE_URL/REDIS_URL/NEO4J_URI здесь указывают на localhost, но контейнеры
# их не увидят: в docker-compose.yml те же переменные заданы через
# environment:, а он перекрывает env_file. Значения нужны для запусков с хоста.
if [[ -f btc-graph/.env && "$FORCE_ENV" != "1" ]]; then
    echo "  btc-graph/.env уже есть — не трогаю"
else
    cat > btc-graph/.env <<'ENV'
# Anthropic. LLM только формулирует объяснения, все числа считает
# детерминированный скорер, поэтому с заглушкой система работает —
# кандидаты просто приходят без текстового разбора.
ANTHROPIC_API_KEY=sk-ant-ЗАПОЛНИ

DATABASE_URL=postgresql://btc_user:btc_pass@localhost:5432/btc_graph
REDIS_URL=redis://localhost:6379/0

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=btc_neo4j_pass

PROFILES_DIR=config/symbols

# API поднят без аутентификации и закрыт на 127.0.0.1. Ручка перечитывания
# профилей меняет скоринг на лету — держим выключенной.
ENABLE_CONFIG_RELOAD=false
ENV
    echo "  btc-graph/.env создан (ключ Anthropic — заглушка)"
fi

# ── btc-graph-processing ──
if [[ -f btc-graph-processing/.env && "$FORCE_ENV" != "1" ]]; then
    echo "  btc-graph-processing/.env уже есть — не трогаю"
    grep -E '^ADMIN_(USER|PASSWORD)=' btc-graph-processing/.env > "$REMOTE_DIR/logs/.admin_creds"
else
    ADMIN_PASS="\$(openssl rand -base64 24 | tr -dc 'A-Za-z0-9' | head -c 24)"
    ADMIN_KEY="\$(openssl rand -hex 32)"
    cat > btc-graph-processing/.env <<ENV
# ─── Хранилище (общий стек с btc-graph, порты закрыты на 127.0.0.1) ─────────
DATABASE_URL=postgresql://btc_user:btc_pass@localhost:5432/btc_graph
REDIS_URL=redis://localhost:6379/1
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=btc_neo4j_pass
PG_SCHEMA=processing

# ─── Связь с btc-graph ──────────────────────────────────────────────────────
# direct: пакет btc-graph импортируется как библиотека из того же venv,
# что и btcproc. Путь абсолютный и обязан указывать на каталог с src/,
# иначе режим падает с «не похож на репозиторий btc-graph».
SINK_MODE=direct
BTC_GRAPH_PATH=$REMOTE_DIR/btc-graph
BTC_GRAPH_URL=http://localhost:8000
BTC_GRAPH_REDIS_URL=redis://localhost:6379/0
SINK_USE_LLM=false
SINK_BATCH_SIZE=200
# ПУСТО — и это не забывчивость. Пустое значение означает «порог берёт
# btc-graph из профиля КАЖДОЙ монеты» (config/symbols/*.yaml), то есть
# работает калибровка. Любое число здесь перекрывает профили одной линейкой
# на весь батч; поставленный сюда 0.0 отключает фильтр целиком, и в базу
# польётся WEAK (audit 2026-08-09, B1).
SINK_MIN_QUALITY=

# ─── Данные ─────────────────────────────────────────────────────────────────
SYMBOL=$TRAIN_SYMBOL
BASE_TIMEFRAME=15m
CONTEXT_TIMEFRAMES=1h,4h,1d
HISTORY_START=2017-08-01
HORIZON=24h
DATA_DIR=./data
# Сервер в США, а api.binance.com отвечает оттуда 451 Unavailable For Legal
# Reasons. Месячные дампы качаются с data.binance.vision и от этого не
# страдают, но свежий хвост баров добирается через REST — то есть без замены
# адреса live не работал бы вовсе, а train падал бы на последнем месяце.
# data-api.binance.vision — тот же публичный /api/v3/klines, без ключей.
BINANCE_REST_URL=$BINANCE_REST_URL

# ─── Источники сигнала ──────────────────────────────────────────────────────
# У каждого источника ДВА флага, и это не дублирование. Контекстные атомы
# бесплатны и обратимы: в event_block_id не входят, переобучения не требуют,
# live со старой моделью просто пишет их в bar_events.context_atoms. Признаки
# дороги: метка набора меняется, нужен полный train, group_id
# перенумеровываются, накопленный граф Neo4j сносится.
#
# Все ТРИ источника прошли механизм docs/extending_features.md целиком, и
# признаки не заведены ни у одного — по замерам, а не по осторожности:
# SMC (журнал 22.5) и Fear & Greed (34) не предсказывают ни направление, ни
# размах; у деривативов (36) гейт по размаху прошла одна величина из шести
# (oi_chg_1h), но не прошла гейт по градации. Атомы включены не потому, что
# работают, а потому что размечают историю даром — «зона ликвидности»,
# «крайний страх», «набор лонгов» на узле графа и в инспекторе бара.
SMC_ENABLED=true
SMC_FEATURES_ENABLED=false
# FGI требует наполненной external_daily: без неё атомы молча всегда False.
# Наполняет крон fetch-external (FETCH_CRON), включённый ниже вместе с флагом.
FGI_ENABLED=true
FGI_FEATURES_ENABLED=false
# Деривативные метрики Binance USD-M — третий источник (docs/tz_deriv_ingest_14-08-26.md).
# Требует наполненной deriv_metrics — наполняет крон ingest-metrics (DERIV_CRON)
# ниже. Порядок выкатки как у FGI: сначала код, потом наполнение таблицы, и
# только потом флаг — иначе пустая таблица даёт сплошной False молча
# (урок 34.10). Именно в этом порядке источник и введён 2026-08-15: архив
# залит по всем четырём монетам (208 тыс. строк по BTC с 2020-09, 99.7%
# полных баров), после чего поднят флаг. На чистой машине тот же порядок
# держит шаг «Наполняю таблицы внешних источников» ниже — он идёт ДО первого
# train намеренно.
DERIV_ENABLED=true
DERIV_FEATURES_ENABLED=false

# ─── Размах: целевые величины и квантильный регрессор ───────────────────────
# Разметка исходов с 2026-08-19 считает не только направление и путь, но и
# размах (range_pct / rv_fwd / range_ratio). Два дополнительных горизонта
# кладутся на склад под замеры — цена в один проход скользящих окон за train
# и около миллиона строк на горизонт по шести монетам. В live цены нет: он
# исходы не сохраняет.
OUTCOME_EXTRA_HORIZONS=4h,12h
# Квантильный регрессор размаха (журнал 48) — единственная величина проекта,
# прошедшая критерий на отложенной части. Учится в train на ПРИЗНАКАХ (не на
# разметке графа: она этот сигнал теряет), там же проходит гейт из трёх
# условий и попадает в кандидата описательными полями. Порядок ввода тот же,
# что у FGI и деривативов: сначала код, потом флаг — здесь он тем более
# уместен, потому что до первого train с флагом модели просто нет, и
# кандидаты идут без размаха, ничего не ломая.
RANGE_FORECAST_ENABLED=true
RANGE_FORECAST_NORM=atr14
RANGE_FORECAST_SEED=42
RANGE_FORECAST_TRAIN_FRAC=0.7
RANGE_FORECAST_GATE_BOOT=500

# ─── Пороги кластеризации ───────────────────────────────────────────────────
STATES_MIN_GROUP_SHARE=0.0025
STATES_MIN_GROUP_SIZE=300

# ─── Порог дробления состояний (gap statistic) ──────────────────────────────
# Референсов несколько, а порог выражен в их собственных сигмах. Абсолютный
# порог (STATES_SPLIT_GAIN, до 2026-08-11) зависел от размерности: константа,
# откалиброванная на 32 признаках, в 44 означала другую строгость — и граф
# обрушивали двенадцать новых признаков ЛЮБОЙ природы, включая чистый шум.
# 2.0 — результат калибровки по устойчивости числа состояний между окнами
# разной длины (scripts/calibrate_split_gain.py), а не подгонка под прежнее
# число состояний.
STATES_SPLIT_REFERENCE_DRAWS=10
STATES_SPLIT_GAIN_SIGMA=2.0

# ─── Админка (единственный сервис, доступный снаружи — пароль сгенерирован) ──
ADMIN_USER=admin
ADMIN_PASSWORD=\$ADMIN_PASS
ADMIN_SECRET_KEY=\$ADMIN_KEY
# Прогоны идут BackgroundTasks в процессе админки: каждый занимает ядро
# на кластеризации и заметный кусок памяти. На 8 ГБ двойка — потолок.
ADMIN_MAX_CONCURRENT_RUNS=2
ADMIN_SESSION_TTL=43200
ADMIN_MAX_LOGIN_ATTEMPTS=5
ADMIN_LOCKOUT_SECONDS=900
ADMIN_IP_ALLOWLIST=
# X-Forwarded-For можно читать ТОЛЬКО когда до админки нельзя достучаться в
# обход nginx: иначе заголовок ставит сам клиент, и подделанный адрес обходит
# ADMIN_IP_ALLOWLIST и обнуляет счётчик неудачных входов на каждой попытке
# (audit 2026-08-09, B6). Поэтому две настройки ниже связаны жёстко и
# переключаются вместе с ADMIN_DOMAIN — врозь их менять нельзя.
ADMIN_TRUST_PROXY=$ADMIN_TRUST_PROXY_VALUE
ADMIN_HOST=$ADMIN_HOST_VALUE
ADMIN_PORT=8100

# ─── Прогоны ────────────────────────────────────────────────────────────────
# Через сколько молчания прогон в статусе running считается мёртвым. Здесь
# это не теория: расчёт признаков без swap ловит OOM killer, и убитый прогон
# навсегда оставался бы running — крон с --skip-if-busy молча пропускал бы
# каждый следующий live, а админка отдавала бы 409. Значение должно быть
# заметно больше самой долгой стадии train.
RUN_STALE_AFTER_MINUTES=120

# ─── Монитор ресурсов хоста (страница «Сервер» в админке) ───────────────────
# Замеры пишет сервис btcproc-hostmon в SQLite. Путь — вне каталогов
# подпроектов: 01_deploy.sh гонит их rsync --delete, и файл внутри проекта
# сносило бы при каждой выкатке кода.
HOSTMON_DB=$REMOTE_DIR/logs/hostmon.sqlite
HOSTMON_INTERVAL_SECONDS=60
HOSTMON_KEEP_DAYS=30
HOSTMON_MOUNTS=/
HOSTMON_DOCKER=true

# Уведомления о нагрузке в Telegram. Токен и чат заполняются руками — их
# скрипту взять негде; до тех пор алерты считаются, но не отправляются, и
# страница «Сервер» об этом прямо говорит. Проверка: btcproc hostmon --test-telegram
HOSTMON_ALERTS_ENABLED=true
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
HOSTMON_ALERT_COOLDOWN_MINUTES=5
HOSTMON_ALERT_HYSTERESIS=5
HOSTMON_ALERT_SUSTAIN_TICKS=5
HOSTMON_ALERT_DISK_PCT=90
HOSTMON_ALERT_DISK_CRITICAL_PCT=96
HOSTMON_ALERT_MEM_PCT=90
# Swap на этой машине — ранний признак близкого OOM, а не резерв памяти:
# расчёт признаков без него уже ловил killer. Отсюда порог ниже остальных.
HOSTMON_ALERT_SWAP_PCT=60
HOSTMON_ALERT_CPU_PCT=90
HOSTMON_ALERT_LOAD_PER_CORE=2
ENV
    chmod 600 btc-graph-processing/.env
    printf 'ADMIN_USER=admin\nADMIN_PASSWORD=%s\n' "\$ADMIN_PASS" > "$REMOTE_DIR/logs/.admin_creds"
    chmod 600 "$REMOTE_DIR/logs/.admin_creds"
    echo "  btc-graph-processing/.env создан, пароль админки сгенерирован"
fi

EOF

# Настройки монитора ресурсов досыпаются отдельно от шага 1: существующий .env
# скрипт не перезаписывает, поэтому на уже развёрнутой машине новые переменные
# иначе не появились бы вовсе.
hostmon_env

# ─── 2. Обёртка для запусков ─────────────────────────────────────────────────
# Один вход для cron, systemd и рук. Важен cd: btcproc читает .env через
# load_dotenv() без явного пути, то есть относительно текущего каталога.
step "Ставлю обёртку $REMOTE_DIR/bin/btcproc"
REMOTE_DIR="$REMOTE_DIR" remote <<EOF
mkdir -p "$REMOTE_DIR/bin"
cat > "$REMOTE_DIR/bin/btcproc" <<'SH'
#!/usr/bin/env bash
# Единая точка входа в btcproc: правильный интерпретатор и правильный cwd.
set -euo pipefail
cd "REMOTE_DIR_PLACEHOLDER/btc-graph-processing"
exec "REMOTE_DIR_PLACEHOLDER/venv/bin/python" -m btcproc.cli "\$@"
SH
sed -i "s|REMOTE_DIR_PLACEHOLDER|$REMOTE_DIR|g" "$REMOTE_DIR/bin/btcproc"
chmod +x "$REMOTE_DIR/bin/btcproc"

cat > "$REMOTE_DIR/bin/first_train.sh" <<'SH'
#!/usr/bin/env bash
# Первый прогон. Код возврата кладётся в файл: transient-юнит systemd после
# завершения исчезает вместе со своим статусом, а знать результат надо.
set -uo pipefail
SYMBOL="\${1:-BTCUSDT}"
rm -f REMOTE_DIR_PLACEHOLDER/logs/train.exit
REMOTE_DIR_PLACEHOLDER/bin/btcproc train --symbol "\$SYMBOL"
echo \$? > REMOTE_DIR_PLACEHOLDER/logs/train.exit
SH
sed -i "s|REMOTE_DIR_PLACEHOLDER|$REMOTE_DIR|g" "$REMOTE_DIR/bin/first_train.sh"
chmod +x "$REMOTE_DIR/bin/first_train.sh"
echo "  готово"
EOF

# ─── 3. Стек ─────────────────────────────────────────────────────────────────
# Отдельный флаг, а не автоматика: сносить тома — необратимо. Нужен, если
# миграция оставила схему в промежуточном состоянии (alembic_version отстал
# от фактического DDL) — тогда повторный upgrade падает на дубликатах, и
# чистая база быстрее любого ручного разбора. На накопленных данных так делать
# нельзя: уйдёт и граф Neo4j, и вся история кандидатов.
if [[ $RESET_DB -eq 1 ]]; then
    step "Сношу тома PostgreSQL, Redis и Neo4j (--reset-db)"
    REMOTE_DIR="$REMOTE_DIR" remote <<EOF
cd "$REMOTE_DIR/btc-graph"
docker compose down -v --remove-orphans 2>&1 | tail -3
EOF
    warn "данные стека удалены — база будет создана заново"
fi

step "Поднимаю docker-стек btc-graph"
REMOTE_DIR="$REMOTE_DIR" remote <<EOF
cd "$REMOTE_DIR/btc-graph"
docker compose up -d

# Ждём именно healthcheck, а не «контейнер запущен»: миграции пойдут сразу
# следом, и alembic на непрогретом postgres падает с невнятной ошибкой связи.
wait_healthy() {
    local name="\$1" limit="\$2" st=none
    for _ in \$(seq 1 "\$limit"); do
        st=\$(docker inspect --format '{{.State.Health.Status}}' "\$name" 2>/dev/null || echo none)
        if [[ "\$st" == healthy ]]; then
            echo "  \$name здоров"
            return 0
        fi
        sleep 5
    done
    echo "\$name не поднялся (статус: \$st)" >&2
    docker compose logs --tail 40 "\${name#btc_}" >&2 || true
    return 1
}
wait_healthy btc_postgres 60
wait_healthy btc_redis 60
wait_healthy btc_neo4j 90
docker compose ps --format '{{.Name}} — {{.Status}}' | sed 's/^/  /'
EOF

# ─── 4. Миграции ─────────────────────────────────────────────────────────────
step "Миграции PostgreSQL"
# Почему вокруг миграции глушатся фоновые воркеры TimescaleDB.
#
# Миграция 004 пересоздаёт оба continuous aggregate, ставит им политики
# обновления и следом наполняет их: CALL refresh_continuous_aggregate внутри
# транзакции запрещён, поэтому перед вызовом в миграции стоит явный COMMIT.
# Ровно в этот момент только что созданные политики становятся видны фоновому
# воркеру TimescaleDB, он запускает свой первый refresh — и они с миграцией
# дерутся за одно окно:
#
#     LockNotAvailable: could not refresh continuous aggregate
#     "daily_group_stats" due to a concurrent refresh
#
# Хуже, что после COMMIT откатывается не всё: alembic_version остаётся 003,
# а часть DDL уже закоммичена, и повторный upgrade падает на дубликатах.
# На быстрой машине гонка воспроизводится стабильно, поэтому воркеры
# выключаются на время миграции. Это дефект самой миграции — здесь он
# обходится, а не чинится: править чужую миграцию задним числом опаснее,
# чем не давать ей гоняться с собой.
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

# Если применять нечего, не трогаем базу вовсе: обход стоит двух рестартов
# postgres, и платить их на каждом повторном запуске скрипта незачем.
current=\$(docker compose exec -T postgres psql -U btc_user -d btc_graph -tAc \
    "SELECT version_num FROM alembic_version" 2>/dev/null | tr -d '[:space:]' || true)
head_rev=\$(docker compose exec -T api alembic heads 2>/dev/null | awk '{print \$1; exit}' | tr -d '[:space:]' || true)
if [[ -n "\$current" && -n "\$head_rev" && "\$current" == "\$head_rev" ]]; then
    echo "  схема уже на \$current — миграций нет"
    exit 0
fi

# max_background_workers читается только при старте — отсюда рестарт, а не reload.
psql_c "ALTER SYSTEM SET timescaledb.max_background_workers = 0;"
docker compose restart postgres >/dev/null
wait_pg
echo "  фоновые воркеры TimescaleDB выключены на время миграции"

set +e
docker compose exec -T api alembic upgrade head 2>&1 | tail -15
rc=\${PIPESTATUS[0]}
set -e

# Возвращаем воркеры в любом случае: оставить базу без них хуже, чем упавшая
# миграция — политики обновления агрегатов молча перестали бы работать.
psql_c "ALTER SYSTEM RESET timescaledb.max_background_workers;"
docker compose restart postgres >/dev/null
wait_pg
echo "  фоновые воркеры TimescaleDB возвращены"

# Сервисы держат пул соединений к пересозданному postgres — пусть переподключатся.
docker compose restart api celery-worker celery-beat >/dev/null
[[ \$rc -eq 0 ]] || { echo "alembic upgrade head упал (код \$rc)" >&2; exit \$rc; }
echo "  версия схемы: \$(docker compose exec -T postgres psql -U btc_user -d btc_graph -tAc 'SELECT version_num FROM alembic_version')"
EOF

# 004 пересоздаёт continuous aggregates, а migrate-graph проставляет symbol
# узлам Neo4j. На чистой базе обе операции ничего не теряют — важно прогнать
# их ДО первого кандидата не по BTCUSDT, иначе узлы монет схлопнутся молча.
step "Миграция графа Neo4j (ключ symbol+group_id)"
REMOTE_DIR="$REMOTE_DIR" remote <<EOF
cd "$REMOTE_DIR/btc-graph"
docker compose exec -T neo4j cypher-shell -u neo4j -p btc_neo4j_pass \
    < scripts/migrate_graph_symbol.cypher 2>&1 | tail -8
EOF

# Свойство ребра называлось avg_horizon_return, а хранило win rate — имя
# обещало одно, лежало другое, и Cypher-запрос «средний return перехода»
# молча получал не ту величину (audit 2026-08-09, M1). На чистой базе
# миграция ничего не находит и отрабатывает вхолостую; на существующей —
# переносит значение и убирает старое свойство. Идемпотентна.
step "Миграция графа Neo4j (avg_horizon_return → avg_win_rate)"
REMOTE_DIR="$REMOTE_DIR" remote <<EOF
cd "$REMOTE_DIR/btc-graph"
docker compose exec -T neo4j cypher-shell -u neo4j -p btc_neo4j_pass \
    < scripts/migrate_graph_avg_win_rate.cypher 2>&1 | tail -5
EOF

step "Схема processing для btcproc"
REMOTE_DIR="$REMOTE_DIR" remote <<EOF
"$REMOTE_DIR/bin/btcproc" init-db 2>&1 | tail -10
EOF

# ─── 4а. Полнота TRAIN_SYMBOLS ───────────────────────────────────────────────
# Шаг 9 переписывает crontab ЦЕЛИКОМ из TRAIN_SYMBOLS: прежние строки
# вычищаются фильтрами, новые собираются заново из переменной. Поэтому монета,
# обученная раньше и выпавшая из списка (типичный случай — «добавляю SUIUSDT»,
# TRAIN_SYMBOLS="SUIUSDT"), молча пропадает из live и train. Ошибки нет:
# кандидаты просто перестают появляться, а заметно это станет через дни.
#
# Сверяем переменную с тем, что реально обучено в базе, и требуем решения.
# Проверка стоит здесь, а не перед шагом 9: падать надо ДО часового обучения,
# а не после него. На чистой базе обученного нет и шаг молчит.
step "Сверяю TRAIN_SYMBOLS с обученными монетами"
# Неудачу запроса от «обученного нет» отличаем явно: молча пропущенная
# сверка вернула бы ровно ту тихую поломку, ради которой шаг и написан.
if ! trained=$(ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
    "docker exec btc_postgres psql -U btc_user -d btc_graph -tAc \"SELECT DISTINCT symbol FROM processing.runs WHERE kind='train' AND status='done' AND symbol IS NOT NULL ORDER BY 1\"" \
    </dev/null 2>/dev/null); then
    warn "не смог прочитать processing.runs — сверка пропущена, проверь crontab после запуска"
    trained=""
fi

dropped=""
for symbol in $trained; do
    case " $TRAIN_SYMBOLS " in
        *" $symbol "*) ;;
        *) dropped="$dropped $symbol" ;;
    esac
done

if [[ -n "${dropped// /}" ]]; then
    if [[ $DROP_SYMBOLS -eq 1 ]]; then
        warn "Обученные монеты вне TRAIN_SYMBOLS:${dropped} — убираю из расписания (--drop-symbols)"
    else
        die "$(cat <<MSG
Обученные монеты вне TRAIN_SYMBOLS:${dropped}

TRAIN_SYMBOLS — это ПОЛНЫЙ список монет контура, а не «что доучить»:
уже обученные шаг 8 пропустит сам, а шаг 9 соберёт из переменной всё
расписание. Оставь как есть — и эти монеты тихо выпадут из live и train.

Добавляешь монету (BTCUSDT держим первым: первая из списка становится
монетой по умолчанию, SYMBOL в .env):
  TRAIN_SYMBOLS="${dropped# } $TRAIN_SYMBOLS" ./02_configure.sh

Выводишь монету из расписания осознанно:
  TRAIN_SYMBOLS="$TRAIN_SYMBOLS" ./02_configure.sh --drop-symbols
MSG
)"
    fi
else
    info "расписание получат: $TRAIN_SYMBOLS"
fi

# ─── 5. Firewall ─────────────────────────────────────────────────────────────
# 8000 (API без аутентификации) и 5555 (flower) закрыты на уровне compose —
# ufw до опубликованных docker-портов всё равно не дотягивается. Здесь
# закрывается всё остальное: наружу остаются ssh и вход в админку.
#
# 8100 открывается только когда прокси нет. С nginx правило снимает сам шаг
# nginx, а не этот: снимать его здесь означало бы отрезать единственный
# рабочий вход до того, как заработает новый.
step "Настраиваю firewall"
ADMIN_DOMAIN="$ADMIN_DOMAIN" remote <<EOF
sudo ufw allow 22/tcp >/dev/null
if [[ -n "$ADMIN_DOMAIN" ]]; then
    sudo ufw allow 80/tcp  >/dev/null
    sudo ufw allow 443/tcp >/dev/null
else
    sudo ufw allow 8100/tcp >/dev/null
fi
sudo ufw default deny incoming  >/dev/null
sudo ufw default allow outgoing >/dev/null
sudo ufw --force enable >/dev/null
sudo ufw status numbered | sed 's/^/  /'
EOF

# ─── 6. Админка ──────────────────────────────────────────────────────────────
step "Ставлю админку как systemd-сервис"
REMOTE_DIR="$REMOTE_DIR" VPS_USER="$VPS_USER" remote <<EOF
sudo tee /etc/systemd/system/btcproc-admin.service >/dev/null <<UNIT
[Unit]
Description=btcproc admin (crypto-graph)
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
User=$VPS_USER
Group=$VPS_USER
WorkingDirectory=$REMOTE_DIR/btc-graph-processing
ExecStart=$REMOTE_DIR/venv/bin/python -m btcproc.cli admin
Restart=on-failure
RestartSec=10
StandardOutput=append:$REMOTE_DIR/logs/admin.log
StandardError=append:$REMOTE_DIR/logs/admin.log

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now btcproc-admin >/dev/null
# Перезапускаем явно: если сервис уже стоял, enable --now его не тронет,
# а .env мог смениться этим же прогоном скрипта.
sudo systemctl restart btcproc-admin
sleep 5
if ! systemctl is-active --quiet btcproc-admin; then
    echo "админка не стартовала:" >&2
    sudo journalctl -u btcproc-admin -n 25 --no-pager >&2
    exit 1
fi
if curl -fsS -o /dev/null --max-time 10 http://127.0.0.1:8100/health; then
    echo "  админка отвечает на :8100"
else
    echo "  админка запущена, но /health не ответил — смотри logs/admin.log" >&2
fi
EOF

# ─── 6b. Монитор ресурсов ────────────────────────────────────────────────────
hostmon_service

# ─── 7. nginx ────────────────────────────────────────────────────────────────
setup_nginx

# ─── 8. Первый прогон ────────────────────────────────────────────────────────
if [[ $SKIP_TRAIN -eq 1 ]]; then
    warn "Первый прогон пропущен (--skip-train)"
else
# Крон снимается на время обучения. Причина не в блокировке прогонов — она
# работает и так, — а в памяти: cron-овый live и train держат в pandas всю
# историю каждый, и два таких процесса разом упираются в потолок 8 ГБ.
# Расписание вернётся в шаге 9, после того как все монеты обучатся.
step "Снимаю расписание на время обучения"
REMOTE_DIR="$REMOTE_DIR" remote <<'EOF'
# Три фильтра, а не два: у задания обслуживания в команде нет bin/btcproc,
# и без своего фильтра оно пережило бы снятие расписания, а в шаге 9
# добавилось бы вторым экземпляром. Ровно так когда-то задвоился live.
(crontab -l 2>/dev/null || true) | grep -v "# crypto-graph" | grep -v "bin/btcproc" \
    | grep -v "scripts/maintenance.py" > /tmp/ct.pause.$$ || true
crontab /tmp/ct.pause.$$ 2>/dev/null || crontab -r 2>/dev/null || true
rm -f /tmp/ct.pause.$$
echo "  снято"
EOF

# ─── Внешние источники ДО обучения ──────────────────────────────────────────
# Порядок здесь не косметический. Атомы FGI и деривативов считаются джойном из
# external_daily / deriv_metrics, а пишет bar_events ТОЛЬКО train. Если первый
# train пройдёт по пустым таблицам, вся история получит False по этим атомам —
# без единой ошибки в логе, — и починится это лишь следующим еженедельным
# переобучением, то есть через неделю. Ровно тот тихий разъезд, о котором
# предупреждает урок 34.10, только с другой стороны: там забыли завести крон,
# здесь крон есть, но первый раз он отработает уже после обучения.
#
# Обе команды идемпотентны и инкрементальны: на повторном запуске
# ingest-metrics берёт несколько последних суток, а не весь архив.
# Сеть — единственный риск, поэтому падение здесь не роняет установку:
# пустая таблица даёт выключенные атомы, а не сломанный контур.
# Heredoc НЕ закавычен намеренно: $REMOTE_DIR подставляется локально, как во
# всех остальных шагах этого скрипта — функция remote переменные окружения на
# сервер не проносит, она гонит туда только текст скрипта.
step "Наполняю таблицы внешних источников (до первого train)"
remote <<EOF
if [[ ! -x "$REMOTE_DIR/bin/btcproc" ]]; then
    echo "  обёртки ещё нет — пропускаю"
    exit 0
fi
# Внутри remote стоит set -e, поэтому недоступность чужого API иначе уронила
# бы всю установку. Пустая таблица — это выключенные атомы, а не сломанный
# контур, и остановка установки такой цены не стоит.
"$REMOTE_DIR/bin/btcproc" fetch-external 2>&1 | tail -3 || echo "  ВНИМАНИЕ: fetch-external не отработал — FGI-атомы будут False до крона"
"$REMOTE_DIR/bin/btcproc" ingest-metrics --all 2>&1 | tail -5 || echo "  ВНИМАНИЕ: ingest-metrics не отработал — deriv-атомы будут False до крона"
EOF

for symbol in $TRAIN_SYMBOLS; do
    # Уже обученную монету пропускаем: train переобучает модель состояний
    # с нуля, group_id нового прогона несопоставим со старым, и накопленный
    # в Neo4j граф смешал бы две нумерации.
    already=$(ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
        "docker exec btc_postgres psql -U btc_user -d btc_graph -tAc \"SELECT count(*) FROM processing.runs WHERE symbol='$symbol' AND kind='train' AND status='done'\"" </dev/null 2>/dev/null || echo 0)
    if [[ "${already//[^0-9]/}" -gt 0 && $RETRAIN -eq 0 ]]; then
        info "$symbol — уже обучена (прогонов: ${already//[^0-9]/}), пропускаю"
        continue
    fi

    step "Прогон: train --symbol $symbol"
    info "идёт под systemd, обрыв ssh его не убьёт; полная история — 15–60 минут"
    REMOTE_DIR="$REMOTE_DIR" remote <<EOF
sudo systemctl reset-failed btcproc-train.service 2>/dev/null || true
if systemctl is-active --quiet btcproc-train.service; then
    echo "  прогон уже идёт — просто жду его"
    exit 0
fi
rm -f "$REMOTE_DIR/logs/train.exit"
sudo systemd-run \
    --unit=btcproc-train \
    --description="crypto-graph: train по $symbol" \
    --property=User=$VPS_USER \
    --property=Group=$VPS_USER \
    --property=WorkingDirectory=$REMOTE_DIR/btc-graph-processing \
    --property=StandardOutput=append:$REMOTE_DIR/logs/train.log \
    --property=StandardError=append:$REMOTE_DIR/logs/train.log \
    "$REMOTE_DIR/bin/first_train.sh" "$symbol" >/dev/null
echo "  запущен"
EOF

    # Опрос короткими ssh-сессиями: прогон живёт на сервере сам по себе,
    # поэтому обрыв связи здесь ничего не ломает — можно просто перезапустить
    # 02_configure.sh, он подхватит идущий прогон.
    printf '  ждём'
    deadline=$(( $(date +%s) + 4*3600 ))
    while :; do
        exit_code=$(ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
            "cat $REMOTE_DIR/logs/train.exit 2>/dev/null || true" </dev/null 2>/dev/null || true)
        if [[ -n "$exit_code" ]]; then
            printf '\n'
            break
        fi
        [[ $(date +%s) -gt $deadline ]] && die "Прогон $symbol идёт больше 4 часов — смотри $REMOTE_DIR/logs/train.log"
        printf '.'
        sleep 60
    done

    if [[ "$exit_code" != "0" ]]; then
        ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "tail -40 $REMOTE_DIR/logs/train.log" </dev/null || true
        die "Прогон $symbol завершился с кодом $exit_code. Кроны НЕ поставлены — разберись с прогоном и перезапусти скрипт."
    fi
    info "$symbol обучена"
    ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "tail -2 $REMOTE_DIR/logs/train.log" </dev/null | sed 's/^/    /'
done
fi

# ─── 9. Кроны ────────────────────────────────────────────────────────────────
# Ставятся последними и только после успешного train: live продолжает работу
# обученной модели, до неё ему нечего размечать. Плюс train и live одной
# монеты друг друга блокируют — cron поверх идущего прогона просто отвалился бы.
step "Ставлю расписание"
# Монеты перечисляются явными --symbol, а не флагом --all: --all берёт весь
# реестр, и монета, которую ещё не обучали, роняла бы весь пакетный прогон
# ненулевым кодом каждые полчаса.
LIVE_ARGS=""
for symbol in $TRAIN_SYMBOLS; do LIVE_ARGS="$LIVE_ARGS --symbol $symbol"; done
REMOTE_DIR="$REMOTE_DIR" LIVE_ARGS="$LIVE_ARGS" LIVE_CRON="$LIVE_CRON" \
    TRAIN_CRON="$TRAIN_CRON" MAINT_CRON="$MAINT_CRON" FETCH_CRON="$FETCH_CRON" \
    DERIV_CRON="$DERIV_CRON" remote <<EOF
MARK="# crypto-graph"
# Вычищаются И строки-маркеры, И сами команды — иначе повторный запуск
# скрипта плодит дубли расписания. Маркер стоит только в комментариях, а
# команда его не содержит: фильтра по одному маркеру недостаточно, и с ним
# после второго запуска live висел в кроне дважды. Команд теперь две разных
# формы (обёртка bin/btcproc и прямой вызов scripts/maintenance.py), поэтому
# фильтров три. Тот же набор стоит в шаге «снимаю расписание» — они обязаны
# совпадать: пропущенная там строка переживёт паузу и задвоится здесь.
(crontab -l 2>/dev/null || true) \
    | grep -v "\$MARK" | grep -v "bin/btcproc" \
    | grep -v "scripts/maintenance.py" > /tmp/ct.\$\$ || true
cat >> /tmp/ct.\$\$ <<CRON
\$MARK — инкрементальный прогон: догружает бары, размечает, шлёт кандидатов
\$MARK   --skip-if-busy: пока идёт еженедельный train, монета занята, и это
\$MARK   не ошибка — точка продолжения берётся из данных, следующий запуск догонит
$LIVE_CRON $REMOTE_DIR/bin/btcproc live$LIVE_ARGS --skip-if-busy >> $REMOTE_DIR/logs/live.log 2>&1
\$MARK — еженедельное переобучение модели состояний
\$MARK   --no-emit обязателен: train пересобирает кандидатов по ВСЕЙ истории
\$MARK   (~40 тыс. на монету), и слать их в btc-graph заново каждую неделю
\$MARK   незачем — исторические уже отправлены, новые пойдут через live.
\$MARK   ВНИМАНИЕ: train перенумеровывает group_id. Сравнивать кандидатов
\$MARK   разных прогонов можно только через runs.model_run_scope().
$TRAIN_CRON $REMOTE_DIR/bin/btcproc train$LIVE_ARGS --no-emit --skip-if-busy >> $REMOTE_DIR/logs/train.log 2>&1
\$MARK — внешние дневные ряды (Fear & Greed) в external_daily
\$MARK   ЕДИНСТВЕННЫЙ поход в чужой API во всём расписании. В live его нет
\$MARK   намеренно: недоступность alternative.me не должна ронять прогон.
\$MARK   Обратная сторона — если этот крон молча перестанет отрабатывать,
\$MARK   FGI-атомы на свежих барах станут False без единой ошибки в логе.
\$MARK   Проверять по logs/fetch-external.log: он печатает диапазон суток.
$FETCH_CRON $REMOTE_DIR/bin/btcproc fetch-external >> $REMOTE_DIR/logs/fetch-external.log 2>&1
\$MARK — деривативные метрики Binance USD-M (ОИ, long/short ratio, тейкеры)
\$MARK   в deriv_metrics. Тот же принцип, что FETCH_CRON: файл суток
\$MARK   появляется только на следующие сутки, поэтому отдельно от live/train.
\$MARK   DERIV_ENABLED остаётся false, пока фаза 2 (гейты) не даст решения —
\$MARK   крон наполняет таблицу заранее, это не включает атомы само по себе.
\$MARK   Без --start команда ИНКРЕМЕНТАЛЬНА: продолжает от последнего
\$MARK   заполненного бара с откатом на пару суток. Раньше она ежедневно
\$MARK   перемалывала весь архив с начала листинга (~2170 файлов BTC) ради
\$MARK   одного нового дня. Полный бэкфилл — только явной датой, руками.
$DERIV_CRON $REMOTE_DIR/bin/btcproc ingest-metrics --all >> $REMOTE_DIR/logs/ingest-metrics.log 2>&1
\$MARK — ежедневная сводка: покрытие истории, отставание, очередь отправки
17 6 * * * $REMOTE_DIR/bin/btcproc status >> $REMOTE_DIR/logs/status.log 2>&1
\$MARK — недельная уборка базы: разметка старых live-прогонов + вакуум.
\$MARK   Разметку train НЕ трогает — она и есть главный источник роста,
\$MARK   но решение о ней отложено; отчёт скрипта показывает, когда пора.
\$MARK   При идущем прогоне откладывается сам, ждать неделю не страшно.
\$MARK   cd обязателен: .env читается относительно рабочего каталога,
\$MARK   ровно как в обёртке bin/btcproc.
$MAINT_CRON cd $REMOTE_DIR/btc-graph-processing && $REMOTE_DIR/venv/bin/python scripts/maintenance.py >> $REMOTE_DIR/logs/maintenance.log 2>&1
CRON
crontab /tmp/ct.\$\$
rm -f /tmp/ct.\$\$
echo "  crontab:"
crontab -l | sed 's/^/    /'

# Логи прогонов пишутся вечно — без ротации через полгода это гигабайты.
sudo tee /etc/logrotate.d/crypto-graph >/dev/null <<'LR'
/opt/crypto-graph/logs/*.log {
    weekly
    rotate 8
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
LR
echo "  logrotate настроен"
EOF

# ─── 10. Итог ─────────────────────────────────────────────────────────────────
step "Проверка состояния"
REMOTE_DIR="$REMOTE_DIR" remote <<EOF
"$REMOTE_DIR/bin/btcproc" status 2>&1 | tail -25
EOF

ADMIN_CREDS=$(ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "cat $REMOTE_DIR/logs/.admin_creds" 2>/dev/null || true)

if [[ -z "$ADMIN_DOMAIN" ]]; then
    ADMIN_URL="http://$VPS_HOST:8100"
elif [[ "$ENABLE_TLS" == "1" ]]; then
    ADMIN_URL="https://$ADMIN_DOMAIN/login"
else
    ADMIN_URL="http://$ADMIN_DOMAIN/login"
fi

printf '\n\033[1;32m✓ Готово\033[0m\n'
cat <<TXT

  Админка:  $ADMIN_URL
$(printf '%s\n' "$ADMIN_CREDS" | sed 's/^/    /')

  Внутренние сервисы закрыты на 127.0.0.1 — только через ssh-туннель:
    ssh -L 8000:127.0.0.1:8000 -L 5555:127.0.0.1:5555 -L 7474:127.0.0.1:7474 $SSH_TARGET

  Расписание:  live по ${TRAIN_SYMBOLS} (${LIVE_CRON}), train (${TRAIN_CRON}),
               сводка status в 06:17, уборка базы (${MAINT_CRON})
  Логи:        $REMOTE_DIR/logs/{train,live,status,admin,hostmon,maintenance}.log

  Осталось руками:
    • вписать ANTHROPIC_API_KEY в $REMOTE_DIR/btc-graph/.env и перезапустить
      стек (cd $REMOTE_DIR/btc-graph && docker compose up -d --force-recreate),
      иначе кандидаты приходят без текстового объяснения.

  Новая монета (порядок важен, см. CLAUDE.md): SymbolSpec в btcproc/symbols.py
  → профиль оценки в btc-graph/config/symbols/ → добавить тикер в TRAIN_SYMBOLS
  и перезапустить этот скрипт. Без профиля монета получит unknown_symbol_profile
  и оценку по линейке биткоина.
TXT
