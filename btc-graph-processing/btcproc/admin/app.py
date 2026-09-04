"""
Админка crypto-graph.

Страницы:
  /            сводка: покрытие истории, размер графа, кандидаты, приёмник
  /graph       граф состояний (Cytoscape.js), клик по узлу — его переходы
  /chart       свечи, раскрашенные по состояниям, с маркерами кандидатов
  /candidates  таблица кандидатов с фильтрами
  /runs        прогоны: запуск, прогресс, лог
  /server      нагрузка хоста: CPU, память, swap, диск, процессы, контейнеры

Всё за авторизацией: middleware пускает без сессии только на /login и /static.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from btcproc import config, symbols
from btcproc.admin import auth, queries
from btcproc.db import activity as activity_mod
from btcproc.db import runs as runs_repo
from btcproc.db.session import init_schema
from btcproc.hostmon import store as hostmon_store
from btcproc.sink import graph_sink

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ─── Форматирование для шаблонов ────────────────────────────────────────────
# Байты, длительности и «сколько прошло» встречаются на странице «Сервер»
# десятками, и считать их в шаблоне значило бы размазать формулы по разметке.
# Байты форматирует db-слой (`activity.human_bytes`): те же числа он
# показывает в настройках памяти и размерах таблиц, и две реализации
# разошлись бы на первом же округлении. Зависимость идёт в правильную
# сторону — админка знает про db, db про админку не знает.
human_bytes = activity_mod.human_bytes


def human_seconds(value) -> str:
    """Длительность словами: «3 сут 4 ч», «12 мин», «40 с»."""
    if value is None:
        return "—"
    seconds = int(value)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days} сут {hours} ч"
    if hours:
        return f"{hours} ч {minutes} мин"
    if minutes:
        return f"{minutes} мин"
    return f"{seconds} с"


def human_ago(ts) -> str:
    if ts is None:
        return "—"
    delta = int(time.time()) - int(ts)
    return "только что" if delta < 5 else f"{human_seconds(delta)} назад"


def human_ts(ts) -> str:
    if ts is None:
        return "—"
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def utc_minutes(value) -> str:
    """
    Время бара в UTC, до минут.

    Соединение с БД пояс не задаёт, поэтому TIMESTAMPTZ приезжает в поясе
    процесса: на боевом это UTC, на ноутбуке разработчика — нет. Подписать
    колонку «UTC» и напечатать местное время значило бы разойтись с графиком,
    который ось честно рисует в UTC, — и выглядело бы это как сдвиг данных,
    а не как разница поясов.
    """
    if value is None:
        return "—"
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc)
    return value.strftime("%Y-%m-%d %H:%M")


templates.env.filters["bytes"] = human_bytes
templates.env.filters["duration"] = human_seconds
templates.env.filters["ago"] = human_ago
templates.env.filters["ts"] = human_ts
templates.env.filters["utc"] = utc_minutes

app = FastAPI(title="crypto-graph admin", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

PUBLIC_PATHS = {"/login", "/logout", "/health"}


# ─── Разбор параметров форм ─────────────────────────────────────────────────
# HTML-форма отправляет незаполненные поля пустыми строками. Типизированные
# параметры FastAPI (`int | None`, `float | None`) на `?run=&min_quality=`
# отвечают 422, и вся фильтрация ломается разом. Поэтому query-параметры
# страниц принимаются строками и разбираются этими помощниками: пусто и мусор
# означают «фильтр не задан», а не ошибку запроса.
def opt_str(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def opt_int(value: str | None) -> int | None:
    value = opt_str(value)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        logger.warning("Неразбираемое целое в параметре: %r", value)
        return None


def opt_float(value: str | None) -> float | None:
    value = opt_str(value)
    if value is None:
        return None
    try:
        return float(value.replace(",", "."))
    except ValueError:
        logger.warning("Неразбираемое число в параметре: %r", value)
        return None


@app.on_event("startup")
def _startup() -> None:
    # Конфигурация проверяется до первого запроса: лучше не подняться вовсе,
    # чем работать с пустым паролем.
    config.admin.validate()
    # Дефолтная монета, которой нет в реестре, — это ссылки в никуда на всех
    # страницах. Проверяем рядом с паролями, по той же причине: лучше
    # не подняться, чем работать неправильно.
    symbols.validate_default()
    try:
        init_schema()
    except Exception:  # noqa: BLE001 — админка полезна и для диагностики БД
        logger.exception("Не удалось применить схему БД на старте")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    ip = auth.client_ip(request)
    if not auth.ip_allowed(ip):
        logger.warning("Отклонён запрос с IP %s (вне ADMIN_IP_ALLOWLIST)", ip)
        return JSONResponse({"detail": "Доступ с этого адреса запрещён"}, status_code=403)

    path = request.url.path
    if path in PUBLIC_PATHS or path.startswith("/static"):
        return await call_next(request)

    if auth.current_user(request) is None:
        if path.startswith("/api/"):
            return JSONResponse({"detail": "Требуется авторизация"}, status_code=401)
        return RedirectResponse("/login", status_code=303)

    return await call_next(request)


def current_symbol(request: Request) -> str:
    """
    Монета, выбранная в шапке. Хранится в query-параметре `?symbol=`,
    дефолт — из .env.

    Неизвестный тикер не роняет страницу и не подставляется молча: он был бы
    ссылкой, ведущей в пустую выдачу без объяснения. Откатываемся на дефолт.
    """
    raw = (request.query_params.get("symbol") or "").strip()
    if not raw:
        return config.data.symbol
    try:
        return symbols.get(raw).ticker
    except symbols.UnknownSymbolError:
        logger.warning("Неизвестная монета в ?symbol=%r — беру дефолт", raw)
        return config.data.symbol

# Площадка монеты → префикс биржи в тикере TradingView. Ключи — те же, что
# в `symbols.VENUES`: неизвестная площадка означала бы ссылку в пустой график,
# поэтому такую монету оставляем без ссылки вовсе.
TV_EXCHANGES: dict[str, str] = {
    "binance_spot": "BINANCE",
    "bybit_spot": "BYBIT",
}


#: Окна страницы «все инструменты». Часы, а не дни: страница отвечает на
#: вопрос «что происходит прямо сейчас», и всё, что длиннее суток, читается
#: уже по одной монете на /chart.
CHARTS_WINDOWS: tuple[int, ...] = (4, 8, 12, 24, 48)
CHARTS_DEFAULT_HOURS = 12


def tradingview_url(symbol: str) -> str | None:
    """
    Ссылка на тот же инструмент в TradingView, на базовом таймфрейме проекта.

    Внешний график нужен для того, чего у нас нет: стакан, свои разметки,
    сравнение с другими инструментами. Тикер собирается из площадки монеты
    (`SymbolSpec.venue`) — у HYPEUSDT это Bybit, и биржа зашитая константой
    увела бы на несуществующую пару.
    """
    try:
        spec = symbols.get(symbol)
    except symbols.UnknownSymbolError:
        return None
    exchange = TV_EXCHANGES.get(spec.venue)
    if not exchange:
        logger.warning("Нет биржи TradingView для площадки %r", spec.venue)
        return None
    minutes = config.TIMEFRAME_MINUTES.get(config.data.base_tf, 60)
    interval = "1D" if minutes >= 1440 else str(minutes)
    query = urlencode({"symbol": f"{exchange}:{spec.ticker}", "interval": interval})
    return f"https://www.tradingview.com/chart/?{query}"


def page(request: Request, name: str, **context) -> HTMLResponse:
    context.setdefault("user", auth.current_user(request))
    symbol = context.setdefault("symbol", current_symbol(request))
    # Список для селектора и признак «монет больше одной» — шаблон решает,
    # показывать ли переключатель.
    context.setdefault("symbols", symbols.tickers(only_enabled=True))
    context.setdefault("active", "")
    # Хвост для ссылок между страницами: без него переход на соседнюю
    # вкладку молча сбрасывал бы выбранную монету на дефолтную.
    context.setdefault("symbol_qs", f"symbol={symbol}")
    context.setdefault("tv_url", tradingview_url(symbol))
    return templates.TemplateResponse(request, name, context)


# ─── Авторизация ────────────────────────────────────────────────────────────
@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    if auth.current_user(request):
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login")
async def login_submit(request: Request, user: str = Form(...), password: str = Form(...)):
    ip = auth.client_ip(request)
    locked = auth.guard.locked_for(ip)
    if locked:
        return templates.TemplateResponse(
            request, "login.html",
            {"error": f"Слишком много попыток. Попробуй через {locked} сек."},
            status_code=429,
        )

    if not auth.check_credentials(user, password):
        auth.guard.register_failure(ip)
        # Небольшая задержка гасит перебор «в лоб» без блокировки процесса.
        await asyncio.sleep(0.5)
        return templates.TemplateResponse(
            request, "login.html", {"error": "Неверный логин или пароль"}, status_code=401
        )

    auth.guard.reset(ip)
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        auth.COOKIE_NAME,
        auth.issue_session(user),
        max_age=config.admin.session_ttl,
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
    )
    return response


@app.get("/logout")
def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(auth.COOKIE_NAME)
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


# ─── Страницы ───────────────────────────────────────────────────────────────
#: Окно общей сводки на дашборде. Сутки — потому что `live` идёт раз в
#: полчаса: за меньшее окно у половины монет не окажется ни одного бара с
#: заметным кандидатом, и блок будет выглядеть сломанным.
HIGHLIGHT_HOURS = 24


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    symbol = current_symbol(request)
    data = queries.overview(symbol)
    run_id = data["last_train"]["run_id"] if data["last_train"] else None
    return page(
        request, "dashboard.html",
        active="dashboard",
        symbol=symbol,
        data=data,
        ratings=queries.rating_distribution(run_id, symbol),
        sink=graph_sink.sink_status(),
        recent_runs=runs_repo.list_runs(6),
        top_groups=queries.top_groups(run_id, 10) if run_id else [],
        top_transitions=queries.transitions_table(run_id, 10) if run_id else [],
        # Единственный блок страницы, который смотрит поперёк монет:
        # заголовок и все карточки выше — про выбранную.
        highlights=queries.recent_highlights(HIGHLIGHT_HOURS, 50),
        highlight_hours=HIGHLIGHT_HOURS,
        run_id=run_id,
    )


@app.get("/graph", response_class=HTMLResponse)
def graph_page(request: Request, run: str | None = None, find: str | None = None):
    # Список прогонов фильтруется по монете: выбрать в нём чужой прогон
    # значило бы получить граф другого инструмента под заголовком этого.
    #
    # kind="train" — обязательно на этом запросе, а не только в шаблоне.
    # Граф (market_groups/transitions) пишет исключительно train, а live
    # значительно чаще: без фильтра тут топ-20 ЛЮБЫХ прогонов монеты, и через
    # ~10 часов после недельного train (20 live-прогонов по расписанию раз в
    # 30 мин) train вытесняется из этого окна целиком — селектор показывает
    # пустой список до следующего train, хотя граф на месте и прекрасно
    # выбирается по прямой ссылке ?run=. Заметили на бою: у монеты, которую
    # только что переобучили, список был с одной опцией, а через несколько
    # часов стал бы пустым для всех монет разом.
    symbol = current_symbol(request)
    run_id = opt_int(run) or _latest_train_id(symbol)
    # ?find= — наводка с дашборда: «это состояние, покажи его на графе».
    # Значение только подставляется в поле поиска, дальше работает та же
    # клиентская логика, что и при ручном вводе.
    return page(request, "graph.html", active="graph", symbol=symbol, run_id=run_id,
                find=(find or "").strip(),
                runs=runs_repo.list_runs(20, symbol, kind="train"))


@app.get("/states", response_class=HTMLResponse)
def states_page(request: Request, run: str | None = None):
    """
    Полный список состояний модели: цвет, имя, доля, чем выделяется.

    Отдельная страница, а не блок на сводке: состояний у монеты бывает
    полсотни, и на сводке это вытеснило бы всё остальное. Раньше их не было
    нигде целиком — сводка показывала десять крупнейших, граф по одному,
    график раскрашивал бары, — и легенда графика была единственным местом,
    где цвета перечислялись подряд. Место под графиком она занимала большое,
    а прочесть по ней было нечего: полсотни чипов подряд.

    `kind="train"` в списке прогонов по той же причине, что у графа:
    `market_groups` пишет только он, и live-прогон в селекторе означал бы
    пустую страницу.
    """
    symbol = current_symbol(request)
    run_id = opt_int(run) or _latest_train_id(symbol)
    return page(request, "states.html", active="states", symbol=symbol,
                run_id=run_id,
                states=queries.states_page(run_id) if run_id else [],
                runs=runs_repo.list_runs(20, symbol, kind="train"))


@app.get("/chart", response_class=HTMLResponse)
def chart_page(request: Request, run: str | None = None, focus: str | None = None):
    # ?focus=<unix-время> — переход «показать этого кандидата на графике».
    # Прогон в ссылке не нужен и намеренно не передаётся: раскраска берётся по
    # КОРНЮ модели (model_run_scope), а не по одному run_id, поэтому свежий
    # train показывает разметку любого своего live-прогона. Маркеры кандидатов
    # с 2026-08-20 моделью не ограничены вовсе (журнал 51) — кандидат либо был
    # выпущен на этом баре, либо нет, и переобучение этого не отменяет.
    symbol = current_symbol(request)
    run_id = opt_int(run) or _latest_train_id(symbol)
    return page(request, "chart.html", active="chart", symbol=symbol, run_id=run_id,
                focus=opt_int(focus),
                runs=runs_repo.list_runs(20, symbol))


@app.get("/charts", response_class=HTMLResponse)
def charts_all_page(request: Request, hours: str | None = None):
    """
    Все инструменты одним экраном: свечи каждой монеты за последние часы.

    Отдельная страница, а не режим `/chart`: там смысл — разметка одной монеты
    моделью её прогона (состояния, кандидаты, признаки), и всё это по
    построению помонетно. Здесь смысл ровно обратный — сравнить движение
    инструментов между собой, — поэтому ни прогона, ни фильтров тут нет.

    Монета из шапки не участвует: страница показывает все включённые сразу.
    """
    window = opt_int(hours) or CHARTS_DEFAULT_HOURS
    window = max(1, min(window, queries.OVERVIEW_MAX_HOURS))
    return page(request, "charts_all.html", active="charts", hours=window,
                windows=CHARTS_WINDOWS,
                tickers=symbols.tickers(only_enabled=True))


@app.get("/candidates", response_class=HTMLResponse)
def candidates_page(
    request: Request,
    run: str | None = None,
    rating: str | None = None,
    direction: str | None = None,
    min_quality: str | None = None,
    transition: str | None = None,
    emitted: str | None = None,
    page_num: str | None = None,
):
    # Все фильтры принимаются строками: форма отправляет незаполненные поля
    # как пустые строки (`min_quality=&run=`), а типизированные параметры
    # FastAPI на такое отвечают 422 — фильтрация переставала работать целиком.
    symbol = current_symbol(request)
    run_id = opt_int(run)
    data = queries.candidates_page(
        run_id=run_id,
        symbol=symbol,
        rating=opt_str(rating),
        direction=opt_str(direction),
        min_quality=opt_float(min_quality),
        transition=opt_str(transition),
        emitted=opt_str(emitted),
        page=opt_int(page_num) or 1,
    )
    # В шаблон возвращаем то, что пришло, — чтобы выбранные значения
    # остались в полях формы после перерисовки.
    filters = {
        "run": run_id, "rating": opt_str(rating), "direction": opt_str(direction),
        "min_quality": opt_float(min_quality), "transition": opt_str(transition),
        "emitted": opt_str(emitted),
    }
    template = "partials/candidates_table.html" if request.headers.get("hx-request") \
        else "candidates.html"
    return page(request, template, active="candidates", symbol=symbol, data=data,
                filters=filters, runs=runs_repo.list_runs(20, symbol))


@app.get("/candidates/{candidate_id}", response_class=HTMLResponse)
def candidate_detail(request: Request, candidate_id: str):
    row = queries.candidate_detail(candidate_id)
    if not row:
        raise HTTPException(status_code=404, detail="Кандидат не найден")
    return page(request, "candidate_detail.html", active="candidates", row=row)


@app.get("/help", response_class=HTMLResponse)
def help_page(request: Request):
    """
    Шпаргалка для оператора: принципы работы и то, к чему возвращаются каждый
    день. Полная инструкция с процедурами живёт в docs/operator_guide.md —
    страница не заменяет её, а избавляет от похода в репозиторий за мелочью.
    """
    return page(request, "help.html", active="help")


# ─── Сервер: нагрузка хоста ─────────────────────────────────────────────────
# Окна графиков: ключ уходит в ?window=, подпись — на кнопку, число — глубина
# в секундах.
SERVER_WINDOWS = {
    "1h": ("1 час", 3600),
    "6h": ("6 часов", 6 * 3600),
    "24h": ("сутки", 86400),
    "7d": ("неделя", 7 * 86400),
    "30d": ("месяц", 30 * 86400),
}
SERVER_DEFAULT_WINDOW = "6h"
# Сколько пропущенных тактов означают «сэмплер молчит». Три, а не один:
# перезапуск сервиса при выкатке кода штатно съедает такт, и ругаться на это
# каждый раз значило бы приучить не смотреть на предупреждение.
SERVER_STALE_TICKS = 3

#: Контейнер, чью память разбирает страница «PostgreSQL». Имя задано в
#: docker-compose.yml btc-graph и в переменную не вынесено намеренно: страница
#: разбирает КОНКРЕТНУЮ базу этого контура, а не произвольный контейнер.
PG_CONTAINER = "btc_postgres"


def monitor_state() -> dict:
    """
    Что монитор знает о машине: последний замер, покрытие, журнал алертов.

    Отсутствие базы — не ошибка страницы, а понятный диагноз: сэмплер не
    запущен. Поэтому она открывается всегда и говорит, что именно поднять.
    """
    interval = max(1, config.hostmon.interval)
    empty = {
        "ok": False, "error": None, "latest": None, "fresh": False,
        "coverage": None, "alerts": [], "states": {},
    }
    try:
        with contextlib.closing(hostmon_store.connect(read_only=True)) as conn:
            latest = hostmon_store.latest(conn)
            coverage = hostmon_store.coverage(conn)
            recent = hostmon_store.recent_alerts(conn, 15)
            states = hostmon_store.alert_states(conn)
    except FileNotFoundError:
        return {**empty, "error": (
            f"Базы замеров нет ({config.hostmon.db_path}) — сэмплер ни разу не "
            "отработал. Запусти сервис: systemctl status btcproc-hostmon "
            "(локально — make hostmon)."
        )}
    except sqlite3.Error as exc:
        logger.warning("База монитора не читается: %s", exc)
        return {**empty, "error": f"База замеров не читается: {exc}"}

    fresh = bool(latest) and (time.time() - latest["ts"]) < interval * SERVER_STALE_TICKS
    return {
        "ok": True, "error": None, "latest": latest, "fresh": fresh,
        "coverage": coverage, "alerts": recent, "states": states,
    }


def _live_snapshot() -> dict:
    """
    Живой срез процессов и контейнеров. Собирается на каждый запрос — истории
    по ним монитор не ведёт (обоснование — докстринг `hostmon/collect.py`).
    """
    try:
        from btcproc.hostmon import collect
    except ImportError as exc:      # psutil не установлен в этом интерпретаторе
        logger.warning("Монитор процессов недоступен: %s", exc)
        return {"procs": None, "containers": None, "collect_error": str(exc)}

    try:
        procs = collect.processes()
    except Exception as exc:        # noqa: BLE001 — страница обязана открыться
        logger.warning("Снимок процессов не собран: %s", exc)
        procs = None
    return {
        "procs": procs,
        "containers": collect.containers_cached(),
        "collect_error": None,
    }


def _runs_state() -> dict:
    """
    Прогоны рядом с нагрузкой: график в потолке объясняется тем, что идёт
    `train`, а не поломкой.

    Postgres живёт в docker-стеке, и именно его недоступность — сама по себе
    ценный факт для этой страницы. Поэтому ошибка не роняет её, а показывается
    как состояние базы.
    """
    try:
        active = runs_repo.active_runs()
        return {
            "ok": True, "error": None, "active": active,
            "recent": runs_repo.list_runs(8),
            "stale": [run for run in active if runs_repo.is_stale(run)],
        }
    except Exception as exc:        # noqa: BLE001
        logger.warning("Состояние прогонов недоступно: %s", exc)
        return {"ok": False, "error": str(exc)[:300], "active": [], "recent": [], "stale": []}


def _pg_activity() -> dict:
    """
    Что база делает прямо сейчас. Собирается на каждое обновление страницы —
    истории по запросам монитор не ведёт, `pg_stat_activity` её и не хранит.

    Как и у прогонов, недоступный Postgres здесь не сбой страницы, а факт:
    он живёт в том же стеке, за которым мы следим.
    """
    from btcproc.db import activity

    try:
        data = activity.snapshot(config.admin.pg_slow_seconds)
        return {"ok": True, "error": None, **data}
    except Exception as exc:        # noqa: BLE001 — страница обязана открыться
        logger.warning("Срез запросов PostgreSQL не собран: %s", exc)
        return {"ok": False, "error": str(exc)[:300], "rows": [], "summary": {},
                "slow_seconds": config.admin.pg_slow_seconds}


def _server_context(window: str) -> dict:
    return {
        "mon": monitor_state(),
        "runs_state": _runs_state(),
        "pg": _pg_activity(),
        "hostmon": config.hostmon,
        "alerts_cfg": config.alerts,
        "window": window,
        "windows": SERVER_WINDOWS,
        "stale_ticks": SERVER_STALE_TICKS,
        "now_ts": int(time.time()),
        # Имя контейнера базы шаблон знает по нему одному: строка таблицы
        # контейнеров становится ссылкой на разбор его памяти.
        "pg_container": PG_CONTAINER,
        **_live_snapshot(),
    }


@app.get("/server", response_class=HTMLResponse)
def server_page(request: Request, window: str | None = None,
                pg_note: str | None = None):
    """
    Состояние машины: CPU, память, swap, диск в динамике плюс живой срез
    процессов, контейнеров и прогонов.

    Страница про хост, а не про монету, поэтому `?symbol=` она не читает — но
    в ссылках шапки его сохраняет (см. `page`).

    Верхняя половина обновляется htmx-запросом сюда же (partial), графики —
    отдельным запросом к /api/server/series: их данные приходят JSON'ом и
    перерисовываются реже, чем цифры.
    """
    window = window if window in SERVER_WINDOWS else SERVER_DEFAULT_WINDOW
    template = "partials/server_live.html" if request.headers.get("hx-request") \
        else "server.html"
    # Итог снятия бэкенда показывается в СТАТИЧНОЙ половине страницы: живая
    # перерисовывается каждые 10 секунд и сообщение унесла бы раньше, чем
    # оператор дочитает.
    return page(request, template, active="server",
                pg_note=opt_str(pg_note), **_server_context(window))


@app.get("/api/server/series")
def api_server_series(window: str | None = None):
    """Ряды нагрузки за окно — для графиков страницы «Сервер»."""
    key = window if window in SERVER_WINDOWS else SERVER_DEFAULT_WINDOW
    now = int(time.time())
    since = now - SERVER_WINDOWS[key][1]
    try:
        with contextlib.closing(hostmon_store.connect(read_only=True)) as conn:
            data = hostmon_store.series(conn, since, now)
            latest = hostmon_store.latest(conn)
    except FileNotFoundError:
        return JSONResponse(
            {"detail": "Сэмплер не запущен — базы замеров нет"}, status_code=503,
        )
    except sqlite3.Error as exc:
        return JSONResponse({"detail": f"База замеров: {exc}"}, status_code=503)

    data.update(window=key, since=since, now=now,
                interval=config.hostmon.interval, latest=latest)
    return JSONResponse(data)


# ─── Уведомления ────────────────────────────────────────────────────────────
def parse_headers(raw: str) -> dict[str, str]:
    """
    Заголовки из текстового поля: по строке на заголовок, `Имя: значение`.

    Формат выбран вместо JSON намеренно: оператор вставляет сюда токен
    авторизации, и заставлять его экранировать кавычки ради одной строки —
    лишний способ ошибиться. Пустые строки и строки без двоеточия
    игнорируются молча: это опечатка в необязательном поле, а не повод
    отказать в сохранении правила.
    """
    headers: dict[str, str] = {}
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        name, _, value = line.partition(":")
        name, value = name.strip(), value.strip()
        if name and value:
            headers[name] = value
    return headers


def format_headers(headers: dict) -> str:
    return "\n".join(f"{name}: {value}" for name, value in (headers or {}).items())


def event_families() -> list[str]:
    """
    Семейства событий для селектора — из ATOM_FAMILY, а не списком в шаблоне.
    Захардкоженный список разошёлся бы с детекторами при первом же новом
    атоме, причём молча: фильтр просто перестал бы предлагать новое семейство.
    Импорт ленивый — модуль тянет pandas.
    """
    from btcproc.features.events import ATOM_FAMILY

    return sorted(set(ATOM_FAMILY.values()))


@app.get("/notifications", response_class=HTMLResponse)
def notifications_page(request: Request, edit: str | None = None,
                       status: str | None = None):
    """
    Настройка вебхуков: список правил, форма правила, журнал доставок.

    Журнал здесь не для полноты картины: отправка идёт в фоновом потоке и
    следов больше не оставляет, поэтому вопрос «почему получателю ничего
    не пришло» отвечается только отсюда.
    """
    from btcproc.notify import rules as rules_repo

    symbol = current_symbol(request)
    edit_id = opt_int(edit)
    rule = rules_repo.get_rule(edit_id) if edit_id else None
    return page(
        request, "notifications.html", active="notifications", symbol=symbol,
        rules=rules_repo.list_rules(),
        rule=rule,
        headers_text=format_headers(rule.headers) if rule else "",
        deliveries=queries.deliveries(limit=40),
        totals=queries.delivery_totals(),
        families=event_families(),
        transitions=queries.transition_options(_latest_train_id(symbol), 100),
        notify_config=config.notify,
        status=opt_str(status),
    )


@app.post("/notifications/save")
def notifications_save(
    rule_id: str = Form(""),
    name: str = Form(""),
    url: str = Form(""),
    enabled: bool = Form(False),
    payload_mode: str = Form("full"),
    headers: str = Form(""),
    symbol: str = Form(""),
    ratings: list[str] = Form([]),
    directions: list[str] = Form([]),
    min_quality: str = Form(""),
    min_research_score: str = Form(""),
    min_sample_size: str = Form(""),
    transitions: str = Form(""),
    event_families: list[str] = Form([]),
    require_evaluation: bool = Form(False),
):
    """
    Создать или обновить правило.

    Пустые поля означают «фильтр не задан», а не ноль: `min_quality=""` — это
    «любое качество», а `min_quality=0` формально то же самое, но выглядит
    как заданный порог. Поэтому разбор идёт через opt_*.
    """
    from btcproc.notify import rules as rules_repo

    values = {
        "name": name.strip(),
        "url": url.strip(),
        "enabled": enabled,
        "payload_mode": payload_mode,
        "headers": parse_headers(headers),
        "symbol": opt_str(symbol),
        "ratings": rules_repo.clean_list(ratings),
        "directions": rules_repo.clean_list(directions),
        "min_quality": opt_float(min_quality),
        "min_research_score": opt_float(min_research_score),
        "min_sample_size": opt_int(min_sample_size),
        # Переходы вводятся строкой через запятую: их сотни, селектором не
        # выбрать, а нужный оператор копирует со страницы кандидатов.
        "transitions": rules_repo.clean_list(transitions.split(",")),
        "event_families": rules_repo.clean_list(event_families),
        "require_evaluation": require_evaluation,
    }
    try:
        rules_repo.save_rule(values, opt_int(rule_id))
    except rules_repo.RuleError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RedirectResponse("/notifications?status=saved", status_code=303)


@app.post("/notifications/{rule_id}/delete")
def notifications_delete(rule_id: int):
    from btcproc.notify import rules as rules_repo

    rules_repo.delete_rule(rule_id)
    return RedirectResponse("/notifications?status=deleted", status_code=303)


@app.post("/notifications/{rule_id}/test")
def notifications_test(rule_id: int):
    """
    Разовая отправка на адрес правила — мимо очереди, мимо журнала и мимо
    фильтров. Ждём ответа: оператор нажал кнопку и смотрит на результат.

    Кандидат берётся настоящий — последний по монете правила. Если их ещё нет
    (новая установка), уходит показательный пример из payload.example_row():
    принимающей системе нужно тело, а не отказ.
    """
    from btcproc.notify import service as notify_service
    from btcproc.notify import payload as payload_mod
    from btcproc.notify import rules as rules_repo

    rule = rules_repo.get_rule(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="Правило не найдено")
    row = queries.latest_candidate_row(rule.symbol) or payload_mod.example_row()
    result = notify_service.send_one(rule, row)
    detail = result["error"] or f"HTTP {result['http_status']}"
    # Сообщение уезжает в query-параметр, а в нём бывают пробелы, кавычки и
    # двоеточия из чужого текста ошибки — без экранирования это битый Location.
    return RedirectResponse(
        "/notifications?status=" + quote(f"{result['status']}: {detail}"[:300]),
        status_code=303,
    )


RUNS_PER_PAGE = 50


@app.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request, page_no: str | None = None, kind: str | None = None):
    """
    Список прогонов с пагинацией.

    Пагинация появилась не для красоты: `live` идёт по крону каждые полчаса на
    три монеты, то есть полторы сотни прогонов в сутки. Одним списком это
    перестаёт быть читаемым за пару дней.

    Номер страницы обязан переживать автообновление таблицы (оно раз в три
    секунды), иначе пролистать список невозможно в принципе — он будет
    прыгать на первую страницу быстрее, чем читаешь. Поэтому и hx-get в
    шаблоне несёт текущие page и kind.
    """
    template = "partials/runs_table.html" if request.headers.get("hx-request") \
        else "runs.html"
    # Прогоны показываем по всем монетам: это страница про загрузку машины,
    # а не про конкретный инструмент.
    active_runs = runs_repo.active_runs()
    limit = config.admin.max_concurrent_runs

    kind = opt_str(kind)
    current = max(1, opt_int(page_no) or 1)
    total = runs_repo.count_runs(kind=kind)
    pages = max(1, -(-total // RUNS_PER_PAGE))       # деление вверх
    current = min(current, pages)

    return page(request, template, active="runs",
                runs=runs_repo.list_runs(
                    RUNS_PER_PAGE, offset=(current - 1) * RUNS_PER_PAGE, kind=kind,
                ),
                active_run=runs_repo.active_run(),
                active_runs=active_runs,
                max_concurrent=limit,
                at_capacity=len(active_runs) >= limit,
                page_no=current, pages=pages, total_runs=total, kind=kind)


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: int):
    run = runs_repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Прогон не найден")
    template = "partials/run_log.html" if request.headers.get("hx-request") \
        else "run_detail.html"
    return page(request, template, active="runs", run=run)


# ─── Управление прогонами ───────────────────────────────────────────────────

def _form_symbols(symbol: str) -> list[str]:
    """
    Разбор поля формы: конкретная монета либо «все активные».

    Пустое значение означает монету по умолчанию — так форма без селектора
    (и старые закладки) продолжают работать.
    """
    value = (symbol or "").strip()
    if value.lower() == "all":
        return [spec.ticker for spec in symbols.enabled()]
    if not value:
        return [config.data.symbol]
    try:
        return [symbols.get(value).ticker]
    except symbols.UnknownSymbolError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _guard_capacity(tickers: list[str]) -> None:
    """
    Проверка перед запуском.

    Одну монету дважды одновременно считать нельзя — это гонка за одни и те же
    строки. Разные монеты параллельно можно и нужно, но не бесконечно: прогоны
    идут BackgroundTasks в процессе админки, и каждый занимает ядро под
    кластеризацию и заметный кусок памяти.
    """
    # Мёртвый прогон (процесс убит OOM-killer'ом или ребутом) остаётся
    # `running` навсегда и занимает слот лимита + блокирует свою монету
    # ответом 409. Снимаем такие перед проверкой, а не после.
    active = [
        run for run in runs_repo.active_runs() if not runs_repo.reap_if_stale(run)
    ]
    busy = {run["symbol"] for run in active if run.get("symbol")}

    clash = sorted(busy & set(tickers))
    if clash:
        raise HTTPException(
            status_code=409,
            detail=f"По {', '.join(clash)} уже идёт прогон — дождись окончания",
        )

    limit = config.admin.max_concurrent_runs
    if len(active) + len(tickers) > limit:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Одновременно выполняется {len(active)} прогонов при лимите {limit}. "
                f"Запуск ещё {len(tickers)} превысил бы его: кластеризация упирается "
                f"в CPU и память. Дождись окончания или подними "
                f"ADMIN_MAX_CONCURRENT_RUNS."
            ),
        )


@app.post("/runs/train")
def start_train(
    background: BackgroundTasks,
    symbol: str = Form(""),
    # Дефолт False, а не True, — это не опечатка: снятый чекбокс браузер не
    # отправляет ВОВСЕ, и FastAPI подставил бы дефолт. С Form(True) галки были
    # декорацией — выключить ingest/emit из админки было невозможно.
    # Состояние «включено по умолчанию» задаёт `checked` в шаблоне.
    ingest: bool = Form(False),
    emit: bool = Form(False),
    start: str = Form(""),
    end: str = Form(""),
):
    from btcproc.pipeline.train import run_train

    tickers = _form_symbols(symbol)
    _guard_capacity(tickers)

    for ticker in tickers:
        background.add_task(
            _safe_run, run_train,
            symbol=ticker, do_ingest=ingest, do_emit=emit,
            start=start or None, end=end or None,
        )
    return RedirectResponse("/runs", status_code=303)


@app.post("/runs/live")
def start_live(background: BackgroundTasks, symbol: str = Form(""),
               emit: bool = Form(False), lookback: str = Form("")):
    """
    Пустой lookback = авто-режим: продолжить с последнего выпущенного
    кандидата (`resolve_cutoff`). Раньше поле было `int = Form(240)` и пустым
    быть не могло в принципе, то есть админка ВСЕГДА задавала окно явно.
    Это оставляло невосполнимую дыру: после паузы длиннее окна запуск из
    админки выпускал кандидатов только за последние 240 минут, но двигал
    `last_candidate_ts` в свежее время — и следующий крон-live продолжал уже
    с него. Пропущенный интервал не закрывал никто.
    """
    from btcproc.pipeline.live import run_live

    tickers = _form_symbols(symbol)
    _guard_capacity(tickers)

    for ticker in tickers:
        background.add_task(
            _safe_run, run_live,
            symbol=ticker, lookback_minutes=opt_int(lookback), do_emit=emit,
        )
    return RedirectResponse("/runs", status_code=303)


@app.post("/runs/{run_id}/emit")
def resend(background: BackgroundTasks, run_id: int):
    from btcproc.pipeline.train import emit_pending

    background.add_task(_safe_run, emit_pending, run_id)
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


def _pg_container_memory() -> dict:
    """
    Разбор памяти контейнера базы плюс вывод, ради которого он и делается.

    `docker stats` показывает у Postgres одно число, куда входит
    `shared_buffers` — пул, выделенный при старте раз и навсегда. Оператор
    видит «два гигабайта» и идёт искать тяжёлые запросы там, где нагрузки нет
    вовсе. Поэтому строки таблицы здесь дополняются готовым выводом: сколько
    из показанного докером — статика, а сколько действительно тратят
    процессы.
    """
    try:
        from btcproc.hostmon import collect
    except ImportError as exc:      # psutil не установлен в этом интерпретаторе
        return {"enabled": False, "rows": [], "error": str(exc), "summary": None}

    try:
        data = collect.container_memory(PG_CONTAINER)
    except Exception as exc:        # noqa: BLE001 — страница обязана открыться
        # `container_memory` штатные отказы docker возвращает полем `error`,
        # но неожиданное исключение обязано остаться внутри блока, а не унести
        # с собой запросы и настройки, ради которых страницу и открыли.
        logger.warning("Память контейнера %s не разобрана: %s", PG_CONTAINER, exc)
        return {"enabled": True, "rows": [], "error": str(exc)[:300],
                "summary": None, "container": PG_CONTAINER}

    values = data.get("values") or {}
    rows = [
        {"key": key, "label": label, "note": note,
         "bytes": values.get(key), "value": activity_mod.human_bytes(values.get(key))}
        for key, label, note in collect.CGROUP_MEMORY_KEYS
        if values.get(key) is not None
    ]

    summary = None
    if values:
        shmem = values.get("shmem") or 0
        anon = values.get("anon") or 0
        # `docker stats` считает MEM USAGE как anon + shmem (страничный кэш в
        # него не входит) — повторяем ту же арифметику, чтобы число на этой
        # странице сходилось с числом в таблице контейнеров, а не спорило с ним.
        counted = shmem + anon
        summary = {
            "counted": activity_mod.human_bytes(counted),
            "shmem": activity_mod.human_bytes(shmem),
            "anon": activity_mod.human_bytes(anon),
            "shmem_pct": round(100.0 * shmem / counted, 1) if counted else None,
        }
    return {"enabled": data.get("enabled", False), "rows": rows,
            "error": data.get("error"), "summary": summary,
            "container": PG_CONTAINER}


def _pg_profile() -> dict:
    """
    Всё, что страница «PostgreSQL» знает о базе, четырьмя независимыми
    кусками.

    Независимыми буквально: каждый блок ловит свою ошибку и отдаёт `None`.
    База, которой плохо, — главный повод открыть эту страницу, и падение
    одного запроса не имеет права унести с собой остальные три. Тот же
    принцип, что у `_runs_state` и `_pg_activity`.
    """
    result: dict = {"ok": True, "error": None}
    parts = (
        ("settings", activity_mod.memory_settings),
        ("stats", activity_mod.database_stats),
        ("tables", activity_mod.top_tables),
        ("statements", activity_mod.top_statements),
    )
    for name, func in parts:
        try:
            result[name] = func()
        except Exception as exc:    # noqa: BLE001 — страница обязана открыться
            logger.warning("Блок «%s» страницы PostgreSQL не собран: %s", name, exc)
            result[name] = None
            # Первая же ошибка обычно означает недоступную базу, и показать её
            # надо один раз наверху, а не четырьмя одинаковыми строками.
            result["ok"] = False
            result.setdefault("error", None)
            result["error"] = result["error"] or str(exc)[:300]
    return result


@app.get("/server/postgres", response_class=HTMLResponse)
def server_postgres_page(request: Request, pg_note: str | None = None):
    """
    Что занимает память базы и что её грузит — вся диагностика Postgres на
    одной странице.

    Отдельная страница, а не блок на «Сервере»: собирается она дороже
    (`docker exec` в контейнер плюс четыре запроса к системным вьюхам), а
    открывают её редко и по конкретному поводу. Живая половина «Сервера»
    перерисовывается каждые десять секунд, и тащить этот сбор в каждый её
    такт значило бы сделать монитор заметной нагрузкой на машину, за которой
    он следит.

    Попадают сюда кликом по строке контейнера `btc_postgres` в таблице
    контейнеров.
    """
    return page(request, "server_postgres.html", active="server",
                pg_note=opt_str(pg_note),
                memory=_pg_container_memory(),
                pg=_pg_activity(),
                profile=_pg_profile())


@app.post("/server/postgres/statements/reset")
def server_postgres_reset(request: Request):
    """
    Обнулить накопленную статистику запросов.

    Действие пишущее, поэтому только POST и только по сессии админки. Оно
    ничего не ломает — счётчики `pg_stat_statements` не данные, — но
    необратимо: предыдущий период после нажатия не восстановить. Отсюда
    подтверждение в интерфейсе.
    """
    ok = activity_mod.reset_statements()
    note = ("Статистика запросов обнулена — счёт периода пошёл заново."
            if ok else
            "PostgreSQL отказал в сбросе статистики: у роли btc_user нет прав "
            "на pg_stat_statements_reset(), либо расширение не загружено.")
    return RedirectResponse(f"/server/postgres?{urlencode({'pg_note': note})}",
                            status_code=303)


@app.post("/server/pg/{pid}/stop")
def server_pg_stop(pid: int, window: str | None = None):
    """
    Снять один бэкенд PostgreSQL: отменить запрос, а для висящей транзакции —
    закрыть соединение. Выбор способа — в `activity.stop_query`.

    Действие пишущее и в чужой процесс, поэтому оно: только POST, только по
    сессии админки (как и вся страница), с подтверждением в интерфейсе и
    записью в лог. Результат уезжает в query-строку редиректа, чтобы
    оператор увидел, что именно произошло, — «кнопка нажалась и ничего»
    здесь худший из исходов.
    """
    from btcproc.db import activity

    try:
        result = activity.stop_query(pid)
        note = result["note"]
    except Exception as exc:        # noqa: BLE001
        logger.warning("Снять бэкенд %s не удалось: %s", pid, exc)
        note = f"Не удалось: {exc}"[:300]
    params = {"pg_note": note}
    if window:
        params["window"] = window
    return RedirectResponse(f"/server?{urlencode(params)}", status_code=303)


def _safe_run(func, *args, **kwargs) -> None:
    """Фоновая задача не должна ронять процесс — диагноз уже лёг в runs."""
    try:
        func(*args, **kwargs)
    except Exception:  # noqa: BLE001
        logger.exception("Фоновая задача %s завершилась ошибкой", func.__name__)


# ─── JSON для страниц ───────────────────────────────────────────────────────
@app.get("/api/graph")
def api_graph(request: Request, run: str | None = None, min_count: str | None = None,
              rarity: str | None = None):
    run_id = opt_int(run) or _latest_train_id(current_symbol(request))
    if run_id is None:
        return {"nodes": [], "edges": []}
    return queries.graph_payload(
        run_id, min_count=opt_int(min_count) or 1, rarity=opt_str(rarity)
    )


@app.get("/api/graph/group/{group_id}")
def api_group(request: Request, group_id: float, run: str | None = None):
    run_id = opt_int(run) or _latest_train_id(current_symbol(request))
    node = queries.group_detail(run_id, group_id) if run_id else None
    if not node:
        raise HTTPException(status_code=404, detail="Состояние не найдено")
    # Монета подписывается явно: «group_id 7» без неё бессмысленен —
    # номера состояний у монет свои и между собой несопоставимы.
    node["symbol"] = queries.run_symbol(run_id)
    return node


@app.get("/api/graph/context")
def api_graph_context(request: Request, run: str | None = None):
    """
    Фон состояний целиком: {group_id: [атомы]}.

    Существует ради прогрева, и с 2026-08-16 прогревать почти нечего: агрегат
    считает train, ручка читает готовые строки. Остаётся она по двум
    причинам — держит кэш процесса тёплым и досчитывает фон прогонам, сделанным
    до появления таблицы (единственный тяжёлый случай, один раз на прогон;
    заранее его закрывает `scripts/backfill_state_context.py`). Страница графа
    дёргает ручку после отрисовки и результат выбрасывает.
    """
    symbol = current_symbol(request)
    run_id = opt_int(run) or _latest_train_id(symbol)
    if run_id is None:
        return {}
    root = runs_repo.model_root(run_id)
    owner = queries.run_symbol(root) or symbol
    return {str(gid): atoms
            for gid, atoms in queries.state_context_atoms(root, owner).items()}


@app.get("/api/chart")
def api_chart(request: Request, run: str | None = None, start: str | None = None,
              end: str | None = None, limit: str | None = None,
              rating: str | None = None, layer: str | None = None):
    symbol = current_symbol(request)
    run_id = opt_int(run) or _latest_train_id(symbol)
    if run_id is None:
        return {"bars": [], "markers": [], "groups": []}
    try:
        return queries.chart_data(
            run_id,
            symbol=symbol,
            start=opt_str(start),
            end=opt_str(end),
            limit=min(opt_int(limit) or 1500, 5000),
            rating=opt_str(rating),
            # Значение из строки запроса не доверяем: слой участвует в SQL
            # только через сравнение с "all", но проверка здесь делает набор
            # допустимых значений явным, а не выводимым из ветки ниже.
            layer="all" if opt_str(layer) == "all" else "issued",
        )
    except queries.SymbolRunMismatch as exc:
        # 422, а не пустой график: несогласованность монеты и прогона иначе
        # выглядит как «состояний не нашлось».
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/chart/indicator")
def api_chart_indicator(request: Request, name: str, run: str | None = None,
                        start: str | None = None, end: str | None = None):
    """
    Одна серия признака под свечами.

    Отдельным запросом от `/api/chart`, а не полем в нём: смена индикатора не
    должна перезагружать бары — иначе график сбрасывал бы позицию и масштаб на
    каждое переключение панели.
    """
    symbol = current_symbol(request)
    run_id = opt_int(run) or _latest_train_id(symbol)
    if run_id is None:
        return {"name": name, "points": [], "note": "нет ни одного прогона монеты"}
    return queries.indicator_series(
        run_id, symbol, name, start=opt_str(start), end=opt_str(end)
    )


@app.get("/api/chart/freshness")
def api_chart_freshness(request: Request):
    """
    Отметка данных монеты для автообновления графика.

    Отдельный дешёвый запрос вместо перезагрузки всего графика по часам:
    страница спрашивает «приехало ли что-то новое» и грузит бары только
    тогда, когда приехало. Обоснование — докстринг `queries.chart_freshness`.
    """
    return queries.chart_freshness(current_symbol(request))


@app.get("/api/charts/overview")
def api_charts_overview(hours: str | None = None):
    """
    Свечи всех включённых монет за окно — одним ответом.

    Один запрос на страницу, а не по запросу на плитку: монет семь, и семь
    параллельных походов в ту же таблицу дали бы ровно тот же ответ дороже,
    зато с разъезжающимся временем среза между плитками.
    """
    return queries.overview_charts(symbols.tickers(only_enabled=True),
                                   hours=opt_int(hours) or CHARTS_DEFAULT_HOURS)


@app.get("/api/charts/overview/states")
def api_charts_overview_states(hours: str | None = None):
    """
    Раскраска плиток состояниями — вторым запросом после свечей.

    Так же, как панель признака на /chart: свечи появляются сразу, разметка
    доезжает следом и не задерживает страницу, когда у монеты ещё нет модели.
    """
    return queries.overview_states(symbols.tickers(only_enabled=True),
                                   hours=opt_int(hours) or CHARTS_DEFAULT_HOURS)


@app.get("/api/runs/{run_id}")
def api_run(run_id: int):
    run = runs_repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Прогон не найден")
    return run


def _latest_train_id(symbol: str | None = None) -> int | None:
    """Последний train монеты — источник графа и раскраски по умолчанию."""
    run = (
        runs_repo.latest_completed_run("train", symbol)
        or runs_repo.active_run("train", symbol)
    )
    return int(run["run_id"]) if run else None
