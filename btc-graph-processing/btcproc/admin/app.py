"""
Админка crypto-graph.

Страницы:
  /            сводка: покрытие истории, размер графа, кандидаты, приёмник
  /graph       граф состояний (Cytoscape.js), клик по узлу — его переходы
  /chart       свечи, раскрашенные по состояниям, с маркерами кандидатов
  /candidates  таблица кандидатов с фильтрами
  /runs        прогоны: запуск, прогресс, лог

Всё за авторизацией: middleware пускает без сессии только на /login и /static.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from btcproc import config, symbols
from btcproc.admin import auth, queries
from btcproc.db import runs as runs_repo
from btcproc.db.session import init_schema
from btcproc.sink import graph_sink

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

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
        run_id=run_id,
    )


@app.get("/graph", response_class=HTMLResponse)
def graph_page(request: Request, run: str | None = None):
    # Список прогонов фильтруется по монете: выбрать в нём чужой прогон
    # значило бы получить граф другого инструмента под заголовком этого.
    symbol = current_symbol(request)
    run_id = opt_int(run) or _latest_train_id(symbol)
    return page(request, "graph.html", active="graph", symbol=symbol, run_id=run_id,
                runs=runs_repo.list_runs(20, symbol))


@app.get("/chart", response_class=HTMLResponse)
def chart_page(request: Request, run: str | None = None):
    symbol = current_symbol(request)
    run_id = opt_int(run) or _latest_train_id(symbol)
    return page(request, "chart.html", active="chart", symbol=symbol, run_id=run_id,
                runs=runs_repo.list_runs(20, symbol))


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


@app.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request):
    template = "partials/runs_table.html" if request.headers.get("hx-request") \
        else "runs.html"
    # Прогоны показываем по всем монетам: это страница про загрузку машины,
    # а не про конкретный инструмент.
    active_runs = runs_repo.active_runs()
    limit = config.admin.max_concurrent_runs
    return page(request, template, active="runs", runs=runs_repo.list_runs(50),
                active_run=runs_repo.active_run(),
                active_runs=active_runs,
                max_concurrent=limit,
                at_capacity=len(active_runs) >= limit)


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
    active = runs_repo.active_runs()
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
    ingest: bool = Form(True),
    emit: bool = Form(True),
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
               emit: bool = Form(True), lookback: int = Form(240)):
    from btcproc.pipeline.live import run_live

    tickers = _form_symbols(symbol)
    _guard_capacity(tickers)

    for ticker in tickers:
        background.add_task(
            _safe_run, run_live,
            symbol=ticker, lookback_minutes=lookback, do_emit=emit,
        )
    return RedirectResponse("/runs", status_code=303)


@app.post("/runs/{run_id}/emit")
def resend(background: BackgroundTasks, run_id: int):
    from btcproc.pipeline.train import emit_pending

    background.add_task(_safe_run, emit_pending, run_id)
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


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


@app.get("/api/chart")
def api_chart(request: Request, run: str | None = None, start: str | None = None,
              end: str | None = None, limit: str | None = None,
              rating: str | None = None):
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
        )
    except queries.SymbolRunMismatch as exc:
        # 422, а не пустой график: несогласованность монеты и прогона иначе
        # выглядит как «состояний не нашлось».
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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
