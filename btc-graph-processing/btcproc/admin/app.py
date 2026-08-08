"""
Админка btc-graph-processing.

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

from btcproc import config
from btcproc.admin import auth, queries
from btcproc.db import runs as runs_repo
from btcproc.db.session import init_schema
from btcproc.sink import graph_sink

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="btc-graph-processing admin", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

PUBLIC_PATHS = {"/login", "/logout", "/health"}


@app.on_event("startup")
def _startup() -> None:
    # Конфигурация проверяется до первого запроса: лучше не подняться вовсе,
    # чем работать с пустым паролем.
    config.admin.validate()
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


def page(request: Request, name: str, **context) -> HTMLResponse:
    context.setdefault("user", auth.current_user(request))
    context.setdefault("symbol", config.data.symbol)
    context.setdefault("active", "")
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
    data = queries.overview()
    run_id = data["last_train"]["run_id"] if data["last_train"] else None
    return page(
        request, "dashboard.html",
        active="dashboard",
        data=data,
        ratings=queries.rating_distribution(run_id),
        sink=graph_sink.sink_status(),
        recent_runs=runs_repo.list_runs(8),
    )


@app.get("/graph", response_class=HTMLResponse)
def graph_page(request: Request, run: int | None = None):
    run_id = run or _latest_train_id()
    return page(request, "graph.html", active="graph", run_id=run_id,
                runs=runs_repo.list_runs(20))


@app.get("/chart", response_class=HTMLResponse)
def chart_page(request: Request, run: int | None = None):
    run_id = run or _latest_train_id()
    return page(request, "chart.html", active="chart", run_id=run_id,
                runs=runs_repo.list_runs(20))


@app.get("/candidates", response_class=HTMLResponse)
def candidates_page(
    request: Request,
    run: int | None = None,
    rating: str | None = None,
    direction: str | None = None,
    min_quality: float | None = None,
    transition: str | None = None,
    emitted: str | None = None,
    page_num: int = 1,
):
    data = queries.candidates_page(
        run_id=run, rating=rating, direction=direction, min_quality=min_quality,
        transition=transition, emitted=emitted, page=page_num,
    )
    filters = {
        "run": run, "rating": rating, "direction": direction,
        "min_quality": min_quality, "transition": transition, "emitted": emitted,
    }
    template = "partials/candidates_table.html" if request.headers.get("hx-request") \
        else "candidates.html"
    return page(request, template, active="candidates", data=data, filters=filters,
                runs=runs_repo.list_runs(20))


@app.get("/candidates/{candidate_id}", response_class=HTMLResponse)
def candidate_detail(request: Request, candidate_id: str):
    row = queries.candidate_detail(candidate_id)
    if not row:
        raise HTTPException(status_code=404, detail="Кандидат не найден")
    return page(request, "candidate_detail.html", active="candidates", row=row)


@app.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request):
    template = "partials/runs_table.html" if request.headers.get("hx-request") \
        else "runs.html"
    return page(request, template, active="runs", runs=runs_repo.list_runs(50),
                active_run=runs_repo.active_run())


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: int):
    run = runs_repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Прогон не найден")
    template = "partials/run_log.html" if request.headers.get("hx-request") \
        else "run_detail.html"
    return page(request, template, active="runs", run=run)


# ─── Управление прогонами ───────────────────────────────────────────────────
@app.post("/runs/train")
def start_train(
    background: BackgroundTasks,
    ingest: bool = Form(True),
    emit: bool = Form(True),
    start: str = Form(""),
    end: str = Form(""),
):
    if runs_repo.active_run():
        raise HTTPException(status_code=409, detail="Уже идёт прогон — дождись окончания")

    from btcproc.pipeline.train import run_train

    background.add_task(
        _safe_run, run_train,
        do_ingest=ingest, do_emit=emit,
        start=start or None, end=end or None,
    )
    return RedirectResponse("/runs", status_code=303)


@app.post("/runs/live")
def start_live(background: BackgroundTasks, emit: bool = Form(True),
               lookback: int = Form(240)):
    if runs_repo.active_run():
        raise HTTPException(status_code=409, detail="Уже идёт прогон — дождись окончания")

    from btcproc.pipeline.live import run_live

    background.add_task(_safe_run, run_live, lookback_minutes=lookback, do_emit=emit)
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
def api_graph(run: int | None = None, min_count: int = 1, rarity: str | None = None):
    run_id = run or _latest_train_id()
    if run_id is None:
        return {"nodes": [], "edges": []}
    return queries.graph_payload(run_id, min_count=min_count, rarity=rarity)


@app.get("/api/graph/group/{group_id}")
def api_group(group_id: float, run: int | None = None):
    run_id = run or _latest_train_id()
    node = queries.group_detail(run_id, group_id) if run_id else None
    if not node:
        raise HTTPException(status_code=404, detail="Состояние не найдено")
    return node


@app.get("/api/chart")
def api_chart(run: int | None = None, start: str | None = None,
              end: str | None = None, limit: int = 1500):
    run_id = run or _latest_train_id()
    if run_id is None:
        return {"bars": [], "markers": [], "groups": []}
    return queries.chart_data(run_id, start=start, end=end, limit=min(limit, 5000))


@app.get("/api/runs/{run_id}")
def api_run(run_id: int):
    run = runs_repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Прогон не найден")
    return run


def _latest_train_id() -> int | None:
    run = runs_repo.latest_completed_run("train") or runs_repo.active_run("train")
    return int(run["run_id"]) if run else None
