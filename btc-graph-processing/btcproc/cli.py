"""
Командная строка btc-graph-processing.

    python -m btcproc.cli init-db
    python -m btcproc.cli ingest --symbol ETHUSDT
    python -m btcproc.cli train --all --no-emit
    python -m btcproc.cli live --all
    python -m btcproc.cli admin

Общий принцип мультимонетности: у команд, работающих с данными, есть
`--symbol` (можно указать несколько раз) и `--all` (все активные монеты
реестра). Без флагов поведение прежнее — монета из `.env`, чтобы существующие
скрипты и `make train` не сломались.
"""
from __future__ import annotations

import logging
import time

import typer

from btcproc import config, symbols
from btcproc.db import runs as runs_repo
from btcproc.db import session

app = typer.Typer(add_completion=False, help="Генератор кандидатов для btc-graph")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _resolve(symbol: list[str] | None, all_symbols: bool) -> list[symbols.SymbolSpec]:
    """Разбор --symbol/--all с понятной ошибкой вместо трейсбека."""
    try:
        return symbols.resolve_many(symbol, all_symbols)
    except symbols.UnknownSymbolError as exc:
        raise typer.BadParameter(str(exc)) from exc


class SkipBusy(Exception):
    """Прогон пропущен, потому что по монете уже идёт другой. Не ошибка."""


def _run_for_each(
    specs: list[symbols.SymbolSpec],
    action,
    what: str,
) -> None:
    """
    Последовательный пакетный прогон по монетам.

    Три требования, все существенные:
      * последовательно, а не параллельно — узкое место CPU на кластеризации,
        два одновременных train просто отнимут память друг у друга;
      * падение одной монеты НЕ останавливает остальные: прогон помечается
        failed с диагнозом, цикл идёт дальше, в конце печатается сводка.
        Это ровно тот случай, ради которого в проекте оставлен except Exception;
      * ненулевой exit code, если упала хотя бы одна монета — иначе cron будет
        рапортовать об успехе.
    """
    failures: list[tuple[str, str]] = []
    skipped: list[str] = []
    ok = 0

    for spec in specs:
        if len(specs) > 1:
            typer.echo(f"\n=== {spec.ticker} ({what}) ===")
        try:
            result = action(spec)
        except SkipBusy as skip:
            # Не провал: монета занята другим прогоном, и так и задумано.
            # В failures не попадает, значит exit code остаётся нулевым.
            skipped.append(spec.ticker)
            typer.echo(typer.style(f"  пропуск: {skip}", fg=typer.colors.YELLOW))
            continue
        except Exception as exc:  # noqa: BLE001 — падение одной монеты не рушит пакет
            failures.append((spec.ticker, f"{type(exc).__name__}: {exc}"))
            typer.echo(typer.style(f"  ОШИБКА {spec.ticker}: {exc}", fg=typer.colors.RED))
            continue
        ok += 1
        if result is not None:
            typer.echo(runs_repo.dumps(result))

    if len(specs) > 1:
        tail = f", пропущено {len(skipped)}" if skipped else ""
        typer.echo(f"\nИтог: {ok} из {len(specs)} успешно{tail}")
        for ticker, error in failures:
            typer.echo(f"  {ticker}: {error}")

    if failures:
        raise typer.Exit(code=1)


@app.command("init-db")
def init_db() -> None:
    """Создать схему processing и таблицы."""
    session.init_schema()
    typer.echo(f"Схема {config.db.schema} готова")
    # Дефолтная монета, которой нет в реестре, — это дефолт, указывающий
    # в никуда. Обнаружиться это должно здесь, а не на середине прогона.
    symbols.validate_default()
    typer.echo(
        f"Монеты: {', '.join(symbols.tickers(only_enabled=True))} "
        f"(по умолчанию {config.data.symbol})"
    )


@app.command()
def ingest(
    symbol: list[str] = typer.Option(None, "--symbol", help="Тикер; можно указать несколько раз"),
    all_symbols: bool = typer.Option(False, "--all", help="Все активные монеты реестра"),
    start: str = typer.Option(None, help="Дата начала, YYYY-MM-DD (перекрывает реестр)"),
    context: bool = typer.Option(True, help="Пересобрать старшие ТФ из базового"),
) -> None:
    """
    Скачать историю монеты в БД — с той площадки, что указана в её реестре.

    Цикл по монетам живёт в sources.sync_many: он нужен и админке, и обкатке
    новой монеты, а не только этой команде.
    """
    from btcproc.ingest import bars, sources

    specs = _resolve(symbol, all_symbols)

    result = sources.sync_many(
        [spec.ticker for spec in specs],
        tf=config.data.base_tf,
        start=start,
        context=context,
        progress=lambda i, n, msg: typer.echo(f"  [{i}/{n}] {msg}"),
        on_symbol=lambda i, n, ticker: typer.echo(
            f"\n=== {ticker} ({i}/{n}) ===" if n > 1 else f"=== {ticker} ==="
        ),
    )

    for ticker, report in result["symbols"].items():
        typer.echo(f"{ticker}: баров записано {report['rows']}")
        if report["missing_months"]:
            # Текущий месяц в этом списке — норма: месячный архив публикуется
            # только после его закрытия, до тех пор работают дневные дампы и REST.
            # Длинный хвост пропусков В НАЧАЛЕ — признак того, что history_start
            # монеты занижен: уточни symbols.resolve_history_start(ticker).
            typer.echo(
                f"  нет месячных дампов за: {', '.join(report['missing_months'])} "
                "(текущий месяц добирается дневными архивами и REST — это ожидаемо)"
            )
        if report.get("context"):
            typer.echo(f"  старшие ТФ: {report['context']}")

    for ticker, error in result["failed"].items():
        typer.echo(typer.style(f"ОШИБКА {ticker}: {error}", fg=typer.colors.RED))

    typer.echo(
        f"\nИтог: {result['ok']} из {result['total']} монет, "
        f"всего баров {result['rows_total']}"
    )
    if result["failed"]:
        raise typer.Exit(code=1)


def _guard_active_run(force: bool, symbol: str, skip_if_busy: bool = False) -> None:
    """
    Два тяжёлых прогона ОДНОЙ монеты одновременно — это гонка за одни и те же
    строки и вдвое большая нагрузка на БД. Прогоны разных монет независимы
    и блокировать друг друга не должны: иначе мультимонетность сводится
    к очереди из одной монеты за раз.

    skip_if_busy меняет реакцию, а не саму проверку. Для человека за
    клавиатурой «идёт другой прогон» — ошибка: он хотел запустить и не
    запустил. Для расписания это норма: раз в неделю `train` занимает монету
    на полчаса, и попавший на него получасовой `live` должен молча уступить,
    а не сыпать в лог ошибками до конца обучения. Догонит на следующем
    запуске — точка продолжения берётся из данных, а не из календаря.
    """
    active = runs_repo.active_run(symbol=symbol)
    if not active or force:
        return
    # Прогон, убитый насмерть, остаётся `running` навсегда — сам он статус уже
    # не снимет. Без этой проверки крон с --skip-if-busy молча пропускал бы
    # каждый следующий live монеты, и обновление данных прекращалось бы тихо.
    if runs_repo.reap_if_stale(active):
        typer.echo(
            f"Прогон #{active['run_id']} по {symbol} признан мёртвым "
            f"(нет heartbeat) и помечен failed — продолжаю."
        )
        return
    detail = (f"#{active['run_id']} ({active['kind']}, стадия "
              f"{active['stage'] or '—'}, {active['progress']:.0%})")
    if skip_if_busy:
        raise SkipBusy(f"по {symbol} идёт прогон {detail} — пропускаю")
    raise typer.BadParameter(
        f"По {symbol} уже идёт прогон {detail}. "
        "Дождись окончания или запусти с --force."
    )


@app.command()
def train(
    symbol: list[str] = typer.Option(None, "--symbol", help="Тикер; можно несколько раз"),
    all_symbols: bool = typer.Option(False, "--all", help="Все активные монеты реестра"),
    start: str = typer.Option(None, help="Дата начала истории (перекрывает реестр)"),
    end: str = typer.Option(None, help="Дата конца (для воспроизводимых прогонов)"),
    ingest_data: bool = typer.Option(True, "--ingest/--no-ingest", help="Качать историю"),
    emit: bool = typer.Option(True, "--emit/--no-emit", help="Слать кандидатов в btc-graph"),
    force: bool = typer.Option(False, "--force", help="Запустить, даже если идёт другой прогон"),
    skip_if_busy: bool = typer.Option(
        False, "--skip-if-busy",
        help="Занятую монету пропустить без ошибки — для расписания",
    ),
) -> None:
    """
    Полный прогон истории: от баров до кандидатов в btc-graph.

    ВНИМАНИЕ: состояния переобучаются с нуля, поэтому номера group_id
    в новом прогоне не соответствуют старым. Для регулярного обновления
    нужен live, а не train.

    С --all монеты обрабатываются последовательно, каждая своим run_id;
    падение одной не останавливает остальные.
    """
    from btcproc.pipeline.train import run_train

    specs = _resolve(symbol, all_symbols)
    typer.echo(
        "Модель состояний будет переобучена: group_id нового прогона "
        "несопоставимы с предыдущими.\n"
    )

    def action(spec: symbols.SymbolSpec) -> dict:
        _guard_active_run(force, spec.ticker, skip_if_busy)
        return run_train(
            symbol=spec.ticker, start=start, end=end,
            do_ingest=ingest_data, do_emit=emit,
        )

    _run_for_each(specs, action, "train")


@app.command()
def live(
    symbol: list[str] = typer.Option(None, "--symbol", help="Тикер; можно несколько раз"),
    all_symbols: bool = typer.Option(False, "--all", help="Все активные монеты реестра"),
    model_run: int = typer.Option(None, help="run_id прогона с нужной моделью"),
    lookback: int = typer.Option(
        None,
        help="Минут назад считать кандидатов свежими. "
             "По умолчанию — продолжить с последнего выпущенного кандидата",
    ),
    max_lookback_days: int = typer.Option(
        30, help="Предел, если кандидатов не было очень давно"
    ),
    emit: bool = typer.Option(True, "--emit/--no-emit"),
    force: bool = typer.Option(False, "--force", help="Запустить, даже если идёт другой прогон"),
    skip_if_busy: bool = typer.Option(
        False, "--skip-if-busy",
        help="Занятую монету пропустить без ошибки — для расписания",
    ),
) -> None:
    """
    Инкрементальный прогон по свежим барам.

    Точка продолжения берётся из данных и считается по своей монете: бары
    качаются с последнего сохранённого, кандидаты — с последнего выпущенного.
    Пропуск в несколько дней закрывается одним запуском.
    """
    from btcproc.pipeline.live import run_live

    specs = _resolve(symbol, all_symbols)
    if model_run is not None and len(specs) > 1:
        raise typer.BadParameter(
            "--model-run задаёт одну конкретную модель и несовместим с несколькими "
            "монетами: модель состояний применима только к своему инструменту."
        )

    def action(spec: symbols.SymbolSpec) -> dict:
        _guard_active_run(force, spec.ticker, skip_if_busy)
        return run_live(
            symbol=spec.ticker,
            model_run_id=model_run,
            lookback_minutes=lookback,
            max_lookback_days=max_lookback_days,
            do_emit=emit,
        )

    _run_for_each(specs, action, "live")


@app.command()
def emit(
    run: int = typer.Option(..., help="run_id, чьих кандидатов отправляем"),
    limit: int = typer.Option(None, help="Ограничить число кандидатов"),
) -> None:
    """
    Дослать кандидатов прогона в btc-graph.

    Отправляются только те, у кого ещё нет отметки об отправке, — повторный
    запуск безопасен и ничего не дублирует. Монета берётся из самого прогона.
    """
    from btcproc.pipeline.train import emit_pending

    # Без этой проверки опечатка в номере прогона выглядела как «всё
    # отправлено»: пустая выборка и нули в ответе.
    row = runs_repo.get_run(run)
    if row is None:
        raise typer.BadParameter(
            f"Прогона #{run} нет. Список: python3 -m btcproc.cli status"
        )
    typer.echo(f"Прогон #{run} ({row.get('symbol') or '?'})")

    # Исторический бэкфилл разбирается часами. Без прогресса команда
    # неотличима от зависшей — и выглядит зависшей ровно в тот момент,
    # когда работает штатно.
    started = time.time()

    def progress(done: int, total: int) -> None:
        if total == 0:
            typer.echo("  выбираю очередь…")
            return
        elapsed = time.time() - started
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        typer.echo(
            f"  {done}/{total} ({done / total:.0%}) · {rate * 60:.0f}/мин · "
            f"осталось ~{eta / 60:.0f} мин",
        )

    typer.echo(runs_repo.dumps(emit_pending(run, limit=limit, on_batch=progress)))


@app.command()
def status(
    symbol: list[str] = typer.Option(None, "--symbol", help="Только эти монеты"),
    all_symbols: bool = typer.Option(
        True, "--all/--default-only",
        help="Все активные монеты (по умолчанию) или только монета из .env",
    ),
) -> None:
    """
    Что есть в базе: покрытие истории, отставание, кандидаты, приёмник.

    Строка на монету. Диагностика окружения общая — печатается один раз.
    """
    from datetime import datetime, timezone

    from btcproc.db.session import fetch_all
    from btcproc.ingest import bars
    from btcproc.sink import graph_sink

    specs = _resolve(symbol, all_symbols and not symbol)
    tickers = [spec.ticker for spec in specs]
    now = datetime.now(timezone.utc)

    # Одним запросом на всё, а не N: страница status не должна линейно
    # тормозить от числа монет.
    coverage = {
        row["symbol"]: row
        for row in bars.coverage_all(tickers, tf=config.data.base_tf)
    }
    cand_rows = {
        row["symbol"]: row
        for row in fetch_all(
            "SELECT symbol, count(*) AS total, "
            "count(*) FILTER (WHERE emitted_at IS NULL) AS pending, "
            "max(ts) AS last_ts FROM candidates WHERE symbol = ANY(%s) GROUP BY symbol",
            (tickers,),
        )
    }

    header = (
        f"{'монета':<10} {'баров':>9} {'история':>23} {'отставание':>11} "
        f"{'кандидатов':>11} {'в очереди':>10}  последний train"
    )
    typer.echo(header)
    typer.echo("─" * len(header))

    for spec in specs:
        cov = coverage.get(spec.ticker)
        cand = cand_rows.get(spec.ticker, {})
        last_train = runs_repo.latest_completed_run("train", spec.ticker)

        if cov:
            period = f"{cov['first_ts']:%Y-%m-%d}…{cov['last_ts']:%Y-%m-%d}"
            lag = f"{(now - cov['last_ts']).total_seconds() / 60:.0f} мин"
            bars = f"{cov['bars']:,}".replace(",", " ")
        else:
            period, lag, bars = "нет данных", "—", "0"

        train_note = (
            f"#{last_train['run_id']} {last_train['finished_at']:%Y-%m-%d %H:%M}"
            if last_train and last_train.get("finished_at")
            else typer.style("нет модели", fg=typer.colors.YELLOW)
        )

        typer.echo(
            f"{spec.ticker:<10} {bars:>9} {period:>23} {lag:>11} "
            f"{cand.get('total', 0):>11} {cand.get('pending', 0):>10}  {train_note}"
        )

    disabled = [s.ticker for s in symbols.SYMBOLS if not s.enabled]
    if disabled and all_symbols:
        typer.echo(f"\nВыключены в реестре: {', '.join(disabled)}")

    # ── Общая диагностика: одна на все монеты ────────────────────────────────
    typer.echo("\nПоследние прогоны:")
    for row in runs_repo.list_runs(5):
        line = (
            f"  #{row['run_id']} {row.get('symbol') or '?':<9} {row['kind']:<6} "
            f"{row['status']:<8} {row['progress']:.0%} {row['stage'] or ''}"
        )
        # Молчащий running — почти наверняка убитый процесс. Пока он числится
        # идущим, крон с --skip-if-busy пропускает live этой монеты молча,
        # поэтому в status это единственное место, где симптом виден заранее.
        if runs_repo.is_stale(row):
            line += "  ← нет heartbeat, прогон скорее всего мёртв"
            line = typer.style(line, fg=typer.colors.YELLOW)
        typer.echo(line)

    sink = graph_sink.sink_status()
    typer.echo(f"\nПриёмник ({sink['mode']}): "
               f"{'OK' if sink['ok'] else 'ПРОБЛЕМА'} — {sink['detail']}")

    # Расхождение окружений (venv против системного python) обнаруживается
    # иначе только по трейсбеку в середине прогона.
    import sys

    typer.echo(f"Интерпретатор: {sys.executable}")
    missing = graph_sink.missing_direct_dependencies()
    if missing and config.sink.mode == "direct":
        typer.echo(
            typer.style(
                "  ВНИМАНИЕ: нет " + ", ".join(missing)
                + " — оценки не будут сохраняться в PostgreSQL и Neo4j.\n"
                f"  Лечится: {sys.executable} -m pip install -r requirements.txt",
                fg=typer.colors.YELLOW,
            )
        )


@app.command("fetch-external")
def fetch_external(
    series: str = typer.Option("fear_greed", help="Какой внешний ряд качать"),
) -> None:
    """
    Скачать внешний дневной ряд (Fear & Greed Index и т.п.) в external_daily.

    Отдельная команда намеренно: сетевой поход в чужой API не должен идти
    внутри train/live и рисковать их регулярным расписанием
    (docs/task_fear_greed.md, раздел 8). Гоняется отдельно, обычно раз в сутки.
    """
    from btcproc.ingest import external

    if series != "fear_greed":
        raise typer.BadParameter(f"Неизвестный ряд {series!r}. Известен: fear_greed.")

    result = external.sync_fear_greed()
    if result["rows"] == 0:
        typer.echo("Поставщик не вернул строк.")
        raise typer.Exit(code=1)

    typer.echo(f"{series}: сохранено {result['rows']} суток ({result.get('span', '—')})")
    if result["revised_days"]:
        typer.echo(
            typer.style(
                f"  ВНИМАНИЕ: {result['revised_days']} суток разошлись со "
                "старым снимком — поставщик пересчитал историю задним числом. "
                "Запиши в development_log.md.",
                fg=typer.colors.YELLOW,
            )
        )


@app.command("notify-example")
def notify_example(
    mode: str = typer.Option("full", help="Формат payload: full | compact"),
) -> None:
    """
    Напечатать пример тела уведомления.

    Существует ради чужой системы: её автору нужен образец JSON, а не описание
    полей словами. Пример собирается тем же кодом, что и боевое тело, поэтому
    протухнуть молча не может — в отличие от куска JSON, выписанного в
    документацию руками.
    """
    import json

    from btcproc.notify import payload as payload_mod
    from btcproc.notify import rules as rules_mod

    if mode not in rules_mod.PAYLOAD_MODES:
        raise typer.BadParameter(
            f"допустимы {', '.join(rules_mod.PAYLOAD_MODES)}"
        )
    rule = rules_mod.Rule(rule_id=1, name="пример", url="https://example.com/hook",
                          payload_mode=mode)
    body = payload_mod.build(rule, payload_mod.example_row())
    typer.echo(json.dumps(body, ensure_ascii=False, indent=2))


@app.command("notify-test")
def notify_test(
    rule: int = typer.Option(..., "--rule", help="rule_id правила из админки"),
) -> None:
    """
    Отправить на адрес правила последнего кандидата — проверка связи.

    Идёт мимо очереди, мимо фильтров и мимо журнала доставок: это ручная
    проверка адреса, а не событие. Ответ печатается — здесь ожидание и есть
    смысл команды.
    """
    from btcproc.admin import queries
    from btcproc.notify import service as notify_service
    from btcproc.notify import payload as payload_mod
    from btcproc.notify import rules as rules_mod

    spec = rules_mod.get_rule(rule)
    if spec is None:
        raise typer.BadParameter(f"Правила #{rule} нет")
    row = queries.latest_candidate_row(spec.symbol) or payload_mod.example_row()
    result = notify_service.send_one(spec, row)
    typer.echo(runs_repo.dumps(result))
    if result["status"] != "sent":
        raise typer.Exit(code=1)


@app.command()
def admin(
    host: str = typer.Option(None),
    port: int = typer.Option(None),
    reload: bool = typer.Option(False, help="Автоперезагрузка при правке кода"),
) -> None:
    """Запустить админку."""
    import uvicorn

    config.admin.validate()
    symbols.validate_default()
    uvicorn.run(
        "btcproc.admin.app:app",
        host=host or config.admin.host,
        port=port or config.admin.port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
