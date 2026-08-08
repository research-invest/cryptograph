"""
Командная строка btc-graph-processing.

    python -m btcproc.cli init-db
    python -m btcproc.cli ingest
    python -m btcproc.cli train --no-emit
    python -m btcproc.cli live
    python -m btcproc.cli admin
"""
from __future__ import annotations

import logging

import typer

from btcproc import config
from btcproc.db import runs as runs_repo
from btcproc.db import session

app = typer.Typer(add_completion=False, help="Генератор кандидатов BTC для btc-graph")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@app.command("init-db")
def init_db() -> None:
    """Создать схему processing и таблицы."""
    session.init_schema()
    typer.echo(f"Схема {config.db.schema} готова")


@app.command()
def ingest(
    symbol: str = typer.Option(None, help="Тикер, по умолчанию из .env"),
    start: str = typer.Option(None, help="Дата начала, YYYY-MM-DD"),
    context: bool = typer.Option(True, help="Пересобрать старшие ТФ из базового"),
) -> None:
    """Скачать историю Binance в БД."""
    from btcproc.ingest import binance

    result = binance.sync_history(
        symbol, config.data.base_tf, start,
        progress=lambda i, n, msg: typer.echo(f"  [{i}/{n}] {msg}"),
    )
    typer.echo(f"Баров записано: {result['rows']}")
    if result["missing_months"]:
        # Текущий месяц в этом списке — норма: месячный архив публикуется
        # только после его закрытия, до тех пор работают дневные дампы и REST.
        typer.echo(
            f"Нет месячных дампов за: {', '.join(result['missing_months'])} "
            "(текущий месяц добирается дневными архивами и REST — это ожидаемо)"
        )
    if context:
        typer.echo(f"Старшие ТФ: {binance.rebuild_context_timeframes(symbol)}")


def _guard_active_run(force: bool) -> None:
    """
    Два тяжёлых прогона одновременно — это гонка за одни и те же таблицы
    и вдвое большая нагрузка на БД. Админка такое блокирует, CLI до сих пор
    молчал: если live висит в cron, ручной train легко попадал на него.
    """
    active = runs_repo.active_run()
    if active and not force:
        raise typer.BadParameter(
            f"Уже идёт прогон #{active['run_id']} ({active['kind']}, "
            f"стадия {active['stage'] or '—'}, {active['progress']:.0%}). "
            "Дождись окончания или запусти с --force."
        )


@app.command()
def train(
    symbol: str = typer.Option(None),
    start: str = typer.Option(None, help="Дата начала истории"),
    end: str = typer.Option(None, help="Дата конца (для воспроизводимых прогонов)"),
    ingest_data: bool = typer.Option(True, "--ingest/--no-ingest", help="Качать историю"),
    emit: bool = typer.Option(True, "--emit/--no-emit", help="Слать кандидатов в btc-graph"),
    force: bool = typer.Option(False, "--force", help="Запустить, даже если идёт другой прогон"),
) -> None:
    """
    Полный прогон истории: от баров до кандидатов в btc-graph.

    ВНИМАНИЕ: состояния переобучаются с нуля, поэтому номера group_id
    в новом прогоне не соответствуют старым. Для регулярного обновления
    нужен live, а не train.
    """
    from btcproc.pipeline.train import run_train

    _guard_active_run(force)
    typer.echo(
        "Модель состояний будет переобучена: group_id нового прогона "
        "несопоставимы с предыдущими.\n"
    )
    stats = run_train(
        symbol=symbol, start=start, end=end, do_ingest=ingest_data, do_emit=emit
    )
    typer.echo(runs_repo.dumps(stats))


@app.command()
def live(
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
) -> None:
    """
    Инкрементальный прогон по свежим барам.

    Точка продолжения берётся из данных: бары качаются с последнего
    сохранённого, кандидаты — с последнего выпущенного. Пропуск в несколько
    дней закрывается одним запуском.
    """
    from btcproc.pipeline.live import run_live

    _guard_active_run(force)
    stats = run_live(
        model_run_id=model_run,
        lookback_minutes=lookback,
        max_lookback_days=max_lookback_days,
        do_emit=emit,
    )
    typer.echo(runs_repo.dumps(stats))


@app.command()
def emit(
    run: int = typer.Option(..., help="run_id, чьих кандидатов отправляем"),
    limit: int = typer.Option(None, help="Ограничить число кандидатов"),
) -> None:
    """
    Дослать кандидатов прогона в btc-graph.

    Отправляются только те, у кого ещё нет отметки об отправке, — повторный
    запуск безопасен и ничего не дублирует.
    """
    from btcproc.pipeline.train import emit_pending

    # Без этой проверки опечатка в номере прогона выглядела как «всё
    # отправлено»: пустая выборка и нули в ответе.
    if runs_repo.get_run(run) is None:
        raise typer.BadParameter(
            f"Прогона #{run} нет. Список: python3 -m btcproc.cli status"
        )
    typer.echo(runs_repo.dumps(emit_pending(run, limit=limit)))


@app.command()
def status() -> None:
    """Что есть в базе: покрытие истории, отставание, прогоны, приёмник."""
    from datetime import datetime, timezone

    from btcproc.db.session import fetch_one
    from btcproc.ingest import binance
    from btcproc.sink import graph_sink

    typer.echo("Покрытие истории:")
    for row in binance.coverage():
        typer.echo(
            f"  {row['tf']:>4}: {row['bars']:>8} баров  "
            f"{row['first_ts']:%Y-%m-%d} … {row['last_ts']:%Y-%m-%d %H:%M}"
        )

    # Главный вопрос к системе в работе — не «сколько всего», а «насколько
    # отстали»: свежесть баров и очередь неотправленных кандидатов.
    last_bar = binance.last_ts()
    if last_bar:
        lag_min = (datetime.now(timezone.utc) - last_bar).total_seconds() / 60
        typer.echo(f"\nОтставание баров: {lag_min:.0f} мин от текущего момента")

    stats = fetch_one(
        "SELECT count(*) AS total, "
        "count(*) FILTER (WHERE emitted_at IS NULL) AS pending, "
        "max(ts) AS last_ts FROM candidates WHERE symbol = %s",
        (config.data.symbol,),
    ) or {}
    if stats.get("total"):
        typer.echo(
            f"Кандидатов: {stats['total']}, в очереди отправки: {stats['pending']}"
        )
        if stats.get("last_ts"):
            gap_h = (last_bar - stats["last_ts"]).total_seconds() / 3600 if last_bar else 0
            typer.echo(
                f"Последний кандидат: {stats['last_ts']:%Y-%m-%d %H:%M} "
                f"(следующий live закроет {gap_h:.1f} ч)"
            )
    else:
        typer.echo("\nКандидатов ещё нет — нужен полный прогон (train)")

    typer.echo("\nПоследние прогоны:")
    for row in runs_repo.list_runs(5):
        typer.echo(
            f"  #{row['run_id']} {row['kind']:<6} {row['status']:<8} "
            f"{row['progress']:.0%} {row['stage'] or ''}"
        )

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


@app.command()
def admin(
    host: str = typer.Option(None),
    port: int = typer.Option(None),
    reload: bool = typer.Option(False, help="Автоперезагрузка при правке кода"),
) -> None:
    """Запустить админку."""
    import uvicorn

    config.admin.validate()
    uvicorn.run(
        "btcproc.admin.app:app",
        host=host or config.admin.host,
        port=port or config.admin.port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
