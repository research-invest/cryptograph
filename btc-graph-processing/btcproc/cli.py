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
        typer.echo(f"Нет дампов за: {', '.join(result['missing_months'])}")
    if context:
        typer.echo(f"Старшие ТФ: {binance.rebuild_context_timeframes(symbol)}")


@app.command()
def train(
    symbol: str = typer.Option(None),
    start: str = typer.Option(None, help="Дата начала истории"),
    end: str = typer.Option(None, help="Дата конца (для воспроизводимых прогонов)"),
    ingest_data: bool = typer.Option(True, "--ingest/--no-ingest", help="Качать историю"),
    emit: bool = typer.Option(True, "--emit/--no-emit", help="Слать кандидатов в btc-graph"),
) -> None:
    """Полный прогон истории: от баров до кандидатов в btc-graph."""
    from btcproc.pipeline.train import run_train

    stats = run_train(
        symbol=symbol, start=start, end=end, do_ingest=ingest_data, do_emit=emit
    )
    typer.echo(runs_repo.dumps(stats))


@app.command()
def live(
    model_run: int = typer.Option(None, help="run_id прогона с нужной моделью"),
    lookback: int = typer.Option(240, help="Минут назад считать кандидатов свежими"),
    emit: bool = typer.Option(True, "--emit/--no-emit"),
) -> None:
    """Инкрементальный прогон по свежим барам."""
    from btcproc.pipeline.live import run_live

    stats = run_live(model_run_id=model_run, lookback_minutes=lookback, do_emit=emit)
    typer.echo(runs_repo.dumps(stats))


@app.command()
def emit(
    run: int = typer.Option(..., help="run_id, чьих кандидатов отправляем"),
    limit: int = typer.Option(None, help="Ограничить число кандидатов"),
) -> None:
    """Дослать кандидатов прогона в btc-graph."""
    from btcproc.pipeline.train import emit_pending

    typer.echo(runs_repo.dumps(emit_pending(run, limit=limit)))


@app.command()
def status() -> None:
    """Что есть в базе: покрытие истории, прогоны, приёмник кандидатов."""
    from btcproc.ingest import binance
    from btcproc.sink import graph_sink

    typer.echo("Покрытие истории:")
    for row in binance.coverage():
        typer.echo(
            f"  {row['tf']:>4}: {row['bars']:>8} баров  "
            f"{row['first_ts']:%Y-%m-%d} … {row['last_ts']:%Y-%m-%d %H:%M}"
        )

    typer.echo("\nПоследние прогоны:")
    for row in runs_repo.list_runs(5):
        typer.echo(
            f"  #{row['run_id']} {row['kind']:<6} {row['status']:<8} "
            f"{row['progress']:.0%} {row['stage'] or ''}"
        )

    sink = graph_sink.sink_status()
    typer.echo(f"\nПриёмник ({sink['mode']}): "
               f"{'OK' if sink['ok'] else 'НЕДОСТУПЕН'} — {sink['detail']}")


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
