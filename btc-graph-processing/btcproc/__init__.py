"""
Пакет генератора кандидатов.

Здесь же — единственная проверка версии интерпретатора. Код пользуется
синтаксисом `str | None` в аннотациях, и на 3.9 это ломается НЕ на импорте:
`from __future__ import annotations` откладывает разбор, а падает потом
FastAPI/pydantic, когда вычисляет аннотацию обработчика, — сорока строками
трейсбека внутри чужой библиотеки, из которых версия интерпретатора никак не
следует. Проверка стоит в `__init__`, а не в CLI, потому что скрипты из
`scripts/` импортируют пакет напрямую, минуя `btcproc.cli`.

Ловушка, из-за которой это вообще случается: `make admin` берёт `PY ?= python3`,
а `python3` на macOS без venv — системный 3.9 из CommandLineTools.
"""
import sys

if sys.version_info < (3, 10):
    raise RuntimeError(
        "btcproc требует Python 3.10 или новее (у тебя "
        f"{sys.version_info.major}.{sys.version_info.minor} — {sys.executable}).\n"
        "Синтаксис аннотаций `str | None` появился в 3.10; на 3.9 это падает не "
        "здесь, а внутри FastAPI при разборе обработчика.\n"
        "Как запускать: `make admin PY=.venv/bin/python` (или любой другой "
        "интерпретатор 3.10+); венв заводится `python3.12 -m venv .venv && "
        "make install PY=.venv/bin/python`."
    )
