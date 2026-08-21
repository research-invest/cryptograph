"""
Сторожа документации.

Журнал решений — не приложение к коду, а способ его читать: и `CLAUDE.md`
обоих проектов, и ТЗ, и комментарии в модулях ссылаются на разделы по номеру.
Номер, встретившийся дважды, ломает эти ссылки молча — «раздел 48» перестаёт
что-либо адресовать, и понять это можно только открыв файл. Ровно это и
случилось 2026-08-19: два раздела получили номер 48, и продержались до
2026-08-21 (правка — `docs/plan_geometry_xs_2026-08-21.md`, шаг 0).

Класс сторожа тот же, что у тестов полноты словарей `naming`: проверяется не
поведение кода, а то, что описание остаётся адресуемым.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

LOG = Path(__file__).resolve().parents[1] / "docs" / "development_log.md"

#: «## 47. Заголовок (дата)» — номер раздела верхнего уровня.
SECTION = re.compile(r"^## (\d+)\.", re.M)
#: «### 47.4. Заголовок» — номер подраздела.
SUBSECTION = re.compile(r"^### (\d+)\.(\d+)", re.M)


def _log_text() -> str:
    if not LOG.exists():  # журнал живёт в репозитории, но тест не должен падать в срезе
        pytest.skip(f"нет {LOG}")
    return LOG.read_text(encoding="utf-8")


def test_development_log_sections_are_unique():
    """
    Номера разделов журнала не повторяются.

    Повтор не ломает ни один прогон — он ломает ссылку, и обнаруживается
    только чтением. Поэтому проверка автоматическая.
    """
    numbers = [int(m) for m in SECTION.findall(_log_text())]
    duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
    assert not duplicates, f"номера разделов встречаются дважды: {duplicates}"


def test_development_log_sections_are_consecutive():
    """
    Разделы идут подряд и по возрастанию.

    Дыра в нумерации так же обесценивает ссылку, как и повтор: «раздел 50»
    при отсутствующем 50 читается как опечатка, и найти, что имелось в виду,
    нельзя. Проверка заодно ловит перенумерацию, сделанную наполовину.
    """
    numbers = [int(m) for m in SECTION.findall(_log_text())]
    assert numbers, "в журнале нет ни одного раздела верхнего уровня"
    expected = list(range(numbers[0], numbers[0] + len(numbers)))
    assert numbers == expected, (
        f"нумерация разрывна: {[n for n, e in zip(numbers, expected) if n != e][:5]}"
    )


def test_subsections_belong_to_their_section():
    """
    Подраздел `### N.x` стоит внутри раздела `## N`.

    Это то, что осталось незамеченным при перенумерации 2026-08-19: заголовки
    второго раздела «48» были правильными сами по себе, но принадлежали
    чужому номеру. Проверка держит подразделы привязанными к своему разделу и
    ловит перенумерацию, забывшую про них.
    """
    current: int | None = None
    wrong: list[str] = []
    for line in _log_text().split("\n"):
        section = SECTION.match(line)
        if section:
            current = int(section.group(1))
            continue
        subsection = SUBSECTION.match(line)
        if subsection and current is not None and int(subsection.group(1)) != current:
            wrong.append(line.strip())
    assert not wrong, f"подразделы под чужим номером: {wrong[:5]}"
