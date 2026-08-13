"""
Реестр источников сигнала: SMC, ончейн, индексы, деривативы.

## Зачем

До 2026-08-11 источник был зашит по имени в трёх местах: `if
config.smc.features_on` в `build_features`, `_attach_smc` в `detect_atoms` и
двоичная `feature_version()`. Для одного источника это терпимо, для второго —
уже нет, и ломается оно молча.

Показательна именно версия. Схема «есть SMC → v2, иначе v1» вторым источником
по тому же рецепту превращается в

    if config.smc.features_on:     return "v2"
    if config.onchain.features_on: return "v3"
    return "v1"

и набор «база + SMC + ончейн» (56 признаков) получает метку `"v2"` — ту же,
что «база + SMC» (44). Дальше `save_feature_set` перезаписывает
`feature_sets.names` по первичному ключу, старые строки `features` остаются
массивами на 44 значения, а `load_features` строит из них DataFrame с 56
именами колонок. Ни ошибки, ни предупреждения — просто с какого-то момента
признаки читаются не под теми именами.

Реестр закрывает это тем, что версия собирается ИЗ СОСТАВА:
`"v1"`, `"v1+smc"`, `"v1+onchain"`, `"v1+onchain+smc"`. Композируется на любое
число источников, читается глазами, и база без источников остаётся `"v1"` —
накопленные строки сохраняют идентичность.

## Чего реестр НЕ делает: он не владеет порядком атомов

Порядок ключей `ATOM_FAMILY` задаёт номера битов в маске, то есть
**идентичность всех исторических `event_block_id`**. Если собирать
`ATOM_FAMILY` динамически из реестра, порядок начнёт зависеть от порядка
регистрации источников и от того, какие из них включены, — и биты поедут при
каждом изменении конфигурации, без единой ошибки.

Поэтому граница проведена так: **`ATOM_FAMILY` остаётся единственным
упорядоченным источником правды, с добавлением только в конец.** Источник
поставляет ДЕТЕКТОРЫ (функцию расчёта и список имён), а не позиции в словаре.
Закреплено тестом `test_atom_bit_order_is_pinned`.

## Как завести новый источник

1. модуль `btcproc/features/<источник>.py` с расчётом (образец — `smc.py`);
2. свой раздел в `config.py` с двумя флагами: атомы дёшевы, признаки дороги;
3. `FeatureSource` ниже — имя, флаги, списки колонок, функция расчёта;
4. атомы **в конец** `ATOM_FAMILY` в `events.py` (+ `CONTEXT_ATOMS`);
5. формулировки: `naming.AXES` для признаков, `naming.ATOM_LABELS` для атомов.

Шаг 5 не опциональный и стережётся тестами полноты — см.
`docs/extending_features.md`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from btcproc import config


@dataclass(frozen=True)
class FeatureSource:
    """
    Один источник сигнала: что он даёт, когда включён и как считается.

    `features_enabled` и `atoms_enabled` — функции, а не значения: конфиг
    читается в момент вызова. Значение, вычисленное на импорте модуля,
    заморозило бы состояние флага навсегда, и подмена `config.smc` в тестах
    и в скриптах замера перестала бы работать.

    `compute` возвращает DataFrame с индексом входных баров и всеми
    колонками источника — и признаками, и атомами сразу. Разделять расчёт
    незачем: он один проход, а кому какие колонки нужны, решают вызывающие.
    """

    name: str
    features_enabled: Callable[[], bool]
    atoms_enabled: Callable[[], bool]
    feature_columns: tuple[str, ...]
    atom_columns: tuple[str, ...]
    compute: Callable[[pd.DataFrame], pd.DataFrame]

    @property
    def columns(self) -> tuple[str, ...]:
        return self.feature_columns + self.atom_columns


def _smc_source() -> FeatureSource:
    """
    Smart Money — первый и пока единственный источник.

    Импорт модуля расчёта ленивый: `smc.py` тянет тяжёлый проход, а реестр
    импортируется отовсюду, включая `--help` командной строки.
    """
    from btcproc.features import events, smc

    def compute(base: pd.DataFrame) -> pd.DataFrame:
        return smc.build_smc_cached(base)

    return FeatureSource(
        name="smc",
        features_enabled=lambda: config.smc.features_on,
        atoms_enabled=lambda: config.smc.enabled,
        feature_columns=tuple(smc.FEATURE_CANDIDATES),
        # Список атомов берётся из ATOM_FAMILY, а не из smc.BOOLEAN_COLUMNS:
        # часть детекторов замер отбраковал (eqh_above/eql_below вырождены,
        # structure_bearish — точное отрицание structure_bullish), и в
        # систему они не заводились. Источник объявляет то, что реально
        # доехало до ATOM_FAMILY.
        atom_columns=tuple(events.SMC_ATOMS),
        compute=compute,
    )


def sources() -> tuple[FeatureSource, ...]:
    """
    Все зарегистрированные источники, включённые и нет.

    Функция, а не константа модуля: конструкторы источников читают списки
    колонок из своих модулей, и вычислять их на импорте `registry` значило бы
    тянуть `smc.py` (а завтра — ончейн-клиент) при любом обращении к реестру.
    """
    return (_smc_source(),)


def enabled_feature_sources() -> list[FeatureSource]:
    """Источники, чьи признаки входят в вектор прямо сейчас."""
    return [s for s in sources() if s.features_enabled()]


def enabled_atom_sources() -> list[FeatureSource]:
    """Источники, чьи атомы считаются прямо сейчас."""
    return [s for s in sources() if s.atoms_enabled()]


def compose_version(base: str, active: list[FeatureSource]) -> str:
    """
    Метка набора из состава: `"v1"`, `"v1+smc"`, `"v1+onchain+smc"`.

    Имена сортируются, чтобы порядок объявления источников в реестре не влиял
    на метку: перестановка строк в `sources()` не должна порождать новую
    версию набора при том же составе.
    """
    return base + "".join(f"+{name}" for name in sorted(s.name for s in active))


def all_feature_columns() -> tuple[str, ...]:
    """Признаки ВСЕХ источников, независимо от флагов — для тестов полноты."""
    return tuple(c for s in sources() for c in s.feature_columns)


def all_atom_columns() -> tuple[str, ...]:
    """Атомы ВСЕХ источников, независимо от флагов — для тестов полноты."""
    return tuple(c for s in sources() for c in s.atom_columns)
