"""
Реестр источников сигнала: композиция версии, полнота описания, порядок битов.

Проверяется не то, что реестр «работает», а то, что он не может сломать
накопленное. Три свойства из задачи 11-101, §4:

* метка набора собирается из состава и на двух источниках не схлопывается;
* источник, приехавший наполовину описанным, валит сюиту, а не всплывает
  через недели пустой подписью состояния;
* реестр НЕ владеет порядком атомов — иначе биты маски поедут при каждом
  изменении конфигурации.
"""
from __future__ import annotations

import hashlib

import pytest

from btcproc import config
from btcproc.features import builder, events, registry
from btcproc.states import naming


# ── Композиция версии ───────────────────────────────────────────────────────
def _fake(name: str, features: bool = True, atoms: bool = True):
    return registry.FeatureSource(
        name=name,
        features_enabled=lambda: features,
        atoms_enabled=lambda: atoms,
        feature_columns=(f"{name}_f",),
        atom_columns=(f"{name}_a",),
        compute=lambda base: base,
    )


def test_version_composes_from_active_sources():
    compose = registry.compose_version
    assert compose("v1", []) == "v1"
    assert compose("v1", [_fake("smc")]) == "v1+smc"
    assert compose("v1", [_fake("smc"), _fake("onchain")]) == "v1+onchain+smc"


def test_version_is_stable_against_declaration_order():
    """
    Перестановка источников в реестре не должна порождать новую версию при
    том же составе: иначе безобидная правка порядка строк потребовала бы
    миграции `features`.
    """
    a, b = _fake("indices"), _fake("onchain")
    assert registry.compose_version("v1", [a, b]) == registry.compose_version("v1", [b, a])


def test_two_sources_never_share_a_label():
    """
    Ровно та поломка, ради которой заводился реестр.

    Двоичная схема давала «база + SMC» и «база + SMC + ончейн» одну метку
    "v2". Дальше save_feature_set перезаписывал feature_sets.names по PK, а
    старые строки features оставались массивами прежней длины — и читались
    уже под чужими именами, без единой ошибки.
    """
    smc = _fake("smc")
    onchain = _fake("onchain")
    labels = {
        registry.compose_version("v1", []),
        registry.compose_version("v1", [smc]),
        registry.compose_version("v1", [onchain]),
        registry.compose_version("v1", [smc, onchain]),
    }
    assert len(labels) == 4, "два разных состава получили одну метку"


def test_feature_version_reflects_the_flags(monkeypatch):
    # Каждый ЧУЖОЙ источник обязан быть выключен явно, иначе метка соберётся
    # из состава, зависящего от окружения машины: с 2026-08-14 на боевом VPS
    # стоит FGI_ENABLED=true, и «v1» превращалось в «v1+fgi» — тест зеленел
    # локально и падал при выкатке. Заводя третий источник, добавь его сюда.
    monkeypatch.setattr(config, "fgi",
                        config.FearGreedConfig(enabled=False, features_enabled=False))
    monkeypatch.setattr(config, "smc",
                        config.SMCConfig(enabled=False, features_enabled=False))
    assert builder.feature_version() == "v1"
    assert events.event_version() == "v1"

    # Атомы включены, признаки нет: наборы независимы, и метки тоже.
    monkeypatch.setattr(config, "smc",
                        config.SMCConfig(enabled=True, features_enabled=False))
    assert builder.feature_version() == "v1"
    assert events.event_version() == "v1+smc"

    monkeypatch.setattr(config, "smc",
                        config.SMCConfig(enabled=True, features_enabled=True))
    assert builder.feature_version() == "v1+smc"
    assert events.event_version() == "v1+smc"


def test_flags_are_read_at_call_time(monkeypatch):
    """
    Флаг источника — функция, а не значение, вычисленное на импорте.

    Значение заморозило бы состояние флага навсегда, и подмена config.smc
    в тестах и в скриптах замера перестала бы работать — молча, потому что
    расчёт просто пошёл бы по старой ветке.
    """
    monkeypatch.setattr(config, "smc", config.SMCConfig(enabled=False))
    source = next(s for s in registry.sources() if s.name == "smc")
    assert not source.atoms_enabled()

    monkeypatch.setattr(config, "smc", config.SMCConfig(enabled=True))
    assert source.atoms_enabled()


# ── Полнота описания источника ──────────────────────────────────────────────
def test_every_registered_atom_is_declared_in_atom_family():
    """
    Источник не может объявить атом, которого нет в ATOM_FAMILY: у такого
    атома нет ни семейства, ни бита, ни места в выдаче.
    """
    unknown = set(registry.all_atom_columns()) - set(events.ATOM_FAMILY)
    assert not unknown, f"атомы источников вне ATOM_FAMILY: {sorted(unknown)}"


def test_every_registered_atom_has_a_russian_label():
    """
    Атом без подписи уходит в интерфейс сырым идентификатором. Тест
    реестровый, а не по ATOM_FAMILY: новый источник обязан покрываться
    автоматически, без правки списка в тесте.
    """
    uncovered = set(registry.all_atom_columns()) - set(naming.ATOM_LABELS)
    assert not uncovered, (
        f"атомы источников без русской подписи: {sorted(uncovered)}. "
        "Добавь их в naming.ATOM_LABELS."
    )


def test_every_registered_feature_has_a_phrase():
    """
    Признак без формулировки не даёт ни ошибки, ни предупреждения — он просто
    выпадает из подписи, и состояние, выделяющееся именно им, называется
    «рынок около среднего». То есть подпись начинает врать.
    """
    vocabulary = {f for _axis, mapping in naming.AXES for f in mapping}
    uncovered = set(registry.all_feature_columns()) - vocabulary - naming._UNNAMED
    assert not uncovered, (
        f"признаки источников без формулировки: {sorted(uncovered)}. "
        "Добавь их в naming.AXES или, осознанно, в naming._UNNAMED."
    )


def test_source_columns_do_not_collide():
    """Два источника не могут дать колонку с одним именем — иначе join затрёт."""
    seen: dict[str, str] = {}
    for source in registry.sources():
        for column in source.columns:
            assert column not in seen, (
                f"колонка {column} объявлена и в {seen[column]}, и в {source.name}"
            )
            seen[column] = source.name


# ── Порядок битов реестру не принадлежит ────────────────────────────────────
#
# Хэш от полного списка (атом, бит). Меняется он ровно тогда, когда меняется
# порядок или состав SIGNATURE_ATOMS, — то есть когда меняется смысл всех
# исторических event_block_id. Значение обновляется только вместе с
# осознанным решением пересобрать разметку.
ATOM_BIT_FINGERPRINT = "1cccfacbd316c5b29ca0bb05a0997653"


def _bit_fingerprint() -> str:
    payload = ";".join(f"{atom}={bit}" for atom, bit in events.ATOM_BIT.items())
    return hashlib.md5(payload.encode()).hexdigest()


def test_atom_bit_order_is_pinned():
    """
    Порядок битов маски прибит хэшем.

    Реестр поставляет ДЕТЕКТОРЫ, а не позиции: если бы ATOM_FAMILY собирался
    из реестра динамически, порядок зависел бы от порядка регистрации
    источников и от того, какие включены, — и биты ехали бы при каждом
    изменении конфигурации, без единой ошибки. Этот тест делает такую
    перестройку невозможной незаметно.
    """
    assert _bit_fingerprint() == ATOM_BIT_FINGERPRINT, (
        "изменился порядок или состав SIGNATURE_ATOMS — это меняет смысл ВСЕХ "
        "исторических event_block_id. Если так и задумано, обнови "
        "ATOM_BIT_FINGERPRINT вместе с решением пересобрать разметку."
    )


def test_atom_order_does_not_depend_on_source_flags(monkeypatch):
    """Состав включённых источников на нумерацию битов не влияет вообще."""
    monkeypatch.setattr(config, "smc", config.SMCConfig(enabled=False))
    off = _bit_fingerprint()
    monkeypatch.setattr(config, "smc",
                        config.SMCConfig(enabled=True, features_enabled=True))
    assert _bit_fingerprint() == off


@pytest.mark.parametrize("column", ["mask", "context_mask"])
def test_internal_columns_do_not_leak(bars, column):
    """Служебные маски остаются внутри build_event_blocks."""
    assert column not in events.build_event_blocks(bars).columns
