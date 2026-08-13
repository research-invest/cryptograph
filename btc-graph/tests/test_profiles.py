"""
Тесты конфиг-слоя профилей: слияние с _default, валидация схемы, устойчивость
загрузчика к битым файлам.

Смысл валидации схемы — машинно ловить то, что раньше ловилось только
внимательным чтением кода: недостижимую ступень лесенки (audit #1), веса,
не дающие в сумме единицу, и непокрытое значение enum.
"""
from __future__ import annotations

import textwrap

import pytest
from pydantic import ValidationError

from src.config import profiles


from conftest import FROZEN_PROFILES  # noqa: E402

DEFAULT_YAML = (FROZEN_PROFILES / "_default.yaml").read_text(encoding="utf-8")


@pytest.fixture
def profiles_dir(tmp_path, monkeypatch):
    """
    Изолированный каталог профилей: в нём лежит настоящий _default,
    а тест дописывает нужные ему монеты.
    """
    directory = tmp_path / "symbols"
    directory.mkdir()
    (directory / "_default.yaml").write_text(DEFAULT_YAML, encoding="utf-8")
    monkeypatch.setenv("PROFILES_DIR", str(directory))
    profiles.reload_profiles()
    yield directory
    # Возвращаем ЗАМОРОЖЕННЫЙ каталог, а не «никакой». Удаление переменной
    # переключало сюиту на боевые профили до конца прогона, и тесты ступеней
    # в файлах после этого падали по чужой причине.
    monkeypatch.setenv("PROFILES_DIR", str(FROZEN_PROFILES))
    profiles.reload_profiles()


def write(directory, name: str, body: str) -> None:
    (directory / f"{name}.yaml").write_text(textwrap.dedent(body), encoding="utf-8")
    profiles.reload_profiles()


# ─── Загрузка и слияние ───────────────────────────────────────────────────────

def test_default_profile_matches_hardcoded_calibration():
    """_default обязан описывать ровно ту калибровку, что была в коде шага 1."""
    p = profiles.default_profile()
    assert p.weights.as_dict() == {
        "statistical": 0.30, "directional": 0.35, "context": 0.20, "rarity": 0.15
    }
    assert p.rating.strong_min == 0.75
    assert p.rating.moderate_min == 0.55
    assert p.statistical.sample_size.thresholds == (1000, 500, 200)
    assert p.context.age_bucket == {
        "age_lt_30": 1.0, "age_30_60": 0.75, "age_60_120": 0.5, "age_gt_120": 0.0
    }


def test_frozen_default_matches_shipped(shipped_profiles):
    """
    Замороженная тестовая копия `_default` не должна разойтись с боевой.

    Тесты механики считают по копии; если боевой `_default` поправят, а копию
    забудут, вся сюита продолжит зеленеть, проверяя вчерашнюю линейку.
    """
    shipped = shipped_profiles.default_profile()
    frozen_text = (FROZEN_PROFILES / "_default.yaml").read_text(encoding="utf-8")
    shipped_text = (
        profiles._REPO_ROOT / "config" / "symbols" / "_default.yaml"
    ).read_text(encoding="utf-8")

    # Сравниваем содержимое без шапки копии — она поясняет, зачем копия нужна.
    assert shipped_text in frozen_text, (
        "tests/fixtures/profiles/_default.yaml разошёлся с config/symbols/_default.yaml. "
        "Обнови копию и пересними снапшот скорера, если правка осознанна."
    )
    assert shipped.fingerprint == profiles.default_profile().fingerprint


def test_btcusdt_has_own_calibration(shipped_profiles):
    """
    С версии 2 BTCUSDT — не пустое наследование, а своя калибровка по выгрузке
    прогона #17. Базовая линейка при этом осталась прежней: она общая мерка
    для quality_score_baseline и меняться не должна.

    Раньше здесь проверялось обратное («BTC равен _default»). Это было верно,
    пока у биткоина не было своего профиля, и заодно делало всю сюиту зависимой
    от его калибровки — см. комментарий про замороженные профили в conftest.
    """
    btc = shipped_profiles.get_profile("BTCUSDT")
    base = shipped_profiles.default_profile()

    assert btc.version >= 2
    assert btc.name != base.name
    assert btc.statistical.sample_size.thresholds != base.statistical.sample_size.thresholds
    # Ось, которой калибровка не касалась, наследуется.
    assert btc.context.age_bucket == base.context.age_bucket


def test_partial_override_merges_with_default(profiles_dir):
    """Профиль монеты переопределяет одну ступень, остальное наследуется."""
    write(profiles_dir, "ETHUSDT", """
        symbol: ETHUSDT
        version: 2
        statistical:
          sample_size: {mode: higher_better, ladder: [[400, 1.0], [200, 0.7], [80, 0.4]], floor: 0.1}
        weights: {statistical: 0.35, directional: 0.30, context: 0.20, rarity: 0.15}
    """)
    eth = profiles.get_profile("ETHUSDT")

    assert eth.statistical.sample_size.thresholds == (400, 200, 80)
    assert eth.weights.statistical == 0.35
    # Не переопределённое пришло из _default целиком.
    assert eth.statistical.valid_label_pct.thresholds == (0.85, 0.80, 0.75)
    assert eth.rating.strong_min == 0.75
    assert eth.validator.very_small_sample_size == 100


def test_ladder_list_is_replaced_not_merged(profiles_dir):
    """
    Лесенка заменяется целиком — «частично унаследованной» лесенки не бывает.
    А mode и floor рядом с ней наследуются, если их не переопределить.
    """
    write(profiles_dir, "SOLUSDT", """
        symbol: SOLUSDT
        version: 1
        directional:
          win_rate: {ladder: [[0.62, 1.0]]}
    """)
    sol = profiles.get_profile("SOLUSDT")
    assert sol.directional.win_rate.thresholds == (0.62,)
    assert sol.directional.win_rate.mode == "higher_better"
    assert sol.directional.win_rate.floor == 0.1     # унаследован из _default


def test_enum_map_override_is_partial(profiles_dir):
    """
    Enum-карта — словарь, поэтому сливается по ключам: подвинуть один бакет
    можно, не переписывая все четыре. Полноту покрытия гарантирует родитель.
    """
    write(profiles_dir, "SOLUSDT", """
        symbol: SOLUSDT
        version: 1
        context:
          age_bucket: {age_30_60: 0.9}
    """)
    sol = profiles.get_profile("SOLUSDT")
    assert sol.context.age_bucket == {
        "age_lt_30": 1.0, "age_30_60": 0.9, "age_60_120": 0.5, "age_gt_120": 0.0
    }


def test_symbol_lookup_is_case_insensitive(profiles_dir):
    write(profiles_dir, "ETHUSDT", "symbol: ETHUSDT\nversion: 1\n")
    assert profiles.get_profile("ethusdt").symbol == "ETHUSDT"
    assert profiles.get_profile("  EthUsdt ").symbol == "ETHUSDT"


def test_unknown_symbol_falls_back_to_default(profiles_dir):
    assert profiles.get_profile("DOGEUSDT").symbol == "_default"
    assert profiles.is_known_symbol("DOGEUSDT") is False
    assert profiles.get_profile(None).symbol == "_default"
    assert profiles.get_profile("").symbol == "_default"


def test_known_symbol_reported_only_for_own_profile(profiles_dir):
    write(profiles_dir, "ETHUSDT", "symbol: ETHUSDT\nversion: 1\n")
    assert profiles.is_known_symbol("ETHUSDT") is True
    # `_default` — не монета, а базовая калибровка.
    assert profiles.is_known_symbol("_default") is False


def test_broken_yaml_does_not_break_loader(profiles_dir):
    """
    Опечатка в профиле одной монеты не должна ронять API: она попадает
    в лог и в load_errors, монета считается по _default.
    """
    write(profiles_dir, "ETHUSDT", "symbol: ETHUSDT\nversion: 1\n")
    (profiles_dir / "BADUSDT.yaml").write_text(
        "symbol: BADUSDT\nversion: 1\nweights: {statistical: 9.0}\n", encoding="utf-8"
    )
    profiles.reload_profiles()

    assert "BADUSDT" in profiles.load_errors()
    assert profiles.get_profile("BADUSDT").symbol == "_default"
    # Соседние профили при этом живы.
    assert profiles.get_profile("ETHUSDT").symbol == "ETHUSDT"


def test_missing_default_is_fatal(tmp_path, monkeypatch):
    """Без _default считать нечем — это единственный случай, когда загрузчик падает."""
    monkeypatch.setenv("PROFILES_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        profiles.reload_profiles()
    monkeypatch.setenv("PROFILES_DIR", str(FROZEN_PROFILES))
    profiles.reload_profiles()


def test_extends_other_than_default_is_rejected(profiles_dir):
    """Цепочки наследования не поддерживаем: профиль читается по одному файлу."""
    write(profiles_dir, "ETHUSDT", "symbol: ETHUSDT\nversion: 1\n")
    (profiles_dir / "SOLUSDT.yaml").write_text(
        "symbol: SOLUSDT\nversion: 1\nextends: ETHUSDT\n", encoding="utf-8"
    )
    profiles.reload_profiles()
    assert "SOLUSDT" in profiles.load_errors()


# ─── Валидация схемы ──────────────────────────────────────────────────────────

def test_weights_must_sum_to_one():
    with pytest.raises(ValidationError, match="сумма weights"):
        profiles.WeightsSpec(statistical=0.5, directional=0.3, context=0.1, rarity=0.2)


def test_rating_bounds_must_be_ordered():
    with pytest.raises(ValidationError, match="strong_min"):
        profiles.RatingSpec(strong_min=0.5, moderate_min=0.5)


@pytest.mark.parametrize("mode,ladder", [
    ("higher_better", [[0.5, 1.0], [0.8, 0.7]]),    # порог растёт — вторая ступень мертва
    ("lower_better", [[0.30, 1.0], [0.10, 0.7]]),   # порог убывает — то же самое
    ("higher_better", [[0.5, 1.0], [0.5, 0.7]]),    # равные пороги
])
def test_non_monotonic_ladder_is_rejected(mode, ladder):
    """
    Машинная защита от повторения audit #1: ступень, до которой проверка
    никогда не доходит, — это молча мёртвая ветка калибровки.
    """
    with pytest.raises(ValidationError, match="ladder"):
        profiles.LadderSpec(mode=mode, ladder=ladder, floor=0.0)


@pytest.mark.parametrize("field,mode,ladder", [
    ("valid_label_pct", "higher_better", [[1.0, 1.0], [0.9, 0.7]]),
    ("win_rate", "higher_better", [[1.5, 1.0], [0.9, 0.7]]),
    ("monthly_concentration", "lower_better", [[0.0, 1.0], [0.1, 0.7]]),
])
def test_unreachable_share_threshold_is_rejected(field, mode, ladder):
    """
    Доли лежат в [0, 1], сравнение строгое — порог 1.0 у higher_better
    не сработает никогда. Монотонность такую ступень не ловит: она
    формально убывает.

    Не гипотетика: ровно это предложил калибратор на реальной выгрузке ETH —
    p90 доли валидных меток оказался равен 1.0, потому что у девяти
    кандидатов из десяти метки валидны все до одной.
    """
    spec = profiles.LadderSpec(mode=mode, ladder=ladder, floor=0.0)
    with pytest.raises(ValueError, match="недостижима"):
        profiles._validate_reachable(field, spec)


def test_reachable_check_skips_unbounded_fields():
    """sample_size и fa_ratio не ограничены единицей — их порог 1.0 законен."""
    spec = profiles.LadderSpec(
        mode="higher_better", ladder=[[1.0, 1.0], [0.5, 0.7]], floor=0.0
    )
    profiles._validate_reachable("sample_size", spec)      # не бросает
    profiles._validate_reachable("fa_ratio", spec)


def test_profile_with_unreachable_ladder_does_not_load(profiles_dir):
    """Мёртвая ступень должна валить загрузку профиля, а не тихо жить в нём."""
    (profiles_dir / "ETHUSDT.yaml").write_text(
        "symbol: ETHUSDT\nversion: 1\n"
        "statistical:\n"
        "  valid_label_pct: {mode: higher_better, ladder: [[1.0, 1.0], [0.9, 0.7]], floor: 0.0}\n",
        encoding="utf-8",
    )
    profiles.reload_profiles()
    assert "недостижима" in profiles.load_errors()["ETHUSDT"]


def test_ladder_score_out_of_range_is_rejected():
    with pytest.raises(ValidationError):
        profiles.LadderSpec(mode="higher_better", ladder=[[0.5, 1.5]], floor=0.0)


def test_empty_ladder_is_rejected():
    with pytest.raises(ValidationError):
        profiles.LadderSpec(mode="higher_better", ladder=[], floor=0.5)


def test_enum_map_must_cover_all_values():
    """Непокрытый бакет — это KeyError на живом кандидате, а не «значение по умолчанию»."""
    with pytest.raises(ValidationError, match="age_gt_120"):
        profiles.ContextSpec(
            context_status={"fresh": 1.0, "stale": 0.0},
            age_bucket={"age_lt_30": 1.0, "age_30_60": 0.5},
            trajectory_entropy={"low": 1.0, "medium": 0.5, "high": 0.0},
        )


def test_enum_map_rejects_unknown_key(profiles_dir):
    """
    Опечатка в имени бакета иначе выглядела бы как работающая настройка:
    ключ есть, а на оценку не влияет.
    """
    (profiles_dir / "ETHUSDT.yaml").write_text(
        "symbol: ETHUSDT\nversion: 1\n"
        "context:\n  age_bucket: {age_lt_31: 1.0}\n",
        encoding="utf-8",
    )
    profiles.reload_profiles()
    assert "age_lt_31" in profiles.load_errors()["ETHUSDT"]


# ─── Ступени и fingerprint ────────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (0.90, 1.0), (0.86, 1.0),
    (0.85, 0.7),                 # строгое «>»: ровно порог падает ступенью ниже
    (0.80, 0.4), (0.75, 0.0), (0.10, 0.0),
])
def test_ladder_higher_better_is_strict(value, expected):
    spec = profiles.default_profile().statistical.valid_label_pct
    assert spec.apply(value) == expected


@pytest.mark.parametrize("value,expected", [
    (0.05, 1.0), (0.099, 1.0),
    (0.10, 0.7),                 # строгое «<»
    (0.15, 0.3), (0.30, 0.0), (0.99, 0.0),
])
def test_ladder_lower_better_is_strict(value, expected):
    spec = profiles.default_profile().statistical.monthly_concentration
    assert spec.apply(value) == expected


def test_step_clamps_to_available_thresholds():
    """Профиль может иметь меньше ступеней, чем ожидает текст формулировки."""
    spec = profiles.LadderSpec(mode="higher_better", ladder=[[0.8, 1.0]], floor=0.0)
    assert spec.step(0) == 0.8
    assert spec.step(1) == 0.8
    assert spec.step(-1) == 0.8


def test_fingerprint_changes_with_threshold(profiles_dir):
    """
    Правка порога без бампа version обязана быть видна: иначе записи «до»
    и «после» лежали бы под одной меткой профиля и смешались в средних.
    """
    write(profiles_dir, "ETHUSDT", """
        symbol: ETHUSDT
        version: 1
        statistical:
          sample_size: {mode: higher_better, ladder: [[400, 1.0], [200, 0.7], [80, 0.4]], floor: 0.1}
    """)
    before = profiles.get_profile("ETHUSDT").fingerprint

    write(profiles_dir, "ETHUSDT", """
        symbol: ETHUSDT
        version: 1
        statistical:
          sample_size: {mode: higher_better, ladder: [[401, 1.0], [200, 0.7], [80, 0.4]], floor: 0.1}
    """)
    after = profiles.get_profile("ETHUSDT").fingerprint

    assert before != after
    assert profiles.get_profile("ETHUSDT").name == "ETHUSDT@1"
    assert len(after) == 12


def test_description_is_not_inherited(profiles_dir):
    """
    Описание характеризует конкретный профиль. Унаследованное, оно делает
    реестр дезинформацией: ETHUSDT показывал бы себя как «базовую калибровку
    под BTCUSDT» — ровно то, чем он не является.
    """
    write(profiles_dir, "ETHUSDT", "symbol: ETHUSDT\nversion: 1\n")
    assert profiles.get_profile("ETHUSDT").description == ""
    assert profiles.default_profile().description != ""

    write(profiles_dir, "SOLUSDT", "symbol: SOLUSDT\nversion: 1\ndescription: своё\n")
    assert profiles.get_profile("SOLUSDT").description == "своё"


def test_fingerprint_ignores_description(profiles_dir):
    """Комментарий к профилю — не калибровка, менять из-за него отпечаток незачем."""
    write(profiles_dir, "ETHUSDT", "symbol: ETHUSDT\nversion: 1\ndescription: раз\n")
    first = profiles.get_profile("ETHUSDT").fingerprint
    write(profiles_dir, "ETHUSDT", "symbol: ETHUSDT\nversion: 1\ndescription: два\n")
    assert profiles.get_profile("ETHUSDT").fingerprint == first


def test_fingerprint_is_cached_but_still_the_real_hash(profiles_dir):
    """
    Отпечаток считается один раз на объект профиля: скорер читает его на
    каждого кандидата, и на 31 921 кандидате пересчёт съедал 3.53 c из 3.78 c
    всего скоринга. Кэш обязан отдавать РОВНО то же значение, что честный
    пересчёт, и не появляться в `model_dump()` — из него отпечаток и считается.

    Невидимость правки порога сторожит `test_fingerprint_changes_with_threshold`:
    перезагрузка каталога собирает новые объекты, а не правит эти.
    """
    import hashlib
    import json

    write(profiles_dir, "ETHUSDT", "symbol: ETHUSDT\nversion: 1\n")
    profile = profiles.get_profile("ETHUSDT")

    payload = json.dumps(
        profile.model_dump(mode="json", exclude={"description"}),
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    assert profile.fingerprint == expected
    assert profile.fingerprint == profile.fingerprint
    assert "_fingerprint" not in profile.model_dump()


def test_list_profiles_reports_versions(profiles_dir):
    write(profiles_dir, "ETHUSDT", "symbol: ETHUSDT\nversion: 7\n")
    rows = {row["symbol"]: row for row in profiles.list_profiles()}
    assert rows["ETHUSDT"]["version"] == 7
    assert rows["ETHUSDT"]["profile"] == "ETHUSDT@7"
    assert rows["_default"]["profile"].startswith("_default@")
