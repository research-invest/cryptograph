"""
Тесты имён состояний.

Имя выводится из отклонений признаков и потому не может разойтись
с содержимым состояния — в отличие от ручного справочника, который протух бы
при первом переобучении. Здесь проверяется, что вывод устойчив и читаем.
"""
from __future__ import annotations

import pytest

from btcproc.states import naming


def test_direction_of_deviation_flips_the_phrase():
    """Знак отклонения решает, какая из двух формулировок оси попадёт в имя."""
    up = naming.describe_state({"trend_align": 1.2})
    down = naming.describe_state({"trend_align": -1.2})
    assert up == "тренд вверх"
    assert down == "тренд вниз"


def test_weak_deviation_is_not_mentioned():
    """
    Отклонение меньше SALIENCE — это шум. Подпись «чуть-чуть выше среднего»
    хуже, чем её отсутствие: она создаёт впечатление осмысленного отличия.
    """
    assert naming.describe_state({"trend_align": 0.2}, "neutral") == "рынок около среднего"


def test_one_phrase_per_axis():
    """
    rsi, tf1h_rsi, tf4h_rsi и tf1d_rsi — один смысл на разных таймфреймах.
    В имя должен попасть сильнейший, иначе выйдет «перекуплен · перекуплен ·
    перекуплен» вместо описания состояния.
    """
    name = naming.describe_state({
        "rsi": 0.8, "tf1h_rsi": 0.9, "tf4h_rsi": 1.4, "tf1d_rsi": 1.1,
    })
    assert name == "перекуплен на 4h"


def test_axes_keep_their_order_regardless_of_strength():
    """
    Порядок слов задаётся осями, а не силой отклонения: иначе подписи соседних
    состояний перетасовывались бы и их стало бы труднее сравнивать глазами.
    """
    name = naming.describe_state({"tf1d_rsi": 2.0, "rv_rank": -0.6, "trend_align": 1.0})
    assert name == "затишье · тренд вверх · перекуплен на 1d"


def test_name_is_capped_by_max_parts():
    name = naming.describe_state(
        {"rv_rank": 2.0, "trend_align": 2.0, "pos_1m": 2.0, "rsi": 2.0, "vol_z": 2.0}
    )
    assert len(name.split(" · ")) == naming.MAX_PARTS


@pytest.mark.parametrize("bias,expected", [
    ("long_skew", "невыразительное, чаще рост"),
    ("short_skew", "невыразительное, чаще падение"),
    ("neutral", "рынок около среднего"),
])
def test_fallback_when_nothing_stands_out(bias, expected):
    """Состояние без выраженных отличий описывается через исход, а не молчанием."""
    assert naming.describe_state({}, bias) == expected
    assert naming.describe_state(None, bias) == expected


def test_unknown_and_broken_features_are_ignored():
    """Чужой ключ или мусор вместо числа не должны ронять подпись."""
    name = naming.describe_state(
        {"неизвестный_признак": 5.0, "rv_rank": "не число", "trend_align": 1.1}
    )
    assert name == "тренд вверх"


def test_label_format():
    assert naming.label_for(7.0, "затишье") == "7 (затишье)"
    assert naming.label_for(7.0, "") == "7"


@pytest.fixture(scope="module")
def all_features(bars, context, monkeypatch_module):
    """
    Полный набор признаков — базовый плюс SMC плюс Fear & Greed.

    Словарь имён обязан покрывать все источники, поэтому сверяться с
    набором, собранным при выключенных источниках, нельзя: часть проверки
    просто не состоится. Флаги задаются ЯВНО — дефолты SMCConfig/FearGreedConfig
    читаются из окружения, и на сервере с SMC_ENABLED=true «умолчание»
    означает другое (см. смысл фикстуры smc_off в test_smc.py).

    FGI джойнит внешний ряд из БД (`ingest.external.load_external_daily`) —
    тесты в БД не ходят, поэтому загрузчик подменяется синтетическим рядом,
    покрывающим диапазон дат фикстуры `bars`.
    """
    import pandas as pd

    from btcproc import config
    from btcproc.features.builder import build_features
    from btcproc.ingest import external

    monkeypatch_module.setattr(
        config, "smc", config.SMCConfig(enabled=True, features_enabled=True)
    )
    monkeypatch_module.setattr(
        config, "fgi", config.FearGreedConfig(enabled=True, features_enabled=True)
    )
    days = pd.date_range(
        bars.index[0].normalize() - pd.Timedelta(days=1),
        bars.index[-1].normalize() + pd.Timedelta(days=1),
        freq="1D", tz="UTC",
    )
    synthetic_daily = pd.DataFrame(
        {"value": 50.0 + 40.0 * pd.Series(range(len(days))).mod(7).sub(3) / 3.0},
        index=days,
    )
    monkeypatch_module.setattr(
        external, "load_external_daily", lambda series, start=None, end=None: synthetic_daily
    )
    return build_features(bars, context)


def test_vocabulary_covers_only_real_features(all_features):
    """
    Опечатка в имени признака делает строку словаря мёртвой: она никогда
    не совпадёт с ключом top_features, и это никак не проявится — просто
    подпись станет беднее. Сверяемся с настоящим набором признаков.
    """
    vocabulary = {f for _axis, mapping in naming.AXES for f in mapping}
    unknown = vocabulary - set(all_features.columns)
    assert not unknown, f"словарь ссылается на несуществующие признаки: {sorted(unknown)}"


def test_every_feature_has_a_phrase(all_features):
    """
    Обратное направление, и оно важнее первого.

    Признак без формулировки не даёт ни ошибки, ни предупреждения — он просто
    не участвует в подписи. Состояние, выделяющееся именно им, получает
    «рынок около среднего», то есть подпись врёт: отличие есть, а сказано,
    что его нет. Ровно так двенадцать SMC-признаков были бы невидимы в графе
    после включения SMC_FEATURES_ENABLED.

    Поэтому новый признак обязан либо получить формулировку, либо осознанно
    попасть в _UNNAMED — молча остаться без имени он не может.
    """
    vocabulary = {f for _axis, mapping in naming.AXES for f in mapping}
    uncovered = set(all_features.columns) - vocabulary - naming._UNNAMED

    assert not uncovered, (
        "признаки без формулировки — они молча выпадут из подписи состояния: "
        f"{sorted(uncovered)}. Добавь их в naming.AXES или, если это осознанно, "
        "в naming._UNNAMED с объяснением."
    )


def test_smc_features_are_named(all_features):
    """
    Проверка не ради полноты словаря (её делает тест выше), а ради того,
    что подпись действительно собирается из SMC-величин, а не откатывается
    в «рынок около среднего».
    """
    from btcproc.features import smc

    name = naming.describe_state(
        {"near_liquidity": 2.1, "premium_discount": -1.8, "ob_age_norm": 1.5}
    )

    assert name != "рынок около среднего"
    assert set(smc.FEATURE_CANDIDATES) <= set(all_features.columns)


def test_group_stats_produces_names(states_and_features):
    """Имя должно появляться само при построении графа, а не проставляться потом."""
    from btcproc.states import graph

    states, features, outcomes = states_and_features
    groups = graph.group_stats(states, outcomes, features)

    assert "name" in groups.columns
    assert groups["name"].notna().all()
    assert (groups["name"].str.len() > 0).all()


# ── Подписи атомов ──────────────────────────────────────────────────────────


def test_atom_labels_cover_all_atoms():
    """
    Атом без подписи уходит в интерфейс сырым идентификатором.

    Это не падение и не ошибка — просто рядом с состоянием появляется
    `sweep_low_reclaim`, и оператор читает его ровно так же, как читал бы
    `group_id 42`. Ради этого всё и затевалось, поэтому проверяем полноту,
    а не наличие «хоть каких-то» подписей.
    """
    from btcproc.features import events

    uncovered = set(events.ATOM_FAMILY) - set(naming.ATOM_LABELS)

    assert not uncovered, (
        f"атомы без русской подписи: {sorted(uncovered)}. "
        "Добавь их в naming.ATOM_LABELS."
    )


def test_atom_labels_have_no_strays():
    """Обратное направление: опечатка в ключе делает подпись мёртвой."""
    from btcproc.features import events

    stray = set(naming.ATOM_LABELS) - set(events.ATOM_FAMILY)

    assert not stray, f"подписи для несуществующих атомов: {sorted(stray)}"


def test_unknown_atom_falls_back_to_its_id():
    """Незнакомый атом не должен ломать панель — показываем как есть."""
    assert naming.label_for_atom("нет_такого_атома") == "нет_такого_атома"
    assert naming.label_for_atom("in_premium") == "в премиальной зоне"


def test_feature_catalog_is_exactly_the_vocabulary():
    """
    Плоский каталог для интерфейса обязан совпадать со словарём по составу.
    Он существует ровно затем, чтобы селектор признаков на графике не заводил
    второй словарь: разъехавшись, тот назвал бы признак иначе, чем имя
    состояния, посчитанное из того же признака.
    """
    catalog = naming.feature_catalog()
    axes = dict(naming.AXES)

    assert len(catalog) == sum(len(vocabulary) for _axis, vocabulary in naming.AXES)
    for axis, feature, positive, negative in catalog:
        assert axes[axis][feature] == (positive, negative)
