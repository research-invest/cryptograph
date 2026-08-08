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


def test_vocabulary_covers_only_real_features(features):
    """
    Опечатка в имени признака делает строку словаря мёртвой: она никогда
    не совпадёт с ключом top_features, и это никак не проявится — просто
    подпись станет беднее. Сверяемся с настоящим набором признаков.
    """
    vocabulary = {f for _axis, mapping in naming.AXES for f in mapping}
    unknown = vocabulary - set(features.columns)
    assert not unknown, f"словарь ссылается на несуществующие признаки: {sorted(unknown)}"


def test_group_stats_produces_names(states_and_features):
    """Имя должно появляться само при построении графа, а не проставляться потом."""
    from btcproc.states import graph

    states, features, outcomes = states_and_features
    groups = graph.group_stats(states, outcomes, features)

    assert "name" in groups.columns
    assert groups["name"].notna().all()
    assert (groups["name"].str.len() > 0).all()
