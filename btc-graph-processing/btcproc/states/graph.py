"""
Граф состояний: статистика узлов (состояний) и рёбер (переходов).

`transition_rarity` считается перцентилями частоты, а не абсолютным порогом:
при адаптивном числе состояний абсолютная доля перехода зависит от того,
сколько групп нашлось, и фиксированный порог сделал бы все переходы
одинаково «редкими» на дробном графе.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from btcproc.states import naming

logger = logging.getLogger(__name__)

RARE_QUANTILE = 0.33
UNCOMMON_QUANTILE = 0.66


def rarity_by_quantiles(counts: pd.Series) -> pd.Series:
    """Треть самых редких → rare, следующая треть → uncommon, остальные → common."""
    if counts.empty:
        return pd.Series(dtype=object)
    q_rare = counts.quantile(RARE_QUANTILE)
    q_uncommon = counts.quantile(UNCOMMON_QUANTILE)
    return pd.Series(
        np.select(
            [counts <= q_rare, counts <= q_uncommon],
            ["rare", "uncommon"],
            default="common",
        ),
        index=counts.index,
    )


def transition_stats(states: pd.DataFrame, outcomes: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Статистика по каждому переходу: сколько раз случался, насколько редок,
    какой средний исход давал.

    states  — результат assign_states.
    outcomes — таблица исходов (ret_pct, is_up), выровненная по тому же индексу.
    """
    events = states[states["is_transition"] & states["transition_id"].notna()]
    if events.empty:
        return pd.DataFrame(
            columns=["transition_id", "prev_group_id", "cur_group_id", "count",
                     "share", "rarity", "avg_horizon_return", "up_share"]
        )

    grouped = events.groupby("transition_id").agg(
        prev_group_id=("prev_group_id", "first"),
        cur_group_id=("group_id", "first"),
        count=("group_id", "size"),
    )
    grouped["share"] = grouped["count"] / len(events)
    grouped["rarity"] = rarity_by_quantiles(grouped["count"])

    if outcomes is not None and not outcomes.empty:
        joined = events.join(outcomes[["ret_pct", "is_up", "valid"]], how="left")
        valid = joined[joined["valid"].fillna(False)]
        agg = valid.groupby("transition_id").agg(
            avg_horizon_return=("ret_pct", "mean"),
            up_share=("is_up", "mean"),
        )
        grouped = grouped.join(agg)
    else:
        grouped["avg_horizon_return"] = np.nan
        grouped["up_share"] = np.nan

    return grouped.reset_index()


def group_stats(
    states: pd.DataFrame,
    outcomes: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    bias_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Статистика по состояниям: размер, доля истории, склонность исходов.

    dominant_bias отвечает на вопрос «что обычно бывало дальше, когда рынок
    находился в этом состоянии» — это подпись узла в графе.
    """
    grouped = states.groupby("group_id").agg(count=("state_seq", "size"))
    grouped["share"] = grouped["count"] / len(states)

    if outcomes is not None and not outcomes.empty:
        joined = states.join(outcomes[["ret_pct", "is_up", "valid"]], how="left")
        valid = joined[joined["valid"].fillna(False)]
        agg = valid.groupby("group_id").agg(
            up_share=("is_up", "mean"),
            avg_ret_pct=("ret_pct", "mean"),
            avg_vol_pct=("ret_pct", "std"),
        )
        grouped = grouped.join(agg)
        grouped["dominant_bias"] = np.select(
            [
                grouped["up_share"] > 0.5 + bias_threshold,
                grouped["up_share"] < 0.5 - bias_threshold,
            ],
            ["long_skew", "short_skew"],
            default="neutral",
        )
    else:
        for col in ("up_share", "avg_ret_pct", "avg_vol_pct"):
            grouped[col] = np.nan
        grouped["dominant_bias"] = "neutral"

    if features is not None and not features.empty:
        grouped["top_features"] = _describe_groups(states, features)
    else:
        grouped["top_features"] = None

    # Имя состояния считается здесь же, из его собственных отклонений, и
    # хранится рядом с ними. Задавать имена руками нельзя: train перенумеровывает
    # состояния при каждом прогоне, и ручная подпись начала бы молча врать —
    # номер остался бы, а смысл под ним поменялся.
    grouped["name"] = [
        naming.describe_state(row.top_features, row.dominant_bias)
        for row in grouped.itertuples()
    ]

    return grouped.reset_index()


def _describe_groups(states: pd.DataFrame, features: pd.DataFrame, top_n: int = 5) -> pd.Series:
    """
    Чем состояние отличается от рынка в среднем.

    Берём z-отклонение среднего признака группы от общего среднего и
    оставляем top_n самых выраженных — так узел графа получает человекочитаемую
    подпись вида «низкая волатильность, цена у верхней границы недели».
    """
    common = features.reindex(states.index).dropna()
    if common.empty:
        return pd.Series(dtype=object)

    labels = states.loc[common.index, "group_id"]
    overall_mean = common.mean()
    overall_std = common.std().replace(0, np.nan)

    described = {}
    for gid, block in common.groupby(labels):
        deviation = ((block.mean() - overall_mean) / overall_std).dropna()
        top = deviation.reindex(deviation.abs().sort_values(ascending=False).index)[:top_n]
        described[gid] = {name: round(float(value), 3) for name, value in top.items()}
    return pd.Series(described)


def to_cytoscape(groups: pd.DataFrame, transitions: pd.DataFrame) -> dict:
    """
    Граф в формате Cytoscape.js для админки.

    Узел несёт размер и bias, ребро — частоту, редкость и средний исход.
    """
    nodes = [
        {
            "data": {
                "id": f"g{row.group_id:g}",
                "group_id": float(row.group_id),
                # label — короткая подпись на самом узле: имена в 60 символов
                # на графе из сорока узлов сливаются в кашу. Полное имя лежит
                # рядом, его показывают панель деталей, подсказка и переключатель
                # «имена на узлах».
                "label": f"{row.group_id:g}",
                "name": row.get("name") or "",
                "size": int(row["count"]),
                "share": float(row.share),
                "bias": row.get("dominant_bias") or "neutral",
                "up_share": None if pd.isna(row.get("up_share")) else float(row.up_share),
                "avg_ret_pct": None if pd.isna(row.get("avg_ret_pct")) else float(row.avg_ret_pct),
                "top_features": row.get("top_features") or {},
            }
        }
        for _, row in groups.iterrows()
    ]

    edges = []
    for _, row in transitions.iterrows():
        if pd.isna(row.prev_group_id):
            continue
        edges.append(
            {
                "data": {
                    "id": row.transition_id,
                    "source": f"g{row.prev_group_id:g}",
                    "target": f"g{row.cur_group_id:g}",
                    "label": row.transition_id,
                    "count": int(row["count"]),
                    "share": float(row.share),
                    "rarity": row.rarity,
                    "up_share": None if pd.isna(row.get("up_share")) else float(row.up_share),
                    "avg_horizon_return": (
                        None if pd.isna(row.get("avg_horizon_return"))
                        else float(row.avg_horizon_return)
                    ),
                }
            }
        )
    return {"nodes": nodes, "edges": edges}
