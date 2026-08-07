"""
Neo4j репозиторий: граф состояний рынка.
Узлы: MarketGroup (group_id)
Рёбра: TRANSITION (transition_id, rarity, count, avg_return)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable

from src.models.candidate import Candidate, CandidateEvaluation

logger = logging.getLogger(__name__)

_driver: Optional[Driver] = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.environ.get("NEO4J_USER", "neo4j"),
                os.environ.get("NEO4J_PASSWORD", "btc_neo4j_pass"),
            ),
        )
    return _driver


def close_driver() -> None:
    global _driver
    if _driver:
        _driver.close()
        _driver = None


def upsert_from_candidate(candidate: Candidate, evaluation: CandidateEvaluation) -> bool:
    """
    Создаёт или обновляет узлы MarketGroup и ребро TRANSITION
    на основе данных кандидата после оценки.

    Возвращает True при успешной записи. Недоступность Neo4j не ломает
    pipeline, но попадает в лог.
    """
    try:
        driver = get_driver()
        with driver.session() as session:
            session.execute_write(_upsert_tx, candidate, evaluation)
        return True
    except ServiceUnavailable:
        logger.warning(
            "Neo4j недоступен — граф не обновлён для кандидата %s",
            candidate.candidate_id,
            exc_info=True,
        )
        return False


def _upsert_tx(tx, candidate: Candidate, evaluation: CandidateEvaluation) -> None:
    # Upsert текущей группы
    tx.run(
        """
        MERGE (g:MarketGroup {group_id: $group_id})
        ON CREATE SET g.label = $label, g.sample_count = 1, g.dominant_bias = $bias
        ON MATCH  SET g.sample_count = coalesce(g.sample_count, 0) + 1,
                      g.dominant_bias = $bias
        """,
        group_id=candidate.current_group_id,
        label=f"group_{int(candidate.current_group_id)}",
        bias=candidate.historical_bias_context.value,
    )

    if candidate.previous_group_id is None:
        return

    # Upsert предыдущей группы
    tx.run(
        """
        MERGE (g:MarketGroup {group_id: $group_id})
        ON CREATE SET g.label = $label
        """,
        group_id=candidate.previous_group_id,
        label=f"group_{int(candidate.previous_group_id)}",
    )

    # Upsert перехода: инкрементируем count, обновляем скользящие средние.
    #
    # Правые части SET вычисляются от состояния ДО предложения, поэтому
    # t.count в формуле среднего — ещё старый, и (avg*n + x)/(n+1) корректно.
    # Проверено на Neo4j 5: последовательность [1.0, 0.0, 0.0] даёт 0.3333.
    tx.run(
        """
        MATCH (src:MarketGroup {group_id: $from_id})
        MATCH (dst:MarketGroup {group_id: $to_id})
        MERGE (src)-[t:TRANSITION {transition_id: $tid}]->(dst)
        ON CREATE SET
            t.rarity = $rarity,
            t.count = 1,
            t.avg_horizon_return = $win_rate,
            t.avg_quality_score = $qs
        ON MATCH SET
            t.count = t.count + 1,
            t.avg_horizon_return = (t.avg_horizon_return * t.count + $win_rate) / (t.count + 1),
            t.avg_quality_score  = (t.avg_quality_score  * t.count + $qs)       / (t.count + 1),
            t.rarity = $rarity
        """,
        from_id=candidate.previous_group_id,
        to_id=candidate.current_group_id,
        tid=candidate.transition_id,
        rarity=candidate.transition_rarity.value,
        win_rate=evaluation.win_rate,
        qs=evaluation.quality_score,
    )


def find_transitions_to_group(
    to_group_id: float,
    rarity_filter: list[str] | None = None,
) -> list[dict]:
    """
    Запрос из ТЗ: найти переходы, ведущие в группу to_group_id.
    rarity_filter: список значений ["uncommon", "rare"]
    """
    try:
        driver = get_driver()
        with driver.session() as session:
            result = session.execute_read(
                _find_transitions_tx, to_group_id, rarity_filter or ["uncommon", "rare"]
            )
        return result
    except ServiceUnavailable:
        logger.warning(
            "Neo4j недоступен — переходы в группу %s не получены", to_group_id, exc_info=True
        )
        return []


def _find_transitions_tx(tx, to_group_id: float, rarity_filter: list[str]) -> list[dict]:
    result = tx.run(
        """
        MATCH (src)-[t:TRANSITION]->(dst:MarketGroup {group_id: $to_id})
        WHERE t.rarity IN $rarity_filter
        RETURN src.group_id AS from_group, t.transition_id AS transition_id,
               t.rarity AS rarity, t.count AS count,
               t.avg_horizon_return AS avg_return,
               t.avg_quality_score AS avg_quality_score
        ORDER BY t.count DESC
        """,
        to_id=to_group_id,
        rarity_filter=rarity_filter,
    )
    return [dict(r) for r in result]


def get_group_info(group_id: float) -> dict | None:
    """Возвращает атрибуты узла MarketGroup."""
    try:
        driver = get_driver()
        with driver.session() as session:
            result = session.run(
                "MATCH (g:MarketGroup {group_id: $gid}) RETURN g",
                gid=group_id,
            )
            record = result.single()
            return dict(record["g"]) if record else None
    except ServiceUnavailable:
        logger.warning(
            "Neo4j недоступен — узел группы %s не получен", group_id, exc_info=True
        )
        return None
