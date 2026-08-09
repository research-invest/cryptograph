"""
Neo4j репозиторий: граф состояний рынка.
Узлы: MarketGroup (symbol, group_id)
Рёбра: TRANSITION (symbol, transition_id, rarity, count, avg_win_rate,
        avg_quality_score)

Ключ узла — ПАРА (symbol, group_id), и это не украшательство. `group_id`
осмыслен только внутри графа одной монеты: генератор обучает модель состояний
на каждую монету отдельно, и «группа 1.0» у BTC и у ETH — разные рынки.
С ключом по одному group_id узлы схлопывались бы в один, рёбра «42->1» двух
монет смешивались, а avg_quality_score усреднялся по разным инструментам.
Испортилось бы это молча: граф выглядел бы наполненным.

Существующий граф переводится скриптом scripts/migrate_graph_symbol.cypher —
его надо прогнать ДО первого кандидата не по BTCUSDT.
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
    symbol = candidate.symbol

    # Upsert текущей группы
    tx.run(
        """
        MERGE (g:MarketGroup {symbol: $symbol, group_id: $group_id})
        ON CREATE SET g.label = $label, g.sample_count = 1, g.dominant_bias = $bias
        ON MATCH  SET g.sample_count = coalesce(g.sample_count, 0) + 1,
                      g.dominant_bias = $bias
        """,
        symbol=symbol,
        group_id=candidate.current_group_id,
        label=f"{symbol}:group_{int(candidate.current_group_id)}",
        bias=candidate.historical_bias_context.value,
    )

    if candidate.previous_group_id is None:
        return

    # Upsert предыдущей группы
    tx.run(
        """
        MERGE (g:MarketGroup {symbol: $symbol, group_id: $group_id})
        ON CREATE SET g.label = $label
        """,
        symbol=symbol,
        group_id=candidate.previous_group_id,
        label=f"{symbol}:group_{int(candidate.previous_group_id)}",
    )

    # Upsert перехода: инкрементируем count, обновляем скользящие средние.
    #
    # Правые части SET вычисляются от состояния ДО предложения, поэтому
    # t.count в формуле среднего — ещё старый, и (avg*n + x)/(n+1) корректно.
    # Проверено на Neo4j 5: последовательность [1.0, 0.0, 0.0] даёт 0.3333.
    #
    # symbol дублируется на ребро: так запросы «все переходы монеты» не обязаны
    # обходить узлы, а данные остаются читаемыми при взгляде на одно ребро.
    #
    # avg_win_rate называется так, потому что в нём лежит именно win rate.
    # До 2026-08-09 свойство называлось avg_horizon_return, а значение
    # подставлялось то же самое: любой Cypher-запрос «средний исход на
    # горизонте» молча получал не ту величину. Переименовано вместе с
    # миграцией существующих рёбер (scripts/migrate_graph_avg_win_rate.cypher).
    # Настоящего среднего исхода на ребре нет — если понадобится, это новое
    # свойство и новое значение из evaluation, а не переиспользование этого.
    tx.run(
        """
        MATCH (src:MarketGroup {symbol: $symbol, group_id: $from_id})
        MATCH (dst:MarketGroup {symbol: $symbol, group_id: $to_id})
        MERGE (src)-[t:TRANSITION {symbol: $symbol, transition_id: $tid}]->(dst)
        ON CREATE SET
            t.rarity = $rarity,
            t.count = 1,
            t.avg_win_rate = $win_rate,
            t.avg_quality_score = $qs
        ON MATCH SET
            t.count = t.count + 1,
            t.avg_win_rate       = (t.avg_win_rate       * t.count + $win_rate) / (t.count + 1),
            t.avg_quality_score  = (t.avg_quality_score  * t.count + $qs)       / (t.count + 1),
            t.rarity = $rarity
        """,
        symbol=symbol,
        from_id=candidate.previous_group_id,
        to_id=candidate.current_group_id,
        tid=candidate.transition_id,
        rarity=candidate.transition_rarity.value,
        win_rate=evaluation.win_rate,
        qs=evaluation.quality_score,
    )


def find_transitions_to_group(
    symbol: str,
    to_group_id: float,
    rarity_filter: list[str] | None = None,
) -> list[dict]:
    """
    Запрос из ТЗ: найти переходы монеты, ведущие в группу to_group_id.

    symbol обязателен и идёт первым аргументом намеренно: запрос «переходы
    в группу 7» без указания монеты не имеет смысла — таких групп столько же,
    сколько монет в графе.
    """
    try:
        driver = get_driver()
        with driver.session() as session:
            result = session.execute_read(
                _find_transitions_tx, symbol, to_group_id,
                rarity_filter or ["uncommon", "rare"],
            )
        return result
    except ServiceUnavailable:
        logger.warning(
            "Neo4j недоступен — переходы %s в группу %s не получены",
            symbol, to_group_id, exc_info=True,
        )
        return []


def _find_transitions_tx(
    tx, symbol: str, to_group_id: float, rarity_filter: list[str]
) -> list[dict]:
    result = tx.run(
        """
        MATCH (src:MarketGroup {symbol: $symbol})
              -[t:TRANSITION]->
              (dst:MarketGroup {symbol: $symbol, group_id: $to_id})
        WHERE t.rarity IN $rarity_filter
        RETURN src.group_id AS from_group, t.transition_id AS transition_id,
               t.rarity AS rarity, t.count AS count,
               t.avg_win_rate AS avg_win_rate,
               t.avg_quality_score AS avg_quality_score
        ORDER BY t.count DESC
        """,
        symbol=symbol,
        to_id=to_group_id,
        rarity_filter=rarity_filter,
    )
    return [dict(r) for r in result]


def get_group_info(symbol: str, group_id: float) -> dict | None:
    """Возвращает атрибуты узла MarketGroup конкретной монеты."""
    try:
        driver = get_driver()
        with driver.session() as session:
            result = session.run(
                "MATCH (g:MarketGroup {symbol: $symbol, group_id: $gid}) RETURN g",
                symbol=symbol,
                gid=group_id,
            )
            record = result.single()
            return dict(record["g"]) if record else None
    except ServiceUnavailable:
        logger.warning(
            "Neo4j недоступен — узел группы %s:%s не получен",
            symbol, group_id, exc_info=True,
        )
        return None


def list_symbols() -> list[str]:
    """Монеты, представленные в графе. Нужно API и диагностике изоляции."""
    try:
        driver = get_driver()
        with driver.session() as session:
            result = session.run(
                "MATCH (g:MarketGroup) RETURN DISTINCT g.symbol AS symbol ORDER BY symbol"
            )
            return [r["symbol"] for r in result if r["symbol"]]
    except ServiceUnavailable:
        logger.warning("Neo4j недоступен — список монет графа не получен", exc_info=True)
        return []


def ensure_constraints() -> bool:
    """
    Создаёт constraint уникальности (symbol, group_id).

    Идемпотентно. Вызывать ПОСЛЕ scripts/migrate_graph_symbol.cypher: на узлах
    без symbol constraint не создастся, и это правильно — сначала бэкфилл.
    """
    try:
        driver = get_driver()
        with driver.session() as session:
            session.run(
                "CREATE CONSTRAINT market_group_key IF NOT EXISTS "
                "FOR (g:MarketGroup) REQUIRE (g.symbol, g.group_id) IS UNIQUE"
            )
        return True
    except ServiceUnavailable:
        logger.warning("Neo4j недоступен — constraint не создан", exc_info=True)
        return False
