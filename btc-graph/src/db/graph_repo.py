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

**Измерения МОДЕЛИ в ключе нет, и это компенсируется очисткой.** `group_id`
осмыслен не только внутри монеты, но и внутри одного обучения: каждый `train`
перенумеровывает состояния. Поэтому генератор обязан звать `clear_symbol()`
перед тем, как отправить кандидатов новой модели, — иначе счётчики рёбер
инкрементируются поверх узлов прежнего поколения. Разбор компромисса и
причина, по которой `model_id` в ключ пока не заводится, — в докстринге
`clear_symbol`.

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


def clear_symbol(symbol: str) -> int:
    """
    Сносит граф одной монеты целиком: узлы MarketGroup и все их рёбра.

    Зачем это нужно. Ключ узла — `(symbol, group_id)`, измерения модели в нём
    нет. А `group_id` осмыслен только внутри одного обучения: генератор
    перенумеровывает состояния при каждом `train`, и «группа 7» новой модели
    — другой рынок, чем «группа 7» прежней. Без очистки накопленные `count`,
    `avg_win_rate` и `avg_quality_score` продолжают инкрементироваться поверх
    узлов предыдущего поколения. Еженедельный `train` — еженедельное
    смешивание; через квартал в одних узлах тринадцать поколений нумерации.

    Испортилось бы это молча: узел прежнего поколения и узел нового внешне
    неотличимы — те же свойства, те же диапазоны значений, правдоподобные
    `group_id`. Ровно тот же класс ошибки, что `run_id ≠ model_id` в замере
    лифта и в админке.

    Почему очистка, а не `model_id` в ключе. Ключ с измерением модели —
    правильное решение для фазы ЭКСПЛУАТАЦИИ: поколения сосуществуют,
    накопленная статистика рёбер сохраняется, запросы фильтруют по модели.
    Но `model_id` пришлось бы протащить через контракт между двумя
    проектами, а сейчас терять нечего — система в фазе становления, и
    пересборка графа дешевле, чем усложнение контракта. Когда статистика
    рёбер станет ценной сама по себе, вариант с ключом придётся вернуть;
    причина отложить записана в `development_log.md`.

    Возвращает число удалённых узлов. Недоступность Neo4j — исключение, а не
    ноль: вызывающий обязан отличить «очищено ноль узлов, графа и не было»
    от «очистка не выполнилась», иначе смешивание поколений вернётся тем же
    молчаливым путём.
    """
    driver = get_driver()
    with driver.session() as session:
        deleted = session.execute_write(_clear_symbol_tx, symbol)
    logger.info("Граф %s очищен перед новой моделью: узлов удалено %d", symbol, deleted)
    return deleted


def _clear_symbol_tx(tx, symbol: str) -> int:
    # `RETURN count(g)` ПОСЛЕ `DETACH DELETE` — рабочий идиом: строки к моменту
    # RETURN уже отобраны, удаление на счётчик не влияет. Вариант через
    # `WITH collect(...) CALL { ... }` считает то же самое, но CALL без
    # variable scope clause на Neo4j 5 объявлен deprecated, а форма со скоупом
    # (`CALL (nodes) { ... }`) требует 5.23+ — привязываться к версии сервера
    # ради счётчика незачем.
    result = tx.run(
        """
        MATCH (g:MarketGroup {symbol: $symbol})
        DETACH DELETE g
        RETURN count(g) AS deleted
        """,
        symbol=symbol,
    )
    record = result.single()
    return int(record["deleted"]) if record else 0


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


def upsert_batch(pairs: list[tuple[Candidate, CandidateEvaluation]]) -> bool:
    """
    Пачка кандидатов одной транзакцией: три запроса вместо трёх на кандидата.

    **Результат обязан совпадать с последовательным `upsert_from_candidate`,
    и здесь это не пожелание, а условие применимости.** Свойства узлов и рёбер
    накапливаются (`count`, `sample_count`, скользящие средние), поэтому любое
    расхождение не всплывёт ошибкой — граф просто станет другим. Совпадение
    обеспечивается двумя вещами:

    * **предагрегация в Python.** Инкрементальное среднее, применённое k раз,
      равно среднему по всем n+k наблюдениям, поэтому пачка сворачивается в
      `(n, сумма win_rate, сумма quality_score)` на переход, и Cypher делает
      ОДИН шаг с теми же формулами. Правые части `SET` вычисляются от
      состояния ДО предложения — на этом же свойстве держался и одиночный путь;
    * **порядок пачки.** `dominant_bias` узла и `rarity` ребра в
      последовательном коде перезаписываются каждым кандидатом, то есть
      побеждает последний. Здесь берётся последний в порядке пачки — она
      приходит в том же порядке, в каком шёл бы цикл.

    Порядок между запросами (текущие группы, предыдущие, рёбра) на результат
    не влияет: узел, созданный как «предыдущий», не получает `sample_count`, а
    последующее `ON MATCH` считает его через `coalesce(..., 0)` — ровно как
    если бы кандидаты шли по одному.

    Транзакция одна на всю пачку: при сбое не применяется ничего, и
    вызывающему безопасно повторить пачку по одному кандидату. Частичное
    применение сделало бы повтор двойным счётом.
    """
    if not pairs:
        return True

    groups, prev_groups, edges = _aggregate(pairs)
    try:
        driver = get_driver()
        with driver.session() as session:
            session.execute_write(_upsert_batch_tx, groups, prev_groups, edges)
        return True
    except ServiceUnavailable:
        logger.warning(
            "Neo4j недоступен — граф не обновлён для пачки из %d кандидатов",
            len(pairs), exc_info=True,
        )
        return False


def _aggregate(
    pairs: list[tuple[Candidate, CandidateEvaluation]],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Пачка → строки для трёх UNWIND: текущие группы, предыдущие, рёбра."""
    groups: dict[tuple[str, float], dict] = {}
    prev_groups: dict[tuple[str, float], dict] = {}
    edges: dict[tuple[str, str], dict] = {}

    for candidate, evaluation in pairs:
        symbol = candidate.symbol

        key = (symbol, candidate.current_group_id)
        row = groups.get(key)
        if row is None:
            row = groups[key] = {
                "symbol": symbol,
                "group_id": candidate.current_group_id,
                "label": f"{symbol}:group_{int(candidate.current_group_id)}",
                "n": 0,
                "bias": None,
            }
        row["n"] += 1
        row["bias"] = candidate.historical_bias_context.value

        if candidate.previous_group_id is None:
            continue

        prev_key = (symbol, candidate.previous_group_id)
        if prev_key not in prev_groups:
            prev_groups[prev_key] = {
                "symbol": symbol,
                "group_id": candidate.previous_group_id,
                "label": f"{symbol}:group_{int(candidate.previous_group_id)}",
            }

        edge_key = (symbol, candidate.transition_id)
        edge = edges.get(edge_key)
        if edge is None:
            edge = edges[edge_key] = {
                "symbol": symbol,
                "tid": candidate.transition_id,
                "from_id": candidate.previous_group_id,
                "to_id": candidate.current_group_id,
                "n": 0,
                "win_sum": 0.0,
                "qs_sum": 0.0,
                "rarity": None,
            }
        edge["n"] += 1
        edge["win_sum"] += evaluation.win_rate
        edge["qs_sum"] += evaluation.quality_score
        edge["rarity"] = candidate.transition_rarity.value

    return list(groups.values()), list(prev_groups.values()), list(edges.values())


def _upsert_batch_tx(tx, groups: list[dict], prev_groups: list[dict], edges: list[dict]) -> None:
    if groups:
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (g:MarketGroup {symbol: row.symbol, group_id: row.group_id})
            ON CREATE SET g.label = row.label, g.sample_count = row.n,
                          g.dominant_bias = row.bias
            ON MATCH  SET g.sample_count = coalesce(g.sample_count, 0) + row.n,
                          g.dominant_bias = row.bias
            """,
            rows=groups,
        )

    if prev_groups:
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (g:MarketGroup {symbol: row.symbol, group_id: row.group_id})
            ON CREATE SET g.label = row.label
            """,
            rows=prev_groups,
        )

    if edges:
        # Формулы средних те же, что в одиночном пути, только вместо одного
        # наблюдения приходит сумма n наблюдений. t.count справа — ещё старый.
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (src:MarketGroup {symbol: row.symbol, group_id: row.from_id})
            MATCH (dst:MarketGroup {symbol: row.symbol, group_id: row.to_id})
            MERGE (src)-[t:TRANSITION {symbol: row.symbol, transition_id: row.tid}]->(dst)
            ON CREATE SET
                t.rarity = row.rarity,
                t.count = row.n,
                t.avg_win_rate = row.win_sum / row.n,
                t.avg_quality_score = row.qs_sum / row.n
            ON MATCH SET
                t.count = t.count + row.n,
                t.avg_win_rate      = (t.avg_win_rate      * t.count + row.win_sum) / (t.count + row.n),
                t.avg_quality_score = (t.avg_quality_score * t.count + row.qs_sum)  / (t.count + row.n),
                t.rarity = row.rarity
            """,
            rows=edges,
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
