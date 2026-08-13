// Перевод графа Neo4j на ключ (symbol, group_id) — шаг 4.3.
//
// До этой миграции узел MarketGroup идентифицировался одним group_id. Пока
// монета была одна, этого хватало; с двумя монетами «группа 1.0» BTC и ETH
// схлопнулись бы в один узел, а рёбра «42->1» — в одно ребро с усреднённым
// по разным рынкам avg_quality_score. Порча была бы молчаливой.
//
// Порядок важен: сначала проставить symbol существующим узлам и рёбрам,
// и только потом создавать constraint. На узлах без symbol constraint
// не создастся (NULL в составном ключе), и это защита, а не помеха.
//
// Запуск:
//   docker compose exec neo4j cypher-shell -u neo4j -p btc_neo4j_pass \
//       -f /scripts/migrate_graph_symbol.cypher
// либо скопировать содержимое в Neo4j Browser (http://localhost:7474).
//
// Скрипт идемпотентен: повторный прогон ничего не меняет.

// ─── 1. Бэкфилл узлов ────────────────────────────────────────────────────────
// Всё, что накоплено до шага 4, относится к BTCUSDT — других монет система
// не принимала (валидатор помечал их флагом symbol_not_btcusdt).
MATCH (g:MarketGroup)
WHERE g.symbol IS NULL
SET g.symbol = 'BTCUSDT';

// Метка приводится к формату «<монета>:group_N» БЕЗУСЛОВНО, а не через
// coalesce: до шага 4 узлы назывались просто «group_1», и если оставить их
// как есть, в одном графе окажутся «group_1» (старый BTC) и «ETHUSDT:group_1»
// (новая монета). Читать такой граф глазами — гарантированная путаница
// ровно там, где монету и надо различать.
MATCH (g:MarketGroup)
WHERE g.group_id IS NOT NULL
SET g.label = g.symbol + ':group_' + toString(toInteger(g.group_id));

// ─── 2. Бэкфилл рёбер ────────────────────────────────────────────────────────
MATCH ()-[t:TRANSITION]->()
WHERE t.symbol IS NULL
SET t.symbol = 'BTCUSDT';

// ─── 3. Проверка перед constraint ────────────────────────────────────────────
// Должно вернуть 0. Если нет — constraint создавать нельзя, разбирайся с
// остатками руками: скорее всего, кто-то писал в граф в обход graph_repo.
MATCH (g:MarketGroup)
WHERE g.symbol IS NULL OR g.group_id IS NULL
RETURN count(g) AS nodes_without_key;

// ─── 4. Дубликаты по новому ключу ────────────────────────────────────────────
// Тоже должно быть пусто. Дубликаты возможны, если граф уже успел принять
// кандидатов не по BTC до этой миграции.
MATCH (g:MarketGroup)
WITH g.symbol AS symbol, g.group_id AS group_id, count(*) AS n
WHERE n > 1
RETURN symbol, group_id, n ORDER BY n DESC;

// ─── 5. Constraint ───────────────────────────────────────────────────────────
CREATE CONSTRAINT market_group_key IF NOT EXISTS
FOR (g:MarketGroup) REQUIRE (g.symbol, g.group_id) IS UNIQUE;

// ─── 6. Итог ─────────────────────────────────────────────────────────────────
MATCH (g:MarketGroup)
RETURN g.symbol AS symbol, count(g) AS groups ORDER BY symbol;
