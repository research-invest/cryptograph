"""
Запросы для админки. Здесь только чтение — ничего не считает и не пишет.
"""
from __future__ import annotations

import colorsys
from typing import Any, Sequence

import pandas as pd

from btcproc import config
from btcproc.admin.single_flight import SingleFlight
from btcproc.db import repo, runs as runs_repo
from btcproc.db import session
from btcproc.states import naming

#: Потолок на каждый запрос страницы. Ставится здесь, а не в пуле и не в
#: конфиге сервера, потому что пул общий: фоновые потоки той же админки
#: гоняют train с bulk-вставками на десятки минут, и единый потолок убивал бы
#: их. Смысл потолка — не ускорить страницу, а сделать невозможным сценарий
#: «одна кривая страница положила базу для всех»: запрос, идущий минуту,
#: заведомо сломан, и его лучше оборвать с ошибкой в логе.
STATEMENT_TIMEOUT_MS = 60_000


def fetch_all(sql: str, params: Sequence[Any] | None = None) -> list[dict]:
    return session.fetch_all(sql, params, timeout_ms=STATEMENT_TIMEOUT_MS)


def fetch_one(sql: str, params: Sequence[Any] | None = None) -> dict | None:
    return session.fetch_one(sql, params, timeout_ms=STATEMENT_TIMEOUT_MS)

# Потолок маркеров на графике: больше пятисот стрелок всё равно сливаются в
# кашу и перекрывают раскраску состояний. Факт обрезки отдаётся наружу
# (`markers_truncated`) — молча терять кандидатов нельзя.
MARKER_LIMIT = 500

#: Потолок точек в серии признака на нижней панели графика. Окно графика
#: ограничено 5000 барами, так что на обычном пути он не срабатывает — он
#: страхует прямой запрос к API без границ окна.
INDICATOR_LIMIT = 6000


# Область видимости прогонов живёт в db/runs.py: ею пользуется не только
# админка, но и замеры в scripts/, а дублировать такое условие нельзя —
# разъедутся молча.
model_run_scope = runs_repo.model_run_scope


def overview(symbol: str | None = None) -> dict:
    """
    Сводка для дашборда по выбранной монете.

    symbol прокидывается аргументом, а не берётся из config: селектор в шапке
    должен переключать ВСЕ цифры страницы, а не только заголовок.
    """
    symbol = symbol or config.data.symbol
    coverage = fetch_all(
        "SELECT tf, count(*) AS bars, min(ts) AS first_ts, max(ts) AS last_ts "
        "FROM ohlcv WHERE symbol = %s GROUP BY tf ORDER BY tf",
        (symbol,),
    )
    last_train = runs_repo.latest_completed_run("train", symbol)
    run_id = last_train["run_id"] if last_train else None

    # Фильтр по монете нужен даже при фильтре по run_id: run_id уже задаёт
    # монету, но без symbol запрос без прогона (run_id=None) посчитал бы
    # кандидатов всех монет сразу.
    scope_sql, scope_params = model_run_scope(run_id) if run_id else ("", [])
    totals = fetch_one(
        "SELECT count(*) AS candidates, "
        "count(*) FILTER (WHERE emitted_at IS NOT NULL) AS emitted, "
        "count(*) FILTER (WHERE rating = 'STRONG') AS strong, "
        "count(*) FILTER (WHERE rating = 'MODERATE') AS moderate, "
        "count(*) FILTER (WHERE rating = 'WEAK') AS weak, "
        "avg(quality_score) AS avg_quality "
        "FROM candidates WHERE symbol = %s" + (f" AND {scope_sql}" if run_id else ""),
        [symbol, *scope_params],
    ) or {}

    graph_size = fetch_one(
        "SELECT (SELECT count(*) FROM market_groups WHERE run_id = %s) AS groups, "
        "(SELECT count(*) FROM transitions WHERE run_id = %s) AS transitions",
        (run_id, run_id),
    ) if run_id else {}

    return {
        "symbol": symbol,
        "base_tf": config.data.base_tf,
        "horizon": config.data.horizon,
        "coverage": coverage,
        "last_train": last_train,
        "totals": totals,
        "graph": graph_size or {},
        "active_run": runs_repo.active_run(symbol=symbol),
        "active_runs": runs_repo.active_runs(),
    }


def rating_distribution(run_id: int | None = None, symbol: str | None = None) -> list[dict]:
    sql = (
        "SELECT rating, direction, count(*) AS n, avg(quality_score) AS avg_quality "
        "FROM candidates WHERE rating IS NOT NULL"
    )
    params: list[Any] = []
    if symbol:
        sql += " AND symbol = %s"
        params.append(symbol)
    if run_id:
        scope_sql, scope_params = model_run_scope(run_id)
        sql += f" AND {scope_sql}"
        params.extend(scope_params)
    sql += " GROUP BY rating, direction ORDER BY rating, direction"
    return fetch_all(sql, params)


def graph_payload(run_id: int, min_count: int = 1, rarity: str | None = None) -> dict:
    """
    Узлы и рёбра графа состояний в формате Cytoscape.

    `market_groups` и `transitions` пишет только train, поэтому выбранный в
    селекторе live-прогон разыменовывается в свой train через `model_root` —
    иначе граф для него был бы пуст. Раскраска `/chart` устроена так же.
    """
    run_id = runs_repo.model_root(run_id)
    groups = fetch_all(
        "SELECT group_id, size AS count, share, dominant_bias, up_share, avg_ret_pct, "
        "top_features, name FROM market_groups WHERE run_id = %s ORDER BY group_id",
        (run_id,),
    )
    sql = "SELECT * FROM transitions WHERE run_id = %s AND count >= %s"
    params: list[Any] = [run_id, min_count]
    if rarity:
        sql += " AND rarity = ANY(%s)"
        params.append([r.strip() for r in rarity.split(",") if r.strip()])
    sql += " ORDER BY count DESC"
    transitions = fetch_all(sql, params)

    if not groups:
        return {"nodes": [], "edges": [], "feature_labels": {}}

    from btcproc.states.graph import to_cytoscape

    payload = to_cytoscape(pd.DataFrame(groups), pd.DataFrame(transitions))
    # Подписи признаков едут словарём на весь граф, а не полем в каждом узле:
    # один и тот же признак выделяется у десятков состояний, и русская строка
    # в каждом top_features раздувала бы ответ без единого нового факта. Тот
    # же приём, что с atom_labels на графике.
    seen = {feature for node in payload["nodes"] for feature in node["data"]["top_features"]}
    payload["feature_labels"] = naming.feature_labels(sorted(seen))
    return payload


class SymbolRunMismatch(ValueError):
    """
    run_id принадлежит другой монете.

    Отдельное исключение, потому что симптом обманчив: LEFT JOIN по
    (symbol, ts, run_id) просто не совпадёт, и график BTC с прогоном ETH
    отдаст свечи вообще без раскраски. Выглядит как «граф ничего не нашёл»,
    а на самом деле выбраны несогласованные монета и прогон.
    """


def run_symbol(run_id: int) -> str | None:
    row = fetch_one("SELECT symbol FROM runs WHERE run_id = %s", (run_id,))
    return row["symbol"] if row else None


def _rating_filter(rating: str | None) -> tuple[str, list[Any]]:
    """
    Кусок WHERE по рейтингу для маркеров графика.

    Вынесен из `chart_data`, потому что нужен дважды и в двух запросах:
    одним отбираются сами маркеры, вторым — счётчики по слоям. Разъедься эти
    два условия, и строка «ещё N не показано» начала бы врать.
    """
    if not rating:
        return "", []
    values = [r.strip() for r in rating.split(",") if r.strip()]
    if not values:
        return "", []
    return " AND c.rating = ANY(%s)", [values]


def _marker_counts(symbol: str, first_ts: Any, last_ts: Any,
                   rating_sql: str, rating_params: list[Any]) -> dict:
    """
    Сколько кандидатов в окне выпущено вживую, а сколько посчитано задним
    числом переобучением.

    Считается по ВСЕМ моделям: вопрос «что тут было» не про модель. Дубли по
    одному бару от разных поколений моделей при этом реальны и в счётчик
    входят — это честнее, чем показать одно число и умолчать, что оно из
    нескольких нумераций.
    """
    row = fetch_one(
        "SELECT count(*) FILTER (WHERE r.kind = 'live')  AS issued, "
        "       count(*) FILTER (WHERE r.kind <> 'live') AS retro "
        "FROM candidates c JOIN runs r ON r.run_id = c.run_id "
        f"WHERE c.symbol = %s AND c.ts BETWEEN %s AND %s{rating_sql}",
        [symbol, first_ts, last_ts, *rating_params],
    )
    return {"issued": (row or {}).get("issued", 0) or 0,
            "retro": (row or {}).get("retro", 0) or 0}


def chart_data(run_id: int, symbol: str | None = None, start: str | None = None,
               end: str | None = None, limit: int = 1500,
               rating: str | None = None, layer: str = "issued") -> dict:
    """
    Свечи с раскраской по состоянию + маркеры кандидатов.

    Раскраска — главный смысл этой страницы: видно, где именно граф считает,
    что рынок сменил состояние, и совпадает ли это с тем, что видит глаз.

    symbol и run_id обязаны быть согласованы — иначе раскраски не будет, и
    понять почему по пустому графику невозможно. Проверяем явно.

    **Маркеры кандидатов НЕ ограничены моделью выбранного прогона** (правка
    2026-08-20, журнал 51). Раскраска — свойство модели, и она обязана быть
    из одной нумерации. Кандидат — свойство ИСТОРИИ: он либо был выпущен на
    этом баре, либо нет, и переобучение задним числом этого не меняет.

    До правки маркеры фильтровались тем же `model_run_scope`, что и раскраска,
    и получалось наоборот: после каждого воскресного `train` со старых баров
    пропадало всё, что система на них выпускала, а вместо этого показывалось
    то, что новая модель насчитала на них ретроспективно, — и на дефолтном
    фильтре «STRONG + MODERATE» не показывалось вообще ничего, потому что
    недельный `train` идёт с `--no-emit` и рейтинга у его кандидатов нет.

    `layer` разделяет два разных по смыслу слоя:

    * `issued` (по умолчанию) — кандидаты live-прогонов: то, что система
      сказала в тот момент, на модели, обученной строго до этих баров;
    * `all` — плюс кандидаты train-прогонов. Это ретроспектива: модель
      обучена на всей истории, включая будущее размечаемого бара, поэтому
      её кандидат на историческом баре — воспоминание, а не суждение.
      Тот же довод, по которому прогноз размаха не даёт чисел на барах
      своего обучения (инвариант 13а).
    """
    symbol = symbol or config.data.symbol
    owner = run_symbol(run_id)
    if owner and owner != symbol:
        raise SymbolRunMismatch(
            f"Прогон #{run_id} посчитан по {owner}, а график запрошен для {symbol}. "
            f"Выбери прогон этой монеты — состояния между монетами несопоставимы."
        )

    # Раскраска берётся не по одному run_id, а по всей МОДЕЛИ: train размечает
    # историю раз в неделю, а всё, что появилось после него, размечено только
    # live-прогонами. При join по единственному run_id дефолтный вид графика
    # (последний train) показывал серым хвост длиной до недели — выглядело как
    # «состояния пропали». Маркеры кандидатов на этой же странице через scope
    # уже ходили, то есть источники расходились.
    #
    # Свежая разметка побеждает: LATERAL берёт строку с наибольшим run_id.
    # Это же и дедуплицирует join — прямой join по нескольким run_id размножил
    # бы бары.
    root = runs_repo.model_root(run_id)
    scope_sql, scope_params = model_run_scope(root, alias="b")

    # Лимит всегда режет диапазон с одного конца — важно, с какого. Если задана
    # нижняя граница, окно отсчитывается от неё вперёд: при сортировке DESC
    # оставались последние N баров диапазона, и начало заданного периода молча
    # пропадало (запрос с 1 июля отдавал бары с 15-го). Без start смысл обратный
    # — нужны свежие бары, поэтому берём с конца.
    inner = ("SELECT ts, symbol, open, high, low, close, volume "
             "FROM ohlcv WHERE symbol = %s AND tf = %s")
    inner_params: list[Any] = [symbol, config.data.base_tf]
    if start:
        inner += " AND ts >= %s"
        inner_params.append(start)
    if end:
        inner += " AND ts <= %s"
        inner_params.append(end)
    inner += " ORDER BY ts ASC LIMIT %s" if start else " ORDER BY ts DESC LIMIT %s"
    inner_params.append(limit)

    # Окно баров отбирается ДО lateral-джойна: иначе поиск разметки шёл бы по
    # всей истории, а не по показанным полутора тысячам баров.
    # Атомы бара приезжают тем же запросом: у `bar_events` ключ (symbol, ts)
    # без run_id — событийный слой считается один раз и от модели не зависит.
    # Джойн обычный, а не LATERAL: размножить бары он не может.
    sql = (
        "SELECT o.ts, o.open, o.high, o.low, o.close, o.volume, "
        "s.group_id, s.is_transition, s.transition_id, s.age_bucket, s.entropy, "
        "e.atoms, e.context_atoms "
        f"FROM ({inner}) o "
        "LEFT JOIN LATERAL ("
        "  SELECT b.group_id, b.is_transition, b.transition_id, b.age_bucket, b.entropy "
        "  FROM bar_states b "
        f"  WHERE b.symbol = o.symbol AND b.ts = o.ts AND {scope_sql} "
        "  ORDER BY b.run_id DESC LIMIT 1"
        ") s ON true "
        "LEFT JOIN bar_events e ON e.symbol = o.symbol AND e.ts = o.ts "
        "ORDER BY o.ts ASC"
    )
    bars = fetch_all(sql, [*inner_params, *scope_params])
    if not bars:
        return {"bars": [], "markers": [], "groups": [], "markers_truncated": False,
                "atom_labels": {}, "indicators": indicator_catalog(run_id)}

    first_ts, last_ts = bars[0]["ts"], bars[-1]["ts"]
    # rating="none" — маркеры не нужны вовсе: на длинном окне их сотни,
    # и они перекрывают саму раскраску состояний.
    truncated = False
    counts = {"issued": 0, "retro": 0}
    if rating == "none":
        candidates = []
    else:
        rating_sql, rating_params = _rating_filter(rating)
        counts = _marker_counts(symbol, first_ts, last_ts, rating_sql, rating_params)
        sql_c = (
            "SELECT c.candidate_id, c.ts, c.research_side, c.rating, c.quality_score, "
            "       c.transition_id, r.kind AS run_kind "
            "FROM candidates c JOIN runs r ON r.run_id = c.run_id "
            f"WHERE c.symbol = %s AND c.ts BETWEEN %s AND %s{rating_sql}"
        )
        params_c: list[Any] = [symbol, first_ts, last_ts, *rating_params]
        if layer != "all":
            sql_c += " AND r.kind = 'live'"
        # DESC, а не ASC: при сортировке по возрастанию лимит отрезал самые
        # свежие маркеры — на окне в 5000 баров свечи шли до сегодня, а стрелки
        # обрывались двумя неделями раньше, и выглядело это как «кандидатов нет».
        # Берём последние MARKER_LIMIT, лишнюю строку — чтобы знать про обрезку.
        sql_c += " ORDER BY c.ts DESC LIMIT %s"
        params_c.append(MARKER_LIMIT + 1)
        rows_c = fetch_all(sql_c, params_c)
        truncated = len(rows_c) > MARKER_LIMIT
        # lightweight-charts требует маркеры по возрастанию времени.
        candidates = list(reversed(rows_c[:MARKER_LIMIT]))

    # Палитра — по всем состояниям модели, а не по попавшим в окно: иначе
    # цвет состояния менялся бы при каждом сдвиге периода (см. state_palette).
    palette = state_palette(root)
    window_ids = sorted({b["group_id"] for b in bars if b["group_id"] is not None})
    # Состояние, которого нет в market_groups, — не норма, но и не повод
    # оставлять бар без цвета: разметка live могла приехать раньше, чем
    # досчитался граф. Такие получают оттенки в хвосте палитры.
    missing = [gid for gid in window_ids if gid not in palette]
    for offset, gid in enumerate(missing):
        palette[gid] = _color(len(palette) + offset, len(palette) + len(missing))
    # Имена состояний для легенды: без них «состояние 7» ничего не говорит,
    # а сверяться с таблицей ради каждого цвета никто не будет. Ключ — root,
    # а не run_id: имена, как и сам граф, пишет только train.
    names = {
        float(r["group_id"]): r["name"] or ""
        for r in fetch_all(
            "SELECT group_id, name FROM market_groups WHERE run_id = %s", (root,)
        )
    }

    out_bars = []
    seen_atoms: set[str] = set()
    for b in bars:
        color = palette.get(b["group_id"], "#8892a0")
        # None и [] здесь разные вещи: пустой список — «событий на баре нет»,
        # отсутствие строки в bar_events — «событийный слой этот бар не считал»
        # (хвост свежее последнего прогона). Схлопывать их нельзя: второе
        # читалось бы как спокойный рынок.
        atoms = None if b["atoms"] is None else list(b["atoms"])
        context = None if b["atoms"] is None else list(b["context_atoms"] or ())
        seen_atoms.update(atoms or ())
        seen_atoms.update(context or ())
        out_bars.append({
            "time": int(b["ts"].timestamp()),
            "open": b["open"], "high": b["high"], "low": b["low"], "close": b["close"],
            "color": color, "borderColor": color, "wickColor": color,
            "group_id": b["group_id"],
            # "NaN" строкой прилетает из pandas на барах без предыдущего
            # состояния (первый бар модели). Наружу это уходило подписью
            # «переход NaN» — читается как сбой, хотя перехода просто нет.
            "transition_id": None if b["transition_id"] in (None, "NaN") else b["transition_id"],
            "volume": b["volume"],
            # Атомы отдаём идентификаторами, подписи — общим словарём ниже:
            # атомов на бар до десятка, и русская строка в каждом баре
            # раздувала бы ответ впятеро без единого нового факта.
            "atoms": atoms,
            "context": context,
        })

    markers = []
    for c in candidates:
        quality = "" if c["quality_score"] is None else f" {c['quality_score']:.2f}"
        # Ретроспективный кандидат обязан выглядеть иначе, а не просто стоять
        # рядом: путать «система это сказала» и «переобученная модель так
        # думает про прошлое» нельзя, а стрелка того же вида ровно к этому и
        # приглашает. Кружок и приглушённый цвет вместо стрелки рейтинга.
        retro = c["run_kind"] != "live"
        markers.append({
            "time": int(c["ts"].timestamp()),
            "position": "belowBar" if c["research_side"] == "long" else "aboveBar",
            "color": "#64748b" if retro else _rating_color(c["rating"]),
            "shape": "circle" if retro
                     else ("arrowUp" if c["research_side"] == "long" else "arrowDown"),
            "text": (f"ретро {c['research_side']}" if retro
                     else f"{c['rating'] or '—'} {c['research_side']}{quality}"),
            "id": c["candidate_id"],
        })

    return {
        "bars": out_bars,
        "markers": markers,
        # Состояния ОКНА, а не всей модели: подсказке и подписям на графике
        # нужны только те, что видны. Полный список — на /states.
        "groups": [
            {"group_id": gid, "color": palette[gid], "name": names.get(gid, "")}
            for gid in window_ids
        ],
        "markers_truncated": truncated,
        # Разбивка по слоям — чтобы строка под графиком могла сказать, сколько
        # кандидатов в окне НЕ показано. Молчание здесь и было исходной бедой.
        "marker_counts": counts,
        "atom_labels": {atom: naming.label_for_atom(atom) for atom in sorted(seen_atoms)},
        # Список признаков модели едет вместе с барами, а не отдельным
        # запросом: он зависит от прогона, и рассинхрон селектора с
        # раскраской означал бы запрос несуществующего в модели признака.
        "indicators": indicator_catalog(run_id),
    }


def _feature_version(root: int) -> str | None:
    """Набор признаков, которым обучена модель прогона."""
    row = fetch_one("SELECT feature_ver FROM state_models WHERE run_id = %s", (root,))
    return row["feature_ver"] if row else None


def indicator_catalog(run_id: int) -> list[dict]:
    """
    Признаки, доступные для нижней панели графика.

    Берём не «что умеет считать код», а состав набора, которым обучена модель
    прогона: на v1 SMC-признаков в базе нет вовсе, и предлагать их в селекторе
    значило бы обещать пустую панель. Порядок и подписи — из словаря имён
    состояний (`naming.AXES`), чтобы индикатор и имя состояния говорили об
    одном и том же одними словами.
    """
    version = _feature_version(runs_repo.model_root(run_id))
    if not version:
        return []
    row = fetch_one("SELECT names FROM feature_sets WHERE version = %s", (version,))
    if not row:
        return []
    available = set(row["names"])

    out = [
        {"name": feature, "axis": axis, "title": title, "high": positive, "low": negative}
        for axis, feature, title, positive, negative in naming.feature_catalog()
        if feature in available
    ]
    # Признак без словарной строки — нарушение инварианта 7, и тест полноты
    # его ловит. Но молча прятать такой признак из интерфейса нельзя: пусть
    # видно будет здесь, а не только в упавшем тесте.
    named = {item["name"] for item in out}
    out.extend(
        {"name": feature, "axis": "без описания", "title": feature, "high": "", "low": ""}
        for feature in row["names"] if feature not in named
    )
    return out


def indicator_series(run_id: int, symbol: str, name: str,
                     start: Any = None, end: Any = None) -> dict:
    """
    Значения одного признака по барам окна — для нижней панели графика.

    Читаем ровно те числа, на которых обучалась модель, а не пересчитываем
    индикатор в браузере: пересчёт разошёлся бы с моделью на параметрах окон
    и на сдвиге старших ТФ, и панель показывала бы не то, что видит граф.
    """
    version = _feature_version(runs_repo.model_root(run_id))
    if not version:
        return {"name": name, "points": [], "note": "у прогона нет модели состояний"}
    row = fetch_one("SELECT names FROM feature_sets WHERE version = %s", (version,))
    names: list[str] = list(row["names"]) if row else []
    if name not in names:
        return {"name": name, "points": [],
                "note": f"признака нет в наборе {version} этого прогона"}

    # Массивы в postgres 1-based; индекс берём по составу набора, а не по
    # порядку в коде — набор монеты может быть старым.
    position = names.index(name) + 1
    sql = ("SELECT ts, values[%s] AS value FROM features "
           "WHERE symbol = %s AND version = %s")
    params: list[Any] = [position, symbol, version]
    if start is not None:
        sql += " AND ts >= %s"
        params.append(start)
    if end is not None:
        sql += " AND ts <= %s"
        params.append(end)
    # Потолок на случай запроса без границ: у монеты сотни тысяч баров, и
    # вся история одним ответом — полтора мегабайта JSON ради полутора тысяч
    # видимых точек. Берём последние, а не первые: без границ нужен свежий
    # хвост. Окно графика (максимум 5000 баров) в этот потолок помещается,
    # то есть на обычном пути обрезка не срабатывает.
    sql += " ORDER BY ts DESC LIMIT %s"
    params.append(INDICATOR_LIMIT)

    rows = fetch_all(sql, params)
    points = [
        {"time": int(r["ts"].timestamp()), "value": r["value"]}
        for r in reversed(rows) if r["value"] is not None
    ]
    described = {item["name"]: item for item in indicator_catalog(run_id)}.get(name, {})
    return {
        "name": name,
        "version": version,
        # Русское название — то же, что в селекторе и в панели узла графа:
        # три места, один словарь naming.AXES.
        "title": described.get("title", name),
        "high": described.get("high", ""),
        "low": described.get("low", ""),
        "points": points,
        # Прогрев окон съедает первые недели истории, а набор v2 посчитан не
        # на всей: пустая панель без объяснения читается как поломка.
        "note": "" if points else "нет посчитанных значений в этом окне",
    }


def state_palette(run_id: int) -> dict[float, str]:
    """
    Цвет каждого состояния МОДЕЛИ — по всем её состояниям сразу, а не по тем,
    что попали в окно графика.

    Раньше палитра считалась от набора состояний видимого периода, и цвет был
    свойством ОКНА: сдвинул график на неделю — половина состояний поменяла
    оттенок, потому что изменился их порядковый номер в отсортированном
    списке. Заметить это трудно (все цвета «выглядят правильно»), а вреда два:
    запомнить цвет нельзя, и список состояний на отдельной странице
    показывал бы не те цвета, что график.

    Ключ — корень модели: `market_groups` пишет только `train`, и у всех его
    `live`-прогонов состояния те же самые.
    """
    root = runs_repo.model_root(run_id)
    group_ids = [
        float(r["group_id"]) for r in fetch_all(
            "SELECT group_id FROM market_groups WHERE run_id = %s ORDER BY group_id",
            (root,),
        )
    ]
    return {gid: _color(i, len(group_ids)) for i, gid in enumerate(group_ids)}


def _color(index: int, total: int) -> str:
    """
    Равномерно разнесённые оттенки — соседние состояния не сливаются.

    Возвращаем именно hex: lightweight-charts парсит цвета сам и на строке
    вида `hsl(32, 62%, 55%)` падает с «Cannot parse color».
    """
    hue = 360.0 * index / max(total, 1)
    red, green, blue = colorsys.hls_to_rgb(hue / 360.0, 0.55, 0.62)
    return "#{:02x}{:02x}{:02x}".format(
        round(red * 255), round(green * 255), round(blue * 255)
    )


def _rating_color(rating: str | None) -> str:
    return {"STRONG": "#16a34a", "MODERATE": "#d97706", "WEAK": "#94a3b8"}.get(
        rating or "", "#64748b"
    )


def candidates_page(
    run_id: int | None = None,
    symbol: str | None = None,
    rating: str | None = None,
    direction: str | None = None,
    min_quality: float | None = None,
    transition: str | None = None,
    emitted: str | None = None,
    page: int = 1,
    per_page: int = 50,
) -> dict:
    where, params = [], []
    if symbol:
        where.append("symbol = %s")
        params.append(symbol)
    if run_id:
        where.append("run_id = %s")
        params.append(run_id)
    if rating:
        where.append("rating = %s")
        params.append(rating)
    if direction:
        where.append("research_side = %s")
        params.append(direction)
    if min_quality is not None:
        where.append("quality_score >= %s")
        params.append(min_quality)
    if transition:
        where.append("transition_id = %s")
        params.append(transition)
    if emitted == "yes":
        where.append("emitted_at IS NOT NULL")
    elif emitted == "no":
        where.append("emitted_at IS NULL")

    clause = f" WHERE {' AND '.join(where)}" if where else ""
    total = (fetch_one(f"SELECT count(*) AS n FROM candidates{clause}", params) or {}).get("n", 0)

    offset = max(page - 1, 0) * per_page
    rows = fetch_all(
        "SELECT candidate_id, symbol, ts, transition_id, event_block_id, research_side, "
        "research_score, sample_size, quality_score, rating, warning_flags, "
        "emitted_at, emit_error FROM candidates"
        + clause
        + " ORDER BY ts DESC, quality_score DESC NULLS LAST LIMIT %s OFFSET %s",
        params + [per_page, offset],
    )
    return {
        "rows": rows,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": max(1, (total + per_page - 1) // per_page),
    }


#: Что попадает в общую сводку дашборда. WEAK не попадает намеренно: их
#: большинство, и лента из них перестаёт быть сводкой.
HIGHLIGHT_RATINGS = ("STRONG", "MODERATE")


def recent_highlights(hours: int = 24, limit: int = 50) -> list[dict]:
    """
    Заметные кандидаты ПО ВСЕМ монетам за последние часы.

    Единственное место в админке, которое смотрит поперёк монет: остальные
    страницы работают внутри выбранной. Смысл именно в этом — оператор
    открывает сводку, чтобы увидеть, где вообще что-то произошло, не
    перебирая шесть монет по очереди.

    Рейтинг ставит btc-graph уже после отправки, поэтому свежий кандидат
    несколько секунд живёт с `rating IS NULL` и в выдачу не попадает — это
    не потеря, а задержка.

    Фильтр по `ts`, а не по `created_at`: интересует время бара, к которому
    относится кандидат, а не момент, когда его посчитали (после `train` они
    расходятся на всю историю).
    """
    return fetch_all(
        "SELECT candidate_id, symbol, run_id, ts, transition_id, event_block_id, "
        "research_side, research_score, sample_size, quality_score, rating, "
        "warning_flags, emitted_at FROM candidates "
        "WHERE rating = ANY(%s) AND ts >= now() - make_interval(hours => %s) "
        "ORDER BY ts DESC, quality_score DESC NULLS LAST LIMIT %s",
        (list(HIGHLIGHT_RATINGS), hours, limit),
    )


def candidate_detail(candidate_id: str) -> dict | None:
    return fetch_one("SELECT * FROM candidates WHERE candidate_id = %s", (candidate_id,))


# ─── Уведомления ────────────────────────────────────────────────────────────
def deliveries(rule_id: int | None = None, limit: int = 50) -> list[dict]:
    """
    Журнал доставок — единственное место, где видно, что стало с вебхуком.
    Отправка идёт в фоновом потоке, поэтому больше её следов нигде нет.
    """
    sql = (
        "SELECT d.*, r.name AS rule_name, r.url FROM notification_deliveries d "
        "LEFT JOIN notification_rules r ON r.rule_id = d.rule_id"
    )
    params: list[Any] = []
    if rule_id:
        sql += " WHERE d.rule_id = %s"
        params.append(rule_id)
    sql += " ORDER BY d.created_at DESC LIMIT %s"
    params.append(limit)
    return fetch_all(sql, params)


def delivery_totals() -> list[dict]:
    """Сводка по правилам: сколько ушло, сколько сорвалось."""
    return fetch_all(
        "SELECT rule_id, status, count(*) AS n, max(created_at) AS last_at "
        "FROM notification_deliveries GROUP BY rule_id, status"
    )


def latest_candidate_row(symbol: str | None = None) -> dict | None:
    """
    Свежий кандидат для кнопки «проверить». Берётся настоящий, а не выдуманный:
    смысл проверки в том, чтобы принимающая система увидела реальное тело.
    """
    sql = (
        "SELECT candidate_id, run_id, symbol, ts, payload, quality_score, rating, "
        "direction, warning_flags, evaluation FROM candidates"
    )
    params: list[Any] = []
    if symbol:
        sql += " WHERE symbol = %s"
        params.append(symbol)
    sql += " ORDER BY ts DESC LIMIT 1"
    return fetch_one(sql, params)


def transition_options(run_id: int | None, limit: int = 200) -> list[str]:
    """Переходы последнего прогона — для подсказки в форме правила."""
    if not run_id:
        return []
    return [
        row["transition_id"]
        for row in fetch_all(
            "SELECT transition_id FROM transitions WHERE run_id = %s "
            "ORDER BY count DESC LIMIT %s",
            (run_id, limit),
        )
    ]


def top_groups(run_id: int, limit: int = 10) -> list[dict]:
    """
    Крупнейшие состояния — где рынок проводит больше всего времени.

    `name` берётся вместе с числами намеренно: номер состояния сам по себе
    не значит ничего (он перенумеровывается каждым train и осмыслен только
    в паре `(symbol, run_id)`), поэтому таблица из одних номеров читается
    как список идентификаторов, а не как описание рынка.
    """
    return fetch_all(
        "SELECT group_id, name, size, share, dominant_bias, up_share, avg_ret_pct "
        "FROM market_groups WHERE run_id = %s ORDER BY size DESC LIMIT %s",
        (run_id, limit),
    )


def states_page(run_id: int) -> list[dict]:
    """
    ВСЕ состояния модели — то, чего в интерфейсе не было нигде.

    Сводка показывает десять крупнейших, граф — узлы по одному, график
    раскрашивает бары. Общего списка «какие вообще состояния есть у этой
    модели, как они выглядят и чем отличаются» не было, а он и есть первое,
    что нужно, чтобы читать три остальные страницы.

    Ничего не вычисляется: `market_groups` заполняет `train`, здесь только
    чтение и цвет из общей палитры (инвариант 19). Цвет обязателен — без него
    строку списка не сопоставить со свечой на графике.
    """
    root = runs_repo.model_root(run_id)
    palette = state_palette(root)
    rows = fetch_all(
        "SELECT group_id, name, size, share, dominant_bias, up_share, "
        "       avg_ret_pct, avg_vol_pct, top_features "
        "FROM market_groups WHERE run_id = %s ORDER BY size DESC",
        (root,),
    )
    seen = {name for row in rows for name in (row.get("top_features") or {})}
    labels = naming.feature_labels(sorted(seen))
    for row in rows:
        row["color"] = palette.get(float(row["group_id"]), "#8892a0")
        # top_features — словарь «признак → отклонение в сигмах». В строку
        # идут три самых крупных по модулю, с русской формулировкой из того же
        # словаря, которым названо само состояние: два словаря разъехались бы
        # молча, и подпись начала бы противоречить имени.
        features = row.get("top_features") or {}
        top = sorted(features.items(), key=lambda kv: -abs(kv[1]))[:3]
        row["top"] = [
            {
                "feature": name,
                "value": value,
                "title": labels.get(name, {}).get("title", name),
                "phrase": (labels.get(name, {}).get("high" if value >= 0 else "low")
                           or ""),
            }
            for name, value in top
        ]
    return rows


def transitions_table(run_id: int, limit: int = 200) -> list[dict]:
    return fetch_all(
        "SELECT * FROM transitions WHERE run_id = %s ORDER BY count DESC LIMIT %s",
        (run_id, limit),
    )


# ─── Фон состояния: контекстные атомы ───────────────────────────────────────
#
# Зачем это отдельно от top_features. Контекстные атомы (16 из них — SMC:
# структура, имбалансы, блоки заказов, зоны ликвидности) в вектор признаков не
# входят и в кластеризации не участвуют, поэтому в имя состояния и в
# `top_features` попасть не могут в принципе — это другой канал. Но состояния
# по ним всё равно различаются, и различие содержательное: «здесь цена втрое
# чаще обычного сидит в премиальной зоне» — это про то же состояние, просто
# сказанное другими словами.
#
# Считается как лифт: доля баров состояния с атомом, делённая на долю по всей
# истории. 1.0 — атом встречается ровно как обычно, 2.0 — вдвое чаще.
#
# САМ АГРЕГАТ ЗДЕСЬ БОЛЬШЕ НЕ СЧИТАЕТСЯ. До 2026-08-16 считался, и открытие
# узла графа означало join `bar_states` × `bar_events` по всей истории монеты
# плюс разворот массивов атомов: секунды на прогретом кэше, десятки секунд на
# холодном, параллельные воркеры на каждый запрос. Теперь его пишет train
# (`repo.save_state_context`), а страница читает полторы тысячи готовых строк.
# Разбор — журнал 43.

#: Атом показывается, только если он и заметен, и выражен. Пороги отсекают
#: две разные ерунды: редкий атом даёт огромный лифт на десятке баров, а
#: вездесущий (in_breaker сидит на 83% истории) даёт лифт 1.02 и не значит
#: ничего. Живут здесь, а не в агрегате: менять их можно без пересчёта.
CONTEXT_MIN_SHARE = 0.05
CONTEXT_MIN_LIFT = 1.25
CONTEXT_TOP_N = 6

#: Потолок на разовый досчёт фона прогонам, посчитанным до появления таблицы.
#: Заметно больше страничного: это тот самый тяжёлый агрегат, только один раз
#: за прогон, а не на каждый клик.
CONTEXT_BACKFILL_TIMEOUT_MS = 300_000

#: Кэш «монета+модель → фон по состояниям». Чтение уже дешёвое, но кэш
#: остаётся: он же держит результат разового досчёта старых прогонов.
#: Кэшировать безопасно — разметка train-прогона после завершения не меняется,
#: и ключ включает run_id.
_CONTEXT_CACHE: dict[tuple[str, int], dict[float, list[dict]]] = {}
_CONTEXT_CACHE_LIMIT = 8

#: Пять кликов по узлу — пять потоков uvicorn, и кэш их не спасает: он
#: заполняется после первого ответа, а уходят они в БД одновременно. Первый
#: считает, остальные ждут его результат.
_CONTEXT_FLIGHT = SingleFlight()


def state_context_atoms(run_id: int, symbol: str) -> dict[float, list[dict]]:
    """
    Чем фон каждого состояния отличается от фона рынка в целом.

    Возвращает {group_id: [{atom, share, lift}, ...]} — только выделяющиеся
    атомы, от сильнейшего лифта. Состояния без выраженного фона в словаре
    отсутствуют: пустой список и отсутствие ключа для UI одно и то же, а
    хранить сорок пустых списков незачем.
    """
    key = (symbol, run_id)
    cached = _CONTEXT_CACHE.get(key)
    if cached is not None:
        return cached
    return _CONTEXT_FLIGHT.run(key, lambda: _load_context(key))


def _load_context(key: tuple[str, int]) -> dict[float, list[dict]]:
    symbol, run_id = key
    # Лидер мог посчитать, пока мы стояли в очереди на ключ.
    cached = _CONTEXT_CACHE.get(key)
    if cached is not None:
        return cached

    if not repo.state_context_ready(run_id):
        # Единственное место, где админка что-то считает, и то не сама:
        # зовёт расчёт генератора для прогона, сделанного до появления
        # таблицы. Отметка ставится и при пустом результате, поэтому
        # повторно сюда прогон уже не попадёт.
        repo.save_state_context(run_id, symbol,
                                timeout_ms=CONTEXT_BACKFILL_TIMEOUT_MS)

    result: dict[float, list[dict]] = {}
    for row in repo.load_state_context(run_id):
        if row["share"] < CONTEXT_MIN_SHARE or (row["lift"] or 0) < CONTEXT_MIN_LIFT:
            continue
        bucket = result.setdefault(float(row["group_id"]), [])
        if len(bucket) < CONTEXT_TOP_N:
            bucket.append({
                "atom": row["atom"],
                # Подпись проставляется здесь, а не в шаблоне: словарь
                # формулировок один на проект и живёт в naming.py.
                "label": naming.label_for_atom(row["atom"]),
                "share": float(row["share"]),
                "lift": float(row["lift"]),
            })

    # Простое вытеснение вместо LRU: ключей единицы (монета × модель), а
    # неограниченный рост в долгоживущем процессе админки — это утечка.
    if len(_CONTEXT_CACHE) >= _CONTEXT_CACHE_LIMIT:
        _CONTEXT_CACHE.pop(next(iter(_CONTEXT_CACHE)))
    _CONTEXT_CACHE[key] = result
    return result


def group_detail(run_id: int, group_id: float) -> dict | None:
    """
    Узел графа. Монета не аргумент: run_id уже однозначно её задаёт
    (один прогон = одна монета). Но подписать её в UI обязательно —
    «group_id 7» без монеты бессмысленен.

    `model_root` — по той же причине, что в `graph_payload`: узлы и переходы
    есть только у train-прогона.
    """
    run_id = runs_repo.model_root(run_id)
    node = fetch_one(
        "SELECT * FROM market_groups WHERE run_id = %s AND group_id = %s",
        (run_id, group_id),
    )
    if not node:
        return None
    node["incoming"] = fetch_all(
        "SELECT * FROM transitions WHERE run_id = %s AND cur_group_id = %s "
        "ORDER BY count DESC LIMIT 20",
        (run_id, group_id),
    )
    node["outgoing"] = fetch_all(
        "SELECT * FROM transitions WHERE run_id = %s AND prev_group_id = %s "
        "ORDER BY count DESC LIMIT 20",
        (run_id, group_id),
    )
    # Фон состояния. Отдаётся здесь, а не в graph_payload: запрос тяжёлый,
    # а панель узла и так грузится по клику отдельным вызовом.
    symbol = run_symbol(run_id)
    node["context_atoms"] = (
        state_context_atoms(run_id, symbol).get(float(group_id), []) if symbol else []
    )
    return node
