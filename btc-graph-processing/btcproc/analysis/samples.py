"""
Выборка кандидатов для замеров: одна модель, одна метрика, честный джойн.

Общее для всех скриптов замера. Вынесено сюда не ради экономии строк, а
потому что здесь легко ошибиться одинаково три раза: ограничение выборки
ОДНОЙ моделью состояний — не осторожность, а исправление конкретной ошибки,
которая уже стоила одного ложного вывода.

Как это выглядело. Замер без ограничения брал всех кандидатов монеты, а
выпускали их разные модели: полные прогоны и частичные (`train --start …`).
Частичные покрывают только свежий период, поэтому скапливаются в holdout и
удваивают там плотность кандидатов. На BTC ни один атом не пережил holdout,
корреляция лифтов между половинами −0.09, знак совпал у 32% атомов — хуже
подбрасывания монеты. На ETH и SOL, где лишних прогонов не было, +0.63 и
+0.61. Списать это на «рынок BTC изменился» было очень легко и совершенно
неверно.
"""
from __future__ import annotations

import pandas as pd

from btcproc import config
from btcproc.db import runs as runs_repo
from btcproc.db.session import fetch_all

#: Метрика «ожидание системы»: доля исторических аналогов, закрывшихся вверх.
#: Отвечает на вопрос «коррелирует ли атом с исторически перекошенными
#: конфигурациями», а не «предсказывает ли атом исход».
_EXPECTED = """
SELECT c.ts,
       (c.payload->>'long_outcome_share')::float8 AS metric
       {extra}
FROM candidates c
{join}
WHERE c.symbol = %s
  AND c.payload->>'long_outcome_share' IS NOT NULL
  AND {scope}
ORDER BY c.ts
"""

#: Метрика «фактический исход»: is_up бара кандидата на горизонте. Свободна
#: от зацикливания на ожиданиях системы — выводы делаются по ней.
_REALIZED = """
SELECT c.ts,
       CASE WHEN o.is_up THEN 1.0 ELSE 0.0 END AS metric
       {extra}
FROM candidates c
JOIN outcomes o ON o.symbol = c.symbol AND o.ts = c.ts AND o.horizon = %s
{join}
WHERE c.symbol = %s
  AND o.valid
  AND o.is_up IS NOT NULL
  AND {scope}
ORDER BY c.ts
"""

_EVENTS_JOIN = "JOIN bar_events e ON e.symbol = c.symbol AND e.ts = c.ts"
_EVENTS_COLUMNS = ", e.atoms, e.context_atoms"


def resolve_model_run(symbol: str, run_id: int | None) -> int:
    """Прогон, задающий модель. По умолчанию — последний успешный train монеты."""
    if run_id:
        return run_id
    latest = runs_repo.latest_completed_run("train", symbol)
    if not latest:
        raise SystemExit(f"У {symbol} нет завершённого train — мерить нечего.")
    return int(latest["run_id"])


def load(symbol: str, metric: str, model_run: int,
         with_atoms: bool = False) -> pd.DataFrame:
    """
    Кандидаты ОДНОЙ модели состояний: ts, metric и (опционально) атомы бара.

    `with_atoms=True` добавляет INNER JOIN с `bar_events`. Он же отсекает
    кандидатов на барах, которые никакой прогон ещё не размечал, — это
    честно, но объём выборки от этого зависит, поэтому замеры без атомов
    джойн не делают.
    """
    scope_sql, scope_params = runs_repo.model_run_scope(model_run)
    scope_sql = scope_sql.replace("run_id", "c.run_id", 1)
    extra = _EVENTS_COLUMNS if with_atoms else ""
    join = _EVENTS_JOIN if with_atoms else ""

    if metric == "realized":
        sql = _REALIZED.format(extra=extra, join=join, scope=scope_sql)
        rows = fetch_all(sql, (config.data.horizon, symbol, *scope_params))
    else:
        sql = _EXPECTED.format(extra=extra, join=join, scope=scope_sql)
        rows = fetch_all(sql, (symbol, *scope_params))

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return frame


def merge_atom_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Склеивает `atoms` и `context_atoms` в одну колонку `atoms`.

    Строки с NULL в `context_atoms` выбрасываются и считаются отдельно: NULL
    там означает «бар размечен прогоном до появления колонки», и молча
    подмешивать такие строки нельзя — по ним фоновые атомы неизвестны, а не
    «отсутствуют».
    """
    stale = int(frame["context_atoms"].isna().sum())
    if stale:
        frame = frame[frame["context_atoms"].notna()].copy()
    frame["atoms"] = [
        list(row["atoms"]) + list(row["context_atoms"])
        for _, row in frame.iterrows()
    ]
    return frame, stale
