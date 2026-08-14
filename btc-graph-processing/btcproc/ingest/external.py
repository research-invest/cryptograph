"""
Загрузка внешних дневных рядов — Fear & Greed Index и (в будущем) прочие.

Отделено от `sources.py`/`binance.py`/`bybit.py` намеренно: те качают бары
монет и участвуют в `ingest`/`live`/`train`, а внешние ряды — общерыночные
величины без своей площадки, которые нельзя добирать в регулярном прогоне
(`docs/task_fear_greed.md`, раздел 3.2 и раздел 8):

    * ходить в чужой API из `train` или `live` нельзя — сетевой поход туда не
      должен ронять или замедлять регулярный прогон;
    * загрузка идёт отдельной командой CLI (`fetch-external`), обычно раз
      в сутки из отдельной записи крона, и прогоны только ЧИТАЮТ таблицу.

## Look-ahead — главная ловушка источника

Значение за сутки D характеризует их целиком и известно не раньше конца дня.
Хранится оно здесь без сдвига (сырое значение поставщика на свою дату) — сдвиг
на бар делает уже `features/fear_greed.py` при джойне к барам: значение с
датой D применяется к барам, начинающимся с D+1 00:00 UTC. Здесь этот сдвиг
дублировать нельзя: тогда он оказался бы применён дважды.
"""
from __future__ import annotations

import logging

import httpx
import pandas as pd

from btcproc.db.session import connect, fetch_all
from psycopg2.extras import Json, execute_values

logger = logging.getLogger(__name__)

FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=0&format=json"
FEAR_GREED_SERIES = "fear_greed"

# История индекса начинается 2018-02-01 (docs/task_fear_greed.md, раздел 3.1).
# Заполнять более ранние сутки чем бы то ни было запрещено явно — ни
# константой, ни первым известным значением.
FEAR_GREED_START = pd.Timestamp("2018-02-01", tz="UTC").date()


def fetch_fear_greed(client: httpx.Client | None = None) -> pd.DataFrame:
    """
    Вся история индекса одним запросом. Возвращает day/value/meta.

    Поля ответа не подставляются по памяти, а разбираются фактически:
    `value` (0–100, строкой у поставщика), `value_classification`,
    `timestamp` (unix, начало суток UTC), `time_until_update`.
    """
    own_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        resp = client.get(FEAR_GREED_URL)
        resp.raise_for_status()
        payload = resp.json()
    finally:
        if own_client:
            client.close()

    rows = payload.get("data") or []
    if not rows:
        return pd.DataFrame(columns=["day", "value", "meta"])

    days = pd.to_datetime(
        pd.to_numeric([r["timestamp"] for r in rows]), unit="s", utc=True
    ).date
    values = [float(r["value"]) for r in rows]
    meta = [
        {k: v for k, v in r.items() if k not in ("timestamp", "value")}
        for r in rows
    ]
    frame = pd.DataFrame({"day": days, "value": values, "meta": meta})
    # Поставщик отдаёт свежее первым — для стабильного хранения и для
    # тестов удобнее хронологический порядок.
    return frame.sort_values("day").reset_index(drop=True)


def _existing_values(series: str, days) -> dict:
    if not len(days):
        return {}
    rows = fetch_all(
        "SELECT day, value FROM external_daily WHERE series = %s AND day = ANY(%s)",
        (series, list(days)),
    )
    return {r["day"]: float(r["value"]) for r in rows}


def store_external_daily(frame: pd.DataFrame, series: str) -> dict:
    """
    Upsert ряда в `external_daily`. Возвращает сводку с числом РАЗОШЕДШИХСЯ
    дней — перекрывающихся дней, у которых сохранённое значение отличалось
    от свежего.

    Расхождение — не молчаливый upsert (docs/task_fear_greed.md, раздел 3.3):
    если поставщик задним числом пересчитывает историю, всё измеренное на
    старом значении измерено на данных, которых в тот момент не существовало.
    Строка обновляется на свежую (поставщик авторитетнее нашего снимка), но
    предупреждение обязано попасть в лог, а решение — в журнал разработки.
    """
    if frame.empty:
        return {"rows": 0, "revised_days": 0, "revised": []}

    before = _existing_values(series, frame["day"].tolist())
    revised = [
        {"day": row.day, "old": before[row.day], "new": row.value}
        for row in frame.itertuples()
        if row.day in before and before[row.day] != row.value
    ]
    if revised:
        logger.warning(
            "external_daily[%s]: %d дней разошлись со снимком поставщика "
            "(история пересчитана задним числом) — %s%s",
            series, len(revised),
            ", ".join(f"{r['day']}: {r['old']:g}→{r['new']:g}" for r in revised[:5]),
            " …" if len(revised) > 5 else "",
        )

    rows = [(series, row.day, float(row.value), Json(row.meta)) for row in frame.itertuples()]
    with connect() as conn, conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO external_daily (series, day, value, meta) VALUES %s "
            "ON CONFLICT (series, day) DO UPDATE SET "
            "value = EXCLUDED.value, meta = EXCLUDED.meta, fetched_at = NOW()",
            rows,
        )
    return {"rows": len(rows), "revised_days": len(revised), "revised": revised}


def load_external_daily(
    series: str, start=None, end=None
) -> pd.DataFrame:
    """
    Ряд из БД: DataFrame с DatetimeIndex по суткам (UTC, полночь) и колонками
    value/meta. Пустой DataFrame — валидный результат («данных ещё нет»),
    а не ошибка: прогоны читают только таблицу и не имеют права падать,
    если `fetch-external` ещё не запускался.
    """
    sql = "SELECT day, value, meta FROM external_daily WHERE series = %s"
    params: list = [series]
    if start is not None:
        sql += " AND day >= %s"
        params.append(pd.Timestamp(start).date())
    if end is not None:
        sql += " AND day <= %s"
        params.append(pd.Timestamp(end).date())
    sql += " ORDER BY day"

    rows = fetch_all(sql, params)
    if not rows:
        return pd.DataFrame(
            columns=["value", "meta"],
            index=pd.DatetimeIndex([], tz="UTC", name="day"),
        )
    frame = pd.DataFrame(rows)
    frame["day"] = pd.to_datetime(frame["day"], utc=True)
    return frame.set_index("day")


def sync_fear_greed() -> dict:
    """Качает индекс целиком и сохраняет. Единственная точка входа для CLI."""
    frame = fetch_fear_greed()
    frame = frame[frame["day"] >= FEAR_GREED_START]
    result = store_external_daily(frame, FEAR_GREED_SERIES)
    if not frame.empty:
        result["span"] = f"{frame['day'].min()}…{frame['day'].max()}"
    return result
