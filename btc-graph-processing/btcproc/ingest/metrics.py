"""
Загрузка деривативных метрик Binance USD-M — открытый интерес, long/short
ratio топ-трейдеров и розницы, давление тейкеров.

Постановка, найденные факты и обоснования каждого решения —
`docs/tz_deriv_ingest_14-08-26.md`. Коротко, что важно знать перед правкой:

* источник — ТОЛЬКО daily-архивы `futures/um` (USD-M), месячных агрегатов
  и `cm`-каталога `metrics` не существует (§0.1);
* граница истории — свойство МОНЕТЫ (`SymbolSpec.metrics_start_date()`), а
  не общая константа: у BTC архив глубже общей границы на ~15 месяцев, у
  HYPE метрики (Binance-перп) начинаются раньше баров (Bybit-спот) (§0.3);
* заголовок CSV сверяется со списком колонок на каждом файле — расхождение
  падает с понятной ошибкой, а не читается по позициям (§0.2, тот же класс
  ошибки, что тихая эвристика `ingest/binance.py:_parse_csv`);
* задвоенные строки (§0.9: 2020-09-01 отдаёт 576 строк вместо 288) убираются
  дедупом по `create_time` ДО агрегации — иначе `src_rows` (счётчик полноты
  бара) начинает врать;
* дыры бывают ПО КОЛОНКАМ, а не по дням целиком (§0.9: у 2022-01 живой ОИ
  при полностью пустых ratio-колонках) — NULL значит «данных нет», `ffill`
  на уровне хранения запрещён;
* окно агрегации 5m→базовый ТФ — `[T, T+15m)`, `label='left', closed='left'`
  — безопасно при ОБЕИХ конвенциях `create_time`, см. докстринг `_aggregate`
  и §0.8;
* архив отстаёт на сутки — 404 за сегодня и вчера не ошибка, а норма (§0.7).
"""
from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

from btcproc import config
from btcproc.db.session import bulk_upsert, fetch_all

logger = logging.getLogger(__name__)

VISION_URL = (
    "https://data.binance.vision/data/futures/um/daily/metrics/"
    "{sym}/{sym}-metrics-{ymd}.zip"
)

# Порядок и состав колонок CSV, как отдаёт поставщик (§0.2). README
# binance-public-data состав `metrics` не документирует — схема получена
# чтением файла, и заголовок обязан сверяться с этим списком на каждом файле:
# молчаливое чтение по позициям на источнике без документированной схемы —
# вопрос времени, а не «если».
METRICS_COLUMNS = [
    "create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
    "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
    "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
]

DERIV_METRICS_COLUMNS = [
    "symbol", "tf", "ts", "oi", "oi_value", "ls_top_acc", "ls_top_pos",
    "ls_global", "taker_ratio", "src_rows",
]

# Правило агрегации 5m → базовый ТФ (§1.3 ТЗ): `last` — уровень (снимок
# состояния позиций, а не поток), `mean` — поток за интервал. Оговорка к
# `mean`: это среднее ОТНОШЕНИЙ, а не отношение сумм — объёмов тейкеров в
# файле нет, только их готовое отношение. Известное приближение, не
# корректное по умолчанию.
_RAW_TO_STORED = {
    "sum_open_interest": ("oi", "last"),
    "sum_open_interest_value": ("oi_value", "last"),
    "count_toptrader_long_short_ratio": ("ls_top_acc", "last"),
    "sum_toptrader_long_short_ratio": ("ls_top_pos", "last"),
    "count_long_short_ratio": ("ls_global", "last"),
    "sum_taker_long_short_vol_ratio": ("taker_ratio", "mean"),
}

_RESAMPLE_RULE = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h"}


class HeaderMismatch(ValueError):
    """Заголовок CSV разошёлся с ожидаемым списком колонок (§0.2)."""


def _cache_path(symbol: str, ymd: str) -> Path:
    path = config.data.data_dir / "binance_metrics" / symbol
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{ymd}.zip"


def _fetch_zip(
    url: str, cache: Path, client: httpx.Client, attempts: int = 4
) -> bytes | None:
    """
    Идентично поведению `ingest.binance._fetch_zip` (A1 ТЗ): 403/404 —
    «файла нет», 429/500/502/503/504 — повтор с растущей паузой, 4 попытки.
    Не вызывается оттуда напрямую — загрузчик метрик не должен зависеть от
    модуля другой площадки, хотя поведение то же самое и источник тот же
    (data.binance.vision), только другой каталог архива.
    """
    if cache.exists() and cache.stat().st_size > 0:
        return cache.read_bytes()

    delay = 2.0
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = client.get(url)
            if resp.status_code in (403, 404):
                return None
            if resp.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(
                    f"{resp.status_code} от {url}", request=resp.request, response=resp
                )
            resp.raise_for_status()
            cache.write_bytes(resp.content)
            return resp.content
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            logger.warning("Повтор %s через %.0f с (%s)", url, delay, exc)
            time.sleep(delay)
            delay *= 2

    logger.error("Не удалось скачать %s: %s", url, last_error)
    return None


def _parse_csv(raw: bytes, symbol: str, ymd: str) -> pd.DataFrame:
    """Заголовок обязателен и обязан совпасть — падение, а не чтение по позициям (§0.2)."""
    text = raw.decode("utf-8")
    first_line = text.split("\n", 1)[0].strip()
    header_cols = [c.strip() for c in first_line.split(",")]
    if header_cols != METRICS_COLUMNS:
        raise HeaderMismatch(
            f"{symbol} {ymd}: заголовок CSV {header_cols!r} не совпал с "
            f"ожидаемым {METRICS_COLUMNS!r}. Схема источника могла "
            "поменяться — сверь docs/tz_deriv_ingest_14-08-26.md §0.2 "
            "и обнови METRICS_COLUMNS осознанно."
        )
    return pd.read_csv(io.StringIO(text))


def _unzip(payload: bytes, symbol: str, ymd: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = zf.namelist()[0]
        return _parse_csv(zf.read(name), symbol, ymd)


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Убирает задвоенные строки по `create_time` (§0.9: 2020-09-01 отдаёт 576
    строк вместо 288 при 288 уникальных `create_time`, значения совпадают).
    Без дедупа `mean` уцелеет (задвоенное значение не сдвигает среднее
    задвоенного же значения), а `src_rows` — счётчик реальной полноты бара —
    начнёт врать вдвое.
    """
    if df.empty:
        return df
    return df.drop_duplicates(subset=["create_time"], keep="first")


def aggregate(df: pd.DataFrame, symbol: str, tf: str) -> pd.DataFrame:
    """
    5m → базовый ТФ. Окно `[T, T+15m)`, `label='left', closed='left'` — то же,
    что агрегация klines в `ingest/bars.resample`.

    Почему это окно безопасно при ОБЕИХ конвенциях `create_time` (§0.8),
    доказательство перенесено сюда целиком — иначе следующий читатель
    «исправит» выравнивание обратно, увидев вроде бы очевидный сдвиг на
    полбара. В этом проекте `ohlcv.ts` — это `open_time` бара (не закрытие),
    и значение признака в строке `T` считается по `close` этой же строки,
    то есть становится известно в `T + 15m`. В окне `[T, T+15m)` при
    гранулярности 5 минут лежат метки `T`, `T+5m`, `T+10m`; самая поздняя —
    `T+10m`. При конвенции «начало интервала» она описывает
    `[T+10m, T+15m)`, при конвенции «конец интервала» — `[T+5m, T+10m)`; оба
    интервала целиком внутри бара `T` и известны к его закрытию. Заглянуть
    вперёд нечем ни при одной из двух конвенций — конвенция влияет только на
    ТОЧНОСТЬ выравнивания (какие именно пять минут рынка отражает снимок
    ОИ), а не на честность отдельного бара.

    `src_rows` — число уникальных исходных 5m-строк (после дедупа), попавших
    в окно: `3` = полный бар, `1–2` = частичный, строки нет вовсе = метрик
    за этот бар нет (не ноль).
    """
    if df.empty:
        return pd.DataFrame(columns=DERIV_METRICS_COLUMNS)

    rule = _RESAMPLE_RULE.get(tf)
    if rule is None:
        raise ValueError(f"Агрегация деривативных метрик не определена для tf={tf!r}")

    df = df.copy()
    df["ts"] = pd.to_datetime(df["create_time"], utc=True)
    df = df.set_index("ts").sort_index()
    for col in _RAW_TO_STORED:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = df.resample(rule, label="left", closed="left")
    out = pd.DataFrame()
    for raw_col, (stored_col, how) in _RAW_TO_STORED.items():
        out[stored_col] = grouped[raw_col].mean() if how == "mean" else grouped[raw_col].last()
    # size() — по числу строк группы, не по конкретной колонке: пустая
    # ratio-колонка (§0.9, 2022-01) не должна выглядеть как «строки не было».
    out["src_rows"] = grouped.size()

    out = out.reset_index()  # index уже назван "ts" — set_index("ts") выше
    out["symbol"] = symbol
    out["tf"] = tf
    return out[DERIV_METRICS_COLUMNS]


def _store(df: pd.DataFrame) -> int:
    """
    NaN → SQL NULL перед вставкой — обязательно, не косметика. psycopg2
    адаптирует `float('nan')` литералом `'NaN'::double precision`: это
    ЗНАЧЕНИЕ, отличное от NULL (`x IS NULL` на нём ложно). Без замены «данных
    нет» (§0.9) молча превращалось бы в число, которое проходит мимо любой
    проверки `IS NULL`, а `x = x` для него тоже ложно — типичный источник
    тихо неверных подсчётов ниже по потоку.
    """
    if df.empty:
        return 0
    df = df.copy()
    df["src_rows"] = df["src_rows"].fillna(0).astype(int)
    float_cols = [c for c in DERIV_METRICS_COLUMNS
                  if c not in ("symbol", "tf", "ts", "src_rows")]
    df[float_cols] = df[float_cols].astype(object).where(df[float_cols].notna(), None)
    rows = list(df[DERIV_METRICS_COLUMNS].itertuples(index=False, name=None))
    return bulk_upsert(
        "deriv_metrics", DERIV_METRICS_COLUMNS, rows,
        conflict_columns=["symbol", "tf", "ts"],
    )


def sync_metrics(
    symbol: str | None = None,
    tf: str | None = None,
    start: str | None = None,
    end: datetime | None = None,
    progress=None,
) -> dict:
    """
    Качает daily-архивы метрик за весь доступный период монеты и кладёт в
    `deriv_metrics`. Уже скачанные архивы берутся с диска, уже загруженные
    бары перезаписываются upsert'ом.

    Точка старта — не раньше `metrics_start_date()` монеты (A5: заполнять
    период до неё нельзя, и ходить туда в сеть незачем — там гарантированный
    404). Точка конца — не позже вчерашних суток: файл сегодняшнего дня
    появляется только после его закрытия (§0.7), а запрос за сегодня — не
    ошибка и не логируется как пропуск.
    """
    from btcproc import symbols

    spec = symbols.get(symbol) if symbol else symbols.default()
    symbol = spec.ticker
    tf = tf or config.data.base_tf

    archive_start = pd.Timestamp(spec.metrics_start_date(), tz="UTC")
    start_dt = pd.Timestamp(start, tz="UTC") if start else archive_start
    start_dt = max(start_dt, archive_start)

    yesterday = pd.Timestamp(datetime.now(timezone.utc)).normalize() - pd.Timedelta(days=1)
    end_dt = pd.Timestamp(end, tz="UTC") if end else yesterday
    end_dt = min(end_dt, yesterday)

    if start_dt > end_dt:
        return {"days": 0, "rows": 0, "missing_days": [], "bars": 0,
                "full_bars": 0, "share_full": 0.0}

    days = pd.date_range(start_dt.normalize(), end_dt.normalize(), freq="D")
    total_rows, missing, full_bars, total_bars = 0, [], 0, 0

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for i, day in enumerate(days):
            ymd = day.strftime("%Y-%m-%d")
            url = VISION_URL.format(sym=symbol, ymd=ymd)
            payload = _fetch_zip(url, _cache_path(symbol, ymd), client)
            if payload is None:
                # 404 внутри известного диапазона = пропуск данных, не
                # ошибка (A1): архив бывает неполным на границах листинга.
                missing.append(ymd)
                if progress:
                    progress(i + 1, len(days), f"{ymd}: файла нет")
                continue

            raw = dedupe(_unzip(payload, symbol, ymd))
            agg = aggregate(raw, symbol, tf)
            total_rows += _store(agg)
            total_bars += len(agg)
            full_bars += int((agg["src_rows"] >= 3).sum())
            if progress:
                progress(i + 1, len(days), f"{ymd}: {len(agg)} баров")

    share_full = full_bars / total_bars if total_bars else 0.0
    return {
        "days": len(days), "rows": total_rows, "missing_days": missing,
        "bars": total_bars, "full_bars": full_bars, "share_full": share_full,
    }


def load_deriv_metrics(
    symbol: str, tf: str | None = None, start=None, end=None
) -> pd.DataFrame:
    """
    Метрики из БД: DataFrame с DatetimeIndex по `ts` (UTC). Пустой DataFrame —
    валидный результат («ingest-metrics ещё не запускался»), а не ошибка:
    расчёт признаков читает только таблицу и не имеет права падать.
    """
    tf = tf or config.data.base_tf
    sql = (
        "SELECT ts, oi, oi_value, ls_top_acc, ls_top_pos, ls_global, "
        "taker_ratio, src_rows FROM deriv_metrics WHERE symbol = %s AND tf = %s"
    )
    params: list = [symbol, tf]
    if start is not None:
        ts = pd.Timestamp(start)
        params.append(ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC"))
        sql += " AND ts >= %s"
    if end is not None:
        ts = pd.Timestamp(end)
        params.append(ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC"))
        sql += " AND ts <= %s"
    sql += " ORDER BY ts"

    rows = fetch_all(sql, params)
    columns = ["oi", "oi_value", "ls_top_acc", "ls_top_pos", "ls_global",
               "taker_ratio", "src_rows"]
    if not rows:
        return pd.DataFrame(
            columns=columns, index=pd.DatetimeIndex([], tz="UTC", name="ts")
        )
    frame = pd.DataFrame(rows)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return frame.set_index("ts")


# ─── Конвенция create_time (§0.8, A3) ───────────────────────────────────────
#
# НЕ используется агрегацией (`aggregate` выше безопасна при обеих конвенциях
# по построению) — карта существует как проверенный факт источника, задача
# 8 чек-листа (§8) и материал для будущей точной 5-минутной привязки, если
# она когда-нибудь понадобится.
#
# Определена сверкой `sum_taker_long_short_vol_ratio` (заведомый поток) с тем
# же потоком, посчитанным из 5m-klines фьючерсов как
# `taker_buy_base / (volume - taker_buy_base)`, на выборочных днях. Найдено —
# конвенция НЕПОСТОЯННА (главный вывод §0.8): периоды заданы по факту
# проверки, а не по предположению о постоянстве. Формат — [start, end).
CREATE_TIME_CONVENTION_PERIODS: tuple[tuple[str, str, str], ...] = (
    ("2020-09-01", "2024-07-01", "end"),
    ("2024-07-01", "2026-02-01", "start"),
    ("2026-02-01", "2026-06-15", "end"),
    ("2026-06-15", "2099-01-01", "start"),
)


def create_time_convention(day) -> str:
    """
    `"start"` или `"end"` — что означает `create_time` для суток `day`
    (§0.8). Не используется расчётом (см. докстринг `aggregate`), только
    для разбора и как подтверждаемый тестом факт.
    """
    ts = pd.Timestamp(day)
    for start, end, convention in CREATE_TIME_CONVENTION_PERIODS:
        if pd.Timestamp(start, tz=ts.tz) <= ts < pd.Timestamp(end, tz=ts.tz):
            return convention
    raise ValueError(f"{day}: вне заданных периодов CREATE_TIME_CONVENTION_PERIODS")


def detect_convention(symbol: str, day, client: httpx.Client) -> str | None:
    """
    Определяет конвенцию НА ФАКТЕ для конкретных суток, сверяя
    `sum_taker_long_short_vol_ratio` метрик с той же величиной, посчитанной
    из 5m-klines фьючерсов, при двух возможных сдвигах (§0.8). Ходит в сеть —
    вызывается только сетевым тестом `tests/test_metrics_convention.py`,
    перепроверяющим `CREATE_TIME_CONVENTION_PERIODS` перед каждым изменением
    карты (A3). Возвращает `None`, если колонка пуста в этот день (§0.9,
    2022-01) — сверка на таком дне невозможна, соседние дни решают.
    """
    from btcproc.ingest.binance import KLINE_COLUMNS, _parse_csv as _parse_kline_csv, _to_utc

    ymd = pd.Timestamp(day).strftime("%Y-%m-%d")

    payload = _fetch_zip(VISION_URL.format(sym=symbol, ymd=ymd),
                         _cache_path(symbol, ymd), client)
    if payload is None:
        raise ValueError(f"{symbol} {ymd}: метрик нет")
    metrics = dedupe(_unzip(payload, symbol, ymd))
    metrics["ts"] = pd.to_datetime(metrics["create_time"], utc=True)
    metrics = metrics.set_index("ts").sort_index()
    metrics_ratio = pd.to_numeric(metrics["sum_taker_long_short_vol_ratio"], errors="coerce")
    if metrics_ratio.notna().sum() < 10:
        return None

    kline_url = (
        f"https://data.binance.vision/data/futures/um/daily/klines/"
        f"{symbol}/5m/{symbol}-5m-{ymd}.zip"
    )
    kcache = config.data.data_dir / "binance_metrics" / symbol / f"{ymd}_5m_klines.zip"
    kpayload = _fetch_zip(kline_url, kcache, client)
    if kpayload is None:
        raise ValueError(f"{symbol} {ymd}: 5m klines фьючерса нет")
    with zipfile.ZipFile(io.BytesIO(kpayload)) as zf:
        kdf = _parse_kline_csv(zf.read(zf.namelist()[0]))
    kdf["ts"] = _to_utc(kdf["open_time"])
    kdf = kdf.set_index("ts").sort_index()
    vol = pd.to_numeric(kdf["volume"], errors="coerce")
    taker_buy = pd.to_numeric(kdf["taker_buy_base"], errors="coerce")
    kline_ratio = (taker_buy / (vol - taker_buy)).replace([float("inf"), float("-inf")], float("nan"))

    # "конец": create_time T описывает [T-5m, T) — сравнить со свечой,
    # открывшейся в T-5m.
    end_aligned = metrics_ratio.copy()
    end_aligned.index = end_aligned.index - pd.Timedelta(minutes=5)
    r_end = end_aligned.corr(kline_ratio)

    # "начало": create_time T описывает [T, T+5m) — та же метка, что open_time.
    r_start = metrics_ratio.corr(kline_ratio)

    r_end = r_end if pd.notna(r_end) else -1.0
    r_start = r_start if pd.notna(r_start) else -1.0
    if r_end < 0.5 and r_start < 0.5:
        return None
    return "start" if r_start > r_end else "end"
