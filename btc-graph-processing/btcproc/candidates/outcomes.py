"""
Разметка исходов на горизонте.

Для каждого бара смотрим вперёд ровно на горизонт (24h = 96 баров при 15m).
Это ЕДИНСТВЕННЫЙ модуль расчёта, которому положено заглядывать вперёд:
всё остальное (`features/`) считает только по прошлому, иначе историческая
статистика кандидата становится фикцией.

Считается два набора величин.

**Направление и путь** — были здесь с самого начала:

  ret_pct — куда пришла цена к концу горизонта;
  mfe_pct — максимальное движение «за» (Maximum Favorable Excursion);
  mae_pct — максимальное движение «против» (Maximum Adverse Excursion).

**Размах** — добавлен 2026-08-19 (задача D
`crypto-graph/docs/tz_range_horizons_19-08-26.md`, блок C аудита §C1):

  range_pct   — (max(high) − min(low)) / entry × 100, размах как он есть;
  rv_fwd      — реализованная волатильность ВПЕРЁД по тому же окну;
  range_ratio — range_pct, нормированный на ATR14(t)·√H.

`range_ratio` — ключевая из трёх. Абсолютный размах нестационарен (2018 и
2024 несравнимы), отношение к текущему уровню волатильности — сравнимо.
Заведены они здесь, а не в замере, ровно потому, что замер их уже считает на
лету: раздел 47 журнала измерял `range_ratio` из баров в каждом скрипте
заново, и любая будущая работа по размаху начиналась бы с того же пересчёта.

**Про нормировку `range_ratio` — важное предупреждение.** Знаменатель ATR14 на
15-минутном баре это 3.5 часа, то есть для горизонта 24h он в семь раз короче
числителя, и часть поведения величины принадлежит знаменателю, а не рынку.
Замер 47.3 показал это прямо: «предсказуемость растёт с горизонтом» держится
на нормировке ATR14 и исчезает при ATR по окну горизонта. Здесь хранится
вариант с ATR14 — он совпадает с разделами 36 и 47 и потому сравним с ними, —
а вторая нормировка остаётся выбором ЗАМЕРА (`analysis/range_model.range_target`),
а не свойством хранимых данных. Любой вывод про размах обязан быть проверен на
обеих: `range_pct` хранится рядом, и пересчитать знаменатель по нему можно
всегда.

`valid` = у бара есть полный горизонт вперёд и в нём нет пропусков баров.
Именно это разделение даёт `valid_label_count` / `invalid_label_count`
в кандидате: последние сутки истории и участки с дырами в данных
статистику портить не должны.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from btcproc import config
from btcproc.features import indicators as ind

logger = logging.getLogger(__name__)

#: Колонки размаха. Вынесены константой, потому что их список нужен трём
#: местам сразу (расчёт, сохранение в `outcomes`, тест полноты схемы) и
#: разъехаться им нельзя.
RANGE_COLUMNS = ("range_pct", "rv_fwd", "range_ratio")


def forward_range_ratio(base: pd.DataFrame, atr14: pd.Series,
                        horizon_bars: int) -> pd.Series:
    """
    range_ratio(t) = размах цены за (t, t+H] баров, нормированный на ATR14(t)
    и на √H (размах растёт примерно как корень из времени — без нормировки
    величина мерила бы длину горизонта, а не рынок).

    NaN на последних horizon_bars барах истории (окно не помещается) и там,
    где ATR14 недоступен (прогрев).

    **Живёт здесь, а не в `analysis/range_lift.py`, где была заведена.**
    С 2026-08-19 ту же величину считает разметка исходов, и две реализации
    одной формулы разъехались бы молча: замер говорил бы про одну величину, а
    сохранённая в `outcomes` колонка означала бы другую. Переезд именно в
    модуль исходов, а не в `features/indicators.py`, — потому что функция
    заглядывает вперёд по определению, и в модуле признаков она была бы
    приглашением использовать её как признак.

    `range_lift` продолжает её экспортировать: старый адрес рабочий.
    """
    high, low, close = base["high"], base["low"], base["close"]
    fwd_high = (high.shift(-1).rolling(horizon_bars, min_periods=horizon_bars)
                .max().shift(-(horizon_bars - 1)))
    fwd_low = (low.shift(-1).rolling(horizon_bars, min_periods=horizon_bars)
               .min().shift(-(horizon_bars - 1)))
    range_pct = (fwd_high - fwd_low) / close * 100.0
    atr_pct = (atr14 / close * 100.0) * math.sqrt(horizon_bars)
    return range_pct / atr_pct.replace(0.0, np.nan)


def compute_outcomes(base: pd.DataFrame, horizon_bars: int | None = None) -> pd.DataFrame:
    """
    base — бары базового ТФ. Возвращает DataFrame с тем же индексом:
    ret_pct, mfe_pct, mae_pct, is_up, range_pct, rv_fwd, range_ratio, valid.
    """
    horizon_bars = horizon_bars or config.data.horizon_bars
    n = len(base)
    out = pd.DataFrame(index=base.index)
    if n == 0:
        empty = dict.fromkeys(
            ("ret_pct", "mfe_pct", "mae_pct", "is_up", *RANGE_COLUMNS, "valid"), []
        )
        return out.assign(**empty)

    close = base["close"].to_numpy(dtype=float)
    high = base["high"].to_numpy(dtype=float)
    low = base["low"].to_numpy(dtype=float)

    ret = np.full(n, np.nan)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    range_pct = np.full(n, np.nan)
    rv_fwd = np.full(n, np.nan)

    if n > horizon_bars:
        # Окно [i+1 .. i+H] — исход считается по барам строго после текущего.
        high_windows = np.lib.stride_tricks.sliding_window_view(high, horizon_bars)
        low_windows = np.lib.stride_tricks.sliding_window_view(low, horizon_bars)
        last = n - horizon_bars  # для i >= last горизонта не хватает

        entry = close[:last]
        window_high = high_windows[1:last + 1].max(axis=1)
        window_low = low_windows[1:last + 1].min(axis=1)
        ret[:last] = (close[horizon_bars:] / entry - 1.0) * 100.0
        mfe[:last] = (window_high / entry - 1.0) * 100.0
        mae[:last] = (window_low / entry - 1.0) * 100.0
        # Размах — те же max и min, но разность между ними, а не расстояние
        # каждого до входа. mfe − mae дало бы то же число, но только пока обе
        # величины считаются от одного entry; связывать их формулой значило бы
        # чинить одно место правкой в другом.
        range_pct[:last] = (window_high - window_low) / entry * 100.0

        # Реализованная волатильность ВПЕРЁД по тому же окну: стандартное
        # отклонение лог-доходностей баров (t, t+H]. ddof=1 — как у
        # `indicators.realized_vol`, чтобы прямое и обратное окно были одной
        # величиной, посчитанной в разные стороны, а не двумя разными.
        log_ret = np.full(n, np.nan)
        log_ret[1:] = np.diff(np.log(close))
        ret_windows = np.lib.stride_tricks.sliding_window_view(log_ret, horizon_bars)
        rv_fwd[:last] = np.std(ret_windows[1:last + 1], axis=1, ddof=1)

    out["ret_pct"] = ret
    out["mfe_pct"] = mfe
    out["mae_pct"] = mae
    out["is_up"] = np.where(np.isnan(ret), None, ret > 0)
    out["range_pct"] = range_pct
    out["rv_fwd"] = rv_fwd
    out["range_ratio"] = forward_range_ratio(base, ind.atr(base, 14), horizon_bars)

    # Пропуск баров внутри горизонта делает разметку недостоверной.
    bar_delta = pd.Timedelta(minutes=config.data.base_minutes)
    expected = bar_delta * horizon_bars
    ts = base.index.to_series()
    actual = ts.shift(-horizon_bars) - ts
    out["valid"] = (~np.isnan(ret)) & (actual == expected).to_numpy()

    logger.info(
        "Исходы: %d валидных из %d (горизонт %d баров)",
        int(out["valid"].sum()), n, horizon_bars,
    )
    return out


def extra_horizon_frames(base: pd.DataFrame,
                         index: pd.Index | None = None) -> dict[str, pd.DataFrame]:
    """
    Разметка дополнительных горизонтов (`config.data.extra_horizons`) — только
    для СОХРАНЕНИЯ.

    Ни кандидаты, ни модель состояний этих строк не видят и видеть не должны:
    основной горизонт у системы один (`config.data.horizon`), и появление
    рядом ещё двух наборов исходов не меняет ни одного её решения. Это склад
    для замеров, а не второй предмет расчёта — см. раздел 47.10 журнала о том,
    почему продуктовая половина задачи D осталась несделанной.

    `index` — строки, которые оставить (обычно индекс признаков): разметка
    считается по всем барам, но хранить её осмысленно на тех же строках, что и
    основной горизонт, иначе `outcomes` разных горизонтов покрывают разные
    участки истории и их нельзя сравнивать напрямую.

    Пустой словарь при пустом списке горизонтов — штатный случай, а не ошибка.
    """
    frames: dict[str, pd.DataFrame] = {}
    for label in config.data.extra_horizons:
        if label == config.data.horizon:
            continue  # основной горизонт уже посчитан вызывающим кодом
        frame = compute_outcomes(base, config.data.bars_of_horizon(label))
        frames[label] = frame if index is None else frame.reindex(index)
    return frames
