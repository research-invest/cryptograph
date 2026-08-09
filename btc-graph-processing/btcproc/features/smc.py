"""
Детекторы концепций Smart Money: структура, имбалансы, ликвидность, зоны.

Постановка — `docs/task_smc_integration.md`, разделы 5 и 6. Модуль **ни к чему
не подключён**: ни к признакам, ни к атомам. Он считает величины, чтобы их
можно было измерить (`scripts/smc_frequencies.py`), а решение о включении
принимается по результатам замера, а не по вере в концепцию.

## Цепочка зависимостей

    свинги → BOS / CHoCH → ордер-блоки → брейкеры
       └───→ уровни и пулы ликвидности → снятия и возвраты
    FVG считается отдельно, от свингов не зависит

Свинг подтверждается только когда `right` баров справа его не перебили,
поэтому **всё структурное отстаёт на `right` баров**. Это не задержка, которую
надо победить, а условие корректности: свинг, «видимый на графике» немедленно,
на самом деле опознан по барам, которых в момент t ещё нет.

## Дисциплина «только прошлое»

Каждая величина на баре t считается из баров до t включительно. Три места,
где ошибиться легко, и как они закрыты:

* **свинги** — центрированное окно даёт позицию экстремума, но результат
  сдвигается на `right`, поэтому в момент t виден только экстремум из t−right;
* **заполнение FVG** — считается инкрементально в прямом проходе, статус на
  баре t не пересматривается барами после t;
* **реестры** — обновляются одним проходом вперёд; ни одна запись не знает
  о будущем.

Закреплено тестами `tests/test_smc.py`, в первую очередь
`test_swings_do_not_look_ahead` и `test_fvg_fill_uses_only_past`.

## Производительность

Один проход по барам с реестрами, ограниченными по возрасту
(`*_max_age_bars`). Ограничение нужно не ради точности, а ради сложности: без
выбывания реестр растёт линейно с историей, а проход по нему на каждом баре
делает алгоритм квадратичным. На 314 тысячах баров BTC проход занимает
единицы секунд.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from btcproc import config
from btcproc.features import indicators as ind

logger = logging.getLogger(__name__)

# Редкие происшествия — кандидаты в SIGNATURE_ATOMS. Бюджет 1–2 штуки
# на всю задачу (раздел 4 ТЗ), поэтому список здесь длиннее, чем то, что
# в итоге будет включено.
SIGNATURE_CANDIDATES = [
    "bos_up",
    "bos_down",
    "choch_up",
    "choch_down",
    "fvg_formed_large",
    "sweep_high_reclaim",
    "sweep_low_reclaim",
]

# Состояния — кандидаты в CONTEXT_ATOMS. Блоки событий не дробят, бюджет
# свободный.
CONTEXT_CANDIDATES = [
    "structure_bullish",
    "structure_bearish",
    "in_unfilled_fvg",
    "sweep_high",
    "sweep_low",
    "eqh_above",
    "eql_below",
    "in_bullish_ob",
    "in_bearish_ob",
    "in_breaker",
    "in_discount",
    "in_premium",
]

# Непрерывные величины — кандидаты в вектор признаков. Каждая обязана быть
# стационарной И хорошо обусловленной: счётчики через log1p, доли в [0, 1],
# близости в [0, 1] или [-1, 1].
#
# Близость, а не расстояние. Первая версия отдавала расстояние в ATR с
# потолком 10, и это оказалось прямой ошибкой: «ближайшей зоны нет» —
# случай частый, а после robust-нормировки (IQR ≈ 0.45) он превращался в
# выброс силой ±5 при типичном разбросе ±1. Замер: 2.7–6.7% строк упирались
# в клип apply_scale против 0.4% у худшего из существующих признаков. Такой
# признак работает как бинарный флаг впятеро тяжелее любого другого и
# растаскивает кластеризацию сам по себе, независимо от того, несёт ли он
# информацию.
#
#     близость = 1 / (1 + расстояние в ATR)
#
# Преобразование монотонное, поэтому порядок сохраняется, и ограниченное:
# «зоны нет» естественно даёт 0, а не выброс. Заодно оно осмысленнее по сути
# — разница между 8 и 10 ATR не значит ничего, разница между 0.1 и 0.5 значит
# многое, и шкала теперь это отражает.
FEATURE_CANDIDATES = [
    "bars_since_structure_break",
    "near_fvg_bull",
    "near_fvg_bear",
    "fvg_fill_pct",
    "near_liquidity",
    "near_ob",
    "ob_age_norm",
    "ob_touch_count_norm",
    "premium_discount",
    "near_resistance",
    "near_support",
    "level_strength_norm",
]

# Знаковые близости: модуль — насколько близко, знак — сверху зона или снизу.
SIGNED_FEATURES = {"near_fvg_bull", "near_fvg_bear", "near_ob"}

BOOLEAN_COLUMNS = SIGNATURE_CANDIDATES + CONTEXT_CANDIDATES
ALL_COLUMNS = BOOLEAN_COLUMNS + FEATURE_CANDIDATES

# Индексы полей в записях реестров. Списки вместо dataclass'ов намеренно:
# на 300k баров разница в скорости доступа заметна, а структура здесь
# локальная и живёт двадцать строк.
_DIR, _LO, _HI, _BORN, _FILL = 0, 1, 2, 3, 4          # FVG
_OB_DIR, _OB_LO, _OB_HI, _OB_BORN, _OB_TOUCH, _OB_BREAKER, _OB_INSIDE = range(7)
_LV_PRICE, _LV_STRENGTH, _LV_SEEN, _LV_SWEPT = 0, 1, 2, 3


def confirmed_swings(
    base: pd.DataFrame, left: int = 3, right: int = 3
) -> pd.DataFrame:
    """
    Подтверждённые свинги: цена последнего экстремума, известного на баре.

    Свинг на баре i становится известен только на баре i + right — до этого
    момента его для системы не существует. Возвращает колонки swing_high и
    swing_low; NaN в начале истории означает «подтверждённых свингов ещё нет».

    Почему сдвиг обязателен: `rolling(center=True)` помечает бар i как
    экстремум, глядя на бары до i + right. Без `shift(right)` это значение
    оказалось бы доступно на баре i, то есть система читала бы будущее.
    """
    high, low = base["high"], base["low"]
    window = left + right + 1
    is_high = high == high.rolling(window, center=True).max()
    is_low = low == low.rolling(window, center=True).min()
    return pd.DataFrame(
        {
            "swing_high": high.where(is_high).shift(right).ffill(),
            "swing_low": low.where(is_low).shift(right).ffill(),
        },
        index=base.index,
    )


def detect_fvg(base: pd.DataFrame, atr: pd.Series) -> pd.DataFrame:
    """
    Трёхбарные разрывы справедливой стоимости.

        бычий:    low[i]  > high[i-2]     разрыв = low[i] - high[i-2]
        медвежий: high[i] < low[i-2]      разрыв = low[i-2] - high[i]

    Заглядывания нет: паттерн известен на закрытии третьего бара.

    Размер нормируется на ATR. Абсолютный размер разрыва нестационарен —
    двести долларов это пропасть в 2017 году и рябь в 2025-м.

    Возвращает границы зон и размеры в ATR; NaN там, где паттерна нет.
    """
    high, low = base["high"], base["low"]
    prev_high, prev_low = high.shift(2), low.shift(2)

    bull = low > prev_high
    bear = high < prev_low
    out = pd.DataFrame(index=base.index)
    out["bull_fvg"] = bull
    out["bull_lo"] = prev_high.where(bull)
    out["bull_hi"] = low.where(bull)
    out["bull_atr"] = ((low - prev_high) / atr).where(bull)
    out["bear_fvg"] = bear
    out["bear_lo"] = high.where(bear)
    out["bear_hi"] = prev_low.where(bear)
    out["bear_atr"] = ((prev_low - high) / atr).where(bear)
    return out


def _safe_atr(base: pd.DataFrame) -> np.ndarray:
    """
    ATR без дыр: на прогреве и на нулевом истинном диапазоне деление на него
    дало бы NaN или бесконечность, а те выбросили бы строку целиком.
    """
    atr = ind.atr(base, 14).replace(0.0, np.nan).ffill().bfill()
    values = atr.to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).any():
        # Вырожденный случай (очень короткая история): падать не на чем.
        span = float((base["high"] - base["low"]).mean())
        values[:] = span if span > 0 else 1.0
    return values


def build_smc(base: pd.DataFrame, cfg: config.SMCConfig | None = None) -> pd.DataFrame:
    """
    Все SMC-величины на каждый бар: 7 происшествий, 12 состояний, 12 чисел.

    base — бары базового ТФ. Возвращает DataFrame с индексом base и колонками
    ALL_COLUMNS. Булевы колонки — bool, остальные — float без NaN.
    """
    cfg = cfg or config.smc
    if base.empty:
        return pd.DataFrame(columns=ALL_COLUMNS)

    n = len(base)
    atr = _safe_atr(base)
    op = base["open"].to_numpy(dtype=float)
    hi = base["high"].to_numpy(dtype=float)
    lo = base["low"].to_numpy(dtype=float)
    cl = base["close"].to_numpy(dtype=float)

    swings = confirmed_swings(base, cfg.swing_left, cfg.swing_right)
    swing_high = swings["swing_high"].to_numpy(dtype=float)
    swing_low = swings["swing_low"].to_numpy(dtype=float)

    fvg = detect_fvg(base, pd.Series(atr, index=base.index))
    bull_fvg = fvg["bull_fvg"].to_numpy(dtype=bool)
    bull_lo, bull_hi = fvg["bull_lo"].to_numpy(float), fvg["bull_hi"].to_numpy(float)
    bull_atr = fvg["bull_atr"].to_numpy(float)
    bear_fvg = fvg["bear_fvg"].to_numpy(dtype=bool)
    bear_lo, bear_hi = fvg["bear_lo"].to_numpy(float), fvg["bear_hi"].to_numpy(float)
    bear_atr = fvg["bear_atr"].to_numpy(float)

    out = {name: np.zeros(n, dtype=bool) for name in BOOLEAN_COLUMNS}
    for name in FEATURE_CANDIDATES:
        out[name] = np.zeros(n, dtype=float)

    count_scale = np.log1p(cfg.count_norm_scale)
    age_scale = np.log1p(cfg.level_max_age_bars)

    fvgs: list[list] = []
    obs: list[list] = []
    high_levels: list[list] = []
    low_levels: list[list] = []
    # Снятия, ожидающие возврата: [направление, уровень, бар снятия].
    pending: list[list] = []

    bias = 0                    # 0 — структура ещё не определена
    last_break = -1
    prev_sh = np.nan
    prev_sl = np.nan
    sh_consumed = False
    sl_consumed = False

    for t in range(n):
        atr_t = atr[t]
        close_t, high_t, low_t = cl[t], hi[t], lo[t]

        # ── 1. Заполнение имбалансов баром t ────────────────────────────────
        if fvgs:
            alive = []
            for z in fvgs:
                size = z[_HI] - z[_LO]
                if size > 0:
                    if z[_DIR] > 0:
                        filled = (z[_HI] - low_t) / size
                    else:
                        filled = (high_t - z[_LO]) / size
                    if filled > z[_FILL]:
                        z[_FILL] = min(1.0, filled)
                if z[_FILL] < 1.0 and t - z[_BORN] <= cfg.fvg_max_age_bars:
                    alive.append(z)
            fvgs = alive

        # ── 2. Касания и пробои ордер-блоков ────────────────────────────────
        if obs:
            alive = []
            for z in obs:
                # Касание — вход в зону, а не каждый бар внутри неё. Иначе
                # счётчик считает время, проведённое в зоне, и на недельном
                # сроке жизни блока насыщается у всех блоков одинаково.
                touching = low_t <= z[_OB_HI] and high_t >= z[_OB_LO]
                if touching and not z[_OB_INSIDE]:
                    z[_OB_TOUCH] += 1
                z[_OB_INSIDE] = touching
                # Пробой против собственного направления превращает блок
                # в брейкер и разворачивает его смысл.
                if not z[_OB_BREAKER]:
                    if z[_OB_DIR] > 0 and close_t < z[_OB_LO]:
                        z[_OB_DIR], z[_OB_BREAKER] = -1, True
                    elif z[_OB_DIR] < 0 and close_t > z[_OB_HI]:
                        z[_OB_DIR], z[_OB_BREAKER] = 1, True
                if t - z[_OB_BORN] <= cfg.ob_max_age_bars:
                    alive.append(z)
            obs = alive

        # ── 3. Снятие ликвидности и возврат ─────────────────────────────────
        # Пулом считается уровень, набранный двумя и более свингами: одиночный
        # экстремум — это просто экстремум, скопления стопов за ним нет.
        for lv in high_levels:
            if lv[_LV_STRENGTH] >= 2 and lv[_LV_SWEPT] < 0 and high_t > lv[_LV_PRICE]:
                lv[_LV_SWEPT] = t
                out["sweep_high"][t] = True
                pending.append([1, lv[_LV_PRICE], t])
        for lv in low_levels:
            if lv[_LV_STRENGTH] >= 2 and lv[_LV_SWEPT] < 0 and low_t < lv[_LV_PRICE]:
                lv[_LV_SWEPT] = t
                out["sweep_low"][t] = True
                pending.append([-1, lv[_LV_PRICE], t])

        if pending:
            still = []
            for direction, level, born in pending:
                # Возврат на том же баре — классический случай: фитиль
                # проколол уровень, закрытие вернулось обратно.
                if direction > 0 and close_t < level:
                    out["sweep_high_reclaim"][t] = True
                elif direction < 0 and close_t > level:
                    out["sweep_low_reclaim"][t] = True
                elif t - born < cfg.reclaim_bars:
                    still.append([direction, level, born])
            pending = still

        # ── 4. Новый подтверждённый свинг попадает в реестр уровней ─────────
        sh, sl = swing_high[t], swing_low[t]
        if np.isfinite(sh) and sh != prev_sh:
            _register_level(high_levels, sh, t, atr_t * cfg.level_tolerance_atr)
            prev_sh, sh_consumed = sh, False
        if np.isfinite(sl) and sl != prev_sl:
            _register_level(low_levels, sl, t, atr_t * cfg.level_tolerance_atr)
            prev_sl, sl_consumed = sl, False

        if t % 64 == 0:  # чистка реестров не каждый бар — она недешёвая
            high_levels = [x for x in high_levels
                           if t - x[_LV_SEEN] <= cfg.level_max_age_bars]
            low_levels = [x for x in low_levels
                          if t - x[_LV_SEEN] <= cfg.level_max_age_bars]

        # ── 5. Слом структуры ───────────────────────────────────────────────
        broke_up = np.isfinite(sh) and close_t > sh and not sh_consumed
        broke_down = np.isfinite(sl) and close_t < sl and not sl_consumed

        if broke_up:
            sh_consumed = True
            if bias < 0:
                out["choch_up"][t] = True      # разворот против структуры
            else:
                out["bos_up"][t] = True        # продолжение
            bias, last_break = 1, t
            _create_order_block(obs, t, op, cl, lo, hi, direction=1,
                                lookback=cfg.ob_lookback_bars)
        if broke_down:
            sl_consumed = True
            if bias > 0:
                out["choch_down"][t] = True
            else:
                out["bos_down"][t] = True
            bias, last_break = -1, t
            _create_order_block(obs, t, op, cl, lo, hi, direction=-1,
                                lookback=cfg.ob_lookback_bars)

        out["structure_bullish"][t] = bias > 0
        out["structure_bearish"][t] = bias < 0
        out["bars_since_structure_break"][t] = (
            1.0 if last_break < 0
            else min(1.0, np.log1p(t - last_break) / age_scale)
        )

        # ── 6. Новые имбалансы бара t ───────────────────────────────────────
        # Появляются ПОСЛЕ обновления заполнения: собственный бар заполнить
        # разрыв не может, он его и создал.
        if bull_fvg[t]:
            fvgs.append([1, bull_lo[t], bull_hi[t], t, 0.0])
            if bull_atr[t] > cfg.fvg_min_atr:
                out["fvg_formed_large"][t] = True
        if bear_fvg[t]:
            fvgs.append([-1, bear_lo[t], bear_hi[t], t, 0.0])
            if bear_atr[t] > cfg.fvg_min_atr:
                out["fvg_formed_large"][t] = True

        # ── 7. Снимок состояния на баре t ───────────────────────────────────
        _write_fvg_state(out, t, fvgs, close_t, atr_t)
        _write_ob_state(out, t, obs, close_t, atr_t, count_scale,
                        cfg.ob_max_age_bars)
        _write_level_state(out, t, high_levels, low_levels, close_t, atr_t,
                           count_scale)
        _write_range_state(out, t, sh, sl, close_t, cfg)

    frame = pd.DataFrame(out, index=base.index)
    logger.info(
        "SMC: %d баров, реестры на выходе — FVG %d, OB %d, уровней %d",
        n, len(fvgs), len(obs), len(high_levels) + len(low_levels),
    )
    return frame[ALL_COLUMNS]


_CACHE: dict = {}


def _fingerprint(base: pd.DataFrame, cfg: config.SMCConfig) -> tuple:
    """
    Дешёвый отпечаток набора баров: длина, границы по времени и крайние цены.

    Хэшировать 300 тысяч строк дороже, чем пересчитать; поэтому берётся то, что
    у двух разных монет или двух разных срезов совпасть практически не может.
    """
    close = base["close"]
    return (
        len(base), base.index[0], base.index[-1],
        float(close.iat[0]), float(close.iat[-1]), cfg,
    )


def build_smc_cached(base: pd.DataFrame, cfg: config.SMCConfig | None = None) -> pd.DataFrame:
    """
    То же, что build_smc, но помнит последний результат.

    За один прогон SMC-величины нужны дважды — в признаках и в атомах, — и
    оба раза по одним и тем же барам. Проход стоит около минуты на полной
    истории, а прогонов `live` двенадцать в сутки на каждую монету.

    Кэш на одну запись: прогон обрабатывает монеты последовательно, поэтому
    вторая пара обращений вытесняет первую и ничего не удерживает в памяти
    сверх одного DataFrame.
    """
    cfg = cfg or config.smc
    if base.empty:
        return build_smc(base, cfg)

    key = _fingerprint(base, cfg)
    cached = _CACHE.get("key")
    if cached == key:
        return _CACHE["value"]

    value = build_smc(base, cfg)
    _CACHE["key"], _CACHE["value"] = key, value
    return value


def proximity(distance_atr: float) -> float:
    """
    Расстояние в ATR → близость в (0, 1]. Отсутствие зоны передаётся нулём.

    Монотонно убывает, ограничена, и у неё нет хвоста: именно хвост делал
    признак-расстояние непригодным для кластеризации (см. FEATURE_CANDIDATES).
    """
    return 1.0 / (1.0 + abs(distance_atr))


def signed_proximity(delta_atr: float) -> float:
    """То же, но со знаком: положительное — зона выше цены."""
    near = proximity(delta_atr)
    return near if delta_atr >= 0 else -near


def _register_level(levels: list[list], price: float, t: int, tolerance: float) -> None:
    """
    Свинг либо укрупняет существующий уровень, либо заводит новый.

    Совпадение в пределах допуска — это и есть equal highs/lows: два экстремума
    на одной цене означают скопление стопов за ней. Цена уровня усредняется,
    чтобы уровень не «дрейфовал» к последнему касанию.
    """
    for lv in levels:
        if abs(lv[_LV_PRICE] - price) <= tolerance:
            strength = lv[_LV_STRENGTH] + 1
            lv[_LV_PRICE] = (lv[_LV_PRICE] * lv[_LV_STRENGTH] + price) / strength
            lv[_LV_STRENGTH] = strength
            lv[_LV_SEEN] = t
            # Новый свинг на том же уровне — новая ликвидность за ним.
            lv[_LV_SWEPT] = -1
            return
    levels.append([price, 1, t, -1])


def _create_order_block(obs: list[list], t: int, op: np.ndarray, cl: np.ndarray,
                        lo: np.ndarray, hi: np.ndarray, direction: int,
                        lookback: int) -> None:
    """
    Ордер-блок — последняя противоположная свеча перед сломом структуры.

    Определений в литературе несколько; зафиксировано это, ради
    воспроизводимости. Отсюда же следует, что OB наследует лаг свингов:
    он не может появиться раньше, чем подтвердится слом.
    """
    start = max(0, t - lookback)
    for j in range(t, start - 1, -1):
        bearish = cl[j] < op[j]
        if (direction > 0 and bearish) or (direction < 0 and not bearish):
            obs.append([direction, lo[j], hi[j], t, 0, False, False])
            return


def _write_fvg_state(out: dict, t: int, fvgs: list[list], close: float,
                     atr: float) -> None:
    """Ближайшие незакрытые имбалансы сверху и снизу плюс факт нахождения внутри."""
    best_bull = best_bear = None
    best_bull_d = best_bear_d = np.inf
    inside = False
    nearest_fill = 0.0
    nearest_d = np.inf

    for z in fvgs:
        mid = (z[_LO] + z[_HI]) / 2.0
        distance = abs(mid - close)
        if z[_LO] <= close <= z[_HI]:
            inside = True
        if distance < nearest_d:
            nearest_d, nearest_fill = distance, z[_FILL]
        if z[_DIR] > 0:
            if distance < best_bull_d:
                best_bull_d, best_bull = distance, mid
        elif distance < best_bear_d:
            best_bear_d, best_bear = distance, mid

    out["in_unfilled_fvg"][t] = inside
    out["fvg_fill_pct"][t] = nearest_fill
    # Ноль означает «такого имбаланса нет», и это не путается с «есть вплотную»:
    # у близкой зоны близость около единицы.
    out["near_fvg_bull"][t] = (
        0.0 if best_bull is None else signed_proximity((best_bull - close) / atr)
    )
    out["near_fvg_bear"][t] = (
        0.0 if best_bear is None else signed_proximity((best_bear - close) / atr)
    )


def _write_ob_state(out: dict, t: int, obs: list[list], close: float, atr: float,
                    count_scale: float, max_age: int) -> None:
    nearest = None
    nearest_d = np.inf
    for z in obs:
        if z[_OB_LO] <= close <= z[_OB_HI]:
            if z[_OB_BREAKER]:
                out["in_breaker"][t] = True
            elif z[_OB_DIR] > 0:
                out["in_bullish_ob"][t] = True
            else:
                out["in_bearish_ob"][t] = True
        mid = (z[_OB_LO] + z[_OB_HI]) / 2.0
        distance = abs(mid - close)
        if distance < nearest_d:
            nearest_d, nearest = distance, z

    if nearest is None:
        out["near_ob"][t] = 0.0
        return
    mid = (nearest[_OB_LO] + nearest[_OB_HI]) / 2.0
    out["near_ob"][t] = signed_proximity((mid - close) / atr)
    out["ob_age_norm"][t] = min(1.0, np.log1p(t - nearest[_OB_BORN]) / np.log1p(max_age))
    out["ob_touch_count_norm"][t] = min(
        1.0, np.log1p(nearest[_OB_TOUCH]) / count_scale
    )


def _write_level_state(out: dict, t: int, high_levels: list[list],
                       low_levels: list[list], close: float, atr: float,
                       count_scale: float) -> None:
    """Ближайшие сопротивление сверху и поддержка снизу, плюс висящая ликвидность."""
    resistance = support = None
    res_d = sup_d = np.inf
    strength = 0
    liquidity_d = np.inf

    for lv in high_levels:
        gap = lv[_LV_PRICE] - close
        if gap > 0:
            if gap < res_d:
                res_d, resistance = gap, lv
            if lv[_LV_STRENGTH] >= 2:
                out["eqh_above"][t] = True
                liquidity_d = min(liquidity_d, gap)
    for lv in low_levels:
        gap = close - lv[_LV_PRICE]
        if gap > 0:
            if gap < sup_d:
                sup_d, support = gap, lv
            if lv[_LV_STRENGTH] >= 2:
                out["eql_below"][t] = True
                liquidity_d = min(liquidity_d, gap)

    out["near_resistance"][t] = 0.0 if resistance is None else proximity(res_d / atr)
    out["near_support"][t] = 0.0 if support is None else proximity(sup_d / atr)
    out["near_liquidity"][t] = (
        0.0 if np.isinf(liquidity_d) else proximity(liquidity_d / atr)
    )
    for lv in (resistance, support):
        if lv is not None:
            strength = max(strength, lv[_LV_STRENGTH])
    out["level_strength_norm"][t] = min(1.0, np.log1p(strength) / count_scale)


def _write_range_state(out: dict, t: int, swing_high: float, swing_low: float,
                       close: float, cfg: config.SMCConfig) -> None:
    """
    Положение в структурном диапазоне.

    Отличие от pos_1d/pos_1w из builder.py: границы задаются подтверждёнными
    свингами, а не окном фиксированной длины. Совпадёт ли это с ними на
    практике — вопрос замера, а не рассуждения (раздел 13.2 ТЗ).
    """
    if not (np.isfinite(swing_high) and np.isfinite(swing_low)):
        out["premium_discount"][t] = 0.5
        return
    span = swing_high - swing_low
    if span <= 0:
        out["premium_discount"][t] = 0.5
        return
    position = float(np.clip((close - swing_low) / span, 0.0, 1.0))
    out["premium_discount"][t] = position
    out["in_discount"][t] = position < cfg.discount_below
    out["in_premium"][t] = position > cfg.premium_above
