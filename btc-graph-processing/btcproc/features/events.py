"""
Детекторы рыночных событий и сборка блоков событий.

Каждое событие — «атом» (пробой, всплеск объёма, кросс EMA…), атомы
сгруппированы в семейства. Набор атомов, активных в окне вокруг бара,
образует **блок событий** — тот самый `event_block_id` из схемы кандидата.

Блок кодируется битовой маской: атом i → бит i. Это даёт две вещи сразу —
мгновенное сравнение блоков (одно 64-битное число вместо множества строк)
и стабильный идентификатор, не зависящий от порядка перечисления.
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from btcproc import config
from btcproc.features import indicators as ind

logger = logging.getLogger(__name__)

EVENT_VERSION = "v1"

# Семейство каждого атома. Порядок этого словаря задаёт номера битов —
# менять его задним числом нельзя, иначе старые event_block_id сменят смысл.
ATOM_FAMILY: dict[str, str] = {
    "breakout_1d_high": "price_action_events",
    "breakdown_1d_low": "price_action_events",
    "breakout_1w_high": "price_action_events",
    "breakdown_1w_low": "price_action_events",
    "wide_range_bar": "price_action_events",
    "inside_bar": "price_action_events",

    "vol_expansion": "volatility_events",
    "vol_contraction": "volatility_events",
    "atr_spike": "volatility_events",

    "volume_spike": "volume_events",
    "volume_dry": "volume_events",
    "taker_buy_dominance": "volume_events",
    "taker_sell_dominance": "volume_events",

    "ema_cross_up": "trend_events",
    "ema_cross_down": "trend_events",
    "trend_up_align": "trend_events",
    "trend_down_align": "trend_events",

    "rsi_overbought": "momentum_events",
    "rsi_oversold": "momentum_events",
    "momentum_reversal_up": "momentum_events",
    "momentum_reversal_down": "momentum_events",

    "at_range_high": "zone_context_events",
    "at_range_low": "zone_context_events",
    "mid_range": "zone_context_events",
    "round_level_touch": "zone_context_events",

    "asia_session": "session_events",
    "europe_session": "session_events",
    "us_session": "session_events",
    "weekend": "session_events",

    # ─── Smart Money, добавлены 2026-08-09 ──────────────────────────────────
    # Все шестнадцать — контекстные, то есть в маску не входят и биты 0–19 не
    # сдвигают. Это сознательный порядок работы: сначала завести детекторы
    # фоном и измерить лифт бесплатно, и только потом двигать выжившие в
    # signature, платя дроблением блоков. Обратный порядок означал бы платить
    # за атомы до того, как выяснится, что они дают.
    #
    # Считаются только при SMC_ENABLED=true; иначе колонки есть, но всегда
    # False — набор имён не должен зависеть от флага, иначе CONTEXT_ATOMS
    # описывал бы то одно, то другое.
    "bos_up": "structure_events",
    "bos_down": "structure_events",
    "choch_up": "structure_events",
    "choch_down": "structure_events",
    "structure_bullish": "structure_events",

    "fvg_formed_large": "imbalance_events",
    "in_unfilled_fvg": "imbalance_events",

    "sweep_high": "liquidity_events",
    "sweep_low": "liquidity_events",
    "sweep_high_reclaim": "liquidity_events",
    "sweep_low_reclaim": "liquidity_events",

    "in_bullish_ob": "zone_events",
    "in_bearish_ob": "zone_events",
    "in_breaker": "zone_events",
    "in_discount": "zone_events",
    "in_premium": "zone_events",
}

ATOMS = list(ATOM_FAMILY)

# Что из smc.BOOLEAN_COLUMNS сюда НЕ попало и почему (замер — фаза 2,
# development_log.md, раздел 14):
#   eqh_above, eql_below  — истинны на 95–98% баров, вырождены;
#   structure_bearish     — точное отрицание structure_bullish, лишний бит.
SMC_ATOMS = [a for a, family in ATOM_FAMILY.items()
             if family in {"structure_events", "imbalance_events",
                           "liquidity_events", "zone_events"}]

# В блок событий входят только «происшествия» — пробой, всплеск, кросс,
# разворот. Атомы вроде trend_up_align или us_session описывают не событие,
# а фон: они активны почти всегда, и включение их в блок раздувает число
# уникальных блоков до величины, при которой историческая выборка по
# конкретному блоку никогда не набирается.
#
# Фоновые атомы не пропадают: они уже учтены в векторе признаков (а значит,
# в самом group_id) и сохраняются отдельной колонкой — context_atoms в выдаче
# build_event_blocks и одноимённая колонка bar_events. Отдельной, а не общей
# с atoms, именно чтобы их нельзя было случайно подмешать в маску: atoms,
# atom_count, families и primary_family описывают ровно то, что закодировано
# в event_block_id, и это свойство держит связку «блок ↔ его расшифровка».
CONTEXT_ATOMS = {
    "trend_up_align",
    "trend_down_align",
    "mid_range",
    "taker_buy_dominance",
    "taker_sell_dominance",
    "asia_session",
    "europe_session",
    "us_session",
    "weekend",
    *SMC_ATOMS,
}

SIGNATURE_ATOMS = [a for a in ATOMS if a not in CONTEXT_ATOMS]
ATOM_BIT = {atom: i for i, atom in enumerate(SIGNATURE_ATOMS)}

# Порядок берётся из ATOM_FAMILY, а не из множества CONTEXT_ATOMS: у множества
# порядок обхода не гарантирован между запусками.
CONTEXT_ATOM_LIST = [a for a in ATOMS if a in CONTEXT_ATOMS]
CONTEXT_BIT = {atom: i for i, atom in enumerate(CONTEXT_ATOM_LIST)}

# Порог редкости блока по доле баров, в которых он встречался.
RARE_SHARE = 0.002
UNCOMMON_SHARE = 0.01

# Границы интенсивности по числу атомов в блоке.
SPARSE_MAX = 2
MODERATE_MAX = 5


def detect_atoms(base: pd.DataFrame) -> pd.DataFrame:
    """Булев DataFrame: колонка на атом, строка на бар."""
    w = {
        "1h": config.data.bars_per("1h"),
        "1d": config.data.bars_per("1d"),
        "1w": config.data.bars_per("1d") * 7,
    }
    close, high, low, vol = base["close"], base["high"], base["low"], base["volume"]
    rng = high - low
    atr14 = ind.atr(base, 14)
    rv_d = ind.realized_vol(close, w["1d"])
    rv_w = ind.realized_vol(close, w["1w"])
    rsi14 = ind.rsi(close, 14)
    ema_d = ind.ema(close, w["1d"])
    ema_w = ind.ema(close, w["1w"])
    pos_1w = ind.position_in_range(base, w["1w"])

    # shift(1) в экстремумах: пробой сравнивается с диапазоном ДО текущего бара.
    prev_high_1d = high.rolling(w["1d"], min_periods=w["1h"]).max().shift(1)
    prev_low_1d = low.rolling(w["1d"], min_periods=w["1h"]).min().shift(1)
    prev_high_1w = high.rolling(w["1w"], min_periods=w["1d"]).max().shift(1)
    prev_low_1w = low.rolling(w["1w"], min_periods=w["1d"]).min().shift(1)
    mean_range = rng.rolling(w["1d"], min_periods=w["1h"]).mean()
    mean_vol = vol.rolling(w["1d"], min_periods=w["1h"]).mean()
    dist_d = close - ema_d

    a = pd.DataFrame(index=base.index)

    a["breakout_1d_high"] = close > prev_high_1d
    a["breakdown_1d_low"] = close < prev_low_1d
    a["breakout_1w_high"] = close > prev_high_1w
    a["breakdown_1w_low"] = close < prev_low_1w
    a["wide_range_bar"] = rng > 2.0 * mean_range
    a["inside_bar"] = (high <= high.shift(1)) & (low >= low.shift(1))

    a["vol_expansion"] = rv_d > 1.5 * rv_w
    a["vol_contraction"] = rv_d < 0.6 * rv_w
    a["atr_spike"] = rng > 3.0 * atr14

    a["volume_spike"] = vol > 3.0 * mean_vol
    a["volume_dry"] = vol < 0.3 * mean_vol
    if "taker_buy_base" in base:
        buy_share = (base["taker_buy_base"] / vol.replace(0, np.nan)).clip(0, 1)
        smooth = buy_share.rolling(w["1h"], min_periods=1).mean()
        a["taker_buy_dominance"] = smooth > 0.60
        a["taker_sell_dominance"] = smooth < 0.40
    else:
        a["taker_buy_dominance"] = False
        a["taker_sell_dominance"] = False

    a["ema_cross_up"] = (dist_d > 0) & (dist_d.shift(1) <= 0)
    a["ema_cross_down"] = (dist_d < 0) & (dist_d.shift(1) >= 0)
    a["trend_up_align"] = (close > ema_d) & (ema_d > ema_w)
    a["trend_down_align"] = (close < ema_d) & (ema_d < ema_w)

    a["rsi_overbought"] = rsi14 > 70
    a["rsi_oversold"] = rsi14 < 30
    a["momentum_reversal_up"] = (rsi14 > 30) & (rsi14.shift(1) <= 30)
    a["momentum_reversal_down"] = (rsi14 < 70) & (rsi14.shift(1) >= 70)

    a["at_range_high"] = pos_1w > 0.85
    a["at_range_low"] = pos_1w < 0.15
    a["mid_range"] = pos_1w.between(0.40, 0.60)
    # Круглые уровни: тысячи долларов притягивают заявки и стопы.
    a["round_level_touch"] = ((close % 1000) < 50) | ((close % 1000) > 950)

    hour = base.index.hour
    dow = base.index.dayofweek
    a["asia_session"] = pd.Series((hour >= 0) & (hour < 8), index=base.index)
    a["europe_session"] = pd.Series((hour >= 8) & (hour < 14), index=base.index)
    a["us_session"] = pd.Series((hour >= 14) & (hour < 22), index=base.index)
    a["weekend"] = pd.Series(dow >= 5, index=base.index)

    _attach_smc(a, base)

    return a[ATOMS].fillna(False).astype(bool)


def _attach_smc(a: pd.DataFrame, base: pd.DataFrame) -> None:
    """
    Шестнадцать SMC-атомов — или шестнадцать пустых колонок, если выключено.

    Колонки заводятся в обоих случаях: ATOMS перечисляет их безусловно, и
    отсутствие колонки при выключенном флаге уронило бы срез `a[ATOMS]`.
    Пустые колонки в блок не попадают (все атомы контекстные) и в
    bar_events.context_atoms не пишутся, потому что там хранятся только
    активные имена.

    Импорт внутри функции: smc.py тянет тяжёлый расчёт, а detect_atoms зовут
    и из мест, где SMC выключен.
    """
    from btcproc.features import smc

    if not config.smc.enabled:
        for name in SMC_ATOMS:
            a[name] = False
        return

    values = smc.build_smc_cached(base)
    for name in SMC_ATOMS:
        a[name] = values[name]


def _mask_to_atoms(mask: int) -> list[str]:
    return [atom for atom, bit in ATOM_BIT.items() if mask >> bit & 1]


def _mask_to_context_atoms(mask: int) -> list[str]:
    return [atom for atom, bit in CONTEXT_BIT.items() if mask >> bit & 1]


def block_id(mask: int) -> str:
    """
    Идентификатор блока в формате из ТЗ: event_block_NNNNNN.

    Берём хэш от маски, а не саму маску: она 29-битная и её десятичный вид
    ничего не сообщает, а хэш даёт короткий стабильный ключ.
    """
    digest = hashlib.sha1(str(int(mask)).encode()).hexdigest()
    return f"event_block_{int(digest[:8], 16) % 1_000_000:06d}"


def intensity_bucket(atom_count: int) -> str:
    if atom_count <= SPARSE_MAX:
        return "sparse"
    if atom_count <= MODERATE_MAX:
        return "moderate"
    return "dense"


def rarity_bucket(share: float) -> str:
    if share < RARE_SHARE:
        return "rare"
    if share < UNCOMMON_SHARE:
        return "uncommon"
    return "common"


def build_event_blocks(base: pd.DataFrame, window_bars: int | None = None) -> pd.DataFrame:
    """
    Блок событий на каждый бар.

    window_bars — окно, в котором атомы считаются «сопровождавшими состояние»
    (по умолчанию час). Внутри окна атомы объединяются по ИЛИ: событие часовой
    давности всё ещё описывает контекст текущего бара.

    Возвращает: event_block_id, atoms, context_atoms, families, atom_count,
    family_count, intensity, primary_family.

    В маску (а значит, в event_block_id, atoms, families, atom_count и
    primary_family) входят только SIGNATURE_ATOMS. Контекстные атомы едут
    отдельной колонкой context_atoms: на идентификатор блока они не влияют,
    но нужны для разбора — без них лифт по фоновым признакам не измерить.
    """
    if base.empty:
        return pd.DataFrame()

    window_bars = window_bars or config.data.bars_per("1h")
    detected = detect_atoms(base)
    # Скользящее ИЛИ по окну. Считается сразу по всем атомам: max по окну
    # берётся поколоночно, поэтому срез до или после роллинга даёт одно и то же.
    rolled = detected.rolling(window_bars, min_periods=1).max().astype(bool)
    active = rolled[SIGNATURE_ATOMS]
    context = rolled[CONTEXT_ATOM_LIST]

    bits = np.arange(len(SIGNATURE_ATOMS), dtype=np.int64)
    masks = (active.to_numpy(dtype=np.int64) << bits).sum(axis=1)
    context_bits = np.arange(len(CONTEXT_ATOM_LIST), dtype=np.int64)
    context_masks = (context.to_numpy(dtype=np.int64) << context_bits).sum(axis=1)

    out = pd.DataFrame(index=base.index)
    out["mask"] = masks
    out["context_mask"] = context_masks
    out["atom_count"] = active.sum(axis=1).astype(int)

    # Уникальных масок на порядки меньше, чем баров, — расшифровываем их
    # по одному разу и раздаём по строкам через map.
    unique = pd.unique(masks)
    meta = {}
    for mask in unique:
        names = _mask_to_atoms(int(mask))
        families = sorted({ATOM_FAMILY[n] for n in names})
        counts: dict[str, int] = {}
        for n in names:
            counts[ATOM_FAMILY[n]] = counts.get(ATOM_FAMILY[n], 0) + 1
        primary = max(counts, key=counts.get) if counts else None
        meta[int(mask)] = {
            "event_block_id": block_id(mask),
            "atoms": names,
            "families": families,
            "family_count": len(families),
            "primary_family": primary,
        }

    for field in ("event_block_id", "atoms", "families", "family_count", "primary_family"):
        out[field] = out["mask"].map(lambda m, f=field: meta[int(m)][f])
    out["intensity"] = out["atom_count"].map(intensity_bucket)

    # Контекстных комбинаций максимум 2^9 — расшифровываем так же, по маскам.
    context_meta = {
        int(m): _mask_to_context_atoms(int(m)) for m in pd.unique(context_masks)
    }
    out["context_atoms"] = out["context_mask"].map(lambda m: context_meta[int(m)])

    logger.info("Блоков событий: %d уникальных на %d баров", len(unique), len(out))
    return out.drop(columns=["mask", "context_mask"])


def block_statistics(events: pd.DataFrame) -> pd.DataFrame:
    """
    Частотная статистика блоков: total_rows, row_share, rarity.

    Это ровно те поля кандидата, что отвечают на вопрос «насколько
    специфичен контекст» — считаются по всей доступной истории.
    """
    total = len(events)
    grouped = events.groupby("event_block_id").agg(
        total_rows=("event_block_id", "size"),
        atom_count=("atom_count", "median"),
        family_count=("family_count", "median"),
        intensity=("intensity", lambda s: s.mode().iat[0]),
        primary_family=("primary_family", lambda s: s.mode().iat[0] if s.notna().any() else None),
        families=("families", "last"),
    )
    grouped["row_share"] = grouped["total_rows"] / max(total, 1)
    grouped["rarity"] = grouped["row_share"].map(rarity_bucket)
    grouped["atom_count"] = grouped["atom_count"].astype(int)
    grouped["family_count"] = grouped["family_count"].astype(int)
    return grouped.reset_index()
