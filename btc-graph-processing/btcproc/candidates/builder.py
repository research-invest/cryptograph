"""
Сборка кандидатов — центральный модуль генератора.

Кандидат выпускается в момент, когда рынок находится в состоянии после
перехода, и отвечает на вопрос «что бывало раньше в такой же конфигурации».
Конфигурация = переход между состояниями + блок сопровождающих событий.

Три правила, без которых вся статистика была бы самообманом:

  1. **Только прошлое.** Выборка для кандидата в момент t собирается из
     случаев, которые произошли раньше t.
  2. **Только созревшие исходы.** Случай попадает в выборку не раньше, чем
     пройдёт горизонт: в момент t исход события, случившегося час назад,
     ещё неизвестен.
  3. **Разные возрасты состояния.** Снимки берутся не только в момент
     перехода, но и спустя 45/90/180 минут — иначе все кандидаты в истории
     были бы `age_lt_30` + `fresh`, и оси контекста в btc-graph
     не различались бы вовсе.
"""
from __future__ import annotations

import bisect
import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator

import numpy as np
import pandas as pd

from btcproc import config

logger = logging.getLogger(__name__)

# Смещения снимков от момента перехода, в минутах.
# Дают полный набор age-бакетов схемы: <30, 30-60, 60-120, >120.
SNAPSHOT_OFFSETS_MIN = (0, 45, 90, 180)


@dataclass
class Accumulator:
    """Историческая выборка по одному ключу конфигурации."""

    total: int = 0
    valid: int = 0
    up: int = 0
    ret_sum: float = 0.0
    mfe_up: list[float] = field(default_factory=list)   # отсортирован
    mae_up: list[float] = field(default_factory=list)   # отсортирован, значения ≥ 0
    days: set = field(default_factory=set)
    months: Counter = field(default_factory=Counter)
    # Идентификаторы РЕАЛИЗАЦИЙ перехода, попавших в выборку (`state_seq`).
    # См. свойство `effective` — без него `total` означает строки, а не случаи.
    realizations: set = field(default_factory=set)

    def add(
        self,
        ts: pd.Timestamp,
        ret: float,
        mfe: float,
        mae: float,
        valid: bool,
        realization: object,
    ) -> None:
        self.total += 1
        self.realizations.add(realization)
        self.days.add(ts.date())
        self.months[(ts.year, ts.month)] += 1
        if not valid:
            return
        self.valid += 1
        self.ret_sum += ret
        if ret > 0:
            self.up += 1
            # favorable / adverse описывают путь «хорошего» случая: сколько
            # прошло в нашу сторону и сколько пришлось пересидеть против.
            bisect.insort(self.mfe_up, float(mfe))
            # mae_pct = min(low за горизонт) / entry − 1, то есть величина
            # СО ЗНАКОМ: отрицательная, если цена ходила ниже входа, и
            # положительная, если не ходила вовсе. abs() здесь (до 2026-08-13)
            # переворачивал смысл: чем чище был рост без отката, тем больше
            # записывалась «просадка». Загрязнение било прицельно — список
            # наполняется только по случаям ret > 0, где mae_pct > 0 как раз
            # вероятнее всего.
            #
            # max(0, −mae) даёт глубину просадки в положительных процентах,
            # а ноль означает «цена ни разу не была ниже входа». Это
            # содержательное значение, а не отсутствующее.
            bisect.insort(self.mae_up, max(0.0, -float(mae)))

    @property
    def invalid(self) -> int:
        return self.total - self.valid

    @property
    def effective(self) -> int:
        """
        Число независимых реализаций перехода в выборке.

        `total` считает СТРОКИ снимков, а одна реализация даёт до четырёх
        строк (`SNAPSHOT_OFFSETS_MIN`), окна исходов которых при горизонте
        24h перекрываются минимум на 87.5%. Замер 2026-08-13 на четырёх
        монетах: медиана завышения 2.2 (до старших офсетов доживает не всякая
        реализация — снимок берётся, только если состояние не сменилось).
        То есть «выборка 1000» означала около 450 случаев, и всё, что от
        размера выборки зависит (`research_score`, пороги, ступень
        statistical в btc-graph), оперировало завышенным числом.

        Ломать `sample_size` при этом нельзя: на нём откалиброваны профили
        оценки, поэтому наружу уходят оба числа.
        """
        return len(self.realizations)

    @property
    def long_share(self) -> float:
        return self.up / self.valid if self.valid else 0.0

    @property
    def monthly_concentration(self) -> float:
        if not self.total:
            return 0.0
        return max(self.months.values()) / self.total


def _percentile_sorted(values: list[float], q: float) -> float:
    """Перцентиль по уже отсортированному списку, с линейной интерполяцией."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q / 100.0
    low = int(np.floor(pos))
    high = min(low + 1, len(values) - 1)
    weight = pos - low
    return values[low] * (1 - weight) + values[high] * weight


def _hash(*parts: object, length: int = 20) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode()).hexdigest()[:length]


def _rarity_score(rarity: str) -> float:
    return {"rare": 1.0, "uncommon": 0.66, "common": 0.33}.get(rarity, 0.33)


def research_score(
    acc: Accumulator,
    transition_rarity: str,
    event_rarity: str,
) -> float:
    """
    Синтетическая оценка силы исторической аналогии [0..1].

    Это НЕ вероятность прибыли (см. README btc-graph) — только то, насколько
    выборка велика, устойчива во времени и однонаправлена. btc-graph считает
    свой quality_score независимо и использует research_score лишь как один
    из входов оси rarity.
    """
    skew = abs(2 * acc.long_share - 1)
    valid_pct = acc.valid / acc.total if acc.total else 0.0
    months = len(acc.months)

    s_skew = min(1.0, skew / 0.5)
    # Считается по РЕАЛИЗАЦИЯМ, а не по строкам снимков: до 2026-08-13 здесь
    # стоял acc.total, и «1000 случаев → 1.0» означало около 450 (замеренная
    # медиана завышения — 2.2).
    s_sample = min(1.0, np.log10(acc.effective + 1) / 3.0)      # 1000 случаев → 1.0
    s_repeat = min(1.0, months / 18.0)
    s_spread = 1.0 - min(1.0, acc.monthly_concentration / 0.5)
    s_valid = valid_pct
    s_rare = (_rarity_score(transition_rarity) + _rarity_score(event_rarity)) / 2

    score = (
        0.25 * s_skew
        + 0.20 * s_sample
        + 0.15 * s_repeat
        + 0.15 * s_spread
        + 0.15 * s_valid
        + 0.10 * s_rare
    )
    return float(min(1.0, max(0.0, score)))


def _bias_context(long_share: float, threshold: float) -> str:
    if long_share > 0.5 + threshold:
        return "long_skew"
    if long_share < 0.5 - threshold:
        return "short_skew"
    return "neutral"


def build_snapshots(
    states: pd.DataFrame,
    events: pd.DataFrame,
    outcomes: pd.DataFrame,
    offsets_min: tuple[int, ...] = SNAPSHOT_OFFSETS_MIN,
) -> pd.DataFrame:
    """
    Моменты, в которые имеет смысл выпускать кандидата.

    Снимок берётся, только если состояние к этому времени не сменилось —
    иначе он описывал бы уже другую конфигурацию.
    """
    transitions = states[states["is_transition"] & states["transition_id"].notna()]
    if transitions.empty:
        return pd.DataFrame()

    bar_minutes = config.data.base_minutes
    rows = []
    index = states.index

    for offset in offsets_min:
        shift_bars = offset // bar_minutes
        if shift_bars == 0:
            snap_index = transitions.index
            source = transitions
        else:
            positions = index.get_indexer(transitions.index) + shift_bars
            keep = positions < len(index)
            positions, source = positions[keep], transitions[keep]
            snap_index = index[positions]

        snap_states = states.loc[snap_index]
        # Состояние не должно было смениться за время offset.
        same_state = snap_states["state_seq"].to_numpy() == source["state_seq"].to_numpy()
        snap_index = snap_index[same_state]
        if len(snap_index) == 0:
            continue
        source = source[same_state]
        snap_states = states.loc[snap_index]

        block = pd.DataFrame(
            {
                "ts": snap_index,
                "offset_min": offset,
                # Идентификатор реализации перехода. Все снимки одной
                # реализации (offsets 0/45/90/180) имеют один state_seq —
                # именно по нему считается effective_sample_size, иначе одна
                # реализация выглядела бы четырьмя независимыми случаями.
                "state_seq": source["state_seq"].to_numpy(),
                "transition_id": source["transition_id"].to_numpy(),
                "prev_group_id": source["prev_group_id"].to_numpy(),
                "current_group_id": snap_states["group_id"].to_numpy(),
                "age_minutes": snap_states["age_minutes"].to_numpy(),
                "age_bucket": snap_states["age_bucket"].to_numpy(),
                "entropy": snap_states["entropy"].to_numpy(),
            }
        )
        rows.append(block)

    snapshots = (
        pd.concat(rows, ignore_index=True).sort_values("ts").reset_index(drop=True)
    )
    event_cols = events[
        ["event_block_id", "atom_count", "family_count", "intensity", "primary_family"]
    ].reindex(snapshots["ts"]).reset_index(drop=True)
    outcome_cols = (
        outcomes[["ret_pct", "mfe_pct", "mae_pct", "valid"]]
        .reindex(snapshots["ts"])
        .reset_index(drop=True)
    )
    snapshots = pd.concat([snapshots, event_cols, outcome_cols], axis=1)

    # Снимок без событий или без разметки исхода в выборку не годится:
    # valid после reindex может стать NaN, а NaN в bool() превратился бы в True.
    snapshots["valid"] = snapshots["valid"].fillna(False).astype(bool)
    snapshots[["ret_pct", "mfe_pct", "mae_pct"]] = snapshots[
        ["ret_pct", "mfe_pct", "mae_pct"]
    ].fillna(0.0)
    snapshots = snapshots.dropna(subset=["event_block_id"])
    logger.info("Снимков конфигураций: %d", len(snapshots))
    return snapshots.reset_index(drop=True)


def generate(
    snapshots: pd.DataFrame,
    transition_rarity: dict[str, str],
    block_meta: dict[str, dict],
    symbol: str | None = None,
    cfg: config.CandidateConfig | None = None,
    progress=None,
) -> Iterator[dict]:
    """
    Проходит историю снимков в хронологическом порядке и выпускает кандидатов.

    transition_rarity — редкость перехода из статистики графа;
    block_meta       — статистика блока событий (total_rows, row_share, rarity…).

    Генератор, а не список: кандидатов на 8 лет истории — сотни тысяч, держать
    их все в памяти незачем, вызывающий код пишет их пачками.
    """
    cfg = cfg or config.candidates
    symbol = symbol or config.data.symbol
    horizon = pd.Timedelta(minutes=config.data.horizon_minutes)

    by_full: dict[tuple[str, str], Accumulator] = {}
    by_transition: dict[str, Accumulator] = {}

    records = snapshots.to_dict("records")
    total = len(records)
    pending = 0  # индекс первого ещё не «созревшего» снимка

    for i, row in enumerate(records):
        ts = row["ts"]

        # Шаг 2: доливаем в накопители всё, чей горизонт уже закрылся.
        while pending < total and records[pending]["ts"] + horizon <= ts:
            done = records[pending]
            key = (done["transition_id"], done["event_block_id"])
            by_full.setdefault(key, Accumulator()).add(
                done["ts"], done["ret_pct"], done["mfe_pct"], done["mae_pct"],
                bool(done["valid"]), done["state_seq"],
            )
            by_transition.setdefault(done["transition_id"], Accumulator()).add(
                done["ts"], done["ret_pct"], done["mfe_pct"], done["mae_pct"],
                bool(done["valid"]), done["state_seq"],
            )
            pending += 1

        acc = by_full.get((row["transition_id"], row["event_block_id"]))
        scope = "transition+event_block"
        if (acc is None or not _sample_enough(acc, cfg)) and cfg.fallback_to_transition:
            acc = by_transition.get(row["transition_id"])
            scope = "transition"
        if acc is None or not _sample_enough(acc, cfg) or acc.valid == 0:
            continue

        candidate = _assemble(row, acc, scope, transition_rarity, block_meta, symbol, cfg)
        if candidate is not None:
            yield candidate

        if progress and i % 5000 == 0:
            progress(i, total)


def _sample_enough(acc: Accumulator, cfg: config.CandidateConfig) -> bool:
    """
    Хватает ли выборки, чтобы выпускать по ней кандидата.

    Порогов ДВА, и они про разное. `min_sample_size` — про строки снимков,
    он оставлен ради совместимости со старой калибровкой. Работающий порог —
    `min_effective_sample_size`: он считает независимые реализации перехода,
    а прежние 30 строк снимков означали на деле около четырнадцати случаев.
    """
    return (
        acc.total >= cfg.min_sample_size
        and acc.effective >= cfg.min_effective_sample_size
    )


def _assemble(
    row: dict,
    acc: Accumulator,
    scope: str,
    transition_rarity: dict[str, str],
    block_meta: dict[str, dict],
    symbol: str,
    cfg: config.CandidateConfig,
) -> dict | None:
    long_share = acc.long_share
    skew = 2 * long_share - 1
    if abs(skew) < cfg.min_abs_skew:
        # Без перекоса кандидата нет: направление не определено.
        return None

    side = "long" if skew > 0 else "short"
    block = block_meta.get(row["event_block_id"], {})
    t_rarity = transition_rarity.get(row["transition_id"], "common")
    e_rarity = block.get("rarity", "common")

    # Редкость блока событий описывает бар, на котором выпущен кандидат, и в
    # этом качестве она честна всегда. Но НАЧИСЛЯТЬ за неё баллы можно только
    # тогда, когда историческая выборка на блок обусловлена: при откате на
    # scope="transition" статистика собрана по переходу целиком, и «редкий
    # блок» к ней отношения не имеет. То же ограничение стоит в btc-graph
    # (`_score_rarity` по sample_scope) — здесь его двойник для research_score.
    e_rarity_scored = e_rarity if scope == "transition+event_block" else "common"

    favorable = _percentile_sorted(acc.mfe_up, cfg.favorable_percentile)
    adverse = _percentile_sorted(acc.mae_up, cfg.adverse_percentile)
    ratio = favorable / adverse if adverse > 1e-9 else float(favorable)

    ts: pd.Timestamp = row["ts"]
    group_label = _fmt_group(row["current_group_id"])
    bias = _bias_context(long_share, cfg.bias_skew_threshold)

    # Контекст считается устаревшим, когда с момента перехода прошло больше
    # порога: конфигурация ещё формально та же, но снимок уже не «сейчас».
    context_status = "fresh" if row["age_minutes"] <= cfg.fresh_max_lag_minutes else "stale"

    candidate = {
        # Identity
        #
        # symbol входит во ВСЕ три идентификатора, и это не косметика.
        # transition_id, event_block_id и group_id — номера внутри графа одной
        # монеты: модель состояний обучается на каждую монету отдельно, и
        # «42->1» у BTC и у ETH — разные переходы. Без символа:
        #   * configuration_hash совпадёт, и btc-graph отдаст ETH готовую
        #     оценку BTC (дедуп по хэшу, TTL 30 мин) — кандидат при этом
        #     выглядит нормально оценённым, по данным это не диагностируется;
        #   * candidate_family_key совпадёт, и его фильтр схлопнет семьи двух
        #     монет в одну, погасив кандидата одной из них.
        "candidate_id": _hash(symbol, ts, row["transition_id"], row["event_block_id"],
                              row["offset_min"]),
        "symbol": symbol,
        "configuration_hash": _hash(symbol, row["transition_id"], row["event_block_id"],
                                    row["age_bucket"], row["entropy"], length=16),
        # side входит в ключ семьи, потому что bias его не заменяет: пороги
        # разные (min_abs_skew 0.06 против bias_skew_threshold 0.10), и в
        # полосе между ними long-кандидат и short-кандидат по одной
        # конфигурации получают bias="neutral" оба. До 2026-08-13 они падали
        # в одну семью и вытесняли друг друга в select_best_per_family
        # btc-graph — выживал не более сильный по направлению, а более
        # свежий. Конфликт направлений при этом и так детектируется отдельно
        # (`candidate_filter.detect_conflicts`), схлопывать их в ключе незачем.
        "candidate_family_key": (
            f"{symbol}|{row['current_group_id']:g}|{row['transition_id']}"
            f"|{row['event_block_id']}|{bias}|{side}"
        ),
        "research_score": research_score(acc, t_rarity, e_rarity_scored),

        # State / Trajectory
        "previous_group_id": _as_float(row["prev_group_id"]),
        "current_group_id": float(row["current_group_id"]),
        "transition_id": row["transition_id"],
        "current_group_age_bucket": row["age_bucket"],
        "context_status": context_status,
        "trajectory_entropy": row["entropy"],
        "transition_rarity": t_rarity,

        # Event context
        "event_block_id": row["event_block_id"],
        # NaN вместо None: reindex по снимкам возвращает пропуски float'ом,
        # а в схеме btc-graph это Optional[str] — NaN валидацию не проходит.
        "primary_event_family": _as_str_or_none(row.get("primary_family")),
        "event_intensity_bucket": row["intensity"],
        "event_rarity_bucket": e_rarity,
        "signature_atom_count": int(row["atom_count"]),
        "event_family_count": int(row["family_count"]),
        "event_block_total_rows": int(block.get("total_rows", acc.total)),
        "event_block_row_share": float(block.get("row_share", 0.0)),

        # Historical sample
        "horizon": config.data.horizon,
        "sample_size": acc.total,
        # Число независимых реализаций перехода — см. Accumulator.effective.
        # Отношение sample_size / effective_sample_size — около 2.2 по медиане
        # (замер 29.2 в журнале); именно effective отвечает на вопрос
        # «сколько раз такое было».
        "effective_sample_size": acc.effective,
        # На что обусловлена выборка. "transition" означает откат: пары
        # (переход × блок событий) не набралось, статистика собрана по
        # переходу целиком. Поля блока при этом остаются заполненными — они
        # описывают текущий бар, — но выборка про блок ничего не знает, и
        # приёмник обязан это учитывать (иначе кандидат утверждает «редкий
        # блок» на статистике, к блоку не относящейся).
        "sample_scope": scope,
        "valid_label_count": acc.valid,
        "invalid_label_count": acc.invalid,
        "valid_label_pct": acc.valid / acc.total if acc.total else 0.0,
        "repeatability_days": len(acc.days),
        "repeatability_months": len(acc.months),
        "monthly_concentration": acc.monthly_concentration,

        # Outcome profile
        "historical_bias_context": bias,
        "research_side": side,
        "long_outcome_count": acc.up,
        "short_outcome_count": acc.valid - acc.up,
        "long_outcome_share": long_share,
        "historical_outcome_skew": skew,

        # Favorable / adverse (всегда в терминах long — как в схеме btc-graph)
        "p70_long_favorable_pct": favorable,
        "p80_long_adverse_pct": adverse,
        "long_favorable_adverse_ratio_p70_p80": ratio,
    }

    # Служебные поля: в btc-graph не уходят, нужны нам для админки и разбора.
    # `scope` здесь больше нет — он переехал в тело кандидата как
    # `sample_scope`: strip_meta() вырезает `_meta` перед отправкой, и
    # приёмник о разнице между обусловленной и необусловленной выборкой
    # не знал вовсе.
    candidate["_meta"] = {
        "ts": ts.isoformat(),
        "group_label": group_label,
        "age_minutes": int(row["age_minutes"]),
        "offset_min": int(row["offset_min"]),
        "avg_ret_pct": acc.ret_sum / acc.valid if acc.valid else None,
    }
    return candidate


def _as_float(value) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


def _as_str_or_none(value) -> str | None:
    """pandas отдаёт пропуски как NaN — в схеме кандидата это должен быть None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return str(value)


def _fmt_group(value: float) -> str:
    return f"{value:g}"


def strip_meta(candidate: dict) -> dict:
    """Кандидат без служебных полей — 39 полей схемы btc-graph.

    Было 37; 2026-08-13 добавились `effective_sample_size` и `sample_scope`.
    Оба заведены в модели приёмника как Optional с дефолтом None, поэтому
    старые выгрузки его валидатор проходят по-прежнему.
    """
    return {k: v for k, v in candidate.items() if not k.startswith("_")}


def meta_of(candidate: dict) -> dict:
    return candidate.get("_meta", {})


def parse_ts(candidate: dict) -> datetime:
    return pd.Timestamp(candidate["_meta"]["ts"]).to_pydatetime()
