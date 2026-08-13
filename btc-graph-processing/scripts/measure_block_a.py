"""
Замер сдвига от блока A аудита 2026-08-13 (ТЗ `docs/tz_btcproc_audit_13-08-2026.md`).

    python3 scripts/measure_block_a.py --all
    python3 scripts/measure_block_a.py --symbol BTCUSDT --end 2026-08-01

Отвечает ровно на те вопросы, которые ТЗ требует закрыть числом, а не
рассуждением. Ни одна правка блока A не делает систему предсказательной —
раздел 26 журнала об этом однозначен. Замер нужен для другого: понять
масштаб искажения, которое из системы убрали, и решить, что делать с
перекалибровкой профилей оценки в btc-graph.

## Что меряется

**A1 — abs(mae).** Доля случаев `ret > 0 и mae_pct > 0` по всей истории
монеты: именно они записывались как «просадка», которой не было. Рядом —
парный сдвиг `p80_long_adverse_pct` и `long_favorable_adverse_ratio_p70_p80`
по одним и тем же кандидатам, в медиане и квартилях.

**A2 — sample_size против effective_sample_size.** Отношение по выдаче и
доля конфигураций, которые перестают проходить порог, когда он применяется
к реализациям, а не к строкам снимков.

**A3 — доля выдачи с `sample_scope="transition"`**, то есть доля кандидатов,
у которых поля блока событий и историческая статистика относятся к разным
вещам.

**A4 — доля кандидатов в полосе `|skew| ∈ [min_abs_skew, bias_skew_threshold)`**
и число ключей семьи, в которых до правки сходились обе стороны.

## Как устроено «до»

Прогон делается ТРИЖДЫ по одним и тем же снимкам, и это не расточительство,
а условие корректности замера. Правки A1 и A2 нельзя мерить одним общим
«до»: порог по реализациям (A2) меняет то, на какую выборку кандидат
опирается — при более строгом пороге чаще срабатывает откат на статистику по
переходу, — и разница p80 между «до всего» и «после всего» смешала бы
исправление `abs(mae)` со сменой самой выборки. Поэтому:

* `LegacyMae` — отличается от нового кода ТОЛЬКО `abs(mae)`. Пара с новым
  прогоном даёт чистый эффект A1.
* `LegacyAll` — плюс `effective == total`, то есть код целиком в прежнем
  виде. Пара с новым прогоном даёт эффект A2: сколько кандидатов отсеялось
  и как поехал `research_score`.

A3 и A4 «до» не требуют: там сравнивать нечего, вопрос в доле выдачи, и она
одинакова в обеих версиях.

## Чего замер НЕ делает

Не трогает БД на запись и не отправляет ничего в btc-graph. Модель состояний
берётся готовая — последнего успешного `train` монеты; переобучения здесь
нет и быть не должно (инвариант: `train` перенумеровывает `group_id` и сносит
граф монеты в Neo4j).
"""
from __future__ import annotations

import argparse
import bisect
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ДО импорта btcproc.config: конфиг читает окружение на импорте, а любой
# режим sink, кроме none, означал бы отправку кандидатов замера наружу.
os.environ["SINK_MODE"] = "none"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from btcproc import config, symbols  # noqa: E402
from btcproc.analysis import replay  # noqa: E402
from btcproc.candidates import builder as cand  # noqa: E402

#: Та же замороженная граница, что у validate_holdout: сравнивать замеры,
#: сделанные на разных концах истории, нельзя (раздел 0 плана 2026-08-12).
FROZEN_END = "2026-08-01"


class LegacyMae(cand.Accumulator):
    """
    Новый накопитель с ОДНИМ отличием: `abs(mae)` вместо `max(0, −mae)`.

    Изолирует эффект A1. Пороги и `research_score` при этом остаются
    новыми, поэтому набор кандидатов совпадает с новым прогоном строка в
    строку — и парная разница описывает ровно исправление знака.
    """

    def add(self, ts, ret, mfe, mae, valid, realization) -> None:  # noqa: D102
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
            bisect.insort(self.mfe_up, float(mfe))
            bisect.insort(self.mae_up, abs(float(mae)))   # ← прежнее поведение


class LegacyAll(LegacyMae):
    """
    Накопитель в том виде, в котором он работал до 2026-08-13, целиком.

    `effective == total` возвращает и прежние пороги выборки, и прежний
    `s_sample` в `research_score`: обе величины выражены через `effective`.
    """

    @property
    def effective(self) -> int:
        return self.total


# ─── Подготовка данных ──────────────────────────────────────────────────────
def prepare(symbol: str, end: str) -> replay.Replay:
    """
    Снимки, карты редкости и исходы по модели последнего train'а монеты.

    Разметка — общий код с выгрузкой кандидатов (`btcproc/analysis/replay.py`):
    замер обязан идти по тем же данным, по которым потом калибруются профили.
    """
    return replay.label_history(symbol, end=end, log=print)


def generate_frame(data: replay.Replay, accumulator: type | None = None) -> pd.DataFrame:
    """Выдача кандидатов с указанным накопителем (None — текущий код)."""
    original = cand.Accumulator
    if accumulator is not None:
        cand.Accumulator = accumulator
    try:
        rows = [
            {
                "candidate_id": c["candidate_id"],
                "side": c["research_side"],
                "skew": c["historical_outcome_skew"],
                "bias": c["historical_bias_context"],
                "family_key": c["candidate_family_key"],
                "sample_size": c["sample_size"],
                "effective": c["effective_sample_size"],
                "scope": c["sample_scope"],
                "research_score": c["research_score"],
                "favorable": c["p70_long_favorable_pct"],
                "adverse": c["p80_long_adverse_pct"],
                "ratio": c["long_favorable_adverse_ratio_p70_p80"],
            }
            for c in data.candidates()
        ]
    finally:
        cand.Accumulator = original
    return pd.DataFrame(rows)


# ─── Отдельные замеры ───────────────────────────────────────────────────────
def measure_mae_contamination(outcomes: pd.DataFrame) -> dict:
    """A1: сколько исходов вообще были загрязнены и насколько сильно."""
    valid = outcomes[outcomes["valid"].fillna(False)]
    up = valid[valid["ret_pct"] > 0]
    dirty = up[up["mae_pct"] > 0]
    return {
        "valid_bars": len(valid),
        "up_bars": len(up),
        "dirty_bars": len(dirty),
        "dirty_share_of_up": len(dirty) / len(up) if len(up) else 0.0,
        "dirty_median_pct": float(dirty["mae_pct"].median()) if len(dirty) else 0.0,
        "dirty_p90_pct": float(dirty["mae_pct"].quantile(0.9)) if len(dirty) else 0.0,
    }


def _quartiles(series: pd.Series) -> tuple[float, float, float]:
    return (float(series.quantile(0.25)), float(series.median()),
            float(series.quantile(0.75)))


def measure_shift(new: pd.DataFrame, old: pd.DataFrame) -> dict:
    """
    A1: парный сдвиг p80_adverse и F/A ratio.

    `old` здесь обязан отличаться от `new` ТОЛЬКО знаком mae (LegacyMae),
    иначе в разницу подмешается смена выборки из-за порога A2.
    """
    merged = new.merge(old, on="candidate_id", suffixes=("_new", "_old"))
    long_only = merged[merged["side_new"] == "long"]
    if long_only.empty:
        return {"common": 0}
    changed = long_only["adverse_new"] != long_only["adverse_old"]
    return {
        "common": len(merged),
        "common_long": len(long_only),
        "changed_share": float(changed.mean()),
        "max_abs_delta": float((long_only["adverse_old"]
                                - long_only["adverse_new"]).abs().max()),
        "adverse_old": _quartiles(long_only["adverse_old"]),
        "adverse_new": _quartiles(long_only["adverse_new"]),
        "ratio_old": _quartiles(long_only["ratio_old"]),
        "ratio_new": _quartiles(long_only["ratio_new"]),
        "adverse_zeroed": float((long_only["adverse_new"] == 0).mean()),
        "ratio_grew": float((long_only["ratio_new"] > long_only["ratio_old"]).mean()),
    }


def measure_sample(new: pd.DataFrame, old: pd.DataFrame) -> dict:
    """A2: во сколько раз sample_size завышал число случаев и что отсеялось."""
    ratio = new["sample_size"] / new["effective"]
    return {
        "produced_old": len(old),
        "produced_new": len(new),
        "dropped_share": 1.0 - len(new) / len(old) if len(old) else 0.0,
        "sample_q": _quartiles(new["sample_size"]),
        "effective_q": _quartiles(new["effective"]),
        "inflation_q": _quartiles(ratio),
        "research_score_old": _quartiles(old["research_score"]),
        "research_score_new": _quartiles(new["research_score"]),
    }


def measure_scope(new: pd.DataFrame) -> dict:
    """A3: какая доля выдачи несёт признаки блока, которого нет в статистике."""
    counts = new["scope"].value_counts()
    fallback = int(counts.get("transition", 0))
    return {
        "fallback": fallback,
        "fallback_share": fallback / len(new) if len(new) else 0.0,
        "conditioned": int(counts.get("transition+event_block", 0)),
    }


def measure_family(new: pd.DataFrame) -> dict:
    """A4: сколько кандидатов сидит в полосе, где семьи схлопывались."""
    cfg = config.candidates
    band = new[
        (new["skew"].abs() >= cfg.min_abs_skew)
        & (new["skew"].abs() < cfg.bias_skew_threshold)
    ]
    # Ключ БЕЗ стороны — тот, что был до правки.
    legacy_key = new["family_key"].str.rsplit("|", n=1).str[0]
    sides_per_key = new.groupby(legacy_key)["side"].nunique()
    collided = int((sides_per_key > 1).sum())
    return {
        "band_share": len(band) / len(new) if len(new) else 0.0,
        "band_rows": len(band),
        "legacy_keys": int(len(sides_per_key)),
        "collided_keys": collided,
        "collided_share": collided / len(sides_per_key) if len(sides_per_key) else 0.0,
    }


# ─── Печать ─────────────────────────────────────────────────────────────────
def _fmt_q(q) -> str:
    return f"p25 {q[0]:.3f} | med {q[1]:.3f} | p75 {q[2]:.3f}"


def report(symbol: str, run_id: int, mae: dict, shift: dict, sample: dict,
           scope: dict, family: dict) -> None:
    print(f"\n{'=' * 72}\n{symbol}  (модель прогона #{run_id})\n{'=' * 72}")

    print("\nA1. abs(mae) — «просадка», которой не было")
    print(f"  баров с валидным исходом: {mae['valid_bars']}, из них рост: {mae['up_bars']}")
    print(f"  рост БЕЗ захода ниже входа: {mae['dirty_bars']} "
          f"({mae['dirty_share_of_up']:.1%} всех ростов)")
    print(f"  записывалось как просадка: медиана {mae['dirty_median_pct']:.2f}%, "
          f"p90 {mae['dirty_p90_pct']:.2f}%")
    if shift.get("common"):
        print(f"  общих кандидатов для парного сравнения: {shift['common']} "
              f"(long: {shift['common_long']})")
        print(f"  у скольких long-кандидатов p80 вообще изменился: "
              f"{shift['changed_share']:.2%}, максимальный сдвиг "
              f"{shift['max_abs_delta']:.3f} п.п.")
        print(f"  p80_long_adverse_pct  было: {_fmt_q(shift['adverse_old'])}")
        print(f"  p80_long_adverse_pct стало: {_fmt_q(shift['adverse_new'])}")
        print(f"  F/A ratio             было: {_fmt_q(shift['ratio_old'])}")
        print(f"  F/A ratio            стало: {_fmt_q(shift['ratio_new'])}")
        print(f"  доля long-кандидатов с нулевой просадкой: {shift['adverse_zeroed']:.1%}")
        print(f"  доля long-кандидатов, у которых ratio вырос: {shift['ratio_grew']:.1%}")

    print("\nA2. sample_size против effective_sample_size")
    print(f"  кандидатов было: {sample['produced_old']}, стало: {sample['produced_new']} "
          f"(отсеялось {sample['dropped_share']:.1%})")
    print(f"  sample_size:    {_fmt_q(sample['sample_q'])}")
    print(f"  реализаций:     {_fmt_q(sample['effective_q'])}")
    print(f"  завышение (раз):{_fmt_q(sample['inflation_q'])}")
    print(f"  research_score  было: {_fmt_q(sample['research_score_old'])}")
    print(f"  research_score стало: {_fmt_q(sample['research_score_new'])}")

    print("\nA3. sample_scope")
    print(f"  обусловлено блоком: {scope['conditioned']}, откат на переход: "
          f"{scope['fallback']} ({scope['fallback_share']:.1%})")

    print("\nA4. семья и сторона")
    print(f"  в полосе |skew| ∈ [{config.candidates.min_abs_skew}, "
          f"{config.candidates.bias_skew_threshold}): {family['band_rows']} "
          f"({family['band_share']:.1%})")
    print(f"  ключей семьи (по-старому): {family['legacy_keys']}, из них с обеими "
          f"сторонами: {family['collided_keys']} ({family['collided_share']:.1%})")


def run_symbol(symbol: str, end: str) -> None:
    data = prepare(symbol, end)
    print(f"[{symbol}] прогон новой версии…")
    new = generate_frame(data)
    print(f"[{symbol}] прогон с прежним знаком mae (только A1)…")
    old_mae = generate_frame(data, LegacyMae)
    print(f"[{symbol}] прогон прежней версии целиком (A1 + A2)…")
    old_all = generate_frame(data, LegacyAll)

    if new.empty:
        print(f"[{symbol}] ВНИМАНИЕ: новая версия не выпустила ни одного "
              f"кандидата (прежняя выпускала {len(old_all)}). Порог "
              f"min_effective_sample_size для этой монеты недостижим.")
        return

    report(
        symbol, data.model_run,
        measure_mae_contamination(data.outcomes),
        measure_shift(new, old_mae),
        measure_sample(new, old_all),
        measure_scope(new),
        measure_family(new),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--end", default=FROZEN_END,
                        help=f"граница истории (по умолчанию {FROZEN_END})")
    args = parser.parse_args()

    specs = symbols.resolve_many(args.symbol, args.all)
    print(f"Замер блока A на {len(specs)} монет(ах), граница {args.end}.")
    print(f"Пороги: min_sample_size={config.candidates.min_sample_size}, "
          f"min_effective_sample_size={config.candidates.min_effective_sample_size}")
    for spec in specs:
        try:
            run_symbol(spec.ticker, args.end)
        except (SystemExit, replay.ReplayError) as exc:  # noqa: BLE001
            # Одна монета без модели не должна ронять замер по остальным.
            print(f"[{spec.ticker}] пропущена: {exc}")


if __name__ == "__main__":
    main()
