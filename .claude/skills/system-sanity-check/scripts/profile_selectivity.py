"""
Селективность профиля оценки: на всей истории и на окне, под которое он калибровался.

Отвечает на вопрос «у монеты кандидаты хуже, чем у BTC?». Почти всегда ответ —
«их сравнивают на разных окнах». Профиль калибруется по свежей четверти выгрузки
(`--tail` в `btc-graph/scripts/calibrate_profile.py`), потому что `sample_size`
кандидата монотонно растёт по истории. У монеты с десятью годами торгов старые
кандидаты вытягивает большой `sample_size` свежих; у монеты с двумя годами
вытягивать нечем, и её доля MODERATE+ по ВСЕЙ истории проваливается вдвое,
хотя на своём окне профиль в цели.

Ничего не пишет: читает `processing.candidates` и гоняет скорер btc-graph.

    cd btc-graph-processing
    PYTHONPATH=. ./.venv/bin/python \
        ../.claude/skills/system-sanity-check/scripts/profile_selectivity.py
    ... TAOUSDT PUMPUSDT      # только эти монеты

Цель калибратора — STRONG 1%, MODERATE+ 39%. Читать так: если на свежей
четверти монета в цель попадает, профиль исправен, а разрыв «по всей истории» —
свойство длины истории, а не линейки.
"""
from __future__ import annotations

import statistics as st
import sys

from btcproc.candidates.builder import strip_meta
from btcproc.db.session import fetch_all
from btcproc.sink import graph_sink

graph_sink._load_btc_graph()

from src.config.profiles import get_profile  # noqa: E402
from src.models.candidate import Candidate  # noqa: E402
from src.scorer.candidate_scorer import score_candidate  # noqa: E402

DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "AAVEUSDT",
                   "TAOUSDT", "PUMPUSDT", "HYPEUSDT")


def main(symbols: tuple[str, ...]) -> None:
    print(f"{'монета':10s} {'окно':>18s} {'n':>6s} {'STRONG':>8s} {'MOD+':>7s} "
          f"{'медиана':>8s} {'stat':>6s} {'dir':>6s} {'ctx':>6s} {'rar':>6s}")
    for sym in symbols:
        # ORDER BY ts обязателен: окно «свежая четверть» — это хвост по времени.
        rows = fetch_all(
            "SELECT payload FROM candidates WHERE symbol=%s ORDER BY ts", [sym])
        if not rows:
            print(f"{sym:10s} {'нет кандидатов':>18s}")
            continue
        profile = get_profile(sym)
        windows = (("вся история", rows),
                   ("свежая четверть", rows[int(len(rows) * 0.75):]))
        for label, window in windows:
            scored = [score_candidate(Candidate(**strip_meta(r["payload"])), profile)
                      for r in window]
            n = len(scored)
            strong = sum(1 for s in scored if s.total >= profile.rating.strong_min) / n
            moderate = sum(1 for s in scored if s.total >= profile.rating.moderate_min) / n
            print(f"{sym:10s} {label:>18s} {n:6d} {strong:8.1%} {moderate:7.1%} "
                  f"{st.median(s.total for s in scored):8.3f} "
                  f"{st.median(s.statistical for s in scored):6.2f} "
                  f"{st.median(s.directional for s in scored):6.2f} "
                  f"{st.median(s.context for s in scored):6.2f} "
                  f"{st.median(s.rarity for s in scored):6.2f}")


if __name__ == "__main__":
    main(tuple(a.upper() for a in sys.argv[1:]) or DEFAULT_SYMBOLS)
