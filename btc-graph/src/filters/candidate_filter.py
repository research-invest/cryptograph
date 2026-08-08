"""
Задачи 4 и 5: фильтрация, сравнение и детектирование конфликтов.

Все группировки здесь идут по паре (symbol, ключ). Причина не в аккуратности:
`candidate_family_key`, `transition_id` и `group_id` — идентификаторы ВНУТРИ
графа одной монеты. Генератор выдаёт «42->1» и для BTC, и для ETH, и без
символа в ключе кандидат одной монеты вытеснял бы кандидата другой из своей
«семьи», а BTC-long вместе с ETH-short давали бы ложный конфликт направлений.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.config.profiles import ScoringProfile, get_profile
from src.models.candidate import Candidate, ContextStatus, AgeBucket
from src.scorer.candidate_scorer import score_candidate


@dataclass
class ConflictReport:
    conflict_type: str
    candidate_ids: list[str]
    description: str


def filter_candidates(
    candidates: list[Candidate],
    min_quality_score: float | None = None,
    profiles: dict[str, ScoringProfile] | None = None,
) -> list[tuple[Candidate, float]]:
    """
    Задача 4: фильтрует по quality_score и ранжирует.
    Возвращает список (candidate, quality_score) выше порога, отсортированный по убыванию.

    min_quality_score=None — порог берётся из профиля каждого кандидата: у
    монеты со своей калибровкой и своя граница «достойно рассмотрения».
    Явное число перекрывает профиль для всего списка сразу.

    profiles — заранее отрезолвленные профили по символу; pipeline передаёт их,
    чтобы перезагрузка конфига посреди батча не смешала две калибровки.
    """
    scored = []
    for c in candidates:
        profile = (profiles or {}).get(c.symbol) or get_profile(c.symbol)
        threshold = (
            min_quality_score
            if min_quality_score is not None
            else profile.batch.min_quality_score
        )
        breakdown = score_candidate(c, profile)
        if breakdown.total >= threshold:
            scored.append((c, breakdown.total))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _fresh_bonus(candidate: Candidate, profiles: dict[str, ScoringProfile] | None) -> float:
    profile = (profiles or {}).get(candidate.symbol) or get_profile(candidate.symbol)
    return profile.batch.fresh_bonus


def _family_rank(
    candidate: Candidate,
    quality_score: float,
    profiles: dict[str, ScoringProfile] | None = None,
) -> float:
    """Ключ сортировки внутри семьи: score с небольшой надбавкой за fresh."""
    bonus = (
        _fresh_bonus(candidate, profiles)
        if candidate.context_status == ContextStatus.fresh
        else 0.0
    )
    return quality_score + bonus


def select_best_per_family(
    scored: list[tuple[Candidate, float]],
    profiles: dict[str, ScoringProfile] | None = None,
) -> list[tuple[Candidate, float]]:
    """
    Задача 4: внутри каждой пары (symbol, candidate_family_key) выбирает лучшего.

    Побеждает наибольший quality_score; свежесть добавляет fresh_bonus из
    профиля, то есть при сопоставимых оценках выигрывает fresh, но заметно
    более сильный stale не проигрывает слабому fresh (audit #9).

    Символ входит в ключ обязательно: без него «семья 1.0|42->1|…» у BTC и ETH
    одна и та же, и одна монета молча гасит другую.
    """
    groups: dict[tuple[str, str], list[tuple[Candidate, float]]] = defaultdict(list)
    for candidate, qs in scored:
        key = candidate.candidate_family_key or candidate.candidate_id
        groups[(candidate.symbol, key)].append((candidate, qs))

    result = []
    for group in groups.values():
        group.sort(key=lambda x: -_family_rank(x[0], x[1], profiles))
        result.append(group[0])

    result.sort(key=lambda x: x[1], reverse=True)
    return result


def detect_conflicts(
    candidates: list[Candidate],
    profiles: dict[str, ScoringProfile] | None = None,
) -> list[ConflictReport]:
    """
    Задача 5: обнаруживает противоречивые сигналы среди кандидатов.
    """
    conflicts: list[ConflictReport] = []

    def profile_of(c: Candidate) -> ScoringProfile:
        return (profiles or {}).get(c.symbol) or get_profile(c.symbol)

    # Конфликт 1: одинаковый transition_id одной монеты, разный research_side.
    # Пара (symbol, transition_id) обязательна: «42->1» в графе BTC и в графе
    # ETH — разные переходы, и разное направление по ним не противоречие.
    by_transition: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for c in candidates:
        by_transition[(c.symbol, c.transition_id)].append(c)

    for (symbol, tid), group in by_transition.items():
        sides = {c.research_side for c in group}
        if len(sides) > 1:
            conflicts.append(ConflictReport(
                conflict_type="contradictory_direction",
                candidate_ids=[c.candidate_id for c in group],
                description=(
                    f"{symbol}, переход {tid}: кандидаты указывают в разные стороны "
                    f"{[s.value for s in sides]}"
                ),
            ))

    # Конфликт 2: stale + age_gt_120 одновременно
    for c in candidates:
        if c.context_status == ContextStatus.stale and c.current_group_age_bucket == AgeBucket.age_gt_120:
            conflicts.append(ConflictReport(
                conflict_type="stale_and_old_context",
                candidate_ids=[c.candidate_id],
                description=f"Кандидат {c.candidate_id}: контекст устарел и состояние > 120 мин",
            ))

    # Конфликт 3: высокий research_score при низкой valid_label_pct — ложная уверенность
    for c in candidates:
        limits = profile_of(c).validator
        if (c.research_score > limits.false_confidence_research_score
                and c.valid_label_pct < limits.low_valid_label_pct):
            conflicts.append(ConflictReport(
                conflict_type="false_confidence",
                candidate_ids=[c.candidate_id],
                description=(
                    f"Кандидат {c.candidate_id}: research_score={c.research_score:.3f} высокий, "
                    f"но valid_label_pct={c.valid_label_pct:.1%} < "
                    f"{limits.low_valid_label_pct:.0%} — ложная уверенность"
                ),
            ))

    # Конфликт 4: сезонное смещение
    for c in candidates:
        limits = profile_of(c).validator
        if c.monthly_concentration > limits.high_monthly_concentration:
            conflicts.append(ConflictReport(
                conflict_type="seasonal_concentration",
                candidate_ids=[c.candidate_id],
                description=(
                    f"Кандидат {c.candidate_id}: monthly_concentration="
                    f"{c.monthly_concentration:.1%} > "
                    f"{limits.high_monthly_concentration:.0%} — возможный сезонный артефакт"
                ),
            ))

    return conflicts


def __getattr__(name: str):
    # DEPRECATED: надбавка за свежесть переехала в профиль (batch.fresh_bonus).
    # Имя оставлено для тестов и внешних скриптов, читается лениво.
    if name == "FRESH_BONUS":
        from src.config.profiles import default_profile
        return default_profile().batch.fresh_bonus
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
