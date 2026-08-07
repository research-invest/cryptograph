"""
Задачи 4 и 5: фильтрация, сравнение и детектирование конфликтов.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.models.candidate import Candidate, ContextStatus, AgeBucket
from src.scorer.candidate_scorer import score_candidate


@dataclass
class ConflictReport:
    conflict_type: str
    candidate_ids: list[str]
    description: str


def filter_candidates(
    candidates: list[Candidate],
    min_quality_score: float = 0.60,
) -> list[tuple[Candidate, float]]:
    """
    Задача 4: фильтрует по quality_score и ранжирует.
    Возвращает список (candidate, quality_score) выше порога, отсортированный по убыванию.
    """
    scored = []
    for c in candidates:
        breakdown = score_candidate(c)
        if breakdown.total >= min_quality_score:
            scored.append((c, breakdown.total))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# Насколько свежесть контекста перевешивает разницу в quality_score при выборе
# лучшего кандидата внутри семьи. Раньше fresh был абсолютным приоритетом, и
# кандидат с 0.61 вытеснял кандидата с 0.95 — см. docs/audit_findings.md, №9.
FRESH_BONUS = 0.05


def _family_rank(candidate: Candidate, quality_score: float) -> float:
    """Ключ сортировки внутри семьи: score с небольшой надбавкой за fresh."""
    bonus = FRESH_BONUS if candidate.context_status == ContextStatus.fresh else 0.0
    return quality_score + bonus


def select_best_per_family(
    scored: list[tuple[Candidate, float]],
) -> list[tuple[Candidate, float]]:
    """
    Задача 4: внутри каждого candidate_family_key выбирает лучшего кандидата.

    Побеждает наибольший quality_score; свежесть добавляет FRESH_BONUS, то есть
    при сопоставимых оценках выигрывает fresh, но заметно более сильный stale
    не проигрывает слабому fresh.
    """
    groups: dict[str, list[tuple[Candidate, float]]] = defaultdict(list)
    for candidate, qs in scored:
        key = candidate.candidate_family_key or candidate.candidate_id
        groups[key].append((candidate, qs))

    result = []
    for key, group in groups.items():
        group.sort(key=lambda x: -_family_rank(x[0], x[1]))
        result.append(group[0])

    result.sort(key=lambda x: x[1], reverse=True)
    return result


def detect_conflicts(candidates: list[Candidate]) -> list[ConflictReport]:
    """
    Задача 5: обнаруживает противоречивые сигналы среди кандидатов.
    """
    conflicts: list[ConflictReport] = []

    # Конфликт 1: одинаковый transition_id, разный research_side
    by_transition: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        by_transition[c.transition_id].append(c)

    for tid, group in by_transition.items():
        sides = {c.research_side for c in group}
        if len(sides) > 1:
            conflicts.append(ConflictReport(
                conflict_type="contradictory_direction",
                candidate_ids=[c.candidate_id for c in group],
                description=f"Переход {tid}: кандидаты указывают в разные стороны {[s.value for s in sides]}",
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
        if c.research_score > 0.85 and c.valid_label_pct < 0.70:
            conflicts.append(ConflictReport(
                conflict_type="false_confidence",
                candidate_ids=[c.candidate_id],
                description=(
                    f"Кандидат {c.candidate_id}: research_score={c.research_score:.3f} высокий, "
                    f"но valid_label_pct={c.valid_label_pct:.1%} < 70% — ложная уверенность"
                ),
            ))

    # Конфликт 4: сезонное смещение (monthly_concentration > 0.30)
    for c in candidates:
        if c.monthly_concentration > 0.30:
            conflicts.append(ConflictReport(
                conflict_type="seasonal_concentration",
                candidate_ids=[c.candidate_id],
                description=(
                    f"Кандидат {c.candidate_id}: monthly_concentration={c.monthly_concentration:.1%} > 30% — "
                    f"возможный сезонный артефакт"
                ),
            ))

    return conflicts
