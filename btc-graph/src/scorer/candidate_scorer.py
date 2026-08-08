"""
Scorer: вычисляет quality_score [0..1] по 4 осям согласно ТЗ.

Порогов в этом файле нет — они живут в профиле монеты
(`config/symbols/*.yaml`, загрузка через `src.config.profiles`). Код здесь
описывает только форму формулы: какие признаки входят в какую ось, как ось
усредняется и как оси складываются в итог.

Разделение важно потому, что ступени вида `sample_size > 1000` откалиброваны
под рынок с десятилетней историей. Для монеты с двумя годами торгов та же
лесенка означает «всё WEAK» — и это свойство линейки, а не кандидатов.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config.profiles import (
    DEFAULT_PROFILE_NAME,
    LadderSpec,
    ScoringProfile,
    default_profile,
    get_profile,
)
from src.models.candidate import (
    AgeBucket,
    Candidate,
    ResearchSide,
)


@dataclass
class ScoreBreakdown:
    statistical: float   # A. Статистическая надёжность
    directional: float   # B. Directional asymmetry
    context: float       # C. Контекстуальная свежесть
    rarity: float        # D. Редкость и специфичность
    total: float

    # Каким профилем посчитано. Без этих двух полей сохранённые quality_score
    # разных калибровок неотличимы друг от друга и молча смешиваются в средних.
    profile: str = ""
    fingerprint: str = ""

    # Тот же кандидат, посчитанный профилем `_default`. Профильный total
    # ранжирует кандидатов ВНУТРИ монеты и между монетами не сравним (у ETH
    # ступень sample_size вдвое ниже); baseline существует ровно для вопроса
    # «какая монета сегодня интереснее». В rating не участвует.
    baseline_total: float = 0.0


def _ladder(value: float, spec: LadderSpec) -> float:
    """Балл по ступенчатой шкале профиля."""
    return spec.apply(value)


def _map(key: str, spec: dict[str, float]) -> float:
    """Балл по карте «значение enum → балл». Полнота карты проверена при загрузке."""
    return spec[key]


def _resolve(candidate: Candidate, profile: ScoringProfile | None) -> ScoringProfile:
    return profile if profile is not None else get_profile(candidate.symbol)


def _score_statistical(c: Candidate, profile: ScoringProfile | None = None) -> float:
    """A. Статистическая надёжность выборки."""
    spec = _resolve(c, profile).statistical
    score = (
        _ladder(c.valid_label_pct, spec.valid_label_pct)
        + _ladder(c.sample_size, spec.sample_size)
        + _ladder(c.monthly_concentration, spec.monthly_concentration)
        + _ladder(c.repeatability_months, spec.repeatability_months)
    )
    return score / 4.0


def win_rate_for(c: Candidate) -> float:
    """
    Доля исходов в сторону research_side.

    Единственное место, где long_outcome_share переводится в win rate —
    используется и скорером, и сборкой CandidateEvaluation.

    В профиль не выносится: это свойство данных, а не калибровки.
    """
    if c.research_side == ResearchSide.long:
        return c.long_outcome_share
    return 1.0 - c.long_outcome_share


def fa_ratio_for(c: Candidate) -> float | None:
    """
    Favorable/adverse ratio, применимый к research_side кандидата.

    p70_long_favorable_pct / p80_long_adverse_pct описывают движение ВВЕРХ:
    первое — потенциал long, второе — риск long. Для short-кандидата выгода и
    риск меняются местами, а симметричных short-полей источник данных не даёт.
    Поэтому для short метрика неприменима и возвращается None — вместо того
    чтобы начислять баллы за чужую сторону рынка.

    В профиль не выносится по той же причине, что и win_rate_for.
    """
    if c.research_side == ResearchSide.long:
        return c.long_favorable_adverse_ratio_p70_p80
    return None


def _score_directional(c: Candidate, profile: ScoringProfile | None = None) -> float:
    """
    B. Сила directional asymmetry.

    Для long усредняется по трём критериям, для short — по двум:
    F/A ratio к short-кандидату неприменим (см. fa_ratio_for).
    """
    spec = _resolve(c, profile).directional

    score = (
        _ladder(win_rate_for(c), spec.win_rate)
        + _ladder(abs(c.historical_outcome_skew), spec.abs_outcome_skew)
    )
    criteria = 2

    ratio = fa_ratio_for(c)
    if ratio is not None:
        criteria = 3
        score += _ladder(ratio, spec.fa_ratio)

    return score / criteria


def _score_context(c: Candidate, profile: ScoringProfile | None = None) -> float:
    """
    C. Контекстуальная свежесть.

    Чем дольше рынок сидит в текущем состоянии, тем ближе смена фазы —
    оценка age_bucket убывает по мере старения состояния.
    """
    spec = _resolve(c, profile).context
    score = (
        _map(c.context_status.value, spec.context_status)
        + _map(c.current_group_age_bucket.value, spec.age_bucket)
        + _map(c.trajectory_entropy.value, spec.trajectory_entropy)
    )
    return score / 3.0


def _score_rarity(c: Candidate, profile: ScoringProfile | None = None) -> float:
    """D. Редкость и специфичность."""
    spec = _resolve(c, profile).rarity
    score = (
        _map(c.event_rarity_bucket.value, spec.event_rarity_bucket)
        + _map(c.transition_rarity.value, spec.transition_rarity)
        + _ladder(c.research_score, spec.research_score)
    )
    return score / 3.0


def _weighted_total(
    stat: float, direc: float, ctx: float, rar: float, profile: ScoringProfile
) -> float:
    w = profile.weights
    return (
        stat * w.statistical
        + direc * w.directional
        + ctx * w.context
        + rar * w.rarity
    )


def score_candidate(
    candidate: Candidate,
    profile: ScoringProfile | None = None,
) -> ScoreBreakdown:
    """
    Вычисляет quality_score и разбивку по осям.

    profile=None — берётся профиль монеты кандидата. Явный профиль нужен
    pipeline'у: он резолвит его один раз в начале, чтобы hot-reload посреди
    батча не смешал в одной оценке две версии калибровки.
    """
    profile = _resolve(candidate, profile)

    stat = _score_statistical(candidate, profile)
    direc = _score_directional(candidate, profile)
    ctx = _score_context(candidate, profile)
    rar = _score_rarity(candidate, profile)
    total = _weighted_total(stat, direc, ctx, rar, profile)

    base = default_profile()
    if profile.symbol == base.symbol and profile.version == base.version:
        # Профиль монеты — это и есть базовый: второй проход ничего не изменит.
        baseline = total
    else:
        baseline = _weighted_total(
            _score_statistical(candidate, base),
            _score_directional(candidate, base),
            _score_context(candidate, base),
            _score_rarity(candidate, base),
            base,
        )

    return ScoreBreakdown(
        statistical=round(stat, 4),
        directional=round(direc, 4),
        context=round(ctx, 4),
        rarity=round(rar, 4),
        total=round(total, 4),
        profile=profile.name,
        fingerprint=profile.fingerprint,
        baseline_total=round(baseline, 4),
    )


def get_rating(quality_score: float, profile: ScoringProfile | None = None) -> str:
    """
    STRONG / MODERATE / WEAK по quality_score. Единственный источник порогов.

    profile=None → границы базового профиля. Никогда не сравнивай quality_score
    с числом напрямую: границы у монет разные.
    """
    bounds = (profile or default_profile()).rating
    if quality_score >= bounds.strong_min:
        return "STRONG"
    if quality_score >= bounds.moderate_min:
        return "MODERATE"
    return "WEAK"


# ─── Совместимость ────────────────────────────────────────────────────────────
# Константы, которые до шага 4 были определениями порогов, а теперь — только
# read-only проекция базового профиля. Читаются лениво (профили грузятся при
# первом обращении, а не на импорте), поэтому через модульный __getattr__.
#
# DEPRECATED: новый код обязан брать пороги из профиля кандидата. Эти имена
# живут ради тестов и внешних скриптов, которые их импортируют.
_DEPRECATED_ALIASES = {
    "RATING_STRONG_MIN": lambda p: p.rating.strong_min,
    "RATING_MODERATE_MIN": lambda p: p.rating.moderate_min,
    "WEIGHTS": lambda p: p.weights.as_dict(),
    # Ключи — AgeBucket, как было до шага 4: алиас обязан быть заменяемым
    # один в один, иначе он не алиас, а вторая, слегка другая правда.
    "_AGE_SCORE": lambda p: {
        AgeBucket(key): value for key, value in p.context.age_bucket.items()
    },
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_ALIASES:
        return _DEPRECATED_ALIASES[name](default_profile())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_PROFILE_NAME",
    "ScoreBreakdown",
    "fa_ratio_for",
    "get_rating",
    "score_candidate",
    "win_rate_for",
]
