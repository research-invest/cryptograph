"""
Публичное API чтения: отдать наружу уже посчитанные оценки кандидатов.

Отделено от `routes.py` намеренно и по трём границам сразу:

* **доступ.** Весь роутер закрыт ключом (`src/api/auth.py`), тогда как
  внутренние ручки приёма и оценки остаются открытыми внутри петли. Смешать
  их в одном файле значит рано или поздно завести открытую ручку рядом с
  закрытыми и не заметить;
* **что происходит по запросу.** Здесь только SELECT. Ни LLM (то есть ни
  одного потраченного токена Anthropic), ни записи, ни перезагрузки
  конфигурации: ключ, утёкший наружу, стоит трафика, а не денег и не данных;
* **совместимость.** Префикс `/api/v1` — внешний контракт. `/evaluate/*`
  и `/stats/*` таким контрактом не были и менялись свободно; здесь состав
  полей менять можно только добавлением.

Числа не пересчитываются: отдаётся ровно то, что записал скорер в момент
оценки, вместе с меткой калибровки (`scoring_profile` + `profile_fingerprint`).
Пересчитать «на лету» по текущему профилю было бы удобнее для витрины и
неверно по сути — клиент получил бы число, которого в истории не было.

Что клиент обязан знать про выдачу, поэтому это написано и в ответе:

* `quality_score` сравним **только внутри одной монеты и одной метки
  калибровки**. Между монетами сравнивают `quality_score_baseline`;
* кандидат — исследовательская идея, а не торговый сигнал: ни точки входа,
  ни стопа, ни размера позиции в нём нет и не предполагается.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.auth import require_api_key
from src.config import profiles as profiles_config
from src.models.candidate import Candidate
from src.scorer.candidate_scorer import fa_ratio_for, win_rate_for

# Верхняя граница страницы. Не про производительность SELECT'а, а про то,
# что каждая строка тянет за собой разбор raw_payload в модель.
MAX_LIMIT = 200

RATINGS = {"STRONG", "MODERATE", "WEAK"}
DIRECTIONS = {"long", "short"}

router = APIRouter(
    prefix="/api/v1",
    tags=["public"],
    dependencies=[Depends(require_api_key)],
    responses={
        401: {"description": "Ключ не передан или неизвестен"},
        503: {"description": "API_KEYS не задан — публичное API не настроено"},
    },
)


# ─── Схемы ответа ─────────────────────────────────────────────────────────────

class RangeBlock(BaseModel):
    """
    Прогноз размаха. Есть не у всех кандидатов — и это не аномалия.

    Полей нет у всего, что выпущено до 2026-08-19, у кандидатов из `train`
    (там модель обучалась на этих же барах) и у монет, чья модель не прошла
    гейт калибровки на стороне генератора. `null` означает «система про размах
    здесь ничего не говорит», а не «размах обычный».

    Смотреть надо на `lift` и `regime`: квантили наполовину объясняются часом
    дня и недавней волатильностью, то есть показывают время суток не меньше,
    чем рынок.
    """
    lift: Optional[float] = Field(None, description="Во сколько раз шире тривиального прогноза")
    regime: Optional[str] = Field(None, description="wide / normal / narrow")
    expected_ratio_p50: Optional[float] = None
    expected_ratio_p90: Optional[float] = None


class CandidateItem(BaseModel):
    candidate_id: str
    symbol: str
    evaluated_at: Optional[datetime] = None

    quality_score: Optional[float] = Field(None, description="Сравним только внутри монеты и метки калибровки")
    quality_score_baseline: Optional[float] = Field(None, description="Тот же кандидат по базовой линейке — единственное межмонетно сравнимое число")
    rating: Optional[str] = None
    direction: Optional[str] = None
    scoring_profile: Optional[str] = None
    profile_fingerprint: Optional[str] = None

    win_rate: Optional[float] = Field(None, description="Доля исходов в сторону direction")
    favorable_adverse_ratio: Optional[float] = Field(None, description="p70/p80; для short — null, симметричных полей в данных нет")

    horizon: Optional[str] = None
    sample_size: Optional[int] = None
    valid_label_pct: Optional[float] = None
    repeatability_months: Optional[int] = None
    monthly_concentration: Optional[float] = None
    long_outcome_share: Optional[float] = None
    outcome_skew: Optional[float] = None

    context_status: Optional[str] = None
    transition_id: Optional[str] = None
    transition_rarity: Optional[str] = None
    current_group_id: Optional[float] = None
    previous_group_id: Optional[float] = None
    event_block_id: Optional[str] = None
    family_key: Optional[str] = None
    configuration_hash: Optional[str] = None

    warning_flags: list[str] = []
    range: RangeBlock = RangeBlock()


class CandidateDetail(CandidateItem):
    """Тот же кандидат плюс исходное досье генератора как есть."""
    raw_payload: Optional[dict[str, Any]] = None


class CandidatePage(BaseModel):
    symbol: str
    total: int = Field(description="Сколько строк подходит под фильтр целиком")
    limit: int
    offset: int
    count: int
    order: str
    items: list[CandidateItem]
    notes: list[str]


# ─── Сборка ответа ────────────────────────────────────────────────────────────

# Пишется в каждую страницу выдачи: клиент, который читает только JSON и
# никогда не откроет документацию, — обычный случай, а не исключение.
_NOTES = [
    "quality_score сравним только внутри одной монеты и одной пары "
    "(scoring_profile, profile_fingerprint); между монетами сравнивайте "
    "quality_score_baseline.",
    "Кандидат — исследовательская идея, а не торговый сигнал: точки входа, "
    "стопа и размера позиции в нём нет.",
]


def _as_candidate(raw: dict[str, Any] | None) -> Candidate | None:
    """
    Досье генератора из `raw_payload` обратно в модель.

    Нужно ровно для двух величин — win rate и F/A ratio, — которые нигде не
    хранятся колонками и обязаны считаться единственными функциями перевода
    (`win_rate_for` / `fa_ratio_for`), а не повторной формулой на месте.
    Строки, записанные схемой прошлых версий, просто не дадут этих двух полей.
    """
    if not raw:
        return None
    try:
        return Candidate(**raw)
    except Exception:
        return None


def _range_block(raw: dict[str, Any] | None) -> RangeBlock:
    payload = raw or {}
    return RangeBlock(
        lift=payload.get("range_lift"),
        regime=payload.get("range_regime"),
        expected_ratio_p50=payload.get("expected_range_ratio_p50"),
        expected_ratio_p90=payload.get("expected_range_ratio_p90"),
    )


def _item_fields(record) -> dict[str, Any]:
    raw = record.raw_payload or {}
    candidate = _as_candidate(raw)
    return {
        "candidate_id": record.candidate_id,
        "symbol": record.symbol,
        "evaluated_at": record.evaluated_at,
        "quality_score": record.quality_score,
        "quality_score_baseline": record.quality_score_baseline,
        "rating": record.rating,
        "direction": record.direction,
        "scoring_profile": record.scoring_profile,
        "profile_fingerprint": record.profile_fingerprint,
        "win_rate": win_rate_for(candidate) if candidate else None,
        "favorable_adverse_ratio": fa_ratio_for(candidate) if candidate else None,
        "horizon": record.horizon,
        "sample_size": record.sample_size,
        "valid_label_pct": record.valid_label_pct,
        "repeatability_months": record.repeatability_months,
        "monthly_concentration": record.monthly_concentration,
        "long_outcome_share": record.long_outcome_share,
        "outcome_skew": record.outcome_skew,
        "context_status": record.context_status,
        "transition_id": record.transition_id,
        "transition_rarity": record.transition_rarity,
        "current_group_id": record.current_group_id,
        "previous_group_id": record.previous_group_id,
        "event_block_id": record.event_block_id,
        "family_key": record.family_key,
        "configuration_hash": record.configuration_hash,
        "warning_flags": list(record.warning_flags or []),
        "range": _range_block(raw),
    }


def _to_item(record) -> CandidateItem:
    return CandidateItem(**_item_fields(record))


# ─── Ручки ────────────────────────────────────────────────────────────────────

@router.get("/ping")
def ping(key_label: str = Depends(require_api_key)):
    """
    Проверка ключа. В базу не ходит — отвечает и при лежащем PostgreSQL,
    поэтому годится как первый шаг отладки интеграции.
    """
    return {"status": "ok", "key": key_label}


@router.get("/symbols")
def list_symbols():
    """
    Монеты: у каких есть сохранённые оценки и какой калибровкой они считаны.

    `has_profile=false` означает, что монету считали базовой линейкой —
    её кандидаты помечены флагом `unknown_symbol_profile`.
    """
    from src.db import candidate_repo
    from src.db.connection import get_session

    try:
        with get_session() as session:
            stored = candidate_repo.list_symbols(session)
    except Exception as exc:  # база — единственная зависимость этой ручки
        raise HTTPException(status_code=500, detail=f"Ошибка чтения списка монет: {exc}")

    # Метка калибровки берётся у эффективного профиля, а не из реестра: в
    # реестре лежит `_default`, и монета без своего YAML получила бы «профиль
    # есть» по совпадению ключа. `is_known_symbol` — единственный канонический
    # ответ на вопрос «эта монета откалибрована».
    result = []
    for symbol in stored:
        profile = profiles_config.get_profile(symbol)
        result.append({
            "symbol": symbol,
            "has_profile": profiles_config.is_known_symbol(symbol),
            "scoring_profile": profile.name,
        })
    return {"symbols": result}


@router.get("/candidates", response_model=CandidatePage)
def list_candidates(
    symbol: str = Query(
        ...,
        description="Тикер монеты, например BTCUSDT. `all` — смешанная выдача по всем монетам",
        examples=["BTCUSDT"],
    ),
    rating: Optional[str] = Query(
        None, description="STRONG / MODERATE / WEAK, можно через запятую"
    ),
    direction: Optional[str] = Query(None, description="long или short"),
    min_score: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Нижняя граница quality_score"
    ),
    hours: Optional[int] = Query(
        None, ge=1, description="Только оценки за последние N часов"
    ),
    order: str = Query("recent", description="recent — по времени, score — по качеству"),
    limit: int = Query(50, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
):
    """
    Кандидаты монеты с их оценками — основная ручка API.

    `symbol` обязателен и `all` пишется явно: выдача без монеты перемешивает
    рынки, а профильные `quality_score` между монетами не сравнимы. При
    `symbol=all` сортировка `order=score` идёт по `quality_score_baseline` —
    иначе «топ» означал бы «монета с самой щедрой калибровкой».
    """
    from src.db import candidate_repo
    from src.db.connection import get_session

    symbol_value = (symbol or "").strip()
    symbol_filter = None if symbol_value.lower() == "all" else symbol_value.upper()

    ratings = None
    if rating:
        ratings = [r.strip().upper() for r in rating.split(",") if r.strip()]
        unknown = [r for r in ratings if r not in RATINGS]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Неизвестный рейтинг: {', '.join(unknown)}. Допустимо: STRONG, MODERATE, WEAK",
            )

    direction_filter = None
    if direction:
        direction_filter = direction.strip().lower()
        if direction_filter not in DIRECTIONS:
            raise HTTPException(
                status_code=422, detail="direction: long или short"
            )

    if order not in {"recent", "score"}:
        raise HTTPException(status_code=422, detail="order: recent или score")

    since = None
    if hours is not None:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)

    try:
        with get_session() as session:
            rows, total = candidate_repo.list_candidates(
                session,
                symbol=symbol_filter,
                ratings=ratings,
                direction=direction_filter,
                min_quality_score=min_score,
                since=since,
                order=order,
                limit=limit,
                offset=offset,
            )
            items = [_to_item(row) for row in rows]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка выборки кандидатов: {exc}")

    notes = list(_NOTES)
    if symbol_filter is None:
        notes.insert(
            0,
            "Смешанная выдача по всем монетам: ориентируйтесь на "
            "quality_score_baseline, профильные quality_score между монетами "
            "не сравнимы.",
        )

    return CandidatePage(
        symbol=symbol_filter or "all",
        total=total,
        limit=limit,
        offset=offset,
        count=len(items),
        order=order,
        items=items,
        notes=notes,
    )


@router.get("/candidates/{symbol}/{candidate_id}", response_model=CandidateDetail)
def get_candidate(
    symbol: str,
    candidate_id: str,
    include_raw: bool = Query(
        False, description="Приложить исходное досье генератора (raw_payload)"
    ),
):
    """Один кандидат по паре (монета, идентификатор)."""
    from src.db import candidate_repo
    from src.db.connection import get_session

    try:
        with get_session() as session:
            record = candidate_repo.get_by_id(session, symbol.upper(), candidate_id)
            if record is None:
                raise HTTPException(status_code=404, detail="Кандидат не найден")
            fields = _item_fields(record)
            fields["raw_payload"] = record.raw_payload if include_raw else None
            return CandidateDetail(**fields)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Ошибка чтения кандидата {candidate_id}: {exc}"
        )
