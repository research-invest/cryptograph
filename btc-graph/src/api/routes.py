"""
FastAPI routes: приём кандидатов, оценка, фильтрация, граф, поиск похожих.

Мультимонетность: везде, где выдача или запрос привязаны к графу монеты
(`group_id`, `transition_id`), параметр `symbol` обязателен — эти
идентификаторы осмысленны только внутри одного инструмента. Там, где символ
можно взять из самого кандидата, он не спрашивается.
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ValidationError

from src.agent.pipeline import run_pipeline, run_batch_pipeline
from src.api.public import router as public_router
from src.config import profiles as profiles_config
from src.filters.candidate_filter import detect_conflicts
from src.models.candidate import CandidateEvaluation
from src.parser.candidate_parser import parse_candidate, parse_candidates
from src.validator.candidate_validator import validate_candidate
from src.scorer.candidate_scorer import get_rating, score_candidate

logger = logging.getLogger(__name__)

# Перезагрузка профилей меняет поведение скоринга на лету. Само по себе это
# read-only-безопасно, но API без аутентификации — нет, поэтому по умолчанию
# ручка выключена.
ENABLE_CONFIG_RELOAD = os.environ.get("ENABLE_CONFIG_RELOAD", "false").lower() == "true"

# Явное значение «все монеты» там, где symbol обязателен: чтобы смешанная
# выдача была осознанным выбором, а не следствием забытого параметра.
ALL_SYMBOLS = "all"

app = FastAPI(
    title="Crypto Market Candidate Agent",
    description=(
        "Агент анализа рыночных кандидатов — детерминирование фазы рынка. "
        "Мультимонетный: калибровка оценки задаётся профилем на символ."
    ),
    version="2.1.0",
)


def _fail(exc: Exception, context: str) -> HTTPException:
    """
    Переводит исключение в корректный HTTP-код.

    422 — только когда виноват вход (не прошла валидация Pydantic или парсер
    отверг данные). Всё остальное — 500: сбой БД, недоступность внешнего API,
    баг в коде. Раньше сюда падало всё подряд с кодом 422, и отладка по коду
    ответа была невозможна — см. docs/audit_findings.md, №12.
    """
    if isinstance(exc, (ValidationError, ValueError, TypeError)):
        return HTTPException(status_code=422, detail=str(exc))
    logger.exception("%s", context)
    return HTTPException(status_code=500, detail=f"{context}: {exc}")


def _symbol_filter(symbol: str) -> str | None:
    """`all` → None (без фильтра), иначе нормализованный тикер."""
    value = (symbol or "").strip()
    return None if value.lower() == ALL_SYMBOLS else value.upper()


# ─── Input schemas ────────────────────────────────────────────────────────────

class RawTextInput(BaseModel):
    raw: str
    use_llm: bool = True
    save: bool = True
    # false — пересчитать даже если свежая оценка этого кандидата есть в Redis
    use_cache: bool = True


class JsonCandidateInput(BaseModel):
    candidate: dict
    use_llm: bool = True
    save: bool = True
    use_cache: bool = True


class BatchInput(BaseModel):
    candidates: list[dict]
    use_llm: bool = True
    save: bool = True
    # None — порог берётся из профиля каждой монеты. Число применяется ко
    # всему батчу и перекрывает профили.
    min_quality_score: float | None = None


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}


# Публичное API чтения (`/api/v1/*`) — единственная часть, закрытая ключом из
# API_KEYS. Оно только читает сохранённые оценки: ни LLM, ни записи, ни правки
# конфигурации там нет, поэтому утёкший ключ стоит трафика, а не денег и не
# данных. Всё остальное в этом файле по-прежнему рассчитано на петлю и прокси
# перед ней — см. «Известные ограничения» в CLAUDE.md.
app.include_router(public_router)


# ─── Evaluate ─────────────────────────────────────────────────────────────────

@app.post("/evaluate/raw", response_model=CandidateEvaluation)
def evaluate_raw(body: RawTextInput):
    """Оценить кандидата из raw text. Символ берётся из самого кандидата."""
    try:
        return run_pipeline(
            body.raw, use_llm=body.use_llm, save=body.save, use_cache=body.use_cache
        )
    except Exception as e:
        raise _fail(e, "Ошибка оценки кандидата из raw text")


@app.post("/evaluate/json", response_model=CandidateEvaluation)
def evaluate_json(body: JsonCandidateInput):
    """Оценить кандидата из JSON объекта."""
    try:
        return run_pipeline(
            body.candidate, use_llm=body.use_llm, save=body.save, use_cache=body.use_cache
        )
    except Exception as e:
        raise _fail(e, "Ошибка оценки кандидата из JSON")


@app.post("/evaluate/batch", response_model=list[CandidateEvaluation])
def evaluate_batch(body: BatchInput):
    """
    Оценить список кандидатов с фильтрацией и дедупликацией по family_key.

    Батч может содержать несколько монет: и фильтр, и дедупликация работают
    в пределах монеты.
    """
    try:
        return run_batch_pipeline(
            body.candidates,
            use_llm=body.use_llm,
            min_quality_score=body.min_quality_score,
            save=body.save,
        )
    except Exception as e:
        raise _fail(e, "Ошибка пакетной оценки кандидатов")


@app.post("/score/quick")
def quick_score(body: JsonCandidateInput):
    """Быстрый score без LLM и без сохранения."""
    try:
        candidate = parse_candidate(body.candidate)
        profile = profiles_config.get_profile(candidate.symbol)
        flags = validate_candidate(candidate, profile)
        breakdown = score_candidate(candidate, profile)
        return {
            "candidate_id": candidate.candidate_id,
            "symbol": candidate.symbol,
            "scoring_profile": breakdown.profile,
            "profile_fingerprint": breakdown.fingerprint,
            "quality_score": breakdown.total,
            # Профильный quality_score сравним только внутри монеты.
            "quality_score_baseline": breakdown.baseline_total,
            "rating": get_rating(breakdown.total, profile),
            "score_breakdown": {
                "statistical": breakdown.statistical,
                "directional": breakdown.directional,
                "context": breakdown.context,
                "rarity": breakdown.rarity,
            },
            "warning_flags": flags,
        }
    except Exception as e:
        raise _fail(e, "Ошибка быстрой оценки кандидата")


# ─── Conflicts ────────────────────────────────────────────────────────────────

@app.post("/conflicts")
def detect(body: BatchInput):
    """Найти конфликты среди списка кандидатов (в пределах каждой монеты)."""
    try:
        candidates = parse_candidates(body.candidates)
        conflicts = detect_conflicts(candidates)
        return {
            "total_candidates": len(candidates),
            "symbols": sorted({c.symbol for c in candidates}),
            "conflicts_found": len(conflicts),
            "conflicts": [
                {"type": c.conflict_type, "candidate_ids": c.candidate_ids, "description": c.description}
                for c in conflicts
            ],
        }
    except Exception as e:
        raise _fail(e, "Ошибка поиска конфликтов")


# ─── Database queries ─────────────────────────────────────────────────────────

@app.get("/candidates/strong/{direction}")
def get_strong(
    direction: str,
    symbol: str = Query(
        ...,
        description="Тикер монеты либо 'all' для смешанной выдачи по всем монетам",
    ),
    limit: int = 20,
):
    """
    Последние STRONG-кандидаты по направлению (long/short).

    symbol обязателен: без него выдача перемешивает монеты, а профильные
    quality_score между монетами не сравнимы. Смешанная выдача доступна
    явным `symbol=all` — и тогда ориентироваться надо на quality_score_baseline.
    """
    try:
        from src.db.connection import get_session
        from src.db import candidate_repo
        with get_session() as session:
            records = candidate_repo.get_strong_candidates(
                session, symbol=_symbol_filter(symbol),
                direction=direction, limit=limit,
            )
            return [
                {
                    "candidate_id": r.candidate_id,
                    "symbol": r.symbol,
                    "quality_score": r.quality_score,
                    "quality_score_baseline": r.quality_score_baseline,
                    "scoring_profile": r.scoring_profile,
                    "transition_id": r.transition_id,
                    "context_status": r.context_status,
                    "evaluated_at": r.evaluated_at,
                }
                for r in records
            ]
    except HTTPException:
        raise
    except Exception as e:
        raise _fail(e, f"Ошибка выборки STRONG-кандидатов ({direction})")


@app.get("/candidates/{symbol}/{candidate_id}")
def get_candidate(symbol: str, candidate_id: str):
    """Получить сохранённую оценку из PostgreSQL."""
    try:
        from src.db.connection import get_session
        from src.db import candidate_repo
        with get_session() as session:
            record = candidate_repo.get_by_id(session, symbol.upper(), candidate_id)
            if record is None:
                raise HTTPException(status_code=404, detail="Кандидат не найден")
            return {
                "candidate_id": record.candidate_id,
                "symbol": record.symbol,
                "quality_score": record.quality_score,
                "quality_score_baseline": record.quality_score_baseline,
                "scoring_profile": record.scoring_profile,
                "profile_fingerprint": record.profile_fingerprint,
                "rating": record.rating,
                "direction": record.direction,
                "warning_flags": record.warning_flags,
                "evaluated_at": record.evaluated_at,
            }
    except HTTPException:
        raise
    except Exception as e:
        raise _fail(e, f"Ошибка чтения кандидата {candidate_id}")


@app.post("/candidates/similar")
def find_similar(
    body: JsonCandidateInput,
    limit: int = 10,
    cross_symbol: bool = Query(
        False,
        description="Искать по всем монетам, а не только по монете кандидата",
    ),
):
    """
    Найти исторически похожие кандидаты через pgvector.

    Вектор symbol-agnostic намеренно, поэтому `cross_symbol=true` — рабочий
    сценарий: «а у BTC такая конфигурация уже была?». По умолчанию поиск
    ограничен монетой кандидата.
    """
    try:
        from src.db.connection import get_session
        from src.db import candidate_repo
        from src.db.embedding import build_embedding
        candidate = parse_candidate(body.candidate)
        embedding = build_embedding(candidate)
        with get_session() as session:
            records = candidate_repo.find_similar(
                session, embedding,
                symbol=None if cross_symbol else candidate.symbol,
                limit=limit,
            )
            return [
                {
                    "candidate_id": r.candidate_id,
                    "symbol": r.symbol,
                    "quality_score": r.quality_score,
                    "quality_score_baseline": r.quality_score_baseline,
                    "rating": r.rating,
                    "direction": r.direction,
                    "transition_id": r.transition_id,
                }
                for r in records
            ]
    except Exception as e:
        raise _fail(e, "Ошибка поиска похожих кандидатов")


# ─── Graph queries ─────────────────────────────────────────────────────────────

@app.get("/graph/group/{group_id}")
def get_group(group_id: float, symbol: str = Query(..., description="Тикер монеты")):
    """
    Получить информацию об узле графа MarketGroup.

    symbol обязателен: «группа 7» существует в графе каждой монеты и означает
    в каждом свой рынок.
    """
    try:
        from src.db import graph_repo
        info = graph_repo.get_group_info(symbol.upper(), group_id)
        if info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Группа {group_id} не найдена в графе {symbol.upper()}",
            )
        return info
    except HTTPException:
        raise
    except Exception as e:
        raise _fail(e, "Ошибка чтения узла графа")


@app.get("/graph/transitions/to/{group_id}")
def get_transitions_to(
    group_id: float,
    symbol: str = Query(..., description="Тикер монеты"),
    rarity: str = "uncommon,rare",
):
    """
    Найти переходы монеты, ведущие в группу group_id.
    rarity — через запятую: uncommon,rare или common
    """
    try:
        from src.db import graph_repo
        rarity_filter = [r.strip() for r in rarity.split(",")]
        transitions = graph_repo.find_transitions_to_group(
            symbol.upper(), group_id, rarity_filter
        )
        return {"symbol": symbol.upper(), "group_id": group_id, "transitions": transitions}
    except Exception as e:
        raise _fail(e, "Ошибка чтения переходов графа")


@app.get("/graph/symbols")
def graph_symbols():
    """Монеты, представленные в графе — проверка изоляции после миграции."""
    try:
        from src.db import graph_repo
        return {"symbols": graph_repo.list_symbols()}
    except Exception as e:
        raise _fail(e, "Ошибка чтения списка монет графа")


# ─── Stats (TimescaleDB continuous aggregates) ────────────────────────────────

@app.get("/stats/hourly")
def stats_hourly(hours: int = 24, symbol: str | None = None):
    """Почасовая статистика кандидатов из continuous aggregate."""
    try:
        from src.db.connection import get_session
        from src.db import stats_repo
        with get_session() as session:
            return {
                "symbol": symbol or "all",
                "rows": stats_repo.get_hourly_stats(session, hours=hours, symbol=symbol),
            }
    except Exception as e:
        raise _fail(e, "Ошибка чтения почасовой статистики")


@app.get("/stats/groups")
def stats_groups(
    days: int = 7, symbol: str | None = None, group_id: float | None = None
):
    """
    Дневная статистика по группам состояний.

    Фильтр по group_id требует symbol: номера групп у монет свои.
    """
    try:
        from src.db.connection import get_session
        from src.db import stats_repo
        with get_session() as session:
            return {
                "symbol": symbol or "all",
                "rows": stats_repo.get_daily_group_stats(
                    session, days=days, symbol=symbol, group_id=group_id
                ),
            }
    except Exception as e:
        raise _fail(e, "Ошибка чтения статистики по группам")


@app.get("/stats/ratings")
def stats_ratings(days: int = 7, symbol: str | None = None):
    """Распределение рейтингов STRONG / MODERATE / WEAK."""
    try:
        from src.db.connection import get_session
        from src.db import stats_repo
        with get_session() as session:
            return stats_repo.get_rating_distribution(session, days=days, symbol=symbol)
    except Exception as e:
        raise _fail(e, "Ошибка чтения распределения рейтингов")


@app.get("/stats/events")
def stats_events(limit: int = 50, symbol: str | None = None):
    """Последние события оценки из hypertable candidate_events."""
    try:
        from src.db.connection import get_session
        from src.db import stats_repo
        with get_session() as session:
            return {
                "symbol": symbol or "all",
                "rows": stats_repo.get_recent_events(session, limit=limit, symbol=symbol),
            }
    except Exception as e:
        raise _fail(e, "Ошибка чтения последних событий")


@app.get("/stats/symbols")
def stats_symbols(days: int = 30):
    """Монеты, по которым были оценки за период."""
    try:
        from src.db.connection import get_session
        from src.db import stats_repo
        with get_session() as session:
            return {"symbols": stats_repo.list_symbols(session, days=days)}
    except Exception as e:
        raise _fail(e, "Ошибка чтения списка монет")


# ─── Профили калибровки ───────────────────────────────────────────────────────

@app.get("/config/symbols")
def config_symbols():
    """
    Реестр профилей: какие монеты система знает и какой калибровкой считает.

    Реестр профилей и есть whitelist монет — отдельного списка нет. Монета,
    которой здесь нет, оценивается базовой калибровкой с флагом
    `unknown_symbol_profile`.
    """
    try:
        return {
            "profiles": profiles_config.list_profiles(),
            "errors": profiles_config.load_errors(),
            "directory": str(profiles_config.profiles_dir()),
        }
    except Exception as e:
        raise _fail(e, "Ошибка чтения реестра профилей")


@app.get("/config/symbols/{symbol}")
def config_symbol(symbol: str):
    """
    Эффективный профиль монеты после слияния с `_default`.

    Именно это нужно при отладке калибровки: в YAML лежат только отличия,
    а считается по результату слияния.
    """
    try:
        profile = profiles_config.get_profile(symbol)
        return {
            "requested": symbol.upper(),
            "known": profiles_config.is_known_symbol(symbol),
            "profile": profile.name,
            "fingerprint": profile.fingerprint,
            "effective": profile.model_dump(mode="json"),
        }
    except Exception as e:
        raise _fail(e, f"Ошибка чтения профиля {symbol}")


@app.post("/config/reload")
def config_reload():
    """
    Перечитать профили без рестарта.

    Выключено по умолчанию: ручка меняет поведение скоринга, а API работает
    без аутентификации. Включается ENABLE_CONFIG_RELOAD=true.

    На идущий батч не влияет: pipeline резолвит профили один раз в начале.
    """
    if not ENABLE_CONFIG_RELOAD:
        raise HTTPException(
            status_code=403,
            detail="Перезагрузка профилей выключена (ENABLE_CONFIG_RELOAD=false)",
        )
    try:
        return profiles_config.reload_profiles()
    except HTTPException:
        raise
    except Exception as e:
        raise _fail(e, "Ошибка перезагрузки профилей")


# ─── Stream queue ─────────────────────────────────────────────────────────────

@app.post("/queue/enqueue")
def enqueue(body: JsonCandidateInput, symbol: str | None = None):
    """
    Положить кандидата в Redis Stream для асинхронной обработки.

    Кандидат разбирается сразу: невалидные данные отвергаются здесь (422), а не
    оседают в очереди, где ошибка всплывёт только в логах воркера.

    `?symbol=` проставляет монету payload'у, у которого её нет. Нужно только
    источникам, не умеющим отдавать поле; штатный генератор его всегда шлёт.
    """
    try:
        from src.cache import redis_cache

        payload = dict(body.candidate)
        if symbol and not payload.get("symbol"):
            payload["symbol"] = symbol.upper()

        parsed = parse_candidate(payload)
        msg_id = redis_cache.enqueue_candidate(payload)
        if not msg_id:
            # enqueue_candidate возвращает "" при недоступном Redis — раньше
            # роут всё равно отвечал queued: true (docs/audit_findings.md, №13).
            raise HTTPException(
                status_code=503, detail="Redis недоступен — кандидат не поставлен в очередь"
            )
        return {"queued": True, "msg_id": msg_id, "symbol": parsed.symbol}
    except HTTPException:
        raise
    except Exception as e:
        raise _fail(e, "Ошибка постановки кандидата в очередь")
