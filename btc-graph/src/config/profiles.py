"""
Профили оценки на символ: вся калибровка скорера, валидатора и фильтров.

Зачем это отдельный слой. Пороги вида `sample_size > 1000` или
`repeatability_months > 15` — не универсальные истины, а характеристики рынка
с десятилетней историей. Монета с двумя годами торгов никогда их не наберёт, и
все её кандидаты уедут в WEAK не потому, что они плохие, а потому что линейка
чужая. Профиль — способ дать каждой монете свою линейку, не размножая код.

Почему YAML в каталоге, а не Python-модуль и не таблица в БД:
  * правится без пересборки образа (каталог монтируется volume'ом);
  * лежит в git рядом с кодом — видно в диффе, кто и когда двигал пороги;
  * не требует живого Postgres, чтобы посчитать score.

Реестр профилей заодно работает whitelist'ом известных монет: отдельного
списка символов в проекте нет. Неизвестный символ получает `_default` и
warning-флаг, но оценку получает — валидатор по контракту не бросает исключений.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

DEFAULT_PROFILE_NAME = "_default"

# Каталог профилей. В контейнере это /app/config/symbols (см. docker-compose),
# локально — config/symbols в корне репозитория.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def profiles_dir() -> Path:
    """Каталог с YAML-профилями. Читается на каждый вызов — путь может меняться в тестах."""
    return Path(os.environ.get("PROFILES_DIR") or (_REPO_ROOT / "config" / "symbols"))


# ─── Схема профиля ────────────────────────────────────────────────────────────

class LadderSpec(BaseModel):
    """
    Ступенчатая шкала для числового признака.

    Список [порог, балл] проверяется сверху вниз, сравнение строгое:
    higher_better → `value > порог`, lower_better → `value < порог`.
    Ни одна ступень не сработала → floor. Это ровно поведение хардкода шага 1,
    включая граничные случаи (`valid_label_pct == 0.85` даёт 0.7, а не 1.0).
    """
    model_config = ConfigDict(frozen=True)

    mode: Literal["higher_better", "lower_better"]
    ladder: tuple[tuple[float, float], ...]
    floor: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check(self) -> "LadderSpec":
        if not self.ladder:
            raise ValueError("ladder не может быть пустым — используй floor")
        for threshold, score in self.ladder:
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"балл {score} вне [0, 1] (порог {threshold})")

        # Немонотонная лесенка означает недостижимую ступень: проверка идёт
        # сверху вниз, и порог, который слабее предыдущего, не сработает
        # никогда. Ровно этот дефект был в age_60_120 (audit #1), только там
        # он жил в коде и его нельзя было увидеть иначе как тестом.
        thresholds = [t for t, _ in self.ladder]
        ordered = (
            all(a > b for a, b in zip(thresholds, thresholds[1:]))
            if self.mode == "higher_better"
            else all(a < b for a, b in zip(thresholds, thresholds[1:]))
        )
        if not ordered:
            direction = "убывать" if self.mode == "higher_better" else "возрастать"
            raise ValueError(
                f"пороги ladder должны строго {direction} для mode={self.mode}, "
                f"получено {thresholds} — часть ступеней недостижима"
            )
        return self

    @property
    def thresholds(self) -> tuple[float, ...]:
        return tuple(t for t, _ in self.ladder)

    def apply(self, value: float) -> float:
        """Балл за значение признака."""
        if self.mode == "higher_better":
            for threshold, score in self.ladder:
                if value > threshold:
                    return score
        else:
            for threshold, score in self.ladder:
                if value < threshold:
                    return score
        return self.floor

    def step(self, index: int) -> float:
        """
        Порог ступени по индексу, с зажимом в границы.

        Нужен формулировкам в deterministic.py: «очень высокий research_score»
        — это верхняя ступень профиля, а не литерал 0.85. Зажим позволяет
        профилю иметь меньше ступеней, чем ожидает текст.
        """
        thresholds = self.thresholds
        if index < 0:
            index += len(thresholds)
        return thresholds[max(0, min(index, len(thresholds) - 1))]


class EnumMap(BaseModel):
    """Карта «значение enum → балл». Проверяется на полноту покрытия."""
    model_config = ConfigDict(frozen=True)

    values: dict[str, float]

    def apply(self, key: str) -> float:
        return self.values[key]


#: Признаки, значение которых по схеме кандидата лежит в [0, 1].
#: Для них ступень нужно проверять не только на монотонность, но и
#: на достижимость: сравнение строгое, поэтому порог 1.0 у higher_better
#: не сработает никогда, а порог 0.0 у lower_better — тем более.
#: Ровно такую мёртвую ступень предложил калибратор на реальной выгрузке ETH:
#: p90 доли валидных меток оказался равен 1.0.
_SHARE_FIELDS = frozenset({
    "valid_label_pct", "monthly_concentration",
    "win_rate", "abs_outcome_skew", "research_score",
})


def _validate_reachable(name: str, spec: "LadderSpec") -> None:
    if name not in _SHARE_FIELDS:
        return
    top = spec.ladder[0][0]
    if spec.mode == "higher_better" and top >= 1.0:
        raise ValueError(
            f"{name}: верхняя ступень {top} недостижима — признак не бывает "
            f"больше 1.0, а сравнение строгое. Опусти порог "
            f"(например, до перцентиля ниже 1.0)"
        )
    if spec.mode == "lower_better" and top <= 0.0:
        raise ValueError(
            f"{name}: верхняя ступень {top} недостижима — признак не бывает "
            f"меньше 0.0, а сравнение строгое"
        )


def _validate_enum_map(name: str, mapping: dict[str, float], expected: set[str]) -> None:
    missing = expected - set(mapping)
    if missing:
        raise ValueError(f"{name}: не покрыты значения {sorted(missing)}")
    unknown = set(mapping) - expected
    if unknown:
        raise ValueError(f"{name}: неизвестные ключи {sorted(unknown)}")
    for key, score in mapping.items():
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{name}.{key}: балл {score} вне [0, 1]")


class StatisticalSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    valid_label_pct: LadderSpec
    sample_size: LadderSpec
    monthly_concentration: LadderSpec
    repeatability_months: LadderSpec

    @model_validator(mode="after")
    def _check(self) -> "StatisticalSpec":
        _validate_reachable("valid_label_pct", self.valid_label_pct)
        _validate_reachable("monthly_concentration", self.monthly_concentration)
        return self


class DirectionalSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    win_rate: LadderSpec
    abs_outcome_skew: LadderSpec
    fa_ratio: LadderSpec

    @model_validator(mode="after")
    def _check(self) -> "DirectionalSpec":
        _validate_reachable("win_rate", self.win_rate)
        _validate_reachable("abs_outcome_skew", self.abs_outcome_skew)
        return self


class ContextSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    context_status: dict[str, float]
    age_bucket: dict[str, float]
    trajectory_entropy: dict[str, float]

    @model_validator(mode="after")
    def _check(self) -> "ContextSpec":
        from src.models.candidate import AgeBucket, ContextStatus, TrajectoryEntropy

        _validate_enum_map("context.context_status", self.context_status,
                           {e.value for e in ContextStatus})
        _validate_enum_map("context.age_bucket", self.age_bucket,
                           {e.value for e in AgeBucket})
        _validate_enum_map("context.trajectory_entropy", self.trajectory_entropy,
                           {e.value for e in TrajectoryEntropy})
        return self


class RaritySpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    event_rarity_bucket: dict[str, float]
    transition_rarity: dict[str, float]
    research_score: LadderSpec

    @model_validator(mode="after")
    def _check(self) -> "RaritySpec":
        from src.models.candidate import EventRarityBucket, TransitionRarity

        _validate_reachable("research_score", self.research_score)
        _validate_enum_map("rarity.event_rarity_bucket", self.event_rarity_bucket,
                           {e.value for e in EventRarityBucket})
        _validate_enum_map("rarity.transition_rarity", self.transition_rarity,
                           {e.value for e in TransitionRarity})
        return self


class WeightsSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    statistical: float = Field(ge=0.0, le=1.0)
    directional: float = Field(ge=0.0, le=1.0)
    context: float = Field(ge=0.0, le=1.0)
    rarity: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_sum(self) -> "WeightsSpec":
        total = self.statistical + self.directional + self.context + self.rarity
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"сумма weights = {total}, должна быть 1.0")
        return self

    def as_dict(self) -> dict[str, float]:
        return {
            "statistical": self.statistical,
            "directional": self.directional,
            "context": self.context,
            "rarity": self.rarity,
        }


class RatingSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    strong_min: float = Field(ge=0.0, le=1.0)
    moderate_min: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check(self) -> "RatingSpec":
        if self.strong_min <= self.moderate_min:
            raise ValueError(
                f"strong_min ({self.strong_min}) должен быть строго больше "
                f"moderate_min ({self.moderate_min}) — иначе рейтинг MODERATE недостижим"
            )
        return self


class BatchSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    min_quality_score: float = Field(ge=0.0, le=1.0)
    # Надбавка свежести при выборе лучшего в семье. Не абсолютный приоритет:
    # заметно более сильный stale не должен проигрывать слабому fresh (audit #9).
    fresh_bonus: float = Field(ge=0.0, le=1.0)


class ValidatorSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    low_valid_label_pct: float
    very_small_sample_size: int
    high_monthly_concentration: float
    low_repeatability_months: int
    false_confidence_research_score: float


class LLMSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    market_hint: str = ""


class ScoringProfile(BaseModel):
    """Полная калибровка оценки для одного символа."""
    model_config = ConfigDict(frozen=True)

    symbol: str
    version: int = Field(ge=0)
    description: str = ""

    statistical: StatisticalSpec
    directional: DirectionalSpec
    context: ContextSpec
    rarity: RaritySpec
    weights: WeightsSpec
    rating: RatingSpec
    batch: BatchSpec
    validator: ValidatorSpec
    llm: LLMSpec = Field(default_factory=LLMSpec)

    @property
    def name(self) -> str:
        """Метка профиля, которая пишется в БД рядом с оценкой: «ETHUSDT@3»."""
        return f"{self.symbol}@{self.version}"

    @property
    def fingerprint(self) -> str:
        """
        sha256 нормализованного содержимого, 12 символов.

        Нужен потому, что правка порога без бампа version иначе была бы
        невидима: записи «до» и «после» лежали бы под одной меткой профиля и
        молча смешивались в средних.
        """
        payload = json.dumps(
            self.model_dump(mode="json", exclude={"description"}),
            sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def profile_fingerprint(profile: ScoringProfile) -> str:
    """Функциональная обёртка над ScoringProfile.fingerprint."""
    return profile.fingerprint


# ─── Загрузка ─────────────────────────────────────────────────────────────────

_lock = threading.Lock()
_cache: dict[str, ScoringProfile] = {}
_loaded = False
_load_errors: dict[str, str] = {}


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Слияние профиля с родителем: словари сливаются рекурсивно, списки и скаляры
    заменяются целиком.

    Практические следствия, о которых стоит помнить, правя профиль монеты:
      * `ladder` — список, поэтому заменяется целиком. Частично унаследованной
        лесенки не бывает: её нельзя было бы прочитать глазами по одному файлу.
        А вот `mode` и `floor` рядом с ней унаследуются, если их не указать.
      * enum-карты (`age_bucket` и прочие) — словари, поэтому сливаются
        по ключам: можно подвинуть один бакет, не переписывая все четыре.
        Полнота покрытия при этом гарантирована родителем.
    """
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: ожидался словарь, получен {type(data).__name__}")
    return data


#: Поля, которые НЕ наследуются от родителя. Описание характеризует конкретный
#: профиль: унаследованное, оно превращает реестр в дезинформацию — профиль
#: ETHUSDT показывал бы себя как «базовую калибровку под BTCUSDT».
_NOT_INHERITED = ("description",)


def _build(raw: dict[str, Any], base: dict[str, Any] | None) -> ScoringProfile:
    parent = dict(base or {})
    for field in _NOT_INHERITED:
        parent.pop(field, None)

    merged = _deep_merge(parent, raw)
    merged.pop("extends", None)
    return ScoringProfile(**merged)


def _load_all() -> None:
    """
    Читает каталог профилей. Ленивая, однократная, потокобезопасная.

    Битый YAML одной монеты не должен ронять API: файл пропускается с логом,
    монета получает `_default`. Исключение — сам `_default`: без него считать
    нечем, и молчать об этом нельзя.
    """
    global _loaded

    directory = profiles_dir()
    _cache.clear()
    _load_errors.clear()

    default_path = directory / f"{DEFAULT_PROFILE_NAME}.yaml"
    if not default_path.exists():
        raise FileNotFoundError(
            f"Нет базового профиля {default_path}. Без него оценка невозможна — "
            f"проверь PROFILES_DIR (сейчас {directory})."
        )

    default_raw = _read_yaml(default_path)
    default_profile = _build(default_raw, None)
    _cache[DEFAULT_PROFILE_NAME.upper()] = default_profile

    for path in sorted(directory.glob("*.yaml")):
        if path.stem == DEFAULT_PROFILE_NAME:
            continue
        try:
            raw = _read_yaml(path)
            parent = raw.get("extends", DEFAULT_PROFILE_NAME)
            if parent != DEFAULT_PROFILE_NAME:
                raise ValueError(
                    f"extends: {parent!r} — наследование поддерживается только "
                    f"от {DEFAULT_PROFILE_NAME}"
                )
            profile = _build(raw, default_raw)
        except Exception as exc:  # noqa: BLE001 — порченый профиль не роняет процесс
            _load_errors[path.stem.upper()] = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Профиль %s не загружен (%s) — монета будет считаться по %s",
                path.name, exc, DEFAULT_PROFILE_NAME,
            )
            continue

        if profile.symbol.upper() != path.stem.upper():
            logger.warning(
                "Профиль %s объявляет symbol=%s — ключом реестра остаётся имя файла",
                path.name, profile.symbol,
            )
        _cache[path.stem.upper()] = profile

    _loaded = True
    logger.info(
        "Загружено профилей: %d (%s)", len(_cache), ", ".join(sorted(_cache))
    )


def _ensure_loaded() -> None:
    if _loaded:
        return
    with _lock:
        if not _loaded:
            _load_all()


def reload_profiles() -> dict[str, Any]:
    """
    Перечитывает каталог. Для тестов и для POST /config/reload.

    Возвращает сводку: что загрузилось и что не смогло.
    """
    global _loaded
    with _lock:
        _loaded = False
        _load_all()
    return {
        "loaded": sorted(_cache),
        "errors": dict(_load_errors),
        "directory": str(profiles_dir()),
    }


def get_profile(symbol: str | None) -> ScoringProfile:
    """
    Профиль монеты. Регистронезависимо; неизвестная монета → `_default`.

    Функция намеренно не бросает исключений: оценка кандидата по неизвестной
    монете должна происходить (с флагом), а не падать.
    """
    _ensure_loaded()
    key = (symbol or "").strip().upper()
    return _cache.get(key) or _cache[DEFAULT_PROFILE_NAME.upper()]


def default_profile() -> ScoringProfile:
    """Базовый профиль — им считается quality_score_baseline."""
    _ensure_loaded()
    return _cache[DEFAULT_PROFILE_NAME.upper()]


def is_known_symbol(symbol: str | None) -> bool:
    """Есть ли у монеты собственный профиль. Реестр профилей = whitelist монет."""
    _ensure_loaded()
    key = (symbol or "").strip().upper()
    return bool(key) and key != DEFAULT_PROFILE_NAME.upper() and key in _cache


def list_profiles() -> list[dict[str, Any]]:
    """Список профилей с версиями и fingerprint — для GET /config/symbols."""
    _ensure_loaded()
    return [
        {
            "symbol": profile.symbol,
            "key": key,
            "version": profile.version,
            "profile": profile.name,
            "fingerprint": profile.fingerprint,
            "description": profile.description,
        }
        for key, profile in sorted(_cache.items())
    ]


def load_errors() -> dict[str, str]:
    """Профили, которые не удалось прочитать. Пусто — значит всё в порядке."""
    _ensure_loaded()
    return dict(_load_errors)


# ─── make profiles-check ──────────────────────────────────────────────────────

def _check_cli() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        summary = reload_profiles()
    except Exception as exc:  # noqa: BLE001 — CLI печатает диагноз, а не traceback
        print(f"ОШИБКА: {exc}")
        return 2

    for row in list_profiles():
        print(f"  {row['profile']:<20} {row['fingerprint']}  {row['description']}")
    if summary["errors"]:
        for name, error in sorted(summary["errors"].items()):
            print(f"  БИТЫЙ {name}: {error}")
        return 1
    print(f"Профилей: {len(summary['loaded'])}, ошибок нет ({summary['directory']})")
    return 0


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        raise SystemExit(_check_cli())
    raise SystemExit("Использование: python -m src.config.profiles --check")
