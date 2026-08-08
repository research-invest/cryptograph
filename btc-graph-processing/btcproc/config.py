"""
Единственное место, где читается окружение.

Все модули берут настройки отсюда — так параметры расчёта (таймфреймы,
горизонт, пороги кластеризации) не расползаются по коду и видны целиком.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Значения-заглушки из .env.example: с ними админка стартовать не должна.
_PLACEHOLDERS = {
    "",
    "ЗАМЕНИ_МЕНЯ_длинным_паролем",
    "ЗАМЕНИ_МЕНЯ_на_вывод_openssl_rand_hex_32",
    "changeme",
    "admin",
    "password",
}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    return float(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_list(name: str, default: str) -> list[str]:
    raw = _env(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# Длительность бара в минутах. Ограничиваемся тем, что реально отдаёт Binance
# и что имеет смысл для горизонта 24h.
TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "12h": 720,
    "1d": 1440,
}


@dataclass(frozen=True)
class DataConfig:
    symbol: str = _env("SYMBOL", "BTCUSDT")
    base_tf: str = _env("BASE_TIMEFRAME", "15m")
    context_tfs: list[str] = field(
        default_factory=lambda: _env_list("CONTEXT_TIMEFRAMES", "1h,4h,1d")
    )
    history_start: str = _env("HISTORY_START", "2017-08-01")
    horizon: str = _env("HORIZON", "24h")
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "./data")))

    @property
    def base_minutes(self) -> int:
        return TIMEFRAME_MINUTES[self.base_tf]

    @property
    def horizon_minutes(self) -> int:
        unit = self.horizon[-1]
        value = int(self.horizon[:-1])
        return value * {"m": 1, "h": 60, "d": 1440}[unit]

    @property
    def horizon_bars(self) -> int:
        """Сколько базовых баров укладывается в горизонт оценки."""
        return self.horizon_minutes // self.base_minutes

    def bars_per(self, timeframe: str) -> int:
        """Сколько базовых баров в одном баре старшего ТФ."""
        return TIMEFRAME_MINUTES[timeframe] // self.base_minutes


@dataclass(frozen=True)
class DBConfig:
    url: str = _env(
        "DATABASE_URL", "postgresql://btc_user:btc_pass@localhost:5432/btc_graph"
    )
    schema: str = _env("PG_SCHEMA", "processing")
    redis_url: str = _env("REDIS_URL", "redis://localhost:6379/1")


@dataclass(frozen=True)
class StatesConfig:
    """
    Параметры адаптивной гранулярности графа состояний.

    Логика из ТЗ: неоднородная группа дробится, почти идентичные сливаются.
    Числа подобраны под 15m-историю с 2017 (~300k баров).
    """

    # Стартовое число крупных кластеров, дальше дробление идёт рекурсивно.
    seed_clusters: int = _env_int("STATES_SEED_CLUSTERS", 8)
    # Ниже этого размера группу не дробим — статистика кандидата развалится.
    min_group_size: int = _env_int("STATES_MIN_GROUP_SIZE", 800)
    # Предел глубины рекурсии дробления (2^depth групп максимум на ветку).
    max_depth: int = _env_int("STATES_MAX_DEPTH", 4)
    # Дробим, если разбиение надвое улучшает силуэт хотя бы на столько
    # по сравнению со случайным облаком. Рыночные признаки образуют
    # непрерывное облако, а не разделённые шары, поэтому запас над
    # референсом здесь небольшой по своей природе — порог в десятые доли
    # схлопнул бы весь граф в 2-3 состояния.
    split_gain: float = _env_float("STATES_SPLIT_GAIN", 0.02)
    # Сливаем пары, чьи центроиды разнесены меньше чем на столько
    # «разбросов вдоль оси между ними» (d-prime). Ниже 1.0 облака
    # перекрываются настолько, что говорить о двух состояниях нет смысла.
    merge_separation: float = _env_float("STATES_MERGE_SEPARATION", 1.0)
    # Смена состояния засчитывается только если новое держится столько баров —
    # защита от дребезга на границе кластеров.
    smoothing_bars: int = _env_int("STATES_SMOOTHING_BARS", 2)
    # Глубина траектории для расчёта trajectory_entropy.
    trajectory_window: int = _env_int("STATES_TRAJECTORY_WINDOW", 24)
    # Подвыборка для силуэта — считать его на 300k точках бессмысленно долго.
    silhouette_sample: int = _env_int("STATES_SILHOUETTE_SAMPLE", 4000)
    random_state: int = 42


@dataclass(frozen=True)
class CandidateConfig:
    """Пороги сборки кандидата из исторической выборки."""

    # Меньше этого числа аналогов кандидат не выпускается вовсе.
    min_sample_size: int = _env_int("CAND_MIN_SAMPLE_SIZE", 30)
    # Перекос слабее — кандидата нет, направление не определено.
    min_abs_skew: float = _env_float("CAND_MIN_ABS_SKEW", 0.06)
    # Порог |skew| для historical_bias_context = long_skew / short_skew.
    bias_skew_threshold: float = _env_float("CAND_BIAS_SKEW", 0.10)
    # Перцентили распределений favorable / adverse (см. README, раздел про p70/p80).
    favorable_percentile: float = _env_float("CAND_FAVORABLE_PCT", 70.0)
    adverse_percentile: float = _env_float("CAND_ADVERSE_PCT", 80.0)
    # Граница «свежести» снимка: если данные старше — context_status=stale.
    fresh_max_lag_minutes: int = _env_int("CAND_FRESH_LAG_MIN", 30)
    # Если выборка по (переход + event_block) меньше min_sample_size,
    # откатываемся к выборке только по переходу.
    fallback_to_transition: bool = _env_bool("CAND_FALLBACK_TRANSITION", True)


@dataclass(frozen=True)
class SinkConfig:
    mode: str = _env("SINK_MODE", "direct")  # direct | http | none
    btc_graph_path: Path = field(
        default_factory=lambda: Path(_env("BTC_GRAPH_PATH", "/Volumes/work/btc-graph"))
    )
    btc_graph_url: str = _env("BTC_GRAPH_URL", "http://localhost:8000")
    # Redis самого btc-graph: его дедуп и канал btc:strong_candidates живут
    # в своей базе, смешивать со служебной базой processing не надо.
    btc_graph_redis_url: str = _env("BTC_GRAPH_REDIS_URL", "redis://localhost:6379/0")
    use_llm: bool = _env_bool("SINK_USE_LLM", False)
    batch_size: int = _env_int("SINK_BATCH_SIZE", 200)
    min_quality_score: float = _env_float("SINK_MIN_QUALITY", 0.0)


@dataclass(frozen=True)
class AdminConfig:
    user: str = _env("ADMIN_USER")
    password: str = _env("ADMIN_PASSWORD")
    secret_key: str = _env("ADMIN_SECRET_KEY")
    session_ttl: int = _env_int("ADMIN_SESSION_TTL", 43200)
    max_login_attempts: int = _env_int("ADMIN_MAX_LOGIN_ATTEMPTS", 5)
    lockout_seconds: int = _env_int("ADMIN_LOCKOUT_SECONDS", 900)
    ip_allowlist: list[str] = field(
        default_factory=lambda: _env_list("ADMIN_IP_ALLOWLIST", "")
    )
    host: str = _env("ADMIN_HOST", "127.0.0.1")
    port: int = _env_int("ADMIN_PORT", 8100)

    def validate(self) -> None:
        """
        Жёсткая проверка на старте: пустые или демонстрационные значения
        означают открытую наружу админку, поэтому это ошибка, а не warning.
        """
        problems = []
        if not self.user or self.user in {"", "changeme"}:
            problems.append("ADMIN_USER не задан")
        if self.password in _PLACEHOLDERS or len(self.password) < 12:
            problems.append("ADMIN_PASSWORD не задан или короче 12 символов")
        if self.secret_key in _PLACEHOLDERS or len(self.secret_key) < 32:
            problems.append("ADMIN_SECRET_KEY не задан или короче 32 символов")
        if problems:
            raise RuntimeError(
                "Админка не запущена — проблемы с конфигурацией:\n  - "
                + "\n  - ".join(problems)
                + "\nЗаполни .env (см. .env.example)."
            )


data = DataConfig()
db = DBConfig()
states = StatesConfig()
candidates = CandidateConfig()
sink = SinkConfig()
admin = AdminConfig()
