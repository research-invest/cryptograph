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
    # Монета ПО УМОЛЧАНИЮ для одиночных команд. Полный список пар —
    # в btcproc/symbols.py: дата листинга и переопределения порогов это
    # знание о рынке, оно версионируется вместе с кодом, а не в окружении.
    # Обязана присутствовать в реестре — проверяется symbols.validate_default().
    symbol: str = _env("SYMBOL", "BTCUSDT")
    base_tf: str = _env("BASE_TIMEFRAME", "15m")
    context_tfs: list[str] = field(
        default_factory=lambda: _env_list("CONTEXT_TIMEFRAMES", "1h,4h,1d")
    )
    # Дефолт для монет, у которых history_start в реестре не задан.
    # Спека монеты его перекрывает (см. SymbolSpec.start_date).
    history_start: str = _env("HISTORY_START", "2017-08-01")
    horizon: str = _env("HORIZON", "24h")
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "./data")))
    # Эндпоинт для добора свежего хвоста баров. Вынесен в окружение, потому
    # что api.binance.com отвечает 451 Unavailable For Legal Reasons из ряда
    # юрисдикций (в частности из США), и на таком хосте live не работает вовсе:
    # месячные дампы с data.binance.vision качаются, а последние часы — нет.
    # Замена — data-api.binance.vision: тот же /api/v3/klines, публичный,
    # без ключей, отвечает отовсюду. Дефолт оставлен прежним, чтобы уже
    # работающие установки не поменяли источник данных молча.
    binance_rest_url: str = _env(
        "BINANCE_REST_URL", "https://api.binance.com/api/v3/klines"
    )

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
    # Размер группы, ниже которого дробить нельзя, задаётся ДВУМЯ числами:
    # долей истории и абсолютным полом. Эффективное значение —
    #     max(min_group_size, round(min_group_share * число баров))
    #
    # Доля — основной механизм. Порог обязан быть относительным: 800 баров это
    # 0.27% истории BTC с 2017 года и 1.1% истории двухлетней монеты. С общим
    # абсолютным порогом дробление на короткой истории останавливается рано,
    # и граф выходит грубее не потому, что рынок однороднее, а потому что
    # линейка чужая. 0.0025 на истории BTC даёт ~750 — то же, на чём граф
    # калибровался.
    min_group_share: float = _env_float("STATES_MIN_GROUP_SHARE", 0.0025)
    # Абсолютный пол. Именно пол, а не значение по умолчанию: группа в сотню
    # баров не даёт кандидату статистики при любой длине истории.
    #
    # Раньше здесь стояло 800 — значение, подобранное под BTC. Оно перекрывало
    # долю на всём диапазоне реальных историй (0.0025 × 320 тыс. = 800), то есть
    # относительный порог был бы формальностью. 300 — это «сотни баров»,
    # ниже которых выборка кандидата разваливается независимо от монеты.
    # Побочный эффект: на BTC эффективный порог стал 750 вместо 800. Граф от
    # этого сдвигается в пределах шума, а group_id всё равно перенумеровываются
    # при каждом train (см. README, раздел про train против live).
    min_group_size: int = _env_int("STATES_MIN_GROUP_SIZE", 300)
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
class SMCConfig:
    """
    Параметры детекторов Smart Money (docs/task_smc_integration.md, раздел 9).

    Все пороги заданы в ATR или в барах, ни один — в долларах: детектор,
    зависящий от абсолютной цены, на истории с 2017 года измеряет эпоху,
    а не рынок, и на SOL означает не то же, что на BTC.
    """

    # Выключателя ДВА, потому что у двух половин SMC разная цена.
    #
    # enabled — контекстные атомы. Дёшево и обратимо: в маску они не входят,
    # event_block_id не меняют, переобучения не требуют. live со старой
    # моделью просто пишет их в bar_events.context_atoms, и по ним можно
    # мерить лифт.
    #
    # features_enabled — двенадцать величин в векторе признаков. Дорого:
    # FEATURE_VERSION становится v2, нужен полный train, group_id
    # перенумеровываются. И, по замеру, вредно: добавление двенадцати
    # признаков любой природы обрушивает число состояний (у BTC 43 → 31, у
    # чистого шума той же формы 43 → 22), потому что пороги дробления
    # калибровались на 32 измерениях. Включать только вместе с
    # перекалибровкой split_gain и merge_separation.
    #
    # Раздельно — чтобы включение атомов в бою не делало следующий train
    # молча сорокачетырёхмерным. Один флаг на обе половины ровно это и
    # означал бы.
    enabled: bool = _env_bool("SMC_ENABLED", False)
    features_enabled: bool = _env_bool("SMC_FEATURES_ENABLED", False)

    @property
    def features_on(self) -> bool:
        """Признаки считаются, только если включены обе половины."""
        return self.enabled and self.features_enabled

    # Баров слева и справа для подтверждения свинга. right задаёт лаг ВСЕХ
    # структурных детекторов: свинг на баре i становится известен только на
    # баре i + right. Это не недостаток, а условие честности — центрированное
    # окно без сдвига читает будущее.
    swing_left: int = _env_int("SMC_SWING_LEFT", 3)
    swing_right: int = _env_int("SMC_SWING_RIGHT", 3)

    # Порог «крупного» FVG в ATR. Значение измерено в фазе 2, а не выбрано:
    # при 0.5 детектор срабатывает на 4.6–5.1% баров и в signature не проходит
    # (бюджет — 3%), при 0.7 выходит 2.7–2.9% на всех трёх монетах.
    # Распределение размера разрыва в ATR у BTC, ETH и SOL совпадает с точностью
    # до второго знака (p50 = 0.24 / 0.25 / 0.27), поэтому порог общий и
    # помонетных переопределений не требует.
    fvg_min_atr: float = _env_float("SMC_FVG_MIN_ATR", 0.7)

    # Возраст выбывания из реестров. Нужен не для точности, а чтобы реестры
    # не росли неограниченно: без него один проход по 300k баров деградирует
    # в квадратичный.
    fvg_max_age_bars: int = _env_int("SMC_FVG_MAX_AGE_BARS", 672)      # неделя
    ob_max_age_bars: int = _env_int("SMC_OB_MAX_AGE_BARS", 672)        # неделя
    level_max_age_bars: int = _env_int("SMC_LEVEL_MAX_AGE_BARS", 2688)  # месяц

    # Допуск, в пределах которого два свинга считаются одним уровнем.
    eq_tolerance_atr: float = _env_float("SMC_EQ_TOLERANCE_ATR", 0.15)
    level_tolerance_atr: float = _env_float("SMC_LEVEL_TOLERANCE_ATR", 0.25)

    # Окно возврата после снятия ликвидности. Снятие без возврата — обычный
    # пробой; именно возврат отличает sweep от breakout.
    reclaim_bars: int = _env_int("SMC_RECLAIM_BARS", 4)

    # Сколько баров назад искать свечу ордер-блока от слома структуры.
    ob_lookback_bars: int = _env_int("SMC_OB_LOOKBACK_BARS", 20)

    # Границы premium/discount. 0.382 и 0.618 — числа Фибоначчи, в SMC
    # используются по соглашению; проверять их осмысленность — задача замера,
    # а не кода.
    discount_below: float = _env_float("SMC_DISCOUNT_BELOW", 0.382)
    premium_above: float = _env_float("SMC_PREMIUM_ABOVE", 0.618)

    # Нормировка счётчиков: log1p(x) / log1p(этого). Десять касаний одного
    # уровня — уже «много», дальше разница несущественна.
    count_norm_scale: float = _env_float("SMC_COUNT_NORM_SCALE", 10.0)


@dataclass(frozen=True)
class SinkConfig:
    mode: str = _env("SINK_MODE", "direct")  # direct | http | none
    # Каталог соседнего проекта btc-graph. Дефолт вычисляется от расположения
    # ЭТОГО файла, а не хардкодится: проекты лежат рядом внутри crypto-graph,
    # и абсолютный путь устаревал при каждом переезде. Симптомы были тихими:
    # SINK_MODE=direct падал с «не похож на репозиторий btc-graph», а
    # test_generated_candidates_match_btc_graph_schema молча скипался —
    # то есть контракт схемы переставал проверяться незаметно.
    btc_graph_path: Path = field(
        default_factory=lambda: Path(
            _env("BTC_GRAPH_PATH")
            or (Path(__file__).resolve().parents[2] / "btc-graph")
        )
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
    # Сколько прогонов админка готова вести одновременно. Прогоны разных монет
    # независимы и блокировать друг друга не должны, но идут они BackgroundTasks
    # в процессе админки: три одновременных train займут три ядра на
    # кластеризации и утроят пик памяти. Двойка — компромисс для типичной
    # машины; поднимать осознанно, глядя на RAM.
    max_concurrent_runs: int = _env_int("ADMIN_MAX_CONCURRENT_RUNS", 2)

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
smc = SMCConfig()
sink = SinkConfig()
admin = AdminConfig()
