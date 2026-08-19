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


def _env_float_optional(name: str) -> float | None:
    """
    Как `_env_float`, но умеет вернуть «не задано».

    Обычный `_env_float` всегда отдаёт число, и через окружение невозможно
    выразить «пусть решает приёмник». Для порогов, у которых `None` — это
    самостоятельный режим (а не синоним нуля), нужен именно такой парсер.
    """
    raw = _env(name)
    return float(raw) if raw else None


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
    # Горизонты, которые размечаются В ДОПОЛНЕНИЕ к основному и только
    # СОХРАНЯЮТСЯ: ни кандидаты, ни модель состояний их не видят. Заведены
    # задачей D `crypto-graph/docs/tz_range_horizons_19-08-26.md`: размах, в
    # отличие от направления, на разных горизонтах — разный вопрос
    # (раздел 47 журнала), и считать его каждый раз заново из баров дорого.
    #
    # **Это платная опция.** Каждый лишний горизонт — ещё один комплект строк
    # в `outcomes` (на шести монетах это порядка миллиона строк и сотен
    # мегабайт, навсегда) плюс один проход скользящих окон в `train`. В `live`
    # цены нет вовсе: он исходы не сохраняет, а считает их в памяти и только
    # на основном горизонте. Пустое значение выключает добавку целиком, и
    # ничего, кроме доступности этих строк для замеров, не ломается.
    extra_horizons: list[str] = field(
        default_factory=lambda: _env_list("OUTCOME_EXTRA_HORIZONS", "4h,12h")
    )
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
    # То же самое для Bybit — но с важным отличием: рабочего зеркала у него
    # нет. api.bybit.com, api.bytick.com и api.bybit.nl отвечают 403 с той же
    # CloudFront-заглушкой «configured to block access from your country»,
    # то есть с американского хоста (в том числе с боевого VPS) свежий хвост
    # монеты с Bybit не добирается никак. Переменная оставлена ради установок,
    # где доступ есть, и ради подмены адреса в тестах.
    #
    # Работать без неё загрузчик умеет: тиковые архивы public.bybit.com
    # отдаются отовсюду, и на них он и живёт, отставая на сутки. Это разобрано
    # в bybit.sync_recent — там же причина, почему 403 не роняет прогон.
    bybit_rest_url: str = _env(
        "BYBIT_REST_URL", "https://api.bybit.com/v5/market/kline"
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

    def bars_of_horizon(self, label: str) -> int:
        """
        «4h» → число базовых баров. Отдельно от `bars_per`, потому что
        горизонт — не таймфрейм: `TIMEFRAME_MINUTES` перечисляет сетки, на
        которых есть бары, а горизонтом может быть любая длительность.
        """
        unit = label[-1]
        value = int(label[:-1])
        minutes = value * {"m": 1, "h": 60, "d": 1440}[unit]
        if minutes % self.base_minutes:
            raise ValueError(
                f"горизонт {label} не кратен базовому бару "
                f"({self.base_minutes} мин)"
            )
        return minutes // self.base_minutes


@dataclass(frozen=True)
class FeaturesConfig:
    """
    Сборка вектора признаков.

    Единственный параметр здесь — предохранитель на общий `dropna()` в конце
    `build_features`. Он нужен потому, что признаки внешнего источника
    существуют не на всей истории монеты, а `dropna` по всем колонкам режет
    строку целиком: включение `DERIV_FEATURES_ENABLED` на BTC отрезало бы
    историю до 2020-09 (метрики начинаются там, бары — с 2017-08), то есть
    **минус три года**, и молча — модель просто обучилась бы на трети
    истории. Дыры по отдельным колонкам у поставщика выедают бары ещё и
    из СЕРЕДИНЫ истории (аудит 2026-08-15, B4).

    Порог — доля строк, которую источникам разрешено срезать сверх того, что
    срезала бы базовая тридцатка. Два процента: прогрев окон источника
    (месячные ранги, суточные z-score) стоит недели-другой на границе и в
    эту долю укладывается, а потеря куска истории — нет.

    Превышение — ошибка, а не предупреждение: «включил флаг ради
    эксперимента → модель обучилась не на том» относится ровно к тому классу
    тихой деградации, против которого выстроен весь остальной контур.
    Осознанный эксперимент разрешается переменной окружения.
    """

    max_source_row_loss: float = _env_float("FEATURES_MAX_SOURCE_ROW_LOSS", 0.02)


@dataclass(frozen=True)
class DBConfig:
    url: str = _env(
        "DATABASE_URL", "postgresql://btc_user:btc_pass@localhost:5432/btc_graph"
    )
    schema: str = _env("PG_SCHEMA", "processing")
    redis_url: str = _env("REDIS_URL", "redis://localhost:6379/1")


@dataclass(frozen=True)
class RunsConfig:
    """Учёт прогонов: как отличить идущий прогон от убитого."""

    # Сколько прогон может молчать, прежде чем считаться мёртвым.
    # `update_run` трогает heartbeat на каждой стадии, а самая долгая стадия
    # (кластеризация в train на полной истории) укладывается в десятки минут.
    # Два часа — заведомо больше неё и заведомо меньше получасового интервала
    # крона, помноженного на терпение оператора. Занижать опасно: живой train
    # будет объявлен мёртвым и его слот отдадут второму прогону той же монеты.
    stale_after_minutes: int = _env_int("RUN_STALE_AFTER_MINUTES", 120)


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
    # ── Критерий дробления: gap statistic в сигмах референса ────────────────
    # Разбиение надвое принимается, если силуэт реального разбиения превышает
    # силуэт разбиения СЛУЧАЙНОГО облака той же формы — но не «на 0.02», а на
    # заданное число собственных сигм этого референса.
    #
    # Абсолютный порог (STATES_SPLIT_GAIN = 0.02, до 2026-08-11) был неверен
    # дважды. Во-первых, референс был ОДИН: силуэт случайного облака — сама по
    # себе случайная величина, и решения вблизи границы определялись тем, какой
    # draw выпал. Отсюда наблюдённая нестабильность числа состояний у ETH
    # (29 → 26 → 42 на трёх прогонах) — это не «порог поехал от длины истории»,
    # а хаос вокруг границы решения. Во-вторых, разность силуэтов зависит от
    # размерности: в 44 измерениях концентрация расстояний жмёт и real, и ref,
    # но неодинаково, поэтому константа, откалиброванная на 32 измерениях,
    # означает там другую строгость. Именно из-за этого двенадцать признаков
    # ЛЮБОЙ природы, включая чистый шум, обрушивали граф (43 → 22 у BTC), и
    # это повторилось бы с каждым новым источником — ончейном, индексами,
    # деривативами.
    #
    # Порог в сигмах самонормируется: сигма меряется в тех же единицах, в
    # которых сжался силуэт. Ради этого свойства gap statistic и придуман.
    # `_separation` (d-prime) в этом же модуле нормирован по размерности
    # намеренно и с самого начала — вторая половина механизма просто не была
    # доведена.
    #
    # Число референсов B. Десять — минимум, при котором выборочная сигма
    # осмысленна; стоимость дробления при этом растёт втрое, а не вдесятеро
    # (референсам хватает n_init=1, см. _split_gain).
    split_reference_draws: int = _env_int("STATES_SPLIT_REFERENCE_DRAWS", 10)
    # Порог в сигмах референса.
    #
    # ЧЕСТНО О ТОМ, КАК ВЫБРАНО 2.0. Свип по {0.5, 0.75, 1.0, 1.5, 2.0} × три
    # окна × три монеты был сделан (scripts/calibrate_split_gain.py), но его
    # критерий — разброс числа состояний между окнами — оказался негодной
    # мерой: процедура меняет ответ на 15% уже от ПЯТИДЕСЯТИ баров (0.016%
    # истории BTC, замер в development_log.md, 21.17). То есть «10.3% при
    # 2.0» против «31.6% при 1.0» — две реализации одного шума, а не свойство
    # порога.
    #
    # Поэтому 2.0 стоит по слабому основанию: при нём число состояний
    # попадает в разумный диапазон на всех трёх монетах (BTC 40–50,
    # ETH 24–35, SOL 40–55) и стоимость дробления умеренная. Это выбор по
    # здравому смыслу, а не по замеру, и менять его есть смысл только вместе
    # с починкой самой процедуры.
    #
    # Что от порога НЕ зависит и починено по-настоящему: он больше не
    # определяется одним случайным референсом и не зависит от размерности
    # (test_split_gain_is_independent_of_dimensionality).
    #
    # Число состояний под ним НЕ подбиралось: прежние 43 у BTC сами были
    # артефактом порога, зависящего от размерности.
    split_gain_sigma: float = _env_float("STATES_SPLIT_GAIN_SIGMA", 2.0)
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
class RangeForecastConfig:
    """
    Квантильный регрессор размаха (`analysis/range_forecast.py`, раздел 48).

    **Флаг по умолчанию выключен, и это не осторожность, а порядок ввода**,
    закреплённый уроком 34.10 и повторённый для деривативов (39): сначала
    код и данные, потом флаг, и только на контуре, где уже есть чем считать.
    Включение стоит `train` дороже: гейт обучает модель на 70% истории,
    затем боевая версия переобучается на всей, — то есть примерно
    полуторакратный набор из восьми квантильных бустингов на монету.

    `enabled` включает обучение в `train`. Поля в кандидате появляются, только
    если модель ПРОШЛА гейт: непрошедшая сохраняется для разбора, но её числа
    наружу не идут.
    """

    enabled: bool = _env_bool("RANGE_FORECAST_ENABLED", False)
    #: Нормировка цели. `atr14` совпадает с разделами 36, 47 и 48 и потому
    #: сравнима с ними; вторая (`atr_h`) остаётся выбором ЗАМЕРА.
    normalization: str = _env("RANGE_FORECAST_NORM", "atr14")
    #: Зерно бустинга. Меняет результат слабо (48.6), но фиксируется, чтобы
    #: два прогона одной истории давали одинаковую модель.
    seed: int = _env_int("RANGE_FORECAST_SEED", 42)
    #: Доля истории под обучение при расчёте гейта. Та же, что у Ш0 и D1.
    train_frac: float = _env_float("RANGE_FORECAST_TRAIN_FRAC", 0.7)
    #: Реплик блочного бутстрапа в гейте. Меньше, чем в замере (2000), —
    #: гейт бинарный, а не публикуемое число; разрешения 500 реплик хватает
    #: на порог 0.05 с запасом.
    gate_n_boot: int = _env_int("RANGE_FORECAST_GATE_BOOT", 500)


@dataclass(frozen=True)
class CandidateConfig:
    """Пороги сборки кандидата из исторической выборки."""

    # Меньше этого числа СТРОК снимков кандидат не выпускается вовсе.
    # Мягкий порог, оставлен ради совместимости: строки одной реализации
    # перехода почти дублируют друг друга (см. min_effective_sample_size).
    min_sample_size: int = _env_int("CAND_MIN_SAMPLE_SIZE", 30)
    # Порог по числу НЕЗАВИСИМЫХ реализаций перехода в выборке.
    #
    # Снимки берутся с офсетами 0/45/90/180 минут, то есть одна реализация
    # даёт до четырёх строк, а их окна исходов при горизонте 24h совпадают
    # на 87.5% и более. До 2026-08-13 порог применялся к строкам, и
    # «минимум 30 аналогов» на деле означало около четырнадцати случаев
    # (замеренная медиана завышения — 2.2, а не 4: до старших офсетов
    # доживает не всякая реализация). Зависимость наблюдений давно учтена
    # в измерителях (блочный бутстрап), но в самом поле кандидата — не была.
    #
    # Значение то же, 30, но теперь оно означает то, что написано. Прямое
    # следствие — кандидатов заметно меньше: 20.3% отсева у BTC, 15.6% у ETH,
    # 26.9% у SOL и 74.9% у HYPEUSDT с её годовой историей. Это и есть цель.
    min_effective_sample_size: int = _env_int("CAND_MIN_EFFECTIVE_SAMPLE", 30)
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
class FearGreedConfig:
    """
    Fear & Greed Index (alternative.me) — общерыночный внешний ряд.

    `docs/task_fear_greed.md`: заводится не для предсказания направления
    (это закрыто замерами — раздел 26 и 31 журнала), а как предиктор размаха
    и описательный контекст. Два флага той же формы, что у SMCConfig, и по
    той же причине: атомы бесплатны и обратимы, признаки требуют train.

    В отличие от SMC величина не считается из баров, а джойнится из
    `external_daily` (btcproc/ingest/external.py) — сетевого похода здесь
    нет, таблицу заполняет отдельная команда `fetch-external`.
    """

    enabled: bool = _env_bool("FGI_ENABLED", False)             # контекстные атомы
    features_enabled: bool = _env_bool("FGI_FEATURES_ENABLED", False)  # признаки

    @property
    def features_on(self) -> bool:
        return self.enabled and self.features_enabled

    # Пороги атомов (docs/task_fear_greed.md, §4.1). Не в деньгах и не в ATR —
    # индекс сам по себе безразмерная шкала 0–100, ограниченная по построению,
    # поэтому масштабная инвариантность здесь тривиальна и пороги общие.
    fear_extreme_below: float = _env_float("FGI_FEAR_EXTREME_BELOW", 25.0)
    greed_extreme_above: float = _env_float("FGI_GREED_EXTREME_ABOVE", 75.0)
    flip_midpoint: float = _env_float("FGI_FLIP_MIDPOINT", 50.0)

    # Окно скорости смены настроения (fgi_change_7d).
    change_window_days: int = _env_int("FGI_CHANGE_WINDOW_DAYS", 7)


@dataclass(frozen=True)
class DerivConfig:
    """
    Деривативные метрики Binance USD-M — открытый интерес, long/short ratio,
    давление тейкеров (docs/tz_deriv_ingest_14-08-26.md). Третий внешний
    источник, тот же принцип флагов, что у SMC и FGI, и по той же причине:
    атомы бесплатны и обратимы, признаки требуют train.

    В отличие от FGI величина не общерыночная, а помонетная и уже на сетке
    базового ТФ (`btcproc/ingest/metrics.py`) — джойн в `features/deriv.py`
    идёт без сдвига, сетевого похода здесь нет: таблицу `deriv_metrics`
    заполняет отдельная команда `ingest-metrics`.
    """

    enabled: bool = _env_bool("DERIV_ENABLED", False)             # контекстные атомы
    features_enabled: bool = _env_bool("DERIV_FEATURES_ENABLED", False)  # признаки

    @property
    def features_on(self) -> bool:
        return self.enabled and self.features_enabled


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
    # Порог качества, с которым кандидаты уходят в btc-graph.
    # `None` (дефолт) = «решает btc-graph»: он применит порог
    # `batch.min_quality_score` из профиля КАЖДОЙ монеты — те самые
    # калиброванные значения, ради которых профили и заводились.
    # Число = единая абсолютная линейка на весь батч, перекрывающая профили;
    # это режим разовых экспериментов, а не регулярной работы.
    # Раньше здесь стоял дефолт 0.0, и он молча отключал профильные пороги:
    # фильтр btc-graph пропускал всех, WEAK составлял основную массу базы.
    min_quality_score: float | None = _env_float_optional("SINK_MIN_QUALITY")


@dataclass(frozen=True)
class NotifyConfig:
    """
    Вебхуки: POST с кандидатом на внешний адрес.

    Правила (кому, что и по какому фильтру слать) живут в БД, а не здесь:
    их правит оператор из админки, и перезапуск ради нового адреса — плохая
    цена. В окружении остаётся только то, что относится к механике доставки.
    """

    # Рубильник на весь механизм. Правил может не быть вовсе — тогда флаг
    # ничего не меняет; он нужен, чтобы выключить рассылку разом, не удаляя
    # настроенные правила (например, на время бэкфилла).
    enabled: bool = _env_bool("NOTIFY_ENABLED", True)

    # Сколько ждать ответа ОДНОГО получателя. Прогон этого не ждёт (отправка
    # идёт в фоновом потоке), но и висеть вечно поток не должен: очередь
    # разгребается последовательно, и один мёртвый адрес затормозил бы всех.
    timeout: float = _env_float("NOTIFY_TIMEOUT", 5.0)

    # Потоков-отправителей. Двух хватает: правил единицы, кандидатов за
    # live-прогон — десятки.
    workers: int = _env_int("NOTIFY_WORKERS", 2)

    # Потолок очереди. Переполнение — это не «подождём», а «дропнем с записью
    # в журнал»: копить в памяти неограниченно хуже, чем честно потерять
    # уведомление и увидеть это в админке.
    queue_size: int = _env_int("NOTIFY_QUEUE_SIZE", 2000)

    # Сколько прогон ждёт разбора очереди ПЕРЕД выходом. Крон-процесс живёт
    # ровно столько, сколько идёт прогон, и без этого ожидания фоновые потоки
    # умерли бы вместе с ним, не отправив ничего. Внутри прогона отправка
    # по-прежнему не блокирует ни одной стадии.
    flush_seconds: float = _env_float("NOTIFY_FLUSH_SECONDS", 30.0)

    # Возраст кандидата, старше которого уведомление не отправляется.
    #
    # Это не украшательство, а предохранитель. `train` выпускает сотни тысяч
    # кандидатов на всей истории, и первый же прогон с отправкой без этого
    # порога разослал бы сотни тысяч вебхуков за 2017–2026 годы. Уведомление
    # осмысленно только про «сейчас», поэтому окно и узкое.
    max_candidate_age_minutes: int = _env_int("NOTIFY_MAX_CANDIDATE_AGE_MIN", 180)

    # Хранение журнала доставок. Журнал заодно служит защитой от повторов
    # (PK — пара правило+кандидат), поэтому чистить его агрессивно нельзя:
    # удалённая запись снимает защиту. Но кандидат старше окна выше всё равно
    # не проходит по возрасту, так что месяц — с двойным запасом.
    retention_days: int = _env_int("NOTIFY_RETENTION_DAYS", 30)


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
    # Доверять ли заголовку X-Forwarded-For при определении адреса клиента.
    # Включать ТОЛЬКО если перед админкой действительно стоит прокси, который
    # этот заголовок перезаписывает. Иначе его ставит сам клиент и получает
    # право выбрать себе адрес: обойти ADMIN_IP_ALLOWLIST и обнулять счётчик
    # неудачных входов новым фейковым IP на каждую попытку. В штатной схеме
    # развёртывания прокси нет — отсюда дефолт false.
    trust_proxy: bool = _env_bool("ADMIN_TRUST_PROXY", False)
    # Сколько прогонов админка готова вести одновременно. Прогоны разных монет
    # независимы и блокировать друг друга не должны, но идут они BackgroundTasks
    # в процессе админки: три одновременных train займут три ядра на
    # кластеризации и утроят пик памяти. Двойка — компромисс для типичной
    # машины; поднимать осознанно, глядя на RAM.
    max_concurrent_runs: int = _env_int("ADMIN_MAX_CONCURRENT_RUNS", 2)
    # С какой длительности запрос PostgreSQL считается долгим и подсвечивается
    # на странице «Сервер». Тридцать секунд выбраны между двумя известными
    # величинами: любой запрос страницы админки укладывается в доли секунды и
    # в подсветку не попадёт, а её же потолок (statement_timeout, 60 с) выше
    # порога — то есть подсветка успевает появиться раньше, чем запрос
    # оборвётся. Bulk-вставки прогона в подсветку попадают намеренно: это
    # действительно нагрузка, и объяснять ею занятую базу — правильно.
    pg_slow_seconds: int = _env_int("ADMIN_PG_SLOW_SECONDS", 30)

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


@dataclass(frozen=True)
class HostmonConfig:
    """
    Монитор ресурсов хоста: отдельный процесс-сэмплер + страница «Сервер»
    в админке.

    Хранилище — SQLite-файл, а не Postgres, намеренно. Смотреть на монитор
    приходится ровно тогда, когда стеку плохо: OOM killer, полный диск,
    подвисший контейнер. Ряд в Postgres в этот момент либо недоступен, либо
    пишется с задержкой — то есть истории за момент аварии не осталось бы.
    Файл на хосте от состояния docker не зависит вовсе.

    Путь по умолчанию лежит ВНЕ каталогов подпроектов: `01_deploy.sh` гонит
    их `rsync --delete`, и файл внутри `btc-graph-processing/` сносило бы при
    каждой выкатке кода. `<repo>/logs/` — тот же каталог, где на боевом
    контуре уже живут логи прогонов.
    """
    db_path: Path = field(
        default_factory=lambda: Path(
            _env("HOSTMON_DB")
            or str(Path(__file__).resolve().parents[2] / "logs" / "hostmon.sqlite")
        )
    )
    # Шаг сетки замеров. Минута — компромисс: на ней виден и всплеск от
    # кластеризации (стадия идёт минутами), и месяц истории занимает единицы
    # мегабайт. Секундная сетка ловила бы пики точнее, но монитор перестал бы
    # быть бесплатным для машины, которую он сторожит.
    interval: int = _env_int("HOSTMON_INTERVAL_SECONDS", 60)
    # Сколько держать замеры. 30 суток — чтобы «в прошлый раз память кончилась
    # на train» можно было проверить, а не вспоминать.
    keep_days: int = _env_int("HOSTMON_KEEP_DAYS", 30)
    # Точки монтирования под наблюдением. Первая считается корневой: её
    # заполнение показывает карточка сводки.
    mounts: list[str] = field(
        default_factory=lambda: _env_list("HOSTMON_MOUNTS", "/")
    )
    # Опрашивать ли docker. Требует прав на его сокет (на боевом контуре
    # пользователь в группе docker). Опрос идёт из сэмплера в отдельном
    # процессе с таймаутом: `docker stats` без --no-stream висит вечно, и
    # неудача обязана оставаться неудачей одного поля, а не всего замера.
    docker: bool = _env_bool("HOSTMON_DOCKER", True)
    # Сколько процессов показывать в таблицах «съедает CPU» и «съедает RAM».
    top_processes: int = _env_int("HOSTMON_TOP_PROCESSES", 12)


@dataclass(frozen=True)
class AlertsConfig:
    """
    Пороговые уведомления монитора в Telegram.

    Смысл здесь один: узнать о полном диске или подходящем OOM до того, как
    остановится прогон. Поэтому правила описывают ровно те четыре ресурса,
    исчерпание которых на этой машине уже приводило к тихим отказам, и ни
    одного «на всякий случай».

    Антиспам построен на трёх вещах, и убирать их по отдельности нельзя:

    * **cooldown** (`HOSTMON_ALERT_COOLDOWN_MINUTES`) — пока проблема держится,
      напоминание уходит не чаще, чем раз в это окно. Первое сообщение идёт
      сразу.
    * **гистерезис** (`hysteresis`) — «отпустило» считается не по возврату под
      порог, а по возврату ниже `порог − гистерезис`. Метрика, топчущаяся
      вокруг 90%, иначе прислала бы «сработало/отпустило» десятки раз за час.
    * **выдержка** (`sustain`) — сколько замеров подряд нужно держаться за
      порогом. Диск заполняется медленно и монотонно, там достаточно одного
      замера; CPU и load на этой машине штатно уходят в потолок на время
      кластеризации, и мгновенное значение там означало бы уведомление на
      каждый `train`.
    """
    enabled: bool = _env_bool("HOSTMON_ALERTS_ENABLED", True)
    # Канал. Токен бота — у @BotFather, chat_id — id личного чата или группы
    # (у групп он отрицательный). Не заданы — алерты считаются выключенными,
    # и сэмплер говорит об этом один раз при старте, а не молчит.
    bot_token: str = _env("TELEGRAM_BOT_TOKEN")
    chat_id: str = _env("TELEGRAM_CHAT_ID")
    cooldown_minutes: int = _env_int("HOSTMON_ALERT_COOLDOWN_MINUTES", 5)
    hysteresis: float = _env_float("HOSTMON_ALERT_HYSTERESIS", 5.0)
    # Сколько замеров подряд за порогом нужно шумным метрикам (CPU, load).
    sustain: int = _env_int("HOSTMON_ALERT_SUSTAIN_TICKS", 5)
    # Пороги, %. Диск — два: «пора чистить» и «сейчас встанет всё».
    disk_pct: float = _env_float("HOSTMON_ALERT_DISK_PCT", 90.0)
    disk_critical_pct: float = _env_float("HOSTMON_ALERT_DISK_CRITICAL_PCT", 96.0)
    mem_pct: float = _env_float("HOSTMON_ALERT_MEM_PCT", 90.0)
    # Swap отдельным порогом и ниже остальных: на этой машине он не «резерв
    # памяти», а индикатор близкого OOM — расчёт признаков без swap уже
    # ловил killer, и рост swap идёт раньше, чем память покажет 90%.
    swap_pct: float = _env_float("HOSTMON_ALERT_SWAP_PCT", 60.0)
    cpu_pct: float = _env_float("HOSTMON_ALERT_CPU_PCT", 90.0)
    # Load average на ядро: 2.0 означает «на каждое ядро по два ждущих
    # процесса». В абсолютных числах порог был бы привязан к машине.
    load_per_core: float = _env_float("HOSTMON_ALERT_LOAD_PER_CORE", 2.0)
    # Сообщать ли о возврате в норму. Одно сообщение на событие, спама не
    # даёт, зато закрывает вопрос «оно ещё горит или уже нет».
    notify_recovery: bool = _env_bool("HOSTMON_ALERT_NOTIFY_RECOVERY", True)

    @property
    def configured(self) -> bool:
        """Есть ли куда отправлять. Пустой токен — не ошибка, а «канал не заведён»."""
        return bool(self.enabled and self.bot_token and self.chat_id)


data = DataConfig()
features = FeaturesConfig()
db = DBConfig()
runs = RunsConfig()
states = StatesConfig()
candidates = CandidateConfig()
range_forecast = RangeForecastConfig()
smc = SMCConfig()
fgi = FearGreedConfig()
deriv = DerivConfig()
sink = SinkConfig()
notify = NotifyConfig()
admin = AdminConfig()
hostmon = HostmonConfig()
alerts = AlertsConfig()
