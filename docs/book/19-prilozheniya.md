# Глава 19. Приложения

Справочный материал: поля, признаки, атомы, формулы, параметры, команды,
глоссарий и список мест, где ошибка не даёт ошибки.

---

## 19.1. Все 44 поля кандидата

Единственный источник истины — `btc-graph/src/models/candidate.py`.

### Identity

| Поле | Тип | Описание |
|---|---|---|
| `candidate_id` | str | `sha1(symbol, ts, transition, block, offset)[:20]` |
| `symbol` | str | пара; дефолт `BTCUSDT` (отсутствие ловится флагом) |
| `configuration_hash` | str? | `sha1(symbol, transition, block, age_bucket, entropy)[:16]` |
| `candidate_family_key` | str? | `symbol\|group\|transition\|block\|bias\|side` |
| `research_score` | float [0,1] | синтетическая сила аналогии (глава 10.7) |

### State / Trajectory

| Поле | Тип | Описание |
|---|---|---|
| `previous_group_id` | float? | откуда перешёл |
| `current_group_id` | float | куда перешёл |
| `transition_id` | str | `"42->1"` |
| `current_group_age_bucket` | enum | `age_lt_30` / `age_30_60` / `age_60_120` / `age_gt_120` |
| `context_status` | enum | `fresh` (возраст ≤ 30 мин) / `stale` |
| `trajectory_entropy` | enum | `low` (<0.33) / `medium` (<0.66) / `high` |
| `transition_rarity` | enum | `rare` / `uncommon` / `common`, по терцилям частоты |

### Event context

| Поле | Тип | Описание |
|---|---|---|
| `event_block_id` | str | `event_block_NNNNNN`, хэш битовой маски |
| `primary_event_family` | str? | семейство с наибольшим числом атомов в блоке |
| `event_intensity_bucket` | enum | `sparse` (≤2) / `moderate` (3–5) / `dense` (≥6) |
| `event_rarity_bucket` | enum | `rare` (<0.2%) / `uncommon` (<1%) / `common` |
| `signature_atom_count` | int | атомов в маске |
| `event_family_count` | int | различных семейств |
| `event_block_total_rows` | int | сколько баров в истории с этим блоком |
| `event_block_row_share` | float [0,1] | их доля |

### Historical sample

| Поле | Тип | Описание |
|---|---|---|
| `horizon` | str | `"24h"` |
| `sample_size` | int | **строк снимков** в выборке |
| `effective_sample_size` | int? | **независимых реализаций** (≈ `sample_size` / 2.2) |
| `sample_scope` | enum? | `transition+event_block` / `transition` (откат) |
| `valid_label_count` | int | с созревшим исходом |
| `invalid_label_count` | int | без |
| `valid_label_pct` | float [0,1] | доля |
| `repeatability_days` | int | различных дат |
| `repeatability_months` | int | различных месяцев |
| `monthly_concentration` | float [0,1] | доля случаев в самом плотном месяце |

### Outcome profile

| Поле | Тип | Описание |
|---|---|---|
| `historical_bias_context` | enum | `long_skew` / `short_skew` / `neutral` (порог 0.10) |
| `research_side` | enum | `long` / `short` (порог 0.06) |
| `long_outcome_count` | int | случаев роста |
| `short_outcome_count` | int | случаев «не роста» |
| `long_outcome_share` | float [0,1] | доля роста |
| `historical_outcome_skew` | float [−1,1] | $2p_\uparrow - 1$ |

### Favorable / adverse

| Поле | Тип | Описание |
|---|---|---|
| `p70_long_favorable_pct` | float | p70 MFE по случаям роста |
| `p80_long_adverse_pct` | float | p80 просадки по случаям роста |
| `long_favorable_adverse_ratio_p70_p80` | float | их отношение |
| `short_favorable_adverse_ratio_p70_p80` | float? | зеркало по случаям падения |

### Range profile

| Поле | Тип | Описание |
|---|---|---|
| `expected_range_ratio_p50` | float? | медианный прогноз размаха |
| `expected_range_ratio_p90` | float? | верхний квантиль |
| `range_lift` | float? | **отношение к бенчмарку** — главное поле |
| `range_regime` | enum? | `compressed` (<0.85) / `normal` / `expanded` (>1.15) |

Все четыре `None`, если модели нет, она не прошла гейт или бар лежит до конца её
обучения.

### Служебное (не уходит)

`_meta`: `ts`, `group_label`, `age_minutes`, `offset_min`, `avg_ret_pct`.

---

## 19.2. Все 32 признака

Окна в базовых барах при 15m: $w_{1h}=4$, $w_{4h}=16$, $w_{1d}=96$,
$w_{1w}=672$, $w_{1m}=2688$.

| # | Признак | Формула | Ось словаря |
|---|---|---|---|
| 1–4 | `ret_1h/4h/1d/1w` | $(c_t - c_{t-n}) / (\text{rv}_d\sqrt{n})$ | тренд |
| 5 | `rv_ratio` | $\ln(\text{rv}_d / \text{rv}_w)$ | волатильность |
| 6 | `rv_rank` | $\text{rank}(\text{rv}_d, w_{1m})$ | волатильность |
| 7 | `atr_rank` | $\text{rank}(\text{ATR}_{14}/C, w_{1m})$ | волатильность |
| 8 | `range_exp` | $\ln\!\big(\text{clip}((H-L)/\overline{(H-L)}_{1d}, \ge 10^{-3})\big)$ | волатильность |
| 9–11 | `pos_1d/1w/1m` | $\text{pos}(w)$ | положение |
| 12 | `dd_from_high` | $(C - \max H_{1m}) / \text{ATR}_{14}$ | положение |
| 13 | `dist_ema_1d` | $(C - \text{EMA}_{1d}) / \text{ATR}_{14}$ | тренд |
| 14 | `dist_ema_1w` | $(C - \text{EMA}_{1w}) / \text{ATR}_{14}$ | тренд |
| 15 | `slope_ema_1d` | $(\text{EMA}_{1d}(t) - \text{EMA}_{1d}(t{-}w_{1d})) / \text{ATR}_{14}$ | тренд |
| 16 | `slope_ema_1w` | аналогично на неделе | тренд |
| 17 | `trend_align` | $(\text{EMA}_{1d} - \text{EMA}_{1w}) / \text{ATR}_{14}$ | тренд |
| 18 | `vol_z` | $z(\ln(1+V), w_{1d})$ | объём |
| 19 | `vol_ratio` | $\ln(\overline{V}_{1d} / \overline{V}_{1w})$ | объём |
| 20 | `taker_bias` | $\overline{(V^{\text{tb}}/V)}_{1d} - 0.5$ | поток |
| 21 | `rsi` | $\text{RSI}_{14}/100$ | импульс |
| 22 | `up_bar_share` | $\overline{\mathbb{1}[C>O]}_{1d}$ | импульс |
| 23 | `ret_skew` | $\text{skew}(\Delta c, w_{1d})$ | форма |
| 24–26 | `tf1h_rsi/pos/dist_ema` | на барах 1h, `shift(1)` | по смыслу |
| 27–29 | `tf4h_rsi/pos/dist_ema` | на барах 4h, `shift(1)` | по смыслу |
| 30–32 | `tf1d_rsi/pos/dist_ema` | на барах 1d, `shift(1)` | по смыслу |

Метка набора `v1`. С признаками SMC было бы 44 и метка `v1+smc`.

---

## 19.3. Все 53 атома

### Signature (20) — входят в `event_block_id`

| Бит | Атом | Русская подпись | Семейство |
|---|---|---|---|
| 0 | `breakout_1d_high` | пробой максимума суток | price_action |
| 1 | `breakdown_1d_low` | пробой минимума суток | price_action |
| 2 | `breakout_1w_high` | пробой максимума недели | price_action |
| 3 | `breakdown_1w_low` | пробой минимума недели | price_action |
| 4 | `wide_range_bar` | широкий бар | price_action |
| 5 | `inside_bar` | внутренний бар | price_action |
| 6 | `vol_expansion` | расширение волатильности | volatility |
| 7 | `vol_contraction` | сжатие волатильности | volatility |
| 8 | `atr_spike` | всплеск ATR | volatility |
| 9 | `volume_spike` | всплеск объёма | volume |
| 10 | `volume_dry` | объём пересох | volume |
| 11 | `ema_cross_up` | золотое пересечение EMA | trend |
| 12 | `ema_cross_down` | мёртвое пересечение EMA | trend |
| 13 | `rsi_overbought` | RSI в перекупленности | momentum |
| 14 | `rsi_oversold` | RSI в перепроданности | momentum |
| 15 | `momentum_reversal_up` | разворот импульса вверх | momentum |
| 16 | `momentum_reversal_down` | разворот импульса вниз | momentum |
| 17 | `at_range_high` | у верха диапазона | zone_context |
| 18 | `at_range_low` | у низа диапазона | zone_context |
| 19 | `round_level_touch` | у круглого уровня | zone_context |

> ⚠️ Первые 20 битов прибиты тестом `test_signature_bits_are_pinned`. Новые
> атомы — **только в конец** `ATOM_FAMILY`.

### Context (33) — не входят в маску

**Базовые (9):** `trend_up_align` (тренд согласован вверх),
`trend_down_align` (вниз), `mid_range` (середина диапазона),
`taker_buy_dominance` (давят покупатели), `taker_sell_dominance` (давят
продавцы), `asia_session`, `europe_session`, `us_session`, `weekend`.

**Smart Money (16):**

| Атом | Подпись |
|---|---|
| `bos_up` / `bos_down` | слом структуры вверх / вниз |
| `choch_up` / `choch_down` | смена характера вверх / вниз |
| `structure_bullish` | бычья структура |
| `fvg_formed_large` | образовался крупный имбаланс |
| `in_unfilled_fvg` | внутри незакрытого имбаланса |
| `sweep_high` / `sweep_low` | снятие ликвидности сверху / снизу |
| `sweep_high_reclaim` / `sweep_low_reclaim` | снятие с возвратом |
| `in_bullish_ob` / `in_bearish_ob` | в блоке заказов |
| `in_breaker` | в брейкер-блоке |
| `in_discount` / `in_premium` | в зоне скидки / премиальной |

**Fear & Greed (4):** `fear_extreme` (крайний страх), `greed_extreme` (крайняя
жадность), `sentiment_flip_up`, `sentiment_flip_down`.

**Деривативы (4):** `oi_up_price_up` (набор лонгов), `oi_up_price_down` (набор
шортов), `oi_down_price_up` (закрытие шортов), `oi_down_price_down` (закрытие
лонгов).

---

## 19.4. Все формулы в одном месте

### Индикаторы

$$\text{EMA}_t = \alpha x_t + (1-\alpha)\text{EMA}_{t-1},\quad \alpha = \tfrac{2}{s+1}$$

$$\text{RSI} = 100 - \frac{100}{1 + \overline{\text{gain}}/\overline{\text{loss}}}$$

$$\text{TR}_t = \max(H_t - L_t,\ |H_t - C_{t-1}|,\ |L_t - C_{t-1}|)$$

$$\text{rv}(w) = \operatorname{std}(\Delta \ln C,\ w)$$

$$\text{pos}(w) = \operatorname{clip}\!\left(\frac{C - \min L_w}{\max H_w - \min L_w}, 0, 1\right)$$

### Кластеризация

$$\text{gain} = s_{\text{real}} - \overline{s_{\text{ref}}}
- k\,\sigma_{\text{ref}}\sqrt{1 + 1/B}$$

$$d' = \frac{\|c_B - c_A\|}{\sigma_A^{\parallel} + \sigma_B^{\parallel}}$$

$$\|x-c\|^2 = \|x\|^2 - 2\langle x,c\rangle + \|c\|^2$$

$$\text{min\_group}_{\text{эфф}} = \max(300,\ \operatorname{round}(0.0025 N))$$

### Разметка

$$H_t = -\frac{1}{\ln\min(w,K)}\sum_g p_g \ln p_g$$

### Исходы

$$\text{ret\_pct} = (C_{t+H}/C_t - 1)\cdot 100$$

$$\text{range\_ratio} = \frac{\max H - \min L}{\text{ATR}_{14}\sqrt{H}}$$

### Кандидат

$$\text{research\_score} = 0.25 s_{\text{skew}} + 0.20 s_{\text{sample}}
+ 0.15 s_{\text{repeat}} + 0.15 s_{\text{spread}}
+ 0.15 s_{\text{valid}} + 0.10 s_{\text{rare}}$$

$$s_{\text{sample}} = \min\!\left(1, \frac{\log_{10}(n_{\text{эфф}}+1)}{3}\right)$$

### Оценка

$$Q = 0.30 A + 0.35 B + 0.20 C + 0.15 D$$

### Статистика

$$\text{DEFF} = 1 + (\bar m - 1)\rho, \qquad
n_{\text{эфф}} = n/\text{DEFF}$$

$$p_{\text{boot}} = \frac{1+k}{1+B}$$

$$\text{BH}: k^* = \max\{k : p_{(k)} \le \tfrac{k}{m}\alpha\}$$

$$R^2_{\text{OOS}} = 1 - \frac{\sum(y-\hat y)^2}{\sum(y - \bar y_{\text{train}})^2}$$

$$\Delta R^2 = (1 - R^2_{\text{base}})\cdot\operatorname{corr}(y^\perp, p^\perp)^2$$

### Размах

$$L_\tau(y,\hat y) = \max\!\big(\tau(y-\hat y),\ (\tau-1)(y-\hat y)\big)$$

$$\text{range\_lift} = p_{50}^{\text{модель}} / p_{50}^{\text{бенчмарк}}$$

$$\sigma^2_P = \frac{(\ln H/L)^2}{4\ln 2}$$

$$\sigma^2_{GK} = \tfrac12(\ln H/L)^2 - (2\ln 2 - 1)(\ln C/O)^2$$

$$\sigma^2_{RS} = \ln\tfrac{H}{C}\ln\tfrac{H}{O} + \ln\tfrac{L}{C}\ln\tfrac{L}{O}$$

### Поперечное сечение

$$\text{xs\_fwd\_ret}_i = r_i - \frac{1}{|B|}\sum_{j\in B} r_j$$

$$\text{IC}(t) = \operatorname{corr}_{\text{Sp}}(x_i(t), y_i(t))_{i \in B_t}$$

---

## 19.5. Параметры конфигурации

### Данные (`DataConfig`)

| Переменная | Дефолт | Смысл |
|---|---|---|
| `SYMBOL` | BTCUSDT | монета по умолчанию |
| `BASE_TIMEFRAME` | 15m | базовый бар |
| `CONTEXT_TIMEFRAMES` | 1h,4h,1d | старшие ТФ |
| `HORIZON` | 24h | горизонт исходов |
| `OUTCOME_EXTRA_HORIZONS` | 4h,12h | дополнительные, только на склад |
| `HISTORY_START` | 2017-08-01 | дефолт для монет без даты в реестре |
| `BINANCE_REST_URL` | api.binance.com | **менять на data-api при 451** |

### Состояния (`StatesConfig`)

| Переменная | Дефолт | Смысл |
|---|---|---|
| `STATES_SEED_CLUSTERS` | 8 | стартовое разбиение |
| `STATES_MIN_GROUP_SHARE` | 0.0025 | доля истории — основной порог |
| `STATES_MIN_GROUP_SIZE` | 300 | абсолютный пол |
| `STATES_MAX_DEPTH` | 4 | глубина рекурсии |
| `STATES_SPLIT_REFERENCE_DRAWS` | 10 | число референсов $B$ |
| `STATES_SPLIT_GAIN_SIGMA` | 2.0 | порог в сигмах |
| `STATES_MERGE_SEPARATION` | 1.0 | порог d-prime |
| `STATES_SMOOTHING_BARS` | 2 | подтверждение смены |
| `STATES_TRAJECTORY_WINDOW` | 24 | окно энтропии |
| `STATES_SILHOUETTE_SAMPLE` | 4000 | подвыборка для силуэта |

### Кандидаты (`CandidateConfig`)

| Переменная | Дефолт | Смысл |
|---|---|---|
| `CAND_MIN_SAMPLE_SIZE` | 30 | строк снимков (мягкий) |
| `CAND_MIN_EFFECTIVE_SAMPLE` | 30 | независимых случаев (рабочий) |
| `CAND_MIN_ABS_SKEW` | 0.06 | без перекоса кандидата нет |
| `CAND_BIAS_SKEW` | 0.10 | порог `historical_bias_context` |
| `CAND_FAVORABLE_PCT` | 70.0 | перцентиль выгоды |
| `CAND_ADVERSE_PCT` | 80.0 | перцентиль риска |
| `CAND_FRESH_LAG_MIN` | 30 | граница `fresh` / `stale` |
| `CAND_FALLBACK_TRANSITION` | true | откат на выборку по переходу |

### Размах (`RangeForecastConfig`)

| Переменная | Дефолт | Смысл |
|---|---|---|
| `RANGE_FORECAST_ENABLED` | false | обучение в `train` |
| `RANGE_FORECAST_NORM` | atr14 | нормировка цели |
| `RANGE_FORECAST_SEED` | 42 | зерно бустинга |
| `RANGE_FORECAST_TRAIN_FRAC` | 0.7 | доля под обучение в гейте |
| `RANGE_FORECAST_GATE_BOOT` | 500 | реплик бутстрапа в гейте |

Плюс константы модуля: `MIN_DELTA_R2 = 0.02`, `MAX_COVERAGE_ERROR = 0.05`,
`REGIME_EDGES = (0.85, 1.15)`, `QUANTILES = (0.25, 0.5, 0.75, 0.9)`.

### Источники

| Переменная | Дефолт | Что включает |
|---|---|---|
| `SMC_ENABLED` / `SMC_FEATURES_ENABLED` | false | 16 атомов / 12 признаков |
| `FGI_ENABLED` / `FGI_FEATURES_ENABLED` | false | 4 атома / 2 признака |
| `DERIV_ENABLED` / `DERIV_FEATURES_ENABLED` | false | 4 атома / 6 признаков |
| `FEATURES_MAX_SOURCE_ROW_LOSS` | 0.02 | предохранитель на потерю истории |

### Доставка и приёмник

| Переменная | Дефолт | Смысл |
|---|---|---|
| `SINK_MODE` | direct | `direct` / `http` / `none` |
| `SINK_BATCH_SIZE` | 200 | размер пачки |
| `SINK_MIN_QUALITY` | не задан | **не задавать**: пусть решает профиль |
| `BTC_GRAPH_PATH` | вычисляется | каталог приёмника |
| `USE_DB` / `USE_REDIS` / `USE_GRAPH` | true | хранилища, читаются на импорте |
| `PERSIST_CHUNK_SIZE` | 500 | пачка записи |
| `ENABLE_CONFIG_RELOAD` | false | ручка hot-reload профилей |

### Уведомления и мониторинг

`NOTIFY_ENABLED`, `NOTIFY_MAX_CANDIDATE_AGE_MIN` (180),
`HOSTMON_ALERT_*`, `ADMIN_PG_SLOW_SECONDS`, `ADMIN_MAX_CONCURRENT_RUNS`,
`RUN_STALE_AFTER_MINUTES` (120).

---

## 19.6. Места, где ошибка не даёт ошибки

Сводный список ⚠️ из книги. Каждый пункт — способ получить неправильные числа
без исключения и без падения теста.

| # | Место | Как ломается | Защита |
|---|---|---|---|
| 1 | `configuration_hash` без символа | ETH получает оценку BTC из кэша | `test_configuration_hash_includes_symbol` |
| 2 | Ключ семьи без символа | одна монета гасит другую | группировка по `(symbol, key)` |
| 3 | Ключ узла Neo4j без символа | узлы схлопываются | constraint `(symbol, group_id)` |
| 4 | `train` без очистки графа | счётчики поверх прошлого поколения | ошибка очистки роняет прогон |
| 5 | Порядок битов `ATOM_FAMILY` | все исторические блоки меняют смысл | `test_signature_bits_are_pinned` |
| 6 | Контекстные атомы в маске | выборка по блоку не набирается | `test_context_atoms_do_not_change_block_id` |
| 7 | Признак без строки в `naming.AXES` | подпись состояния врёт | `test_every_feature_has_a_phrase` |
| 8 | Метка набора признаков вручную | старые массивы под чужими именами | `test_two_sources_never_share_a_label` |
| 9 | Разметка членством вместо предикта | расхождение 21–28% баров | `test_fit_labels_equal_predict` |
| 10 | `abs(mae)` в накопителе | чистый рост = большая «просадка» | правка + замер |
| 11 | Ступень `effective` к `sample_size` | оценка смещена вдвое | проверка «обе стороны» |
| 12 | Правка `_default.yaml` | все baseline обесценены | снапшот-тест на 200 кандидатах |
| 13 | Правка порога без бампа версии | записи смешиваются в средних | `profile_fingerprint` |
| 14 | `run_id` вместо модели в агрегатах | смешение шести моделей | `runs.model_run_scope()` |
| 15 | Флаг источника без крона | ряд протухает, атомы `False` | — (известное) |
| 16 | Воркер без `-Q evaluate,celery` | задачи копятся в Redis | — (документировано) |
| 17 | `ALTER SYSTEM` для памяти Postgres | правка не действует | `command` в compose |
| 18 | Уборка «оставить N последних live» | сносит разметку хвоста | правило переписано |
| 19 | `TRAIN_SYMBOLS` как «что доучить» | монеты выпадают из крона | сверка + `--drop-symbols` |
| 20 | Обучающая матрица в `live` | пустой прогноз на свежем баре | `input_matrix` отдельно |
| 21 | Числа модели размаха до `train_end` | запоминание вместо прогноза | срез по `train_end` |
| 22 | Выгрузка без `ORDER BY ts` | holdout становится случайной половиной | — (документировано) |
| 23 | `is_up = None` без отсева | «ошиблись» вместо «факта нет» | обязанность вызывающего |
| 24 | `bool(NaN)` после reindex | невалидный снимок в выборке | `fillna(False)` |
| 25 | Ключ словаря `items` в Jinja | страница падает | конвенция |
| 26 | Неверный `BTC_GRAPH_PATH` | тест схемы **скипается** | дефолт от `config.py` |
| 27 | Кэши источников в двух присваиваниях | монета получает чужие метрики | один слот с кортежем |

---

## 19.7. Команды

### Генератор

```bash
make init-db          # схема processing
make train            # полный прогон (30–60 мин)
make train-all        # по всем активным монетам
make train-fast       # пересчёт по данным из БД, без отправки
make live             # инкрементальный прогон
make live-all
make status           # покрытие, отставание, доступность приёмника
make admin            # админка на 127.0.0.1:8100
make test
make fetch-external   # Fear & Greed → external_daily
make ingest-metrics   # деривативы → deriv_metrics
make maintenance      # уборка + вакуум
make hostmon          # монитор машины
```

Всё это обёртки над `python3 -m btcproc.cli <команда>`. Интерпретатор задаётся
через `make train PY=.venv/bin/python`. У команд с данными есть `--symbol`
(можно несколько раз) и `--all`.

### Замеры

```bash
make validate-holdout   # система целиком на отложенной части
make control-model      # бустинг без графа
make measure-range      # что угодно против размаха
make validate-range     # конфигурация против размаха
make range-forecast     # квантильный регрессор
make measure-candle     # range-оценщики дисперсии
```

### Приёмник

```bash
make up                 # весь стек
make migrate            # alembic upgrade head
make build              # ПОСЛЕ ЛЮБОЙ ПРАВКИ src/ (uvicorn без --reload)
make reload             # после правки .env или compose
make logs / ps / down
make shell-api / shell-pg
make test-local         # pytest локально, без инфраструктуры
make profiles-check     # схема YAML, монотонность, полнота карт
make migrate-graph      # РАЗОВО: symbol узлам Neo4j
```

### Развёртывание

```bash
./01_deploy.sh                        # инфраструктура + код
./01_deploy.sh --code-only            # только выкатка кода
./01_deploy.sh --code-only --no-test  # без тестов на сервере
./02_configure.sh                     # конфигурация + запуск
./02_configure.sh --nginx-only        # только прокси
./02_configure.sh --hostmon-only      # только монитор
```

### Быстрая проверка без полного прогона

```bash
python3 -m btcproc.cli ingest --start 2024-06-01
python3 -m btcproc.cli train --no-ingest --no-emit --start 2024-06-01
```

---

## 19.8. Глоссарий

**Атом** — булев детектор события на баре. 53 штуки в 13 семействах.

**Блок событий** (`event_block_id`) — хэш битовой маски активных
signature-атомов в часовом окне.

**Возраст состояния** (`age_bucket`) — сколько времени рынок в текущем
состоянии: `age_lt_30` … `age_gt_120`.

**Гейт** — критерий годности, объявленный до запуска замера.

**Гранулярность** — число состояний в графе. Характеристика прогона, а не рынка.

**Кандидат** — досье об исторической аналогии, 44 поля.

**Контекстный атом** — атом, не входящий в маску блока. 33 штуки.

**Лифт** — разница долей успеха при активном атоме и без него.

**Модель состояний** (`StateModel`) — набор центроидов плюс параметры
нормировки.

**Перекос** (`skew`) — $2p_\uparrow - 1$, сила отклонения доли роста от половины.

**Профиль** — YAML-файл калибровки оценки для одной монеты.

**Реализация перехода** (`state_seq`) — одно непрерывное пребывание в состоянии.
Единица счёта `effective_sample_size`.

**Редкость** — категория `rare` / `uncommon` / `common`. У переходов по
терцилям частоты, у блоков по абсолютным долям.

**Свежесть** (`context_status`) — `fresh`, если снимку не больше 30 минут.

**Signature-атом** — атом, входящий в маску блока. 20 штук.

**Снимок** — момент выпуска кандидата: переход плюс офсет 0/45/90/180 минут.

**Состояние** (`group_id`) — номер кластера в пространстве признаков. Осмысленен
только в паре `(symbol, run_id)`.

**Траекторная энтропия** — нормированная энтропия Шеннона по окну последних 24
состояний.

**Эффективный размер выборки** — число независимых реализаций перехода. Примерно
вдвое меньше `sample_size`.

**B0 / B1 / B2** — уровни бенчмарка размаха: константа, HAR-RV, HAR-RV плюс
сезонность.

**DEFF** — design effect, во сколько раз зависимость наблюдений сокращает
эффективный размер выборки.

**IC** — information coefficient, ранговая корреляция предиктора с целью внутри
одного среза.

**MFE / MAE** — maximum favorable / adverse excursion, максимальное движение за
и против.

**`range_lift`** — отношение прогноза размаха к прогнозу бенчмарка. Главное поле
размаха.

**`range_ratio`** — размах за горизонт, нормированный на $\text{ATR}_{14}\sqrt{H}$.

**`research_score`** — синтетическая сила аналогии. **Не вероятность.**

**`quality_score`** — оценка приёмника, сравнима только внутри монеты.

**`quality_score_baseline`** — та же оценка по неподвижной общей мерке.
Единственное межмонетно сравнимое число.

---

## 19.9. Карта документации

| Документ | Что в нём |
|---|---|
| `README.md` (корень) | карта репозитория |
| `CLAUDE.md` (корень + подпроекты) | правила работы с кодом, инварианты |
| `btc-graph/README.md` | 50 КБ: словарь, формула score, API, сценарии |
| `btc-graph/README_agent_spec.md` | исходное ТЗ |
| `btc-graph-processing/README.md` | 96 КБ: архитектура и обоснования |
| `btc-graph-processing/docs/operator_guide.md` | эксплуатация без кода |
| `btc-graph-processing/docs/development_log.md` | **журнал решений, 54 раздела** |
| `btc-graph/docs/development_log.md` | журнал решений приёмника, 11 разделов |
| `btc-graph-processing/docs/extending_features.md` | как завести новый источник |
| `btc-graph-processing/docs/notifications.md` | внешний контракт вебхуков |
| `docs/tz_*.md` | технические задания замеров с критериями |
| `docs/audit_*.md` | внешние аудиты и разбор замечаний |
| `docs/retrospective_2026-08-21.md` | сверка ТЗ с построенным, для бизнеса |
| `deploy/README.md` | развёртывание |
| `.claude/skills/` | процедуры: добавление монеты, проверка системы |

> **Историю решений искать в журналах, а не в коде.** Оба проекта ведут
> `development_log.md`: что менялось, почему именно так, что было отвергнуто и
> какие выводы впоследствии снимались.

---

## 19.10. Быстрый старт для нового разработчика

Порядок, в котором стоит входить в проект.

**День 1.** Прочитать главы 1, 2 этой книги и `README.md` корня. Поднять стек
локально:

```bash
cd btc-graph && make up && sleep 15 && make migrate
cd ../btc-graph-processing && make init-db
make status
```

**День 2.** Прогнать тесты обоих проектов, посмотреть, что они проверяют.
Прочитать `CLAUDE.md` обоих подпроектов — там инварианты.

```bash
cd btc-graph && make test-local
cd ../btc-graph-processing && make test
FGI_ENABLED=true SMC_ENABLED=true python3 -m pytest
```

**День 3.** Быстрый прогон на коротком отрезке, разбор результата в админке:

```bash
python3 -m btcproc.cli ingest --start 2024-06-01
python3 -m btcproc.cli train --no-ingest --no-emit --start 2024-06-01
make admin   # 127.0.0.1:8100
```

**День 4.** Главы 7 и 12 книги — центральный алгоритм и методика проверки.
Раздел 26 журнала генератора (валидация на отложенной части) целиком.

**Дальше.** По задаче: правишь признаки — глава 5; правишь оценку — глава 11;
ставишь замер — глава 12 и `docs/tz_*.md` как образец постановки.

**Три правила, которые стоит запомнить до первой правки:**

1. историю решений искать в журналах;
2. если что-то может сломаться без ошибки — оно обязано либо падать, либо
   печатать причину;
3. критерий замера объявляется до запуска.
