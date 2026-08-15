"""
Тесты детекторов Smart Money.

Главное, что здесь проверяется, — отсутствие заглядывания вперёд. Детектор,
подсмотревший будущее, не падает и не выглядит странно: он просто делает всю
историческую статистику фикцией, а обнаруживается это уже на реальных
деньгах. Поэтому проверок на это три, и они устроены одинаково: величина,
посчитанная на префиксе истории, обязана совпасть с посчитанной на полной.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcproc import config
from btcproc.features import smc
from tests.conftest import disable_all_sources


def make_frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Бары из кортежей (open, high, low, close) с минутным индексом."""
    index = pd.date_range("2024-01-01", periods=len(rows), freq="15min", tz="UTC")
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)
    frame["volume"] = 100.0
    return frame


def zigzag(n: int = 600, amplitude: float = 0.004, period: int = 40,
           swing: float = 0.03, growth: float = 0.0) -> pd.DataFrame:
    """
    Пила на медленной волне: даёт свинги, сломы структуры в обе стороны и
    имбалансы.

    Синус вместо случайного блуждания — чтобы экстремумы были в заранее
    известных местах и тест не зависел от зерна генератора.

    Медленная составляющая (`swing`, период в десять раз длиннее) обязательна:
    без неё ряд только растёт, подтверждённые минимумы всё время поднимаются,
    цена никогда не проваливается ниже них — и медвежьей структуры не
    возникает вовсе. Тогда нет ни CHoCH вверх, ни брейкеров, потому что оба
    требуют предшествующего разворота.

    Все размахи заданы **в долях цены**, а рост `growth` — экспоненциальный.
    Это принципиально для теста стационарности: при линейном дрейфе с
    постоянным абсолютным размахом ATR не растёт вместе с ценой, и любая
    величина в ATR честно уезжает в потолок. Так ведёт себя не рынок, а
    неудачная синтетика — рост цены вдвое сопровождается ростом размаха вдвое.
    """
    i = np.arange(n)
    level = 10_000.0 * np.exp(growth * i)
    shape = (
        1.0
        + swing * np.sin(2 * np.pi * i / (period * 10))
        + amplitude * np.sin(2 * np.pi * i / period)
    )
    base = level * shape
    high = base * (1.0 + amplitude * 0.25)
    low = base * (1.0 - amplitude * 0.25)
    open_ = np.concatenate([[base[0]], base[:-1]])
    return make_frame(list(zip(open_, high, low, base)))


# ── Свинги ──────────────────────────────────────────────────────────────────
def test_swings_do_not_look_ahead():
    """
    Свинг на баре i известен только на баре i + right.

    Считаем на полной истории и на её префиксе: общие бары обязаны совпасть
    полностью. Если бы `rolling(center=True)` использовался без `shift`,
    последние `right` баров префикса разошлись бы с полной историей.
    """
    bars = zigzag(600)
    cut = 400
    full = smc.confirmed_swings(bars, left=3, right=3)
    prefix = smc.confirmed_swings(bars.iloc[:cut], left=3, right=3)

    common = prefix.index
    pd.testing.assert_frame_equal(full.loc[common], prefix.loc[common])


def test_swings_lag_by_right_bars():
    """Экстремум не становится свингом раньше, чем через right баров."""
    rows = [(100, 101, 99, 100)] * 5
    rows.append((100, 120, 99, 119))            # бар 5 — очевидный максимум
    rows += [(100, 101, 99, 100)] * 10
    bars = make_frame(rows)

    swings = smc.confirmed_swings(bars, left=3, right=3)
    highs = swings["swing_high"]
    # На баре 5 и двух следующих свинга ещё нет, на 5 + 3 = 8 он появляется.
    assert not (highs.iloc[5:8] == 120).any()
    assert highs.iloc[8] == 120


def test_swings_are_flat_between_confirmations():
    """Между подтверждениями держится последний известный экстремум, не NaN."""
    swings = smc.confirmed_swings(zigzag(400), left=3, right=3)
    tail = swings.iloc[100:]
    assert tail.notna().all().all()


# ── FVG ─────────────────────────────────────────────────────────────────────
def test_fvg_detection_on_synthetic():
    """Заведомый разрыв найден, границы и размер совпадают с построенными."""
    rows = [(100, 102, 98, 100)] * 10
    # Бары 10, 11, 12: low[12] = 108 > high[10] = 102 → бычий разрыв 6.
    rows += [(100, 102, 98, 101), (101, 110, 100, 109), (109, 112, 108, 111)]
    rows += [(111, 112, 110, 111)] * 5
    bars = make_frame(rows)

    atr = pd.Series(2.0, index=bars.index)
    fvg = smc.detect_fvg(bars, atr)

    assert fvg["bull_fvg"].sum() == 1
    hit = fvg[fvg["bull_fvg"]].iloc[0]
    assert hit["bull_lo"] == 102          # high[i-2]
    assert hit["bull_hi"] == 108          # low[i]
    assert hit["bull_atr"] == pytest.approx(3.0)   # разрыв 6 при ATR 2
    assert not fvg["bear_fvg"].any()


def test_fvg_detection_finds_bearish_gap():
    rows = [(100, 102, 98, 100)] * 10
    # low[10] = 98, high[12] = 92 → медвежий разрыв 6.
    rows += [(100, 102, 98, 99), (99, 100, 90, 91), (91, 92, 88, 90)]
    rows += [(90, 92, 88, 90)] * 5
    bars = make_frame(rows)

    fvg = smc.detect_fvg(bars, pd.Series(2.0, index=bars.index))
    assert fvg["bear_fvg"].sum() == 1
    hit = fvg[fvg["bear_fvg"]].iloc[0]
    assert hit["bear_lo"] == 92           # high[i]
    assert hit["bear_hi"] == 98           # low[i-2]
    assert hit["bear_atr"] == pytest.approx(3.0)


def test_fvg_fill_uses_only_past():
    """
    Статус заполнения на баре t не переписывается барами после t.

    Реестр обновляется прямым проходом, поэтому доля заполнения обязана
    совпасть на префиксе и на полной истории.
    """
    bars = zigzag(500)
    cut = 300
    full = smc.build_smc(bars)
    prefix = smc.build_smc(bars.iloc[:cut])

    for column in ("fvg_fill_pct", "in_unfilled_fvg", "near_fvg_bull"):
        pd.testing.assert_series_equal(
            full[column].iloc[:cut], prefix[column], check_names=False
        )


def test_fvg_large_threshold_is_measured_in_atr():
    """Порог «крупного» разрыва — в ATR, а не в деньгах."""
    rows = [(100, 102, 98, 100)] * 10
    rows += [(100, 102, 98, 101), (101, 110, 100, 109), (109, 112, 108, 111)]
    rows += [(111, 112, 110, 111)] * 10
    bars = make_frame(rows)

    strict = config.SMCConfig(fvg_min_atr=100.0)
    loose = config.SMCConfig(fvg_min_atr=0.01)
    assert not smc.build_smc(bars, strict)["fvg_formed_large"].any()
    assert smc.build_smc(bars, loose)["fvg_formed_large"].any()


# ── Структура и ордер-блоки ─────────────────────────────────────────────────
def test_bos_requires_confirmed_swing():
    """
    Слом структуры не может опередить подтверждение свинга.

    Первое срабатывание обязано отстоять от начала истории минимум на
    right баров — раньше подтверждённого экстремума просто не существует.
    """
    cfg = config.SMCConfig(swing_left=3, swing_right=3)
    frame = smc.build_smc(zigzag(400), cfg)
    breaks = frame["bos_up"] | frame["bos_down"] | frame["choch_up"] | frame["choch_down"]
    assert breaks.any(), "на пиле сломы структуры обязаны быть"
    assert int(np.argmax(breaks.to_numpy())) >= cfg.swing_right


def test_structure_flags_are_exclusive():
    frame = smc.build_smc(zigzag(400))
    assert not (frame["structure_bullish"] & frame["structure_bearish"]).any()


def test_choch_flips_structure():
    """CHoCH — слом против текущей структуры, то есть смена направления."""
    frame = smc.build_smc(zigzag(600))
    choch_up = frame.index[frame["choch_up"]]
    assert len(choch_up) > 0
    for ts in choch_up[:5]:
        position = frame.index.get_loc(ts)
        # До разворота структура была медвежьей, после — бычья.
        assert frame["structure_bearish"].iloc[position - 1]
        assert frame["structure_bullish"].iloc[position]


def test_ob_depends_on_bos():
    """
    Без слома структуры ордер-блоков не возникает.

    Плоский рынок не даёт подтверждённых пробоев, значит ни один бар не может
    оказаться «внутри OB».
    """
    flat = make_frame([(100, 100.5, 99.5, 100)] * 300)
    frame = smc.build_smc(flat)
    assert not frame["bos_up"].any() and not frame["bos_down"].any()
    assert not frame["in_bullish_ob"].any()
    assert not frame["in_bearish_ob"].any()
    assert not frame["in_breaker"].any()


def test_breaker_is_flipped_ob():
    """Блок, пробитый против себя, меняет направление и становится брейкером."""
    obs: list[list] = []
    op = np.array([100.0, 100.0])
    cl = np.array([95.0, 95.0])       # медвежья свеча
    lo = np.array([94.0, 94.0])
    hi = np.array([101.0, 101.0])
    smc._create_order_block(obs, 1, op, cl, lo, hi, direction=1, lookback=5)

    assert len(obs) == 1
    block = obs[0]
    assert block[smc._OB_DIR] == 1 and not block[smc._OB_BREAKER]

    # Цена ушла ниже зоны — бычий блок становится медвежьим брейкером.
    if block[smc._OB_DIR] > 0 and 90.0 < block[smc._OB_LO]:
        block[smc._OB_DIR], block[smc._OB_BREAKER] = -1, True
    assert block[smc._OB_DIR] == -1 and block[smc._OB_BREAKER]


def test_breaker_appears_in_full_pass():
    """Тот же переворот, но через build_smc, а не ручной правкой реестра."""
    frame = smc.build_smc(zigzag(800))
    assert frame["in_breaker"].any()


# ── Уровни и ликвидность ────────────────────────────────────────────────────
def test_equal_levels_merge_within_tolerance():
    levels: list[list] = []
    smc._register_level(levels, 100.0, t=0, tolerance=1.0)
    smc._register_level(levels, 100.5, t=5, tolerance=1.0)   # в допуске
    smc._register_level(levels, 110.0, t=9, tolerance=1.0)   # вне допуска

    assert len(levels) == 2
    assert levels[0][smc._LV_STRENGTH] == 2
    assert levels[0][smc._LV_PRICE] == pytest.approx(100.25)
    assert levels[1][smc._LV_STRENGTH] == 1


def test_new_swing_restores_liquidity_of_a_swept_level():
    """Свежий свинг на уровне — новые стопы за ним, снятие засчитывается снова."""
    levels = [[100.0, 2, 0, 7]]                    # уровень уже снят на баре 7
    smc._register_level(levels, 100.0, t=20, tolerance=1.0)
    assert levels[0][smc._LV_SWEPT] == -1


def test_reclaim_is_rarer_than_naked_sweep():
    """Снятие с возвратом — подмножество снятий, иначе детектор бессмыслен."""
    frame = smc.build_smc(zigzag(800))
    assert frame["sweep_high"].sum() >= frame["sweep_high_reclaim"].sum()
    assert frame["sweep_low"].sum() >= frame["sweep_low_reclaim"].sum()


# ── Стационарность и границы ────────────────────────────────────────────────
def test_smc_features_are_scale_invariant():
    """
    Инвариант I7: детектор зависит только от ATR и долей, но не от цены.

    Тот же самый рынок, умноженный на 128, обязан дать **побитово** те же
    величины: ATR масштабируется вместе с ценой, поэтому всякий порог,
    выраженный в ATR, переживает смену масштаба. Любое расхождение означает
    порог, зашитый в деньгах, — и такой детектор на SOL меряет не то же, что
    на BTC.

    Множитель — степень двойки, и это не придирка: умножение на 128 меняет
    только экспоненту и сохраняет все сравнения точно. При множителе вроде 137
    тест ловил бы вдобавок разъезд округления на выборе ближайшей из двух
    равноудалённых зон — на идеально периодической синтетике такие совпадения
    регулярны, на рынке не встречаются. Порог, заданный в деньгах, ×128
    поймает ровно так же.
    """
    cheap = zigzag(1500)
    expensive = cheap.copy()
    for column in ("open", "high", "low", "close"):
        expensive[column] *= 128.0

    pd.testing.assert_frame_equal(
        smc.build_smc(cheap), smc.build_smc(expensive), rtol=1e-9
    )


def test_smc_features_do_not_drift_with_price_level():
    """
    Тройной рост цены не должен сдвигать распределения.

    Прогрев исключён: пока реестры пусты, расстояния сидят на потолке, и
    сравнивать начало истории с серединой бессмысленно — это разница между
    «уровней ещё нет» и «уровни есть», а не между дешёвым и дорогим рынком.
    """
    bars = zigzag(3000, growth=np.log(3.0) / 3000)
    frame = smc.build_smc(bars).iloc[700:]
    half = len(frame) // 2
    head, tail = frame.iloc[:half], frame.iloc[half:]

    for column in ("premium_discount", "near_resistance", "near_support",
                   "fvg_fill_pct"):
        assert abs(head[column].median() - tail[column].median()) < 1.0, column


def test_features_are_finite_and_bounded():
    frame = smc.build_smc(zigzag(1000))
    numeric = frame[smc.FEATURE_CANDIDATES]
    assert numeric.notna().all().all()
    assert np.isfinite(numeric.to_numpy()).all()

    # Все двенадцать ограничены единицей по модулю — это и есть то свойство,
    # ради которого расстояния заменены на близости.
    for column in smc.FEATURE_CANDIDATES:
        assert numeric[column].abs().max() <= 1.0 + 1e-9, column
    for column in smc.FEATURE_CANDIDATES:
        if column in smc.SIGNED_FEATURES:
            assert numeric[column].between(-1.0, 1.0).all(), column
        else:
            assert numeric[column].between(0.0, 1.0).all(), column


def test_proximity_is_monotone_and_bounded():
    """
    Близость убывает с расстоянием, ограничена и не имеет хвоста.

    Хвост — ровно то, из-за чего первая версия признаков не годилась: после
    robust-нормировки редкое «зоны нет» становилось выбросом, который тянул
    кластеризацию сильнее любого настоящего признака.
    """
    assert smc.proximity(0.0) == 1.0
    assert smc.proximity(1.0) == pytest.approx(0.5)
    assert 0.0 < smc.proximity(1000.0) < 0.01
    distances = [0.0, 0.1, 0.5, 1.0, 5.0, 50.0]
    values = [smc.proximity(d) for d in distances]
    assert values == sorted(values, reverse=True)

    assert smc.signed_proximity(2.0) == pytest.approx(smc.proximity(2.0))
    assert smc.signed_proximity(-2.0) == pytest.approx(-smc.proximity(2.0))


def test_no_zone_is_zero_not_an_outlier():
    """
    «Ближайшей зоны нет» кодируется нулём — значением внутри шкалы.

    На плоском рынке уровней не набирается, и раньше здесь стоял потолок
    в 10 ATR, то есть выброс. Теперь это ноль, неотличимый по масштабу от
    «зона очень далеко», чем он по смыслу и является.
    """
    flat = make_frame([(100, 100.5, 99.5, 100)] * 300)
    frame = smc.build_smc(flat)
    assert (frame["near_ob"] == 0.0).all()
    assert frame["near_resistance"].abs().max() <= 1.0


def test_registry_bounded():
    """
    Реестры не растут с длиной истории.

    Проверка косвенная, но честная: если бы записи не выбывали, время прохода
    росло бы квадратично, и вдвое более длинная история заняла бы вчетверо
    больше. Сравниваем время на прогон в пересчёте на бар.
    """
    import time

    short_bars, long_bars = zigzag(2000), zigzag(8000)

    start = time.perf_counter()
    smc.build_smc(short_bars)
    per_bar_short = (time.perf_counter() - start) / len(short_bars)

    start = time.perf_counter()
    smc.build_smc(long_bars)
    per_bar_long = (time.perf_counter() - start) / len(long_bars)

    # При неограниченных реестрах отношение было бы примерно четырёхкратным.
    assert per_bar_long < per_bar_short * 2.5


def test_empty_input_returns_empty_frame():
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    assert smc.build_smc(empty).empty


def test_output_columns_are_declared():
    frame = smc.build_smc(zigzag(300))
    assert list(frame.columns) == smc.ALL_COLUMNS
    assert len(smc.ALL_COLUMNS) == 31
    for column in smc.BOOLEAN_COLUMNS:
        assert frame[column].dtype == bool, column


# ── Фаза 3: подключение к атомам и признакам ────────────────────────────────
@pytest.fixture
def smc_on(monkeypatch):
    """
    SMC включён целиком — и атомы, и признаки.

    Флага два, и по умолчанию признаки выключены даже при включённых атомах:
    у половин разная цена (см. SMCConfig). Тестам, проверяющим вектор,
    нужны оба.

    ЧУЖИЕ источники гасятся целиком: их дефолты тоже читаются из окружения,
    а FGI и деривативы вдобавок джойнят внешние ряды из БД, куда тесты не
    ходят — набор признаков с ними вышел бы пустым (аудит 2026-08-15, B4).
    """
    disable_all_sources(monkeypatch)
    monkeypatch.setattr(
        config, "smc", config.SMCConfig(enabled=True, features_enabled=True)
    )
    return config.smc


@pytest.fixture
def smc_atoms_only(monkeypatch):
    """Только атомы — ровно то, что включается в бою (чужие источники — прочь)."""
    disable_all_sources(monkeypatch)
    monkeypatch.setattr(
        config, "smc", config.SMCConfig(enabled=True, features_enabled=False)
    )
    return config.smc


@pytest.fixture
def smc_off(monkeypatch):
    """
    SMC выключён явно.

    Именно явно, а не «по умолчанию»: дефолты SMCConfig читаются из окружения,
    и на сервере, где SMC_ENABLED=true, «умолчание» означает включено. Тест,
    полагающийся на дефолт, зеленеет на машине разработчика и падает там, где
    флаг поднят, — что и произошло при выкатке. Чужие источники — по той же
    причине и тем же способом.
    """
    disable_all_sources(monkeypatch)
    monkeypatch.setattr(
        config, "smc", config.SMCConfig(enabled=False, features_enabled=False)
    )
    return config.smc


def test_smc_atoms_are_all_context(smc_on):
    """
    Шестнадцать новых атомов — контекстные, и это главное свойство фазы 3.

    Контекстный атом не входит в маску, значит не дробит блоки событий и не
    требует переобучения. Попадание любого из них в signature означало бы
    удвоение числа блоков на атом при 2.4-кратном запасе по выборке.
    """
    from btcproc.features import events

    assert len(events.SMC_ATOMS) == 16
    assert set(events.SMC_ATOMS) <= events.CONTEXT_ATOMS
    assert not (set(events.SMC_ATOMS) & set(events.SIGNATURE_ATOMS))
    # 45 базовых+SMC + 4 контекстных атома Fear & Greed (добавлены 2026-08-14,
    # docs/task_fear_greed.md) + 4 контекстных квадранта деривативов
    # (добавлены 2026-08-15, docs/tz_deriv_ingest_14-08-26.md) — все восемь
    # тоже контекстные, сигнатурные 20 не меняются.
    assert len(events.ATOMS) == 53
    assert len(events.CONTEXT_ATOMS) == 33
    assert len(events.SIGNATURE_ATOMS) == 20

    # Отвергнутые замером фазы 2 в реестр не попали.
    for rejected in ("eqh_above", "eql_below", "structure_bearish"):
        assert rejected not in events.ATOM_FAMILY


def test_smc_atoms_do_not_change_block_id(bars, smc_on):
    """
    Включение SMC не меняет ни один event_block_id.

    Это проверка того же инварианта, что и для старых контекстных атомов, но
    на новых: исторические блоки обязаны сохранить смысл, иначе накопленная
    статистика по ним становится несопоставимой молча.
    """
    from btcproc.features import events

    with_smc = events.build_event_blocks(bars)
    monkey = config.SMCConfig(enabled=False)
    import btcproc.config as config_module
    saved, config_module.smc = config_module.smc, monkey
    try:
        without_smc = events.build_event_blocks(bars)
    finally:
        config_module.smc = saved

    assert with_smc["event_block_id"].equals(without_smc["event_block_id"])
    assert with_smc["atoms"].equals(without_smc["atoms"])
    assert with_smc["atom_count"].equals(without_smc["atom_count"])


def test_smc_atoms_reach_context_column(bars, smc_on):
    from btcproc.features import events

    blocks = events.build_event_blocks(bars)
    seen = {atom for row in blocks["context_atoms"] for atom in row}
    assert seen & set(events.SMC_ATOMS), "ни один SMC-атом не доехал до выдачи"
    # И по-прежнему ни один не протёк в atoms.
    in_atoms = {atom for row in blocks["atoms"] for atom in row}
    assert not (in_atoms & set(events.SMC_ATOMS))


def test_smc_atoms_are_absent_when_disabled(bars, smc_off):
    """При выключенном флаге имена существуют, но не срабатывают никогда."""
    from btcproc.features import events

    detected = events.detect_atoms(bars)
    for name in events.SMC_ATOMS:
        assert name in detected.columns
        assert not detected[name].any()


def test_feature_version_follows_the_features_flag(smc_on):
    """
    Метка набора собирается из СОСТАВА включённых источников, а не нумеруется.

    Раньше здесь стояло `== "v2"`. Двоичная схема второго источника не
    выражала: набор «база + SMC + ончейн» получил бы ту же метку, что
    «база + SMC», `feature_sets.names` перезаписался бы по первичному ключу,
    а старые строки `features` остались бы массивами прежней длины —
    и разошлось бы это молча (btcproc/features/registry.py).
    """
    from btcproc.features import builder

    assert builder.feature_version() == "v1+smc"


def test_atoms_alone_do_not_change_the_feature_set(smc_atoms_only, bars, context):
    """
    Главная развязка: включение атомов не делает вектор сорокачетырёхмерным.

    Один флаг на обе половины означал бы, что включение атомов в бою молча
    меняет FEATURE_VERSION, и следующий же train — хоть кнопкой в админке —
    обучил бы модель на 44 признаках. По замеру это обрушивает граф состояний
    (BTC 43 → 31), причём в основном за счёт размерности, а не информации.
    """
    from btcproc.features import builder

    assert builder.feature_version() == builder.BASE_FEATURE_VERSION
    features = builder.build_features(bars, context)
    for name in smc.FEATURE_CANDIDATES:
        assert name not in features.columns, name


def test_features_need_both_flags():
    """
    Признаки требуют обеих половин, атомы — только своей.

    Обе половины задаются явно: дефолты SMCConfig берутся из окружения, и на
    контуре с SMC_ENABLED=true `SMCConfig(features_enabled=True)` означал бы
    «включены обе» — тест проверял бы не то, что написано.
    """
    both_off = config.SMCConfig(enabled=False, features_enabled=False)
    only_features = config.SMCConfig(enabled=False, features_enabled=True)
    only_atoms = config.SMCConfig(enabled=True, features_enabled=False)
    both_on = config.SMCConfig(enabled=True, features_enabled=True)

    assert not both_off.features_on
    assert not only_features.features_on, "признаки без общего выключателя не считаются"
    assert not only_atoms.features_on, "атомы сами по себе вектор не меняют"
    assert both_on.features_on


def test_feature_version_is_base_when_disabled(smc_off):
    from btcproc.features import builder

    assert builder.feature_version() == builder.BASE_FEATURE_VERSION


def test_smc_features_join_the_vector(bars, context, smc_on):
    """Двенадцать величин добавляются к тридцати двум, без NaN."""
    from btcproc.features import builder

    features = builder.build_features(bars, context)
    for name in smc.FEATURE_CANDIDATES:
        assert name in features.columns, name
    assert features.notna().all().all()
    assert np.isfinite(features.to_numpy()).all()


def test_smc_features_do_not_look_ahead(bars, context, smc_on):
    """
    Тот же тест, что для базовых признаков, но с включённым SMC.

    Структурные детекторы — самое опасное место: свинг «виден» на графике
    сразу, а подтверждается только через right баров.
    """
    from btcproc.features import builder

    cut = len(bars) - 500
    full = builder.build_features(bars, context)
    prefix_context = {tf: df[df.index <= bars.index[cut - 1]] for tf, df in context.items()}
    prefix = builder.build_features(bars.iloc[:cut], prefix_context)

    common = full.index.intersection(prefix.index)[:-1]
    assert len(common) > 100
    pd.testing.assert_frame_equal(
        full.loc[common], prefix.loc[common], check_exact=False, rtol=1e-9
    )


def test_cache_returns_the_same_object(smc_on):
    """Кэш обязан именно переиспользовать результат, а не считать заново."""
    bars = zigzag(400)
    first = smc.build_smc_cached(bars)
    second = smc.build_smc_cached(bars)
    assert first is second

    # Другой набор баров — другой отпечаток, пересчёт.
    third = smc.build_smc_cached(zigzag(400, amplitude=0.008))
    assert third is not first


def test_cache_distinguishes_configs():
    bars = zigzag(400)
    loose = smc.build_smc_cached(bars, config.SMCConfig(fvg_min_atr=0.01))
    strict = smc.build_smc_cached(bars, config.SMCConfig(fvg_min_atr=100.0))
    assert loose["fvg_formed_large"].sum() > strict["fvg_formed_large"].sum()


def test_enabling_smc_does_not_break_an_old_model(bars, context, smc_on):
    """
    Включение SMC без переобучения безопасно: лишние признаки игнорируются.

    Это свойство определяет порядок развёртывания. Модель состояний хранит
    свои feature_names, и `live` отбирает из посчитанного ровно их
    (`features[model.feature_names]`). Значит на сервере можно включить SMC,
    получить контекстные атомы бесплатно и мерить по ним лифт, не трогая ни
    модель, ни нумерацию group_id, ни накопленный в Neo4j граф.

    Ломается это ровно одним способом — если кто-то поменяет отбор колонок на
    «взять все». Тогда предсказание пойдёт по 44 признакам моделью, обученной
    на 32, и упадёт уже внутри predict.
    """
    from btcproc.features import builder

    old_model_features = list(builder.build_features(bars, context).columns)
    monkey = config.SMCConfig(enabled=False)
    import btcproc.config as config_module
    saved, config_module.smc = config_module.smc, monkey
    try:
        base_features = list(builder.build_features(bars, context).columns)
    finally:
        config_module.smc = saved

    # Набор с SMC — надмножество базового, поэтому ничего не «пропадает».
    assert set(base_features) <= set(old_model_features)
    missing = set(base_features) - set(old_model_features)
    assert not missing, "live упал бы с «набор признаков изменился»"

    # И отбор по именам старой модели даёт ровно её вектор.
    with_smc = builder.build_features(bars, context)
    assert list(with_smc[base_features].columns) == base_features
