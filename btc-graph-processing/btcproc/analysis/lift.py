"""
Замер лифта по атомам: даёт ли присутствие атома сдвиг в исходах.

Методика зафиксирована в docs/task_smc_integration.md, раздел 8. Три вещи,
без которых замер бессмыслен и которые здесь обязательны, а не опциональны:

1. **Поправка на множественные сравнения.** Тестируется не один атом, а все
   двадцать девять. При α = 0.05 и 29 тестах полтора «значимых» результата
   появятся из чистого шума. Без поправки замер находит SMC-паттерны там,
   где их нет, — ровно та ошибка, из-за которой они «работают» на бэктесте.
2. **Holdout по времени, а не случайный.** `sample_size` монотонно растёт с
   историей: кандидат 2018 года видит сотню аналогов, кандидат 2026-го —
   тысячи. Случайный сплит перемешивает эти режимы и завышает согласие.
3. **Совпадение знака на holdout.** Значимость на обучающей половине без
   подтверждения на отложенной — это подгонка, а не результат.

## Наблюдения зависимы — и это не мелочь

Двухвыборочный z-тест (`two_proportion_z`) верен ТОЛЬКО для независимых
наблюдений, а у нас они зависимы дважды:

* **перекрытие горизонтов.** Горизонт 24h при базовом ТФ 15m — это 96 баров.
  Два кандидата, отстоящие меньше чем на 96 баров, имеют почти один и тот же
  исход: их окна перекрываются;
* **снимки офсетов.** `SNAPSHOT_OFFSETS_MIN = (0, 45, 90, 180)` даёт до
  четырёх кандидатов на одну реализацию перехода в пределах 12 баров.

Оба эффекта завышают `n`, то есть делают тест **анти-консервативным**: он
находит эффекты, которых нет. По фактическим числам BTC (314 141 бар,
28 818 кандидатов одной модели) в одно окно горизонта попадает около девяти
кандидатов, значит `SE` занижена примерно втрое, и `z` во столько же завышен.

Асимметрия, которую важно понимать: отрицательный результат от починки
становится только надёжнее (не прошло при завышенном z — не пройдёт и при
честном), а положительный требует перепроверки.

Поэтому каждый результат несёт ДВА p-value: наивный (`p_value`) и
бутстрапный (`p_boot`, `block_bootstrap_p`). Значимость считается по
бутстрапному, когда он посчитан; наивный остаётся в таблице ради
сопоставимости со старыми замерами и чтобы разница была видна глазами.

Функции здесь чистые: на вход DataFrame, на выход числа. В БД ходит
scripts/measure_atom_lift.py.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Сколько бутстрап-реплик гоняется по умолчанию. 2000 даёт разрешение
#: p-value до 0.0005 — глубже нет смысла: поправка на множественные
#: сравнения всё равно работает на порядок грубее.
DEFAULT_N_BOOT = 2000

#: Сколько бутстрап-реплик считается за один векторный проход. Индексы реплик
#: — матрица (chunk, n) из int64, поэтому размер подбирается по числу строк, а
#: не константой: на 30 тыс. кандидатов это ~66 реплик, то есть около 16 МБ на
#: массив.
#:
#: Имя без подчёркивания, как и у `stationary_block_indices` с `atom_seed`:
#: их переиспользует соседний `gradation.py` — блочный бутстрап там ровно тот
#: же, меняется только статистика. Подчёркивание означало бы «внутреннее для
#: модуля» и вводило бы в заблуждение.
CHUNK_CELLS = 2_000_000


@dataclass
class AtomLift:
    """Результат одного теста: атом против всех остальных кандидатов."""

    atom: str
    n_with: int
    n_without: int
    p_with: float
    p_without: float
    z: float
    p_value: float
    # Блочный бутстрап: None, если не считался (см. measure_lift).
    p_boot: float | None = None
    # Медиана z по прореживанию непересекающимися окнами — sanity check.
    z_thinned: float | None = None
    # Заполняются после поправки на множественные сравнения.
    significant: bool = False
    # Заполняется, если считался holdout.
    holdout: "AtomLift | None" = field(default=None, repr=False)

    @property
    def lift(self) -> float:
        """Абсолютная разница долей. Именно абсолютная: отношение при
        p_without около нуля даёт бессмысленно большие числа."""
        return self.p_with - self.p_without

    @property
    def effective_p(self) -> float:
        """
        P-value, по которому принимается решение.

        Бутстрапный, если он посчитан: наивный завышает значимость из-за
        зависимости наблюдений (см. шапку модуля). Наивный остаётся
        доступным отдельным полем — он нужен, чтобы видеть размер поправки.
        """
        return self.p_value if self.p_boot is None else self.p_boot

    @property
    def confirmed(self) -> bool:
        """
        Атом остаётся в системе только при выполнении всех условий разом
        (раздел 8.5 ТЗ): значим после поправки, и на holdout лифт сохранил
        знак и порядок величины.
        """
        if not self.significant or self.holdout is None:
            return False
        if self.lift == 0 or self.holdout.lift == 0:
            return False
        if (self.lift > 0) != (self.holdout.lift > 0):
            return False
        ratio = abs(self.holdout.lift) / abs(self.lift)
        return 0.5 <= ratio <= 2.0


def two_sided_p(z: float) -> float:
    """P-value двустороннего нормального теста. erfc вместо scipy — точнее
    в хвостах, чем 1 - cdf, и не тянет зависимость в чистую функцию."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def two_proportion_z(p1: float, n1: int, p2: float, n2: int) -> tuple[float, float]:
    """
    Двухвыборочный z-тест для долей.

        p_pool = (p₁·n₁ + p₂·n₂) / (n₁ + n₂)
        SE     = √( p_pool · (1 − p_pool) · (1/n₁ + 1/n₂) )
        z      = (p₁ − p₂) / SE

    Возвращает (z, p_value). При вырожденных входах — (0.0, 1.0): это
    «не отличили», а не ошибка, и вызывающий код не должен на этом падать.

    ВАЖНО: формула предполагает независимые наблюдения, а у кандидатов они
    зависимы (шапка модуля). Результат этой функции — наивная оценка;
    решение принимается по `block_bootstrap_p`.
    """
    if n1 <= 0 or n2 <= 0:
        return 0.0, 1.0
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    variance = pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2)
    if variance <= 0.0:
        return 0.0, 1.0
    z = (p1 - p2) / math.sqrt(variance)
    return z, two_sided_p(z)


def bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Самая консервативная поправка: α делится на число тестов."""
    if not p_values:
        return []
    threshold = alpha / len(p_values)
    return [p <= threshold for p in p_values]


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Контроль FDR. При числе тестов больше ~20 Бонферрони становится настолько
    консервативным, что не пропускает ничего, — тогда берётся этот критерий.

    Отвергаются гипотезы с рангом до наибольшего k, при котором
    p₍ₖ₎ ≤ (k / m) · α, включительно — включая те, чей собственный p-value
    порог не прошёл (свойство процедуры, а не небрежность).
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    max_rank = 0
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= rank / m * alpha:
            max_rank = rank
    rejected = [False] * m
    for rank, idx in enumerate(order, start=1):
        if rank <= max_rank:
            rejected[idx] = True
    return rejected


def split_by_time(frame: pd.DataFrame, holdout: float = 0.3,
                  ts_column: str = "ts") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разбиение по времени: первые (1 − holdout) строк на обучение, хвост на
    подтверждение. Случайный сплит здесь врёт (см. шапку модуля).
    """
    if not 0.0 < holdout < 1.0:
        raise ValueError(f"holdout должен быть в (0, 1), получено {holdout}")
    ordered = frame.sort_values(ts_column)
    cut = int(len(ordered) * (1.0 - holdout))
    return ordered.iloc[:cut], ordered.iloc[cut:]


# ─── Зависимость наблюдений ─────────────────────────────────────────────────
def block_length_rows(ts: pd.Series, horizon_minutes: int) -> int:
    """
    Длина блока бутстрапа В СТРОКАХ выборки, а не в барах.

    Это единственное место, где легко ошибиться на порядок. Зависимость
    возникает во ВРЕМЕНИ (перекрытие горизонтов), а бутстрап переставляет
    СТРОКИ кандидатов, и одно в другое переводится не единицей: кандидаты
    идут не на каждом баре. Блок в 96 строк (горизонт в барах) накрыл бы у
    BTC около тысячи баров — вдесятеро больше горизонта, и тест стал бы
    абсурдно консервативным.

    Считается ровно та величина, которой блок и обязан быть равен:
    **среднее число строк, попадающих в одно окно горизонта**. Для каждой
    строки берётся, сколько строк лежит в `[t, t + горизонт)`, и результат
    усредняется. Это и есть средний размер кластера зависимых наблюдений —
    то самое `m̄` из `DEFF = 1 + (m̄ − 1)·ρ`.

    Почему не через средний шаг между строками, как было до 2026-08-11.
    Распределение интервалов сильно скошено: у BTC медиана 45 минут (это
    шаг снимков офсетов — соседние строки чаще всего описывают ОДИН переход),
    а среднее 126 минут, потому что его тянут вверх редкие участки ранней
    истории. Деление горизонта на средний шаг давало 11 строк при
    фактических 14.8 — блок оказывался короче нужного примерно на треть, а
    короткий блок делает тест анти-консервативным.

    Почему не по нижнему квантилю интервалов (вариант из задачи 11-102, §4.2).
    Проверено замером: `p10` интервалов у всех трёх монет равен 30 минутам,
    что дало бы блок в 48 строк — перелёт втрое в другую сторону. Квантиль
    интервалов не оценивает плотность окна, он оценивает плотность самого
    тесного места; прямой счёт не требует этого прокси вовсе.

    Возвращает не меньше 2 (блок в одну строку = обычный бутстрап, то есть
    отсутствие поправки) и не больше n // 10 (иначе блоков слишком мало,
    чтобы их перестановка что-то давала).
    """
    n = len(ts)
    if n < 3 or horizon_minutes <= 0:
        return 1
    ordered = pd.to_datetime(pd.Series(ts).reset_index(drop=True)).sort_values()
    minutes = (ordered - ordered.iloc[0]).dt.total_seconds().to_numpy() / 60.0
    if minutes[-1] <= 0:
        return 1
    # Правая граница окна для каждой строки; разность с её собственной
    # позицией и есть число строк в окне.
    right = np.searchsorted(minutes, minutes + horizon_minutes, side="left")
    per_window = right - np.arange(n)

    # У последних строк окно обрезано концом выборки, и их усреднение
    # занижало бы длину блока (на плотном ряду из 2000 строк — 94 вместо
    # 96). Считаем только по строкам с ПОЛНЫМ окном; если таких нет —
    # выборка короче горизонта, и тогда берём что есть.
    complete = minutes + horizon_minutes <= minutes[-1]
    sample = per_window[complete] if complete.any() else per_window
    length = int(round(float(sample.mean())))
    return max(2, min(length, max(2, n // 10)))


def resolution_is_sufficient(n_boot: int, alpha: float, tests: int,
                             correction: str) -> bool:
    """
    Хватает ли реплик, чтобы бутстрапный p-value вообще смог пройти поправку.

    P-value считается как `(1 + k) / (1 + B)`, то есть минимальное достижимое
    значение — `1 / (1 + B)`. Если оно сопоставимо с порогом поправки,
    результат решает шум Монте-Карло, а не данные.

    Числа, ради которых это заведено: при `B = 2000` минимум `0.0005`, а
    порог Бонферрони на 25 тестах — `0.002`. Пройти могут только реплики с
    `k ≤ 3`; при истинной хвостовой вероятности `0.001` ожидаемое `k` равно
    2, и `P(K ≤ 3 | λ = 2) = 0.857`, то есть **14% шанс потерять настоящий
    эффект** просто на шуме генератора.

    Требуем запас в четыре раза: `1/(1+B) ≤ порог/4`. Для Бонферрони это
    даёт правило `B ≥ 4·m/α` (на 45 тестах при α = 0.05 — 3600 реплик).
    """
    if n_boot <= 0:
        return False
    threshold = alpha / max(tests, 1) if correction == "bonferroni" else alpha
    return 1.0 / (1 + n_boot) <= threshold / 4.0


def stationary_block_indices(
    n: int, block_length: int, rng: np.random.Generator, size: int
) -> np.ndarray:
    """
    Индексы `size` реплик стационарного бутстрапа Полица–Романо, (size, n).

    Блоки геометрической длины со средним `block_length`, склеиваются
    циклически до длины исходной выборки. Геометрическая длина (а не
    фиксированная) нужна для стационарности результата: при блоках
    постоянной длины распределение реплики зависит от того, куда попали
    границы блоков.

    Векторизовано целиком: цикл по n на каждую реплику в Python съел бы
    минуты на каждый атом.
    """
    p_block = 1.0 / max(block_length, 1)
    starts = rng.integers(0, n, size=(size, n))
    new_block = rng.random((size, n)) < p_block
    new_block[:, 0] = True

    positions = np.arange(n)
    # Номер блока для каждой позиции и позиция его начала.
    block_id = np.cumsum(new_block, axis=1) - 1
    start_pos = np.maximum.accumulate(np.where(new_block, positions, 0), axis=1)
    offset = positions - start_pos
    return (np.take_along_axis(starts, block_id, axis=1) + offset) % n


def block_bootstrap_p(
    flags: np.ndarray,
    metric: np.ndarray,
    block_length: int,
    n_boot: int = DEFAULT_N_BOOT,
    rng: np.random.Generator | None = None,
) -> float:
    """
    P-value лифта через stationary block bootstrap.

    `flags` — булев признак «атом активен» по строкам выборки в
    хронологическом порядке, `metric` — метрика тех же строк.

    Нулевая гипотеза: связи между признаком и метрикой нет. Она
    воспроизводится так: **блоками пересобирается ПРИЗНАК, а метрика
    остаётся на месте**. Это принципиально. Обычный `shuffle` признака
    разрушил бы его автокорреляцию — «атом держится сериями» — и вернул нас
    ровно к наивному тесту, только другим путём. Блочная пересборка
    сохраняет и автокорреляцию признака, и автокорреляцию метрики, ломая
    только их взаимное выравнивание.

    Возвращает долю реплик, у которых |лифт| не меньше наблюдённого, с
    поправкой на единицу: `(1 + k) / (1 + B)`. Без неё нулевой p-value
    выглядел бы как «эффект достоверен абсолютно», хотя означает лишь
    «реплик не хватило».

    **В знаменатель идут только ПРИГОДНЫЕ реплики.** Реплика, у которой все
    флаги оказались True или все False, лифта не даёт вовсе — она не может
    попасть в числитель ни при каких данных. Считая её в знаменателе, мы
    занижали бы p, то есть смещали результат в сторону значимости. Для
    частого атома доля таких реплик пренебрежима, а для самого редкого,
    проходящего `min_group = 30` при `n ≈ 30 000`, — порядка процентов, и
    смещение попадало бы ровно на те атомы, которые и без того наиболее
    подозрительны. Поэтому реплики добираются до нужного числа пригодных,
    с потолком на число попыток: у вырожденного входа их может не набраться
    никогда, и молча крутиться в этом месте нельзя.
    """
    rng = rng or np.random.default_rng(42)
    flags = np.asarray(flags, dtype=bool)
    metric = np.asarray(metric, dtype=float)
    n = len(flags)
    if n < 10 or not flags.any() or flags.all():
        return 1.0

    total = float(metric.sum())
    observed = abs(metric[flags].mean() - metric[~flags].mean())

    chunk = max(1, min(n_boot, CHUNK_CELLS // max(n, 1)))
    extreme, usable_total, attempts = 0, 0, 0
    attempt_limit = 10 * n_boot
    while usable_total < n_boot and attempts < attempt_limit:
        size = min(chunk, n_boot - usable_total)
        idx = stationary_block_indices(n, block_length, rng, size)
        resampled = flags[idx]                      # (size, n)
        counts = resampled.sum(axis=1)
        sums = (resampled * metric).sum(axis=1)

        usable = (counts > 0) & (counts < n)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_with = sums / counts
            mean_without = (total - sums) / (n - counts)
            lifts = np.abs(mean_with - mean_without)
        extreme += int((usable & (lifts >= observed)).sum())
        usable_total += int(usable.sum())
        attempts += size

    if usable_total == 0:
        return 1.0
    return float((1 + extreme) / (1 + usable_total))


def thinned_z(
    frame: pd.DataFrame,
    atom: str,
    horizon_minutes: int,
    atoms_column: str = "atoms",
    metric_column: str = "metric",
    ts_column: str = "ts",
    phases: int = 24,
) -> float | None:
    """
    Медиана z по прореживанию непересекающимися окнами — грубая проверка.

    Из выборки берётся по одному кандидату на окно длиной в горизонт: окна
    не пересекаются, значит и исходы не перекрываются, значит обычный z-тест
    на этой подвыборке честен. Точка отсчёта окон произвольна, поэтому
    процедура повторяется для `phases` сдвигов и берётся МЕДИАНА: отдельная
    фаза может попасть на вырожденную подвыборку, среднее от этого поехало бы.

    Выбрасывает почти всю выборку и теряет мощность — поэтому это sanity
    check рядом с бутстрапом, а не замена ему. Ценность в том, что здесь
    ошибиться негде: никаких допущений, кроме «непересекающиеся окна
    независимы».

    Возвращает None, если ни одна фаза не дала пригодной подвыборки.
    """
    if frame.empty or horizon_minutes <= 0:
        return None

    ordered = frame.sort_values(ts_column)
    ts = pd.to_datetime(ordered[ts_column])
    origin = ts.iloc[0]
    horizon = pd.Timedelta(minutes=horizon_minutes)
    shift = horizon / max(phases, 1)

    zs: list[float] = []
    for phase in range(max(phases, 1)):
        window = ((ts - (origin - phase * shift)) // horizon).astype("int64")
        picked = ordered.loc[~window.duplicated().to_numpy()]
        result = _measure_one(picked, atom, atoms_column, metric_column)
        if result is not None and min(result.n_with, result.n_without) > 0:
            zs.append(result.z)
    return float(np.median(zs)) if zs else None


# ─── Замер ──────────────────────────────────────────────────────────────────
def atom_seed(atom: str) -> int:
    """Стабильное между запусками зерно по имени атома."""
    return int(hashlib.sha1(atom.encode()).hexdigest()[:8], 16)


def _atom_flags(frame: pd.DataFrame, atom: str, atoms_column: str) -> np.ndarray:
    return frame[atoms_column].map(lambda atoms: atom in atoms).to_numpy(dtype=bool)


def _measure_one(frame: pd.DataFrame, atom: str, atoms_column: str,
                 metric_column: str) -> AtomLift | None:
    has_atom = _atom_flags(frame, atom, atoms_column)
    metric = frame[metric_column].to_numpy(dtype=float)
    with_atom, without_atom = metric[has_atom], metric[~has_atom]
    if with_atom.size == 0 or without_atom.size == 0:
        return None

    p1, n1 = float(with_atom.mean()), int(with_atom.size)
    p2, n2 = float(without_atom.mean()), int(without_atom.size)
    z, p_value = two_proportion_z(p1, n1, p2, n2)
    return AtomLift(atom=atom, n_with=n1, n_without=n2,
                    p_with=p1, p_without=p2, z=z, p_value=p_value)


def measure_lift(
    frame: pd.DataFrame,
    atoms: list[str],
    atoms_column: str = "atoms",
    metric_column: str = "metric",
    alpha: float = 0.05,
    correction: str = "bonferroni",
    holdout: float | None = 0.3,
    min_group: int = 30,
    horizon_minutes: int | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    thinned: bool = False,
    ts_column: str = "ts",
    seed: int = 42,
) -> list[AtomLift]:
    """
    Лифт по каждому атому из списка.

    frame — строка на кандидата, обязательные колонки: ts, `atoms_column`
    (список атомов бара), `metric_column` (метрика в [0, 1]).

    Атомы, у которых меньше `min_group` наблюдений в любой из групп, из
    тестирования исключаются целиком — не помечаются незначимыми, а именно
    исключаются: иначе они раздували бы число тестов и штрафовали поправкой
    те атомы, по которым данные есть.

    `horizon_minutes` — горизонт оценки исходов (`config.data.horizon_minutes`).
    Задан — считается блочный бутстрап, и значимость определяется по нему;
    None — остаётся только наивный z-тест, то есть режим до 2026-08-11.
    Отключать стоит разве что ради сравнения со старыми замерами.

    `thinned=True` добавляет медиану z по прореживанию (дорого, sanity check).

    Возвращает список, отсортированный по убыванию |z|.
    """
    if correction not in ("bonferroni", "bh", "none"):
        raise ValueError(f"неизвестная поправка: {correction}")

    train, test = (frame, None) if holdout is None else split_by_time(frame, holdout)

    block = (
        block_length_rows(train[ts_column], horizon_minutes)
        if horizon_minutes and not train.empty else None
    )
    block_test = (
        block_length_rows(test[ts_column], horizon_minutes)
        if horizon_minutes and test is not None and not test.empty else None
    )

    results: list[AtomLift] = []
    for atom in atoms:
        result = _measure_one(train, atom, atoms_column, metric_column)
        if result is None or min(result.n_with, result.n_without) < min_group:
            continue
        # Своя последовательность на атом: реплики не должны зависеть от
        # того, сколько атомов измерили до него, — иначе замер по одному
        # атому не воспроизводит строку общего замера. sha1, а не hash():
        # встроенный хэш строки солится PYTHONHASHSEED и от запуска к
        # запуску разный, то есть замер перестал бы быть воспроизводимым.
        rng = np.random.default_rng([seed, atom_seed(atom)])
        if block:
            result.p_boot = block_bootstrap_p(
                _atom_flags(train, atom, atoms_column),
                train[metric_column].to_numpy(dtype=float),
                block, n_boot=n_boot, rng=rng,
            )
        if thinned and horizon_minutes:
            result.z_thinned = thinned_z(
                train, atom, horizon_minutes, atoms_column, metric_column, ts_column
            )
        if test is not None and not test.empty:
            result.holdout = _measure_one(test, atom, atoms_column, metric_column)
            if block_test and result.holdout is not None:
                result.holdout.p_boot = block_bootstrap_p(
                    _atom_flags(test, atom, atoms_column),
                    test[metric_column].to_numpy(dtype=float),
                    block_test, n_boot=n_boot, rng=rng,
                )
        results.append(result)

    if not results:
        return []

    p_values = [r.effective_p for r in results]
    if correction == "bonferroni":
        flags = bonferroni(p_values, alpha)
    elif correction == "bh":
        flags = benjamini_hochberg(p_values, alpha)
    else:
        flags = [p <= alpha for p in p_values]
    for result, flag in zip(results, flags):
        result.significant = flag

    return sorted(results, key=lambda r: abs(r.z), reverse=True)


def format_table(results: list[AtomLift], correction: str = "bonferroni",
                 alpha: float = 0.05, n_boot: int | None = None) -> str:
    """
    Текстовая таблица результатов — то, что печатает скрипт.

    `n_boot` нужен только для предупреждения о недостаточном разрешении
    бутстрапа: без него таблица честна во всём, кроме того, что не может
    сказать, хватило ли реплик (см. `resolution_is_sufficient`).
    """
    if not results:
        return "Нет атомов с достаточным числом наблюдений в обеих группах."

    bootstrapped = any(r.p_boot is not None for r in results)
    thinned = any(r.z_thinned is not None for r in results)

    header = (
        f"{'атом':<24} {'n(с)':>7} {'n(без)':>7} {'p(с)':>7} {'p(без)':>7} "
        f"{'лифт':>8} {'z':>7} {'p наив.':>9}"
    )
    if bootstrapped:
        header += f" {'p блок.':>9}"
    if thinned:
        header += f" {'z прор.':>8}"
    header += f" {'знач.':>6} {'holdout':>8} {'итог':>6}"

    lines = [header, "─" * len(header)]
    for r in results:
        holdout_lift = f"{r.holdout.lift:+.4f}" if r.holdout else "—"
        row = (
            f"{r.atom:<24} {r.n_with:>7} {r.n_without:>7} {r.p_with:>7.4f} "
            f"{r.p_without:>7.4f} {r.lift:>+8.4f} {r.z:>7.2f} {r.p_value:>9.5f}"
        )
        if bootstrapped:
            row += f" {'—':>9}" if r.p_boot is None else f" {r.p_boot:>9.5f}"
        if thinned:
            row += f" {'—':>8}" if r.z_thinned is None else f" {r.z_thinned:>8.2f}"
        row += (f" {'да' if r.significant else 'нет':>6} {holdout_lift:>8} "
                f"{'ДА' if r.confirmed else '—':>6}")
        lines.append(row)

    threshold = alpha / len(results) if correction == "bonferroni" else alpha
    lines.append("")
    lines.append(
        f"Тестов: {len(results)}. Поправка: {correction}, α = {alpha}"
        + (f" → порог p ≤ {threshold:.5f}" if correction == "bonferroni" else "")
    )
    if bootstrapped:
        lines.append(
            "Значимость считается по БЛОЧНОМУ p-value: наивный завышен, "
            "потому что горизонты кандидатов перекрываются (шапка lift.py)."
        )
        if n_boot and not resolution_is_sufficient(n_boot, alpha, len(results),
                                                   correction):
            need = int(4 * len(results) / alpha) if correction == "bonferroni" \
                else int(4 / alpha)
            lines.append(
                f"⚠ РЕПЛИК МАЛО: при {n_boot} минимальный достижимый p-value "
                f"{1 / (1 + n_boot):.5f} сопоставим с порогом {threshold:.5f}. "
                f"Настоящий эффект может не пройти на шуме Монте-Карло. "
                f"Нужно --n-boot ≥ {need}."
            )
    else:
        lines.append(
            "ВНИМАНИЕ: блочный бутстрап не считался — значимость по наивному "
            "p-value, а он анти-консервативен (шапка lift.py)."
        )
    if thinned:
        lines.append(
            "«z прор.» — медиана z по прореживанию непересекающимися окнами."
        )
    lines.append(
        "Столбец «итог» = значим после поправки И знак лифта сохранился на "
        "holdout при сопоставимой величине. Только такие атомы засчитываются."
    )
    return "\n".join(lines)
