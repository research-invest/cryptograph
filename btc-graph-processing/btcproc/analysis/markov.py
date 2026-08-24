"""
Граф состояний как марковская модель: проверка в той роли, в которой он
построен.

ТЗ — `crypto-graph/docs/tz_wave_a_24-08-26.md`, задача M; постановка —
`crypto-graph/docs/ideas_math_2026-08-18.md`, §1.1, §1.4 и §2.4.

## Зачем

Самый крупный недобор проекта — не в том, что граф плохо предсказывает цену
(это измерено дважды, разделы 26 и 31), а в том, что **его ни разу не
проверяли как модель динамики**. Матрица переходов считается,
`transition_rarity` считается, кандидат по переходу выпускается — а вопрос
«описывает ли эта разметка процесс» не задан.

Аппарат взят из Markov State Models молекулярной динамики: там та же задача
(высокоразмерный ряд → дискретные состояния → матрица переходов) и те же
болезни (произвол числа состояний, чувствительность разбиения к данным).
Ниже — перенос техники, а не изобретение.

## Три вопроса, на которые отвечает модуль

* **M1, марковость.** Матрица `T(τ)` при ФИКСИРОВАННОМ лаге τ баров — это не
  то же, что `transitions` конвейера: там переход считается по событию смены
  состояния. Собственные значения дают характерные времена
  `t_i(τ) = −τ / ln λ_i(τ)`; у марковской разметки они выходят на плато.
  Тест Чепмена — Колмогорова проверяет то же напрямую: `T(kτ) ≈ T(τ)^k`.
* **M2, порядок.** Условная взаимная информация `I(S_{t+1}; S_{t−1} | S_t)`:
  нужна ли в ключе перехода тройка вместо пары.
* **M3, независимый контроль.** BOCPD по признакам — вторая разметка,
  не пользующаяся НИ ОДНОЙ деталью кластеризации, поэтому не может быть
  отравлена её дефектами.

## Две ловушки, обе уже обжигали проект

**Смесь моделей.** `bar_states` пополняют и `train`, и `live`, под разными
`run_id`, а `group_id` осмыслен только внутри одной модели. Агрегировать
обязательно через `runs.model_run_scope` — иначе получится смесь нумераций
(ошибка раздела 17, стоившая двух тихих багов).

**Самопереходы.** Состояние держится десятками баров, поэтому диагональ
`T(τ)` при малых τ близка к единице, и любая величина, считаемая по барам,
описывает прежде всего липкость. Поэтому M2 считается ДВАЖДЫ: по барам и по
цепи скачков (последовательность различных состояний). Расхождение ответов —
результат про липкость, а не про порядок.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Матрица переходов при фиксированном лаге ───────────────────────────────
def count_matrix(states: np.ndarray, lag: int, n_states: int) -> np.ndarray:
    """
    Матрица счётчиков `C[i, j]` = сколько раз из `i` через `lag` баров
    оказались в `j`.

    Пары берутся со сдвигом на один бар (скользящим окном), а не
    непересекающимися блоками: непересекающиеся дали бы в `lag` раз меньше
    наблюдений без выигрыша в независимости, который всё равно недостижим —
    зависимость здесь и есть предмет измерения.
    """
    if lag < 1 or lag >= len(states):
        raise ValueError(f"недопустимый лаг {lag} при длине {len(states)}")
    source = states[:-lag]
    target = states[lag:]
    flat = np.bincount(source * n_states + target, minlength=n_states * n_states)
    return flat.reshape(n_states, n_states).astype(float)


def transition_matrix(counts: np.ndarray) -> np.ndarray:
    """
    Строчно-стохастическая матрица. Пустая строка (состояние ни разу не
    встретилось источником) остаётся нулевой, а не заполняется равномерно:
    равномерная строка — это выдуманное наблюдение, и в спектре она
    отзовётся собственным значением, которого в данных нет.
    """
    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.where(totals > 0, counts / totals, 0.0)
    return matrix


#: Ниже этого собственного значения время считать нельзя.
#:
#: Проверено положительным контролем (честная марковская цепь, 200 тыс.
#: шагов, t₁ = 4.9 бара): на лагах 1–16 кривая держит плато в пределах 2%, а
#: на лаге 32 подскакивает до 6.3. Причина не в данных: `λ(32) ≈ exp(−32/4.9)
#: = 0.0016`, то есть процесс к этому лагу затух на 99.8%, и оценка такой
#: величины по счётчикам — это оценка шума. Логарифм у нуля растягивает
#: ошибку в разы.
#:
#: Отсюда правило, которое обязано быть в коде, а не в голове: **лаг,
#: превышающий время процесса больше чем втрое, о нём ничего не говорит.**
#: Порог 0.05 — это ровно `exp(−3)`.
MIN_LAMBDA = 0.05


def implied_timescales(matrix: np.ndarray, lag: int, count: int = 5,
                       min_lambda: float = MIN_LAMBDA) -> np.ndarray:
    """
    Характерные времена `t_i = −lag / ln λ_i` для `count` медленнейших
    процессов, в барах.

    Первое собственное значение стационарного распределения равно единице и
    даёт бесконечное время — оно отбрасывается, это не процесс, а сам факт
    существования равновесия. Комплексные собственные значения (матрица не
    обязана быть обратимой) берутся по модулю: осцилляции нас здесь не
    интересуют, интересует скорость затухания.

    Значения ниже `min_lambda` дают NaN — см. константу: там оценивается шум,
    а не время. Подставлять на это место что-либо нельзя.
    """
    values = np.linalg.eigvals(matrix)
    magnitudes = np.sort(np.abs(values))[::-1]
    tail = magnitudes[1:count + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        times = np.where(
            (tail > min_lambda) & (tail < 1.0),
            -lag / np.log(np.clip(tail, 1e-15, 0.999999)),
            np.nan,
        )
    return times


def timescale_curve(states: np.ndarray, lags: list[int], n_states: int,
                    count: int = 3) -> pd.DataFrame:
    """Таблица «лаг × номер процесса» с характерными временами в барах."""
    rows = []
    for lag in lags:
        matrix = transition_matrix(count_matrix(states, lag, n_states))
        times = implied_timescales(matrix, lag, count)
        rows.append({"lag": lag, **{f"t{i + 1}": t for i, t in enumerate(times)}})
    return pd.DataFrame(rows)


def plateau_deviation(curve: pd.DataFrame, column: str,
                      lag_from: int, lag_to: int) -> float:
    """
    Относительный размах кривой на отрезке — мера «вышло ли на плато».

    `(max − min) / медиана` на выбранном диапазоне лагов. Порог заявлен в ТЗ:
    не более 0.20. Медиана в знаменателе, а не среднее: у невышедшей на плато
    кривой хвост тянет среднее и делает отклонение обманчиво малым.
    """
    window = curve[(curve["lag"] >= lag_from) & (curve["lag"] <= lag_to)][column]
    window = window.dropna()
    if len(window) < 2:
        return float("nan")
    median = float(np.median(window))
    if median <= 0:
        return float("nan")
    return float((window.max() - window.min()) / median)


def chapman_kolmogorov(states: np.ndarray, lag: int, factors: list[int],
                       n_states: int, count: int = 3) -> pd.DataFrame:
    """
    Тест Чепмена — Колмогорова через релаксационные кривые.

    Для каждого `k` сравниваются собственные значения `T(k·lag)`, оценённые
    ПРЯМО из данных, с предсказанными марковской моделью — `λ_i(lag)^k`.
    Сравнение по собственным значениям, а не поэлементно по матрице: при 40
    состояниях элементов 1600, большая часть которых — нули с нулевой
    дисперсией, и поэлементная норма измеряла бы прежде всего разреженность.

    Возвращает наблюдённое, предсказанное и относительную ошибку по каждому
    из `count` медленнейших процессов.
    """
    base = transition_matrix(count_matrix(states, lag, n_states))
    base_lambda = np.sort(np.abs(np.linalg.eigvals(base)))[::-1][1:count + 1]

    rows = []
    for k in factors:
        if lag * k >= len(states):
            continue
        actual = transition_matrix(count_matrix(states, lag * k, n_states))
        actual_lambda = np.sort(np.abs(np.linalg.eigvals(actual)))[::-1][1:count + 1]
        predicted = base_lambda ** k
        for i, (a, p) in enumerate(zip(actual_lambda, predicted), start=1):
            # Затухший процесс из теста выпадает: сравнивать 0.002 с 0.003 —
            # значит сравнивать два шума и получать 50% ошибки на любых
            # данных, включая идеально марковские. Та же граница, что у
            # implied_timescales, и по той же причине.
            if p < MIN_LAMBDA:
                continue
            rows.append({
                "k": k, "process": i, "observed": float(a), "predicted": float(p),
                "error": float(abs(a - p) / max(p, 1e-9)),
            })
    return pd.DataFrame(rows)


def bootstrap_lambda_band(states: np.ndarray, lag: int, n_states: int,
                          block: int, n_boot: int, rng: np.random.Generator,
                          count: int = 3) -> pd.DataFrame:
    """
    Блочная доверительная полоса для собственных значений `T(lag)`.

    Бутстрап по ТРАЕКТОРИИ: переставляются блоки подряд идущих состояний,
    длиной в горизонт. Переставлять пары `(s_t, s_{t+lag})` было бы неверно —
    это разрушило бы то самое свойство, которое проверяется.
    """
    n = len(states)
    values = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n - block), size=n // block + 1)
        pieces = [states[s:s + block] for s in starts]
        replica = np.concatenate(pieces)[:n]
        matrix = transition_matrix(count_matrix(replica, lag, n_states))
        lam = np.sort(np.abs(np.linalg.eigvals(matrix)))[::-1][1:count + 1]
        values.append(lam)
    array = np.array(values)
    return pd.DataFrame({
        "process": np.arange(1, array.shape[1] + 1),
        "lo": np.percentile(array, 2.5, axis=0),
        "hi": np.percentile(array, 97.5, axis=0),
    })


# ─── Порядок марковости ─────────────────────────────────────────────────────
def conditional_mutual_information(previous: np.ndarray, current: np.ndarray,
                                   following: np.ndarray, n_states: int) -> float:
    """
    `I(S_{t+1}; S_{t−1} | S_t)` в натах, оценка plug-in по частотам.

    Считается через разложение на энтропии условных распределений, по каждому
    значению `S_t` отдельно, с весами — долями. Оценка **смещена вверх** на
    конечной выборке, и величина смещения растёт с числом состояний: при 40
    состояниях таблица имеет 64 тысячи ячеек, а наблюдений — сотни тысяч.
    Поэтому наружу она отдаётся как есть, а сравнивается не с нулём, а с
    нулёвочным распределением ТОЙ ЖЕ оценки (см. `cmi_null`). Всякий, кто
    сравнит это число с нулём напрямую, получит «зависимость» на любых данных.
    """
    total = len(current)
    if total == 0:
        return 0.0
    result = 0.0
    for state in range(n_states):
        mask = current == state
        weight = mask.sum()
        if weight < 2:
            continue
        table = np.zeros((n_states, n_states))
        np.add.at(table, (previous[mask], following[mask]), 1.0)
        joint = table / table.sum()
        row = joint.sum(axis=1, keepdims=True)
        column = joint.sum(axis=0, keepdims=True)
        expected = row @ column
        nonzero = joint > 0
        contribution = float(np.sum(joint[nonzero] * np.log(
            joint[nonzero] / np.clip(expected[nonzero], 1e-300, None))))
        result += (weight / total) * contribution
    return result


def jump_chain(states: np.ndarray) -> np.ndarray:
    """
    Цепь скачков: подряд идущие повторы схлопываются в одно вхождение.

    Нужна потому, что самопереход доминирует: состояние держится десятками
    баров, и `I(S_{t+1}; S_{t−1} | S_t)` по барам измеряла бы прежде всего
    «состояние обычно не меняется», а не глубину памяти.
    """
    if len(states) == 0:
        return states
    keep = np.empty(len(states), dtype=bool)
    keep[0] = True
    keep[1:] = states[1:] != states[:-1]
    return states[keep]


def cmi_null(sequence: np.ndarray, n_states: int, block: int, n_boot: int,
             rng: np.random.Generator) -> np.ndarray:
    """
    Нулёвочное распределение CMI: блочная перестановка ТРЕТЬЕГО элемента.

    Переставляется `S_{t−1}` блоками, а пары `(S_t, S_{t+1})` остаются на
    месте. Это ломает ровно проверяемую связь — влияние позапрошлого
    состояния — и сохраняет всё остальное, включая липкость и автокорреляцию.
    Наивная перестановка всех трёх рядов разрушила бы и структуру пары, то
    есть сравнивала бы наблюдение с гораздо более бедной моделью, чем надо.
    """
    previous, current, following = sequence[:-2], sequence[1:-1], sequence[2:]
    n = len(previous)
    values = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n - block), size=n // block + 1)
        pieces = [previous[s:s + block] for s in starts]
        shuffled = np.concatenate(pieces)[:n]
        values.append(conditional_mutual_information(shuffled, current, following,
                                                     n_states))
    return np.array(values)


# ─── BOCPD ──────────────────────────────────────────────────────────────────
def bocpd_run_length(data: np.ndarray, hazard: float = 1 / 500.0,
                     mu0: float = 0.0, kappa0: float = 1.0,
                     alpha0: float = 1.0, beta0: float = 1.0,
                     max_run: int = 2000) -> np.ndarray:
    """
    Bayesian online change-point detection (Adams & MacKay, 2007) —
    апостериорная вероятность «длина текущего прогона равна нулю» по барам.

    Модель наблюдений — гауссова с сопряжённым нормально-гамма приором, то
    есть неизвестны и среднее, и дисперсия. Это принципиально для наших
    данных: смена режима волатильности — это смена дисперсии, и модель с
    известной дисперсией её бы не увидела вовсе.

    Одномерный вход. Многомерность (несколько признаков) сводится наружу —
    первой главной компонентой, — а не внутри: полноценная многомерная
    сопряжённая модель потребовала бы матричных апдейтов на каждом баре,
    а выигрыш неизвестен заранее.

    `max_run` обрезает хвост распределения длин прогона: без обрезки массив
    растёт как O(n²) и на трёхстах тысячах баров не помещается никуда.
    Обрезка смещает оценку только для режимов длиннее max_run баров.

    Возвращает вектор `P(runlength = 0)` по барам — вероятность того, что
    именно здесь начался новый режим.
    """
    n = len(data)
    probability = np.zeros(max_run + 1)
    probability[0] = 1.0

    mu = np.full(max_run + 1, mu0)
    kappa = np.full(max_run + 1, kappa0)
    alpha = np.full(max_run + 1, alpha0)
    beta = np.full(max_run + 1, beta0)

    changepoint = np.zeros(n)
    for t in range(n):
        x = data[t]
        # Предиктив — стьюдент с 2α степенями свободы.
        df = 2.0 * alpha
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        z = (x - mu) / scale
        log_predictive = (
            _log_gamma((df + 1.0) / 2.0) - _log_gamma(df / 2.0)
            - 0.5 * np.log(np.pi * df) - np.log(scale)
            - (df + 1.0) / 2.0 * np.log1p(z * z / df)
        )
        predictive = np.exp(log_predictive - log_predictive.max())

        growth = probability * predictive * (1.0 - hazard)
        born = float(np.sum(probability * predictive * hazard))

        new_probability = np.zeros(max_run + 1)
        new_probability[1:] = growth[:-1]
        new_probability[0] = born
        total = new_probability.sum()
        if total <= 0:
            new_probability = np.zeros(max_run + 1)
            new_probability[0] = 1.0
            total = 1.0
        probability = new_probability / total
        changepoint[t] = probability[0]

        # Обновление достаточных статистик, со сдвигом на одну позицию.
        mu_new = np.empty_like(mu)
        kappa_new = np.empty_like(kappa)
        alpha_new = np.empty_like(alpha)
        beta_new = np.empty_like(beta)
        mu_new[0], kappa_new[0], alpha_new[0], beta_new[0] = mu0, kappa0, alpha0, beta0
        mu_new[1:] = (kappa[:-1] * mu[:-1] + x) / (kappa[:-1] + 1.0)
        kappa_new[1:] = kappa[:-1] + 1.0
        alpha_new[1:] = alpha[:-1] + 0.5
        beta_new[1:] = beta[:-1] + kappa[:-1] * (x - mu[:-1]) ** 2 / (
            2.0 * (kappa[:-1] + 1.0))
        mu, kappa, alpha, beta = mu_new, kappa_new, alpha_new, beta_new

    return changepoint


def _log_gamma(values: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln  # локально: scipy тянется только сюда

    return gammaln(values)


def pick_changepoints(probability: np.ndarray, share: float,
                      min_distance: int) -> np.ndarray:
    """
    Точки смены режима из ряда `P(runlength = 0)` — локальные максимумы с
    минимальным разносом, а не просто «всё выше квантиля».

    Порог по квантилю выглядит естественно и не работает: ряд сильно
    автокоррелирован, всплеск растягивается на десятки баров, и «верхние 2%»
    превращаются в 21% помеченных баров (замер 2026-08-24, первая версия).
    При такой плотности случайная точка попадает в окрестность ±2 бара с
    вероятностью 0.7, и тест теряет мощность полностью: наблюдение 0.697
    против случайного 0.696 — это не отрицательный результат, это отсутствие
    измерения.

    Жадный отбор: берём самый высокий незанятый пик, гасим окрестность
    `min_distance`, повторяем, пока не наберём нужную долю баров.
    """
    n = len(probability)
    limit = max(1, int(n * share))
    order = np.argsort(probability)[::-1]
    taken = np.zeros(n, dtype=bool)
    chosen = []
    for index in order:
        if len(chosen) >= limit:
            break
        if taken[index]:
            continue
        chosen.append(int(index))
        low = max(0, index - min_distance)
        taken[low:index + min_distance + 1] = True
    return np.sort(np.array(chosen, dtype=int))


def overlap_rate(a: np.ndarray, b: np.ndarray, tolerance: int) -> float:
    """
    Доля точек `a`, рядом с которыми (в пределах `tolerance` баров) есть точка
    `b`. Именно доля от `a`, а не симметричная мера: вопрос звучит «совпадают
    ли переходы графа со сменами BOCPD», то есть `a` — переходы графа.
    """
    if len(a) == 0 or len(b) == 0:
        return 0.0
    positions = np.searchsorted(b, a)
    hits = 0
    for point, position in zip(a, positions):
        left = b[position - 1] if position > 0 else None
        right = b[position] if position < len(b) else None
        near = [abs(point - x) for x in (left, right) if x is not None]
        if near and min(near) <= tolerance:
            hits += 1
    return hits / len(a)


def shift_null(a: np.ndarray, b: np.ndarray, tolerance: int, length: int,
               n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """
    Нулёвка для совпадения точек — **циклический сдвиг** одного ряда.

    Независимая перестановка здесь была бы неверна: оба ряда сильно
    автокоррелированы (точки идут сериями), и перестановка разрушила бы их
    собственную структуру, завысив значимость. Сдвиг сохраняет обе структуры
    и ломает только взаимное выравнивание — тот же принцип, что у блочного
    бутстрапа в `lift.block_bootstrap_p`.
    """
    shifts = rng.integers(1, length, size=n_boot)
    return np.array([overlap_rate(a, np.sort((b + shift) % length), tolerance)
                     for shift in shifts])
