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

Функции здесь чистые: на вход DataFrame, на выход числа. В БД ходит
scripts/measure_atom_lift.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd


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


def _measure_one(frame: pd.DataFrame, atom: str, atoms_column: str,
                 metric_column: str) -> AtomLift | None:
    has_atom = frame[atoms_column].map(lambda atoms: atom in atoms)
    with_atom = frame.loc[has_atom, metric_column]
    without_atom = frame.loc[~has_atom, metric_column]
    if with_atom.empty or without_atom.empty:
        return None

    p1, n1 = float(with_atom.mean()), len(with_atom)
    p2, n2 = float(without_atom.mean()), len(without_atom)
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
) -> list[AtomLift]:
    """
    Лифт по каждому атому из списка.

    frame — строка на кандидата, обязательные колонки: ts, `atoms_column`
    (список атомов бара), `metric_column` (метрика в [0, 1]).

    Атомы, у которых меньше `min_group` наблюдений в любой из групп, из
    тестирования исключаются целиком — не помечаются незначимыми, а именно
    исключаются: иначе они раздували бы число тестов и штрафовали поправкой
    те атомы, по которым данные есть.

    Возвращает список, отсортированный по убыванию |z|.
    """
    if correction not in ("bonferroni", "bh", "none"):
        raise ValueError(f"неизвестная поправка: {correction}")

    train, test = (frame, None) if holdout is None else split_by_time(frame, holdout)

    results: list[AtomLift] = []
    for atom in atoms:
        result = _measure_one(train, atom, atoms_column, metric_column)
        if result is None or min(result.n_with, result.n_without) < min_group:
            continue
        if test is not None and not test.empty:
            result.holdout = _measure_one(test, atom, atoms_column, metric_column)
        results.append(result)

    if not results:
        return []

    p_values = [r.p_value for r in results]
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
                 alpha: float = 0.05) -> str:
    """Текстовая таблица результатов — то, что печатает скрипт."""
    if not results:
        return "Нет атомов с достаточным числом наблюдений в обеих группах."

    header = (
        f"{'атом':<24} {'n(с)':>7} {'n(без)':>7} {'p(с)':>7} {'p(без)':>7} "
        f"{'лифт':>8} {'z':>7} {'p-value':>9} {'знач.':>6} {'holdout':>8} {'итог':>6}"
    )
    lines = [header, "─" * len(header)]
    for r in results:
        holdout_lift = f"{r.holdout.lift:+.4f}" if r.holdout else "—"
        lines.append(
            f"{r.atom:<24} {r.n_with:>7} {r.n_without:>7} {r.p_with:>7.4f} "
            f"{r.p_without:>7.4f} {r.lift:>+8.4f} {r.z:>7.2f} {r.p_value:>9.5f} "
            f"{'да' if r.significant else 'нет':>6} {holdout_lift:>8} "
            f"{'ДА' if r.confirmed else '—':>6}"
        )

    threshold = alpha / len(results) if correction == "bonferroni" else alpha
    lines.append("")
    lines.append(
        f"Тестов: {len(results)}. Поправка: {correction}, α = {alpha}"
        + (f" → порог p ≤ {threshold:.5f}" if correction == "bonferroni" else "")
    )
    lines.append(
        "Столбец «итог» = значим после поправки И знак лифта сохранился на "
        "holdout при сопоставимой величине. Только такие атомы засчитываются."
    )
    return "\n".join(lines)
