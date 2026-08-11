"""
Адаптивная кластеризация состояний рынка.

Число состояний заранее не задаётся — оно вырастает из данных, как требует ТЗ:

  1. стартовое грубое разбиение на несколько крупных облаков;
  2. **дробление**: группа делится надвое, если внутри неё есть структура —
     проверяется силуэтом разбиения против силуэта нескольких случайных
     равномерных выборок той же формы (gap statistic), с порогом в сигмах
     этих выборок. Без такой поправки KMeans всегда «находит» два кластера
     даже в однородном шаре; без сигм порог зависел бы от размерности, и
     каждый новый источник признаков менял бы гранулярность графа сам по
     себе (см. `_split_gain`);
  3. **слияние**: пара групп схлопывается, если расстояние между центроидами
     меньше их собственного разброса — то есть различие между ними меньше
     внутреннего шума.

Модель — это набор центроидов в нормированном пространстве признаков.
Присвоение состояния новой точке = ближайший центроид.
"""
from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

from btcproc import config

logger = logging.getLogger(__name__)


@dataclass
class StateModel:
    """Всё, что нужно, чтобы присвоить group_id любой точке."""

    feature_names: list[str]
    scale: dict[str, np.ndarray]
    centroids: np.ndarray            # (n_groups, n_features), нормированное пространство
    group_ids: np.ndarray            # (n_groups,) float — те самые group_id из схемы
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def n_groups(self) -> int:
        return len(self.group_ids)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Ближайший центроид для каждой строки. Возвращает float group_id."""
        # (n, 1, d) - (1, k, d) → расстояния до всех центроидов за один проход.
        distances = np.linalg.norm(x[:, None, :] - self.centroids[None, :, :], axis=2)
        return self.group_ids[distances.argmin(axis=1)]

    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "scale": {k: v.tolist() for k, v in self.scale.items()},
            "centroids": self.centroids.tolist(),
            "group_ids": self.group_ids.tolist(),
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "StateModel":
        return cls(
            feature_names=raw["feature_names"],
            scale={k: np.asarray(v, dtype=float) for k, v in raw["scale"].items()},
            centroids=np.asarray(raw["centroids"], dtype=float),
            group_ids=np.asarray(raw["group_ids"], dtype=float),
            params=raw.get("params", {}),
        )


def _subsample(x: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) <= size:
        return x
    idx = rng.choice(len(x), size=size, replace=False)
    return x[idx]


def _split_gain(x: np.ndarray, rng: np.random.Generator, cfg: config.StatesConfig) -> float:
    """
    Насколько разбиение надвое лучше, чем то же разбиение случайного облака.

    Полная форма gap statistic Тибширани: референсов несколько, а порог
    выражен в их собственных сигмах, а не в абсолютных долях силуэта.
    Возвращает уже величину со ВЗЯТЫМ порогом, поэтому решение о дроблении —
    это просто `gain > 0`.

        gain = silhouette(real) − mean(silhouette(refs)) − k·σ(refs)

    Почему не «real − ref» с абсолютным порогом, как было до 2026-08-11:

    * **B = 1.** Силуэт случайного облака — случайная величина со своей
      дисперсией. Сравнение с одним draw при пороге 0.02 означало, что
      решения вблизи границы определял шум этого draw. `rng` засеян, поэтому
      на идентичных данных всё воспроизводилось, — но лишняя неделя баров
      меняла подвыборку, меняла draw, и решения переворачивались. Отсюда
      разброс числа состояний у ETH 29 → 26 → 42 между прогонами.
    * **Абсолютный порог зависит от размерности.** В 44 измерениях
      концентрация расстояний сжимает и real, и ref, но неодинаково, и
      константа, подобранная на 32, означает там другую строгость. Из-за
      этого граф обрушивали двенадцать признаков ЛЮБОЙ природы, включая
      случайные, — свойство калибровки, а не источника.

    Порог в сигмах самонормируется: сигма меряется в тех же единицах, в
    которых сжался силуэт. Стоимость — B прогонов KMeans по референсу вместо
    одного, но с `n_init=1`: референс однороден, множественные старты ему
    ничего не дают, а B прогонов по три старта утроили бы счёт.

    **Чего это НЕ чинит.** Замер 2026-08-11: на крупных группах BTC `gain`
    по восьми разным подвыборкам силуэта даёт +0.032 ± 0.007 (решение
    «дробить» 8 из 8) и −0.019 ± 0.010 (1 из 8). То есть сам критерий вдали
    от границы воспроизводим уверенно, а у границы переворачивается — и
    дальше рекурсия усиливает единственный перевёрнутый выбор во всё
    поддерево. Поэтому число состояний остаётся чувствительным к малым
    изменениям данных у монет, чьё дерево упирается в `max_depth` (у SOL —
    упирается). Лечится это не порогом, а `states_overrides` монеты; разбор —
    development_log.md, 21.14.
    """
    sample = _subsample(x, cfg.silhouette_sample, rng)
    if len(sample) < 50:
        return -1.0

    labels = KMeans(n_clusters=2, n_init=3, random_state=cfg.random_state).fit_predict(sample)
    if len(np.unique(labels)) < 2:
        return -1.0
    real = silhouette_score(sample, labels)

    # Референсы: равномерные облака в том же bounding box.
    low, high = sample.min(axis=0), sample.max(axis=0)
    draws = max(1, cfg.split_reference_draws)
    refs = np.empty(draws, dtype=float)
    for i in range(draws):
        reference = rng.uniform(low, high, size=sample.shape)
        ref_labels = KMeans(
            n_clusters=2, n_init=1, random_state=cfg.random_state + i
        ).fit_predict(reference)
        if len(np.unique(ref_labels)) < 2:
            refs[i] = 0.0
            continue
        refs[i] = silhouette_score(reference, ref_labels)

    if draws < 2:
        # Один референс — сигму оценить не по чему; ведём себя как старая
        # реализация, чтобы конфигурация с B=1 хотя бы не врала знаком.
        return float(real - refs.mean())

    # Поправка √(1 + 1/B) — стандартная у Тибширани: сигма референса сама
    # оценена по B наблюдениям, и её собственная неопределённость входит
    # в порог.
    sigma = float(refs.std(ddof=1)) * math.sqrt(1.0 + 1.0 / draws)
    return float(real - refs.mean() - cfg.split_gain_sigma * sigma)


def effective_min_group_size(cfg: config.StatesConfig, n_samples: int) -> int:
    """
    Порог дробления, приведённый к длине истории.

    `min_group_size = 800` подобран под ~300 тыс. баров BTC 15m с 2017 года,
    то есть это 0.27% истории. На монете с двумя годами торгов (~70 тыс. баров)
    те же 800 — уже 1.1%, и дробление останавливается заметно раньше: граф
    выходит грубее не потому, что рынок однороднее, а потому что порог
    абсолютный.

    Поэтому берём максимум из доли истории и абсолютного минимума. Абсолютный
    минимум остаётся нижней границей осознанно: группа в сотню баров не даёт
    кандидату статистики при любой длине истории.
    """
    scaled = round(cfg.min_group_share * n_samples)
    return max(cfg.min_group_size, int(scaled))


def _recursive_split(
    x: np.ndarray,
    indices: np.ndarray,
    depth: int,
    rng: np.random.Generator,
    cfg: config.StatesConfig,
    out: list[np.ndarray],
    trace: list[dict],
) -> None:
    size = len(indices)
    if depth >= cfg.max_depth or size < 2 * cfg.min_group_size:
        out.append(indices)
        return

    subset = x[indices]
    # Порог уже внутри gain (в сигмах референса), поэтому сравнение с нулём.
    gain = _split_gain(subset, rng, cfg)
    if gain <= 0.0:
        trace.append({"depth": depth, "size": size, "gain": round(gain, 4), "action": "keep"})
        out.append(indices)
        return

    labels = KMeans(n_clusters=2, n_init=5, random_state=cfg.random_state).fit_predict(subset)
    left, right = indices[labels == 0], indices[labels == 1]
    # Разбиение, оставляющее огрызок, пользы не приносит — статистика по нему
    # всё равно не наберётся.
    if min(len(left), len(right)) < cfg.min_group_size:
        trace.append({"depth": depth, "size": size, "gain": round(gain, 4), "action": "too_small"})
        out.append(indices)
        return

    trace.append({
        "depth": depth, "size": size, "gain": round(gain, 4),
        "action": "split", "into": [len(left), len(right)],
    })
    _recursive_split(x, left, depth + 1, rng, cfg, out, trace)
    _recursive_split(x, right, depth + 1, rng, cfg, out, trace)


def _separation(x: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    """
    Насколько две группы различимы — расстояние центроидов, выраженное
    в их собственных разбросах **вдоль оси между центроидами**.

    Полный радиус группы для этого не годится: в 30-мерном пространстве он
    растёт как √d и делает любые две группы «почти одинаковыми», из-за чего
    слияние схлопывает весь граф в пару узлов. Проекция на ось сравнения от
    размерности не зависит: результат — привычное d-prime, где < 1 означает
    сильно перекрывающиеся облака.
    """
    c_left, c_right = x[left].mean(axis=0), x[right].mean(axis=0)
    axis = c_right - c_left
    distance = float(np.linalg.norm(axis))
    if distance < 1e-12:
        return 0.0
    axis /= distance
    std_left = float(np.std(x[left] @ axis))
    std_right = float(np.std(x[right] @ axis))
    total = std_left + std_right
    return distance / total if total > 0 else np.inf


def _merge_close(
    x: np.ndarray,
    groups: list[np.ndarray],
    cfg: config.StatesConfig,
    trace: list[dict],
) -> list[np.ndarray]:
    """
    Слияние неразличимых групп.

    На каждом шаге сливается самая неразличимая пара, пока есть пары с
    separation ниже порога — так схлопываются и цепочки почти одинаковых
    состояний.
    """
    changed = True
    while changed and len(groups) > 1:
        changed = False
        best_pair, best_sep = None, np.inf
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                sep = _separation(x, groups[i], groups[j])
                if sep < cfg.merge_separation and sep < best_sep:
                    best_pair, best_sep = (i, j), sep

        if best_pair is not None:
            i, j = best_pair
            trace.append({
                "action": "merge", "separation": round(best_sep, 4),
                "sizes": [len(groups[i]), len(groups[j])],
            })
            merged = np.concatenate([groups[i], groups[j]])
            groups = [g for k, g in enumerate(groups) if k not in (i, j)] + [merged]
            changed = True

    return groups


def fit_states(
    x: np.ndarray,
    feature_names: list[str],
    scale: dict[str, np.ndarray],
    cfg: config.StatesConfig | None = None,
    progress=None,
) -> tuple[StateModel, np.ndarray]:
    """
    Строит модель состояний по нормированной матрице признаков.

    Возвращает модель и метки для переданных строк (без сглаживания —
    сглаживанием занимается src/states/assign.py).
    """
    cfg = cfg or config.states

    # Порог дробления приводится к длине истории ОДИН раз здесь, а не в каждом
    # месте использования: дальше по коду cfg.min_group_size уже эффективный.
    scaled_min_group = effective_min_group_size(cfg, len(x))
    if scaled_min_group != cfg.min_group_size:
        logger.info(
            "min_group_size: %d → %d (доля %.4f от %d баров)",
            cfg.min_group_size, scaled_min_group, cfg.min_group_share, len(x),
        )
    cfg = dataclasses.replace(cfg, min_group_size=scaled_min_group)

    rng = np.random.default_rng(cfg.random_state)
    trace: list[dict] = []

    if progress:
        progress("стартовое разбиение", 0.1)
    seed_labels = MiniBatchKMeans(
        n_clusters=cfg.seed_clusters,
        n_init=5,
        random_state=cfg.random_state,
        batch_size=4096,
    ).fit_predict(x)

    groups: list[np.ndarray] = []
    for label in np.unique(seed_labels):
        indices = np.flatnonzero(seed_labels == label)
        _recursive_split(x, indices, 0, rng, cfg, groups, trace)
    if progress:
        progress(f"после дробления: {len(groups)} групп", 0.6)

    groups = _merge_close(x, groups, cfg, trace)
    if progress:
        progress(f"после слияния: {len(groups)} групп", 0.8)

    # Крупные группы получают меньшие номера — «1.0» становится самым частым
    # состоянием рынка, что удобно читать в графе.
    groups.sort(key=len, reverse=True)
    centroids = np.array([x[g].mean(axis=0) for g in groups])
    group_ids = np.arange(1, len(groups) + 1, dtype=float)

    labels = np.empty(len(x), dtype=float)
    for gid, indices in zip(group_ids, groups):
        labels[indices] = gid

    model = StateModel(
        feature_names=feature_names,
        scale=scale,
        centroids=centroids,
        group_ids=group_ids,
        params={
            "seed_clusters": cfg.seed_clusters,
            # Уже приведённый к длине истории — именно он применялся.
            "min_group_size": cfg.min_group_size,
            "min_group_share": cfg.min_group_share,
            "max_depth": cfg.max_depth,
            "split_reference_draws": cfg.split_reference_draws,
            "split_gain_sigma": cfg.split_gain_sigma,
            "merge_separation": cfg.merge_separation,
            "n_samples": int(len(x)),
            "trace": trace[-200:],
        },
    )
    logger.info("Состояний найдено: %d на %d точках", model.n_groups, len(x))
    return model, labels
