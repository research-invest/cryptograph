"""
Меняют ли SMC-признаки структуру состояний рынка.

Это самая честная и самая дешёвая проверка гипотезы SMC целиком
(`docs/task_smc_integration.md`, фаза 3). Логика такая:

    Кластеризация ищет разделимые облака в пространстве признаков. Если
    добавление двенадцати структурных величин к тридцати двум существующим
    не меняет ни число состояний, ни их границы, — значит d-prime не нашёл
    ни одного нового разделимого облака. А это ровно означает, что SMC не
    несёт информации сверх того, что уже видно в свечах и объёмах.

Отрицательный результат здесь ценнее положительного, потому что он дешёвый
и однозначный: он закрывает задачу, не требуя ни переобучения на продакшене,
ни перекалибровки профилей оценки.

Скрипт **ничего не пишет** — ни в БД, ни в файлы. Он загружает бары, считает
оба набора признаков, обучает две модели состояний на одной и той же
конфигурации и сравнивает результат.

    python3 scripts/compare_state_granularity.py --symbol BTCUSDT
    python3 scripts/compare_state_granularity.py --all
    python3 scripts/compare_state_granularity.py --start 2022-01-01   # быстрее

Сравнение делается по трём величинам:

  число состояний   прямой ответ на вопрос «стало ли различимее»
  ARI               насколько совпали сами разбиения (1.0 — одинаковые)
  доля SMC в топе   как часто SMC-признак попадает в top_features состояния
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from btcproc import config, symbols  # noqa: E402
from btcproc.features import builder as feat  # noqa: E402
from btcproc.features import smc  # noqa: E402
from btcproc.ingest import bars  # noqa: E402
from btcproc.states import clustering  # noqa: E402


def features_for(symbol: str, start: str | None, enabled: bool):
    """Вектор признаков при заданном состоянии флага SMC."""
    # config.smc читается внутри build_features в момент вызова, поэтому
    # подменять достаточно здесь.
    saved = config.smc
    config.smc = config.SMCConfig(enabled=enabled)
    try:
        base = bars.load_ohlcv(symbol, config.data.base_tf, start, None)
        if base.empty:
            raise SystemExit(f"Нет баров по {symbol}")
        context = {
            tf: bars.load_ohlcv(symbol, tf, start, None)
            for tf in config.data.context_tfs
        }
        return feat.build_features(base, context)
    finally:
        config.smc = saved


def cluster(features, symbol: str):
    """
    Кластеризация с порогами ЭТОЙ монеты.

    Пороги дробления помонетные (`states_overrides` в symbols.py), и брать
    глобальный конфиг значило бы считать ETH по линейке биткоина — то есть
    получать числа состояний, не совпадающие с продакшеном.
    """
    scale = feat.robust_scale_params(features)
    matrix = feat.apply_scale(features, scale)
    started = time.perf_counter()
    model, labels = clustering.fit_states(
        matrix, list(features.columns), scale,
        cfg=symbols.get(symbol).states_config(),
    )
    return model, labels, time.perf_counter() - started


def shuffle_smc(features, seed: int = 42):
    """
    Контроль: SMC-признаки перемешаны по строкам.

    Размерность та же, распределение каждого признака в точности то же,
    связь с рынком разрушена полностью. Это единственный способ отличить
    «SMC изменил структуру состояний» от «структура изменилась оттого, что
    признаков стало 44 вместо 32».

    Различие принципиальное: пороги дробления и слияния (`split_gain`,
    `merge_separation`) калибровались на 32-мерном пространстве, а силуэт и
    d-prime от размерности зависят. Без этого контроля любое изменение числа
    состояний можно с равным успехом объяснить геометрией.
    """
    rng = np.random.default_rng(seed)
    control = features.copy()
    order = rng.permutation(len(control))
    for name in smc.FEATURE_CANDIDATES:
        if name in control.columns:
            control[name] = control[name].to_numpy()[order]
    return control


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    """
    Adjusted Rand Index — совпадение двух разбиений с поправкой на случайность.

    1.0 — разбиения идентичны, 0.0 — совпадают не лучше случайного. Именно ARI,
    а не доля совпавших меток: номера кластеров в двух прогонах произвольны и
    прямое сравнение меток бессмысленно.
    """
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(a, b))


def top_feature_share(features, labels, model) -> float:
    """
    Доля состояний, у которых хотя бы один SMC-признак попал в тройку самых
    отклонившихся от среднего рынка.

    Это то же, из чего строится имя состояния (`states/naming.py`), поэтому
    величина отвечает на понятный вопрос: изменится ли то, как состояния
    описываются человеку.
    """
    scaled = feat.apply_scale(features, model.scale)
    smc_columns = {i for i, name in enumerate(features.columns)
                   if name in smc.FEATURE_CANDIDATES}
    if not smc_columns:
        return 0.0

    hits = 0
    for group in np.unique(labels):
        centroid = scaled[labels == group].mean(axis=0)
        top3 = set(np.argsort(-np.abs(centroid))[:3])
        if top3 & smc_columns:
            hits += 1
    return hits / max(len(np.unique(labels)), 1)


def process(symbol: str, start: str | None) -> None:
    print(f"\n{'=' * 70}\n=== {symbol}\n{'=' * 70}")

    print("Считаю базовый набор (32 признака)…")
    f_base = features_for(symbol, start, enabled=False)
    m_base, l_base, t_base = cluster(f_base, symbol)
    print(f"  признаков {f_base.shape[1]}, строк {len(f_base)}, "
          f"состояний {m_base.n_groups}, кластеризация {t_base:.0f} c")

    print("Считаю набор с SMC (44 признака)…")
    f_smc = features_for(symbol, start, enabled=True)
    m_smc, l_smc, t_smc = cluster(f_smc, symbol)
    print(f"  признаков {f_smc.shape[1]}, строк {len(f_smc)}, "
          f"состояний {m_smc.n_groups}, кластеризация {t_smc:.0f} c")

    print("Контроль: те же 44 признака, SMC-блок перемешан…")
    m_ctl, l_ctl, t_ctl = cluster(shuffle_smc(f_smc), symbol)
    print(f"  состояний {m_ctl.n_groups}, кластеризация {t_ctl:.0f} c")

    # Наборы строк могут разойтись: SMC-признаки без NaN, но join меняет
    # порядок прогрева. Сравниваем по пересечению.
    common = f_base.index.intersection(f_smc.index)
    base_labels = l_base[f_base.index.isin(common)]
    smc_labels = l_smc[f_smc.index.isin(common)]

    ari = adjusted_rand(base_labels, smc_labels)
    share = top_feature_share(f_smc, l_smc, m_smc)

    print(f"\n  состояний:        {m_base.n_groups} → {m_smc.n_groups} "
          f"({m_smc.n_groups - m_base.n_groups:+d})")
    print(f"  контроль (шум):   {m_ctl.n_groups}")
    print(f"  ARI разбиений:    {ari:.3f}")
    print(f"  SMC в топ-3:      {share:.0%} состояний")
    print(f"  общих строк:      {len(common)}")

    print("\n  Вывод:", verdict(m_base.n_groups, m_smc.n_groups, m_ctl.n_groups,
                                ari, share))


def verdict(n_base: int, n_smc: int, n_control: int, ari: float,
            share: float) -> str:
    """
    Вывод делается по сравнению с контролем, а не с базой.

    Контроль отвечает на вопрос «сколько состояний получилось бы от одного
    только роста размерности». Если SMC и шум дают примерно одно и то же,
    информации в SMC нет, сколько бы ни изменилось относительно базы.
    """
    from_dimension = abs(n_control - n_base) / max(n_base, 1)
    from_smc = abs(n_smc - n_control) / max(n_control, 1)

    if from_smc < 0.15:
        return (f"{n_base} → {n_smc} состояний, но перемешанные признаки дают "
                f"{n_control} — то же самое. Изменение объясняется ростом "
                "размерности, а не информацией. SMC не добавляет разделимости.")
    if from_dimension > 0.15:
        return ("часть изменения объясняется размерностью, часть — нет. "
                "Смотреть лифт, но помнить про поправку на размерность.")
    return ("структура изменилась сверх контроля — d-prime нашёл новые "
            "разделимые облака. Требуется замер лифта.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Сравнение гранулярности состояний")
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--start", help="Не вся история — быстрее, но грубее")
    args = parser.parse_args()

    if args.all:
        targets = symbols.tickers(only_enabled=True)
    elif args.symbol:
        targets = args.symbol
    else:
        targets = [config.data.symbol]

    for symbol in targets:
        process(symbol, args.start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
