"""
Гейт N (новизна): out-of-sample R² внешней величины на вектор признаков.

Первым потребителем был Fear & Greed Index (docs/task_fear_greed.md §1.2,
`fgi_residual`), поэтому модуль назывался `fgi_novelty.py`; с появлением
второго потребителя (деривативные метрики, `docs/tz_deriv_ingest_14-08-26.md`
§5) переименован — содержимое обобщено по построению с самого начала
(`novelty_r2` принимает любой ряд), привязки к FGI в коде не было, только в
имени файла.

Живёт отдельно от модулей-источников (`features/fear_greed.py`,
`features/deriv.py`) намеренно — см. докстринг `fear_greed.py`:
`FeatureSource.compute(base, symbol)` получает только сырые бары, а гейту N
нужен уже посчитанный вектор из 32 базовых признаков. Это МЕРА, а не
продакшен-канал: её считают измерительные скрипты (`fgi_frequencies.py`,
`deriv_frequencies.py`, `measure_fgi_range.py`, `measure_feature_gradation.py`),
у которых полный вектор уже есть через `features/builder.py`.

Коэффициенты подбираются ТОЛЬКО на обучающей части (первые `train_frac` по
времени) и применяются к обеим половинам — иначе коэффициенты, подобранные
на полной истории, протащат в остаток информацию из будущего (§1.2 задачи FGI).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def base_feature_columns(features: pd.DataFrame) -> list[str]:
    """
    Признаки БЕЗ вклада зарегистрированных источников (SMC, FGI и т.п.) —
    ровно 32 «базовых» v1, на которые регрессируется индекс по ТЗ.
    """
    from btcproc.features import registry

    foreign = set(registry.all_feature_columns())
    return [c for c in features.columns if c not in foreign]


def novelty_r2(
    fgi_level: pd.Series, features: pd.DataFrame, train_frac: float = 0.7,
    columns: list[str] | None = None,
):
    """
    Out-of-sample R² регрессии fgi_level на columns (по умолчанию —
    base_feature_columns). Возвращает (r2, model, top5, columns, valid_index).
    """
    from sklearn.linear_model import LinearRegression

    columns = columns or base_feature_columns(features)
    y = fgi_level.to_numpy(dtype=float)
    X = features[columns].to_numpy(dtype=float)
    valid = np.isfinite(y)
    y, X = y[valid], X[valid]
    index = features.index[valid]

    cut = int(len(y) * train_frac)
    model = LinearRegression()
    model.fit(X[:cut], y[:cut])
    pred_test = model.predict(X[cut:])
    ss_res = float(np.sum((y[cut:] - pred_test) ** 2))
    ss_tot = float(np.sum((y[cut:] - y[cut:].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    contrib = np.abs(model.coef_) * X[:cut].std(axis=0)
    order = np.argsort(-contrib)[:5]
    top5 = [(columns[i], float(contrib[i])) for i in order]
    return r2, model, top5, columns, index, cut


def compute_fgi_residual(
    fgi_level: pd.Series, features: pd.DataFrame, train_frac: float = 0.7,
    columns: list[str] | None = None,
) -> pd.Series:
    """
    fgi_residual — часть fgi_level, которую вектор признаков не объясняет.

    Коэффициенты OLS подбираются на первых `train_frac` строк (по времени) и
    применяются ко ВСЕЙ истории. Возвращает Series по индексу `features`,
    NaN там, где fgi_level или какой-то из columns недоступен.
    """
    r2, model, _top5, columns, index, _cut = novelty_r2(
        fgi_level, features, train_frac, columns
    )
    X_full = features.loc[index, columns].to_numpy(dtype=float)
    y_full = fgi_level.loc[index].to_numpy(dtype=float)
    residual = y_full - model.predict(X_full)
    return pd.Series(residual, index=index, name="fgi_residual").reindex(features.index)
