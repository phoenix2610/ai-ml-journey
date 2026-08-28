"""The model zoo, and the baseline that keeps everyone honest.

Every candidate is a full Pipeline (preprocessing + estimator), so nothing can
be compared unless it is comparable, and cross-validation re-fits preprocessing
inside each fold.

`DummyRegressor` is in the list on purpose. "R² = 0.83" means nothing until you
know what predicting the median scores. A model that cannot clearly beat the
dummy is not a model.
"""

from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor


def make_pipeline(estimator, *, scale: bool = True) -> Pipeline:
    return Pipeline([("pre", build_preprocessor(scale=scale)), ("model", estimator)])


def candidates(seed: int = 42) -> dict[str, Pipeline]:
    """Name -> pipeline. Ordered from 'does nothing' to 'does a lot'."""
    return {
        # The floor. Predicts the training median for every house.
        "baseline_median": make_pipeline(DummyRegressor(strategy="median")),

        # Linear, scaled. Fast, interpretable, and the honest test of whether
        # the relationship needs anything more than a hyperplane.
        "ridge": make_pipeline(Ridge(alpha=1.0)),

        # Trees split on raw values, so scaling is pointless work for them.
        "random_forest": make_pipeline(
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=seed,
                n_jobs=-1,
            ),
            scale=False,
        ),

        # Usually the winner on tabular data of this size and shape.
        "gradient_boosting": make_pipeline(
            HistGradientBoostingRegressor(
                max_iter=400,
                learning_rate=0.07,
                max_leaf_nodes=31,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=seed,
            ),
            scale=False,
        ),
    }


# Small, deliberately cheap grids. A wider search is not the lesson here, and
# an expensive one makes `run.py` something nobody runs twice.
SEARCH_SPACES: dict[str, dict[str, list]] = {
    "ridge": {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
    "random_forest": {
        "model__max_depth": [None, 16],
        "model__min_samples_leaf": [1, 2, 5],
    },
    "gradient_boosting": {
        "model__learning_rate": [0.05, 0.07, 0.12],
        "model__max_leaf_nodes": [15, 31, 63],
    },
}


__all__ = ["candidates", "make_pipeline", "SEARCH_SPACES"]
