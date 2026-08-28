"""Comparing models with cross-validation, then tuning only the winner.

Two rules:

* **The test set is touched exactly once**, at the very end, by `evaluate.py`.
  Every decision here -- which model, which hyper-parameters -- is made on
  cross-validation over the training set alone. Selecting on test scores makes
  the test score a training score in disguise.

* **Preprocessing lives inside the pipeline**, so each CV fold re-fits it. See
  the docstring in `features.py` for what goes wrong otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, cross_validate

from src.models import SEARCH_SPACES, candidates

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


@dataclass
class CVResult:
    name: str
    mae: float
    mae_std: float
    rmse: float
    r2: float
    fit_seconds: float

    def __str__(self) -> str:
        return (
            f"{self.name:<20} MAE {self.mae:>9,.0f} ±{self.mae_std:>6,.0f}   "
            f"RMSE {self.rmse:>9,.0f}   R² {self.r2:>6.3f}   {self.fit_seconds:>5.1f}s"
        )


def compare(X: pd.DataFrame, y: pd.Series, *, folds: int = 5, seed: int = 42) -> list[CVResult]:
    """Cross-validate every candidate on the training set."""
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    results = []

    for name, pipeline in candidates(seed).items():
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        results.append(
            CVResult(
                name=name,
                mae=-scores["test_mae"].mean(),
                mae_std=scores["test_mae"].std(),
                rmse=-scores["test_rmse"].mean(),
                r2=scores["test_r2"].mean(),
                fit_seconds=scores["fit_time"].sum(),
            )
        )

    return sorted(results, key=lambda r: r.mae)


def tune(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    folds: int = 5,
    seed: int = 42,
) -> GridSearchCV:
    """Grid-search one model. Refits on the full training set when done."""
    pipeline = candidates(seed)[name]
    space = SEARCH_SPACES.get(name, {})

    search = GridSearchCV(
        pipeline,
        space,
        cv=KFold(n_splits=folds, shuffle=True, random_state=seed),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    return search


def save(model, path: Path | str = MODELS_DIR / "house_price_model.joblib") -> Path:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load(path: Path | str = MODELS_DIR / "house_price_model.joblib"):
    import joblib

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no model at {path} -- run `python run.py` first")
    return joblib.load(path)


def train_best(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    folds: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[object, list[CVResult]]:
    """Compare, pick the best by CV MAE, tune it, and return the fitted model."""
    results = compare(X, y, folds=folds, seed=seed)

    if verbose:
        print("  cross-validated on the training set only:\n")
        for result in results:
            print(f"    {result}")

        baseline = next(r for r in results if r.name == "baseline_median")
        best = results[0]
        lift = 100 * (baseline.mae - best.mae) / baseline.mae
        print(f"\n    best: {best.name}, {lift:.0f}% lower MAE than predicting the median")

    winner = next(r.name for r in results if r.name != "baseline_median")
    search = tune(winner, X, y, folds=folds, seed=seed)

    if verbose:
        print(f"\n  tuned {winner}: {search.best_params_}")
        print(f"    CV MAE {-search.best_score_:,.0f}")

    return search.best_estimator_, results


__all__ = ["compare", "tune", "train_best", "save", "load", "CVResult", "MODELS_DIR"]
