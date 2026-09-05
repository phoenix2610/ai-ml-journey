"""The churn model, and why its probabilities have to be trustworthy.

For fraud, scores only had to *rank* — a threshold turned them into decisions
and the absolute value never mattered. Churn is different. Retention budgeting
multiplies probability by customer value:

    expected_loss = P(churn) × monthly_charges × expected_remaining_months

If the model says 0.30 and the true rate for such customers is 0.55, that
arithmetic is wrong and every downstream spending decision inherits the error.
So this module ends with a **calibration check**, not just a discrimination
score. A model with excellent ROC-AUC can still be badly calibrated, and gradient
boosting usually is — it pushes probabilities toward 0 and 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import CATEGORICAL, NUMERIC


def build_preprocessor(scale: bool = True, extra_numeric: tuple = ()) -> ColumnTransformer:
    """Column-explicit, with remainder='drop'.

    Naming the columns rather than taking "everything except the target" is a
    structural defence against leakage: a stray column added upstream cannot
    silently become a feature. `extra_numeric` exists only so run.py can force
    the leaky column in and show what it would have done.
    """
    numeric = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric.append(("scale", StandardScaler()))

    return ColumnTransformer(
        [
            ("num", Pipeline(numeric), NUMERIC + list(extra_numeric)),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(estimator, *, scale: bool = True, extra_numeric: tuple = ()) -> Pipeline:
    return Pipeline(
        [("pre", build_preprocessor(scale=scale, extra_numeric=extra_numeric)), ("model", estimator)]
    )


def candidates(seed: int = 42, extra_numeric: tuple = ()) -> dict[str, Pipeline]:
    return {
        "baseline_majority": make_pipeline(DummyClassifier(strategy="prior"), extra_numeric=extra_numeric),
        # Naturally well calibrated -- it optimises log loss directly.
        "logistic": make_pipeline(LogisticRegression(max_iter=2000, C=1.0), extra_numeric=extra_numeric),
        "random_forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=4, random_state=seed, n_jobs=-1
            ),
            scale=False, extra_numeric=extra_numeric,
        ),
        "gradient_boosting": make_pipeline(
            HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                early_stopping=True, validation_fraction=0.15, random_state=seed,
            ),
            scale=False, extra_numeric=extra_numeric,
        ),
    }


@dataclass
class CVResult:
    name: str
    roc_auc: float
    brier: float
    log_loss: float

    def __str__(self) -> str:
        return (
            f"{self.name:<20} ROC-AUC {self.roc_auc:>6.3f}   "
            f"Brier {self.brier:>6.4f}   log-loss {self.log_loss:>6.4f}"
        )


def compare(X, y, *, folds: int = 5, seed: int = 42) -> list[CVResult]:
    """Discrimination *and* calibration, because both matter here."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scoring = {
        "roc_auc": "roc_auc",
        "brier": "neg_brier_score",
        "log_loss": "neg_log_loss",
    }

    results = []
    for name, pipeline in candidates(seed).items():
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        results.append(
            CVResult(
                name=name,
                roc_auc=scores["test_roc_auc"].mean(),
                brier=-scores["test_brier"].mean(),
                log_loss=-scores["test_log_loss"].mean(),
            )
        )
    return sorted(results, key=lambda r: r.brier)


def calibrate(pipeline, X, y, *, method: str = "isotonic", folds: int = 5):
    """Wrap a fitted-on-CV pipeline so its outputs are honest probabilities.

    Isotonic is used because it makes no shape assumption; with 12k rows there
    is enough data for it. On a few hundred rows, 'sigmoid' is the safer choice.
    """
    calibrated = CalibratedClassifierCV(pipeline, method=method, cv=folds)
    calibrated.fit(X, y)
    return calibrated


def calibration_report(y_true, probabilities, *, bins: int = 10) -> pd.DataFrame:
    """Predicted probability vs observed frequency, bucketed."""
    observed, predicted = calibration_curve(y_true, probabilities, n_bins=bins, strategy="quantile")
    return pd.DataFrame(
        {
            "predicted": predicted,
            "observed": observed,
            "gap": np.asarray(observed) - np.asarray(predicted),
        }
    )


def max_calibration_gap(y_true, probabilities, *, bins: int = 10) -> float:
    """Worst absolute miss between predicted and observed rate."""
    return float(calibration_report(y_true, probabilities, bins=bins)["gap"].abs().max())


def fit_best(X, y, *, folds: int = 5, seed: int = 42, verbose: bool = True):
    results = compare(X, y, folds=folds, seed=seed)

    if verbose:
        print("  cross-validated on the training set:\n")
        for result in results:
            print(f"    {result}")

    winner = next(r.name for r in results if r.name != "baseline_majority")
    raw = candidates(seed)[winner]
    raw.fit(X, y)
    calibrated = calibrate(candidates(seed)[winner], X, y, folds=folds)

    if verbose:
        print(f"\n  selected {winner} (lowest Brier score), then calibrated")

    return raw, calibrated, winner, results


__all__ = [
    "candidates", "compare", "fit_best", "calibrate", "make_pipeline",
    "build_preprocessor", "calibration_report", "max_calibration_gap",
    "CVResult", "brier_score_loss", "roc_auc_score",
]
