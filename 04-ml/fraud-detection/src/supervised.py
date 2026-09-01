"""Supervised models for the labelled case, scored on PR-AUC.

**ROC-AUC is the wrong metric here and it is worth being specific about why.**
ROC plots recall against false-positive rate. With 39,880 legitimate
transactions, moving from 400 false positives to 800 changes FPR from 0.010 to
0.020 -- a rounding error on the ROC curve. To the fraud team it doubles the
review queue. Precision uses the count of *predicted* positives as its
denominator, so it feels that change immediately.

So every comparison here uses **average precision** (area under the
precision-recall curve), and the baseline to beat is the fraud rate itself:
a random classifier scores AP ≈ 0.003.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor

SCORING = {"ap": "average_precision", "roc_auc": "roc_auc", "recall": "recall"}


def make_pipeline(estimator, *, scale: bool = True) -> Pipeline:
    return Pipeline([("pre", build_preprocessor(scale=scale)), ("model", estimator)])


def candidates(seed: int = 42) -> dict[str, Pipeline]:
    return {
        # Always predicts the majority class. Scores 99.7% accuracy, catches
        # zero fraud. It is in the lineup to make that point concrete.
        "baseline_majority": make_pipeline(DummyClassifier(strategy="most_frequent")),

        # class_weight='balanced' re-weights the loss by inverse frequency --
        # the same effect oversampling aims for, without inventing rows or
        # destroying probability calibration.
        "logistic_balanced": make_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
        ),

        "random_forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                random_state=seed, n_jobs=-1,
            ),
            scale=False,
        ),

        "gradient_boosting": make_pipeline(
            HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                class_weight="balanced", early_stopping=True,
                validation_fraction=0.15, random_state=seed,
            ),
            scale=False,
        ),
    }


@dataclass
class CVResult:
    name: str
    average_precision: float
    ap_std: float
    roc_auc: float

    def __str__(self) -> str:
        return (
            f"{self.name:<20} AP {self.average_precision:>6.3f} ±{self.ap_std:.3f}   "
            f"ROC-AUC {self.roc_auc:>6.3f}"
        )


def compare(X, y, *, folds: int = 5, seed: int = 42) -> list[CVResult]:
    """Stratified CV -- unstratified folds can contain zero positives."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results = []

    for name, pipeline in candidates(seed).items():
        scores = cross_validate(
            pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=-1, error_score=0.0
        )
        results.append(
            CVResult(
                name=name,
                average_precision=scores["test_ap"].mean(),
                ap_std=scores["test_ap"].std(),
                roc_auc=scores["test_roc_auc"].mean(),
            )
        )

    return sorted(results, key=lambda r: -r.average_precision)


def fit_best(X, y, *, folds: int = 5, seed: int = 42, verbose: bool = True):
    results = compare(X, y, folds=folds, seed=seed)

    if verbose:
        chance = float(np.mean(y))
        print(f"  a random classifier scores AP ≈ {chance:.4f}\n")
        for result in results:
            print(f"    {result}")

    winner = next(r.name for r in results if r.name != "baseline_majority")
    model = candidates(seed)[winner]
    model.fit(X, y)

    if verbose:
        print(f"\n  fitted: {winner}")
    return model, winner, results


__all__ = ["candidates", "compare", "fit_best", "make_pipeline", "CVResult"]
