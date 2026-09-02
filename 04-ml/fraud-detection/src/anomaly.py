"""Unsupervised detection, for the regime where labels do not exist.

Real fraud teams rarely start with labels. Chargebacks arrive weeks late, most
fraud is never reported, and a brand-new attack pattern has no examples by
definition. So the honest comparison is not "which model is best" but "how much
does supervision actually buy?"

`IsolationForest` is trained here on **legitimate transactions only** —
modelling normal behaviour and flagging departures from it. That is the regime
where it makes sense. Trained on contaminated data it quietly learns that fraud
is normal too.

Expect it to underperform the supervised models badly. That gap is the result
worth reporting: it quantifies the value of labels rather than assuming it.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor


class AnomalyDetector:
    """Wraps an unsupervised detector so it presents a supervised-style API.

    `score(X)` returns "higher = more suspicious", matching the orientation of
    `predict_proba(...)[:, 1]`, so both model families flow through the same
    evaluation and threshold code.
    """

    def __init__(self, estimator, *, scale: bool = True) -> None:
        self.pipeline = Pipeline(
            [("pre", build_preprocessor(scale=scale)), ("model", estimator)]
        )

    def fit(self, X, y=None) -> "AnomalyDetector":
        self.pipeline.fit(X)
        return self

    def fit_on_legitimate(self, X, y) -> "AnomalyDetector":
        """Train on the negative class only -- the point of the method.

        Fitting on everything teaches the detector that fraud is part of normal
        behaviour, which is precisely what it is supposed to find surprising.
        """
        y = np.asarray(y)
        return self.fit(X[y == 0])

    def score(self, X) -> np.ndarray:
        # sklearn's score_samples is "higher = more normal"; invert it.
        return -self.pipeline.named_steps["model"].score_samples(
            self.pipeline.named_steps["pre"].transform(X)
        )

    # Lets an AnomalyDetector be passed anywhere a classifier is expected.
    def predict_proba(self, X) -> np.ndarray:
        raw = self.score(X)
        lo, hi = raw.min(), raw.max()
        normalised = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
        return np.column_stack([1 - normalised, normalised])


def isolation_forest(seed: int = 42, contamination: float = 0.01) -> AnomalyDetector:
    """Isolates points by random splitting -- outliers need fewer splits."""
    return AnomalyDetector(
        IsolationForest(
            n_estimators=200,
            contamination=contamination,
            max_samples=min(4096, 256 * 8),
            random_state=seed,
            n_jobs=-1,
        )
    )


def local_outlier_factor(neighbors: int = 25, contamination: float = 0.01) -> AnomalyDetector:
    """Density-based: flags points in sparser regions than their neighbours."""
    return AnomalyDetector(
        LocalOutlierFactor(
            n_neighbors=neighbors, contamination=contamination, novelty=True, n_jobs=-1
        )
    )


def detectors(seed: int = 42) -> dict[str, AnomalyDetector]:
    return {
        "isolation_forest": isolation_forest(seed),
        "local_outlier_factor": local_outlier_factor(),
    }


__all__ = [
    "AnomalyDetector", "isolation_forest", "local_outlier_factor", "detectors",
]
