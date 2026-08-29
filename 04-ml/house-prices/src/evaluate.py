"""Scoring the final model, and admitting how wrong it is.

A single R² is not an evaluation. What a buyer or an underwriter needs to know
is *how far off* a prediction typically is, whether the error is worse for
cheap houses than expensive ones, and how much confidence to attach to any one
number. So this module reports:

* **MAE** in currency, because "£15k out" is a sentence people can act on.
  RMSE is reported too, but it is dominated by a few large houses.
* **MAPE**, because a £40k error on a £900k mansion is not the same mistake as
  a £40k error on a £120k flat.
* **Error by price decile**, which is where models usually turn out to be bad
  at exactly the segment you cared about.
* **Empirical prediction intervals** from the residual distribution, rather
  than a point estimate pretending to be certain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


@dataclass
class Metrics:
    mae: float
    rmse: float
    mape: float
    r2: float
    median_abs_error: float
    within_10pct: float
    n: int

    def __str__(self) -> str:
        return (
            f"    MAE                {self.mae:>12,.0f}\n"
            f"    RMSE               {self.rmse:>12,.0f}\n"
            f"    median abs error   {self.median_abs_error:>12,.0f}\n"
            f"    MAPE               {self.mape:>11.1f}%\n"
            f"    R²                 {self.r2:>12.3f}\n"
            f"    within 10% of true {self.within_10pct:>11.1f}%\n"
            f"    test rows          {self.n:>12,}"
        )


def score(y_true, y_pred) -> Metrics:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = np.abs(y_true - y_pred)

    return Metrics(
        mae=mean_absolute_error(y_true, y_pred),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mape=100 * mean_absolute_percentage_error(y_true, y_pred),
        r2=r2_score(y_true, y_pred),
        median_abs_error=float(np.median(errors)),
        within_10pct=100 * float(np.mean(errors / y_true <= 0.10)),
        n=len(y_true),
    )


def error_by_decile(y_true, y_pred) -> pd.DataFrame:
    """Where the model is weak. Aggregate metrics hide this completely."""
    frame = pd.DataFrame({"actual": np.asarray(y_true, float), "predicted": np.asarray(y_pred, float)})
    frame["error"] = frame["predicted"] - frame["actual"]
    frame["abs_error"] = frame["error"].abs()
    frame["pct_error"] = 100 * frame["error"] / frame["actual"]
    frame["decile"] = pd.qcut(frame["actual"], 10, labels=False, duplicates="drop") + 1

    out = frame.groupby("decile").agg(
        n=("actual", "size"),
        median_price=("actual", "median"),
        mae=("abs_error", "mean"),
        bias=("error", "mean"),
        mape=("pct_error", lambda s: s.abs().mean()),
    )
    return out.reset_index()


def prediction_interval(residuals, level: float = 0.90) -> tuple[float, float]:
    """Empirical interval from held-out residuals.

    Distribution-free on purpose: house-price residuals are not Gaussian, so a
    ±1.645σ interval would be confidently wrong in both tails.
    """
    alpha = (1 - level) / 2
    residuals = np.asarray(residuals, dtype=float)
    return float(np.quantile(residuals, alpha)), float(np.quantile(residuals, 1 - alpha))


def interval_coverage(y_true, y_pred, lo: float, hi: float) -> float:
    """What fraction of actuals land inside the interval. Should match `level`."""
    residuals = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return 100 * float(np.mean((residuals >= lo) & (residuals <= hi)))


def importances(model, X, y, *, n_repeats: int = 8, seed: int = 42) -> pd.DataFrame:
    """Permutation importance on the *test* set.

    Measured on held-out data because importance on training data rewards
    memorisation. Correlated features share credit, so treat these as a ranking
    rather than an attribution.
    """
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=seed,
        scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


@dataclass
class Evaluation:
    metrics: Metrics
    by_decile: pd.DataFrame
    importance: pd.DataFrame
    interval: tuple[float, float]
    coverage: float
    residuals: np.ndarray = field(repr=False, default=None)


def full_report(model, X_test, y_test, *, level: float = 0.90, seed: int = 42) -> Evaluation:
    predictions = model.predict(X_test)
    residuals = np.asarray(y_test, float) - predictions
    lo, hi = prediction_interval(residuals, level)

    return Evaluation(
        metrics=score(y_test, predictions),
        by_decile=error_by_decile(y_test, predictions),
        importance=importances(model, X_test, y_test, seed=seed),
        interval=(lo, hi),
        coverage=interval_coverage(y_test, predictions, lo, hi),
        residuals=residuals,
    )


__all__ = [
    "score", "Metrics", "error_by_decile", "prediction_interval",
    "interval_coverage", "importances", "full_report", "Evaluation",
]
