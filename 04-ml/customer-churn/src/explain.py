"""Explaining churn well enough that someone can act on it.

Two levels, answering different questions:

* **Global** (`importance`) -- what drives churn across the book? Permutation
  importance on held-out data.
* **Per customer** (`reasons_for`) -- why is *this* account at risk? Needed for
  a retention call, where "the model said 0.71" is not a conversation.

Two cautions the code enforces rather than just mentions:

**Correlated features share credit.** `tenure_months` and `total_charges` move
together, so permutation importance splits the credit between them arbitrarily.
Read it as a ranking, never as an attribution.

**Importance is not actionability.** Tenure is the strongest predictor here and
is completely useless as an intervention — you cannot make a customer older.
`actionable_drivers()` filters to the features someone could actually change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.data import CATEGORICAL, NUMERIC

# Features a retention team can actually influence. Everything else is context.
ACTIONABLE = {
    "contract": "offer a longer contract",
    "support_calls_90d": "proactive support outreach",
    "outages_90d": "fix service reliability",
    "monthly_charges": "review pricing or offer a discount",
    "payment_method": "move off electronic check to auto-pay",
    "internet_service": "review the fibre experience",
}

IMMUTABLE = {"tenure_months", "total_charges", "avg_monthly_gb", "has_dependents"}


def importance(model, X, y, *, n_repeats: int = 8, seed: int = 42) -> pd.DataFrame:
    """Permutation importance on held-out data.

    Held-out because importance on training data rewards memorisation. Scored by
    ROC-AUC drop so it measures ranking damage rather than accuracy at some
    arbitrary threshold.
    """
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=seed, scoring="roc_auc", n_jobs=-1
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
                "actionable": [c in ACTIONABLE for c in X.columns],
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def actionable_drivers(importance_frame: pd.DataFrame) -> pd.DataFrame:
    """Only the drivers a retention team can do something about."""
    out = importance_frame[importance_frame["actionable"]].copy()
    out["lever"] = out["feature"].map(ACTIONABLE)
    return out.reset_index(drop=True)


def reasons_for(
    model,
    row: pd.Series | pd.DataFrame,
    reference: pd.DataFrame,
    *,
    top: int = 3,
) -> list[dict]:
    """Why is *this* customer at risk?

    Counterfactual, one feature at a time: replace each actionable feature with
    the population's typical value and measure how far the churn probability
    falls. The drop is the contribution attributable to that feature, and it
    doubles as an estimate of what fixing it would buy.
    """
    frame = row.to_frame().T if isinstance(row, pd.Series) else row.copy()
    frame = frame.astype(reference.dtypes.to_dict(), errors="ignore")
    baseline = float(model.predict_proba(frame)[0, 1])

    effects = []
    for column in frame.columns:
        if column not in ACTIONABLE:
            continue

        counterfactual = frame.copy()
        if column in NUMERIC:
            typical = reference[column].median()
        else:
            typical = _best_category(model, frame, reference, column)
        if pd.isna(typical):
            continue
        counterfactual[column] = typical

        new_probability = float(model.predict_proba(counterfactual)[0, 1])
        delta = baseline - new_probability
        if delta > 1e-4:
            effects.append(
                {
                    "feature": column,
                    "current": frame[column].iloc[0],
                    "suggested": typical,
                    "risk_reduction": delta,
                    "lever": ACTIONABLE[column],
                }
            )

    effects.sort(key=lambda e: -e["risk_reduction"])
    return effects[:top]


def _best_category(model, frame, reference, column):
    """The category level that minimises predicted churn for this customer."""
    best, best_probability = None, np.inf
    for level in reference[column].dropna().unique():
        trial = frame.copy()
        trial[column] = level
        probability = float(model.predict_proba(trial)[0, 1])
        if probability < best_probability:
            best, best_probability = level, probability
    return best


def risk_table(model, X: pd.DataFrame, *, ids: pd.Series | None = None) -> pd.DataFrame:
    """Every customer with a churn probability and a risk band."""
    probabilities = model.predict_proba(X)[:, 1]
    frame = pd.DataFrame({"churn_probability": probabilities}, index=X.index)
    if ids is not None:
        frame.insert(0, "customer_id", ids.values)

    frame["risk_band"] = pd.cut(
        frame["churn_probability"],
        bins=[0, 0.20, 0.40, 0.60, 1.01],
        labels=["low", "medium", "high", "critical"],
        include_lowest=True,
    )
    return frame.sort_values("churn_probability", ascending=False)


__all__ = [
    "importance", "actionable_drivers", "reasons_for", "risk_table",
    "ACTIONABLE", "IMMUTABLE",
]
