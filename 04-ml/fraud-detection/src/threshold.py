"""Choosing the decision threshold by cost, not by 0.5.

A classifier outputs a score. Turning that into "block this transaction"
requires a threshold, and **0.5 is an arbitrary default that is essentially
never right for imbalanced problems.** With `class_weight='balanced'` the
scores are deliberately re-weighted, so 0.5 does not even mean "50% likely".

The only principled way to pick one is to state what the two errors cost:

* **False negative** -- fraud goes through. Cost is the transaction amount,
  which the dataset actually knows per row.
* **False positive** -- a legitimate customer is declined. Cost is the analyst
  review plus the goodwill hit. Fixed per incident.

Then sweep every threshold and pick the cheapest. The result is a business
decision expressed in currency, which is also a number a fraud team can argue
with -- unlike "we used 0.5".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

# Defaults are illustrative; a real deployment gets these from finance.
REVIEW_COST = 12.0          # analyst time + customer friction per false positive
RECOVERY_RATE = 0.35        # fraction of a caught fraud actually recovered


@dataclass
class ThresholdChoice:
    threshold: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    total_cost: float
    fraud_caught_value: float
    review_cost: float

    def __str__(self) -> str:
        return (
            f"    threshold        {self.threshold:>10.4f}\n"
            f"    precision        {self.precision:>10.1%}\n"
            f"    recall           {self.recall:>10.1%}\n"
            f"    caught / missed  {self.true_positives:>6} / {self.false_negatives}\n"
            f"    false alarms     {self.false_positives:>10,}\n"
            f"    net cost         {self.total_cost:>10,.0f}"
        )


def cost_at(
    y_true,
    scores,
    threshold: float,
    amounts,
    *,
    review_cost: float = REVIEW_COST,
    recovery_rate: float = RECOVERY_RATE,
) -> ThresholdChoice:
    """Expected cost of operating at one threshold."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    amounts = np.asarray(amounts, dtype=float)

    flagged = scores >= threshold
    tp = int(np.sum(flagged & (y_true == 1)))
    fp = int(np.sum(flagged & (y_true == 0)))
    fn = int(np.sum(~flagged & (y_true == 1)))

    # Missed fraud costs the full amount; caught fraud recovers only part of it.
    missed_loss = float(np.sum(amounts[~flagged & (y_true == 1)]))
    recovered = float(np.sum(amounts[flagged & (y_true == 1)])) * recovery_rate
    reviews = fp * review_cost

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return ThresholdChoice(
        threshold=float(threshold),
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        total_cost=missed_loss + reviews - recovered,
        fraud_caught_value=recovered,
        review_cost=reviews,
    )


def sweep(
    y_true,
    scores,
    amounts,
    *,
    review_cost: float = REVIEW_COST,
    recovery_rate: float = RECOVERY_RATE,
    n_points: int = 200,
) -> pd.DataFrame:
    """Evaluate cost across the full range of thresholds."""
    scores = np.asarray(scores, dtype=float)
    # Quantile spacing, because scores cluster near zero on imbalanced data and
    # a linear grid would spend most of its points where nothing changes.
    candidates = np.unique(np.quantile(scores, np.linspace(0.50, 0.99999, n_points)))

    rows = [
        cost_at(y_true, scores, t, amounts,
                review_cost=review_cost, recovery_rate=recovery_rate).__dict__
        for t in candidates
    ]
    return pd.DataFrame(rows)


def best_by_cost(y_true, scores, amounts, **kwargs) -> ThresholdChoice:
    """The cheapest threshold. This is the one to deploy."""
    frame = sweep(y_true, scores, amounts, **kwargs)
    best = frame.loc[frame["total_cost"].idxmin()]
    return ThresholdChoice(**{k: best[k] for k in ThresholdChoice.__dataclass_fields__})


def best_by_f1(y_true, scores, amounts, **kwargs) -> ThresholdChoice:
    """The cheapest threshold if you ignore money. Included for contrast."""
    frame = sweep(y_true, scores, amounts, **kwargs)
    best = frame.loc[frame["f1"].idxmax()]
    return ThresholdChoice(**{k: best[k] for k in ThresholdChoice.__dataclass_fields__})


def at_recall(y_true, scores, amounts, target_recall: float = 0.80, **kwargs) -> ThresholdChoice:
    """Cheapest threshold that still hits a mandated recall floor.

    Regulators and risk committees often set the recall, and the job becomes
    minimising cost subject to it rather than minimising cost outright.
    """
    frame = sweep(y_true, scores, amounts, **kwargs)
    feasible = frame[frame["recall"] >= target_recall]
    if feasible.empty:
        raise ValueError(f"no threshold reaches recall {target_recall:.0%}")
    best = feasible.loc[feasible["total_cost"].idxmin()]
    return ThresholdChoice(**{k: best[k] for k in ThresholdChoice.__dataclass_fields__})


def pr_curve(y_true, scores) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    return pd.DataFrame(
        {
            "threshold": np.append(thresholds, 1.0),
            "precision": precision,
            "recall": recall,
        }
    )


__all__ = [
    "cost_at", "sweep", "best_by_cost", "best_by_f1", "at_recall", "pr_curve",
    "ThresholdChoice", "REVIEW_COST", "RECOVERY_RATE",
]
