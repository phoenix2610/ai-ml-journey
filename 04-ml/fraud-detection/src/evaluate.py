"""Scoring, plus the plots that make an imbalanced problem legible.

Accuracy does not appear anywhere in this module, on purpose. At a 0.3% fraud
rate a model that never fires scores 99.7% and catches nothing; reporting that
number would be actively misleading, so it is not reported at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from src.threshold import pr_curve, sweep

FIGURES = Path(__file__).resolve().parent.parent / "figures"

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"


@dataclass
class Scores:
    average_precision: float
    roc_auc: float
    chance: float

    @property
    def lift(self) -> float:
        """How many times better than random. The honest headline for PR-AUC."""
        return self.average_precision / self.chance if self.chance else 0.0

    def __str__(self) -> str:
        return (
            f"    average precision  {self.average_precision:>8.4f}\n"
            f"    random baseline    {self.chance:>8.4f}   ({self.lift:.0f}x lift)\n"
            f"    ROC-AUC            {self.roc_auc:>8.4f}   (flattering -- see README)"
        )


def score(y_true, scores) -> Scores:
    y_true = np.asarray(y_true).astype(int)
    return Scores(
        average_precision=average_precision_score(y_true, scores),
        roc_auc=roc_auc_score(y_true, scores),
        chance=float(y_true.mean()),
    )


def confusion_at(y_true, scores, threshold: float) -> pd.DataFrame:
    predicted = (np.asarray(scores) >= threshold).astype(int)
    matrix = confusion_matrix(np.asarray(y_true).astype(int), predicted, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=["actually legit", "actually fraud"],
        columns=["passed", "flagged"],
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE, "savefig.bbox": "tight", "savefig.dpi": 150,
            "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "semibold",
            "axes.titlepad": 12, "text.color": INK, "axes.labelcolor": MUTED,
            "xtick.color": MUTED, "ytick.color": MUTED,
            "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
            "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
            "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False,
        }
    )


def _save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_pr_curves(curves: dict[str, tuple], chance: float) -> Path:
    """Precision-recall for every model on one axis. The comparison that counts."""
    _style()
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    colours = [BLUE, ORANGE, "#1baf7a", "#4a3aa7"]

    for (name, (y_true, scores)), colour in zip(curves.items(), colours):
        curve = pr_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(curve["recall"], curve["precision"], color=colour, linewidth=2,
                label=f"{name}  (AP {ap:.3f})", zorder=4)

    ax.axhline(chance, color=MUTED, linewidth=1.4, linestyle=(0, (4, 4)),
               label=f"random  (AP {chance:.4f})", zorder=3)

    ax.set_title("Precision-recall by model")
    ax.set_xlabel("Recall  (share of fraud caught)")
    ax.set_ylabel("Precision  (share of alerts that are real)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    return _save(fig, "01-precision-recall")


def plot_cost_curve(y_true, scores, amounts, chosen: float) -> Path:
    """Total cost against threshold, with the chosen operating point marked."""
    _style()
    frame = sweep(y_true, scores, amounts, n_points=150).sort_values("threshold")

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(frame["threshold"], frame["total_cost"], color=BLUE, linewidth=2, zorder=4)

    best = frame.loc[frame["total_cost"].idxmin()]
    ax.scatter([best["threshold"]], [best["total_cost"]], s=60, color=ORANGE,
               edgecolor=SURFACE, linewidth=2, zorder=6)
    ax.annotate(
        f"cheapest: {best['threshold']:.3f}\n{best['total_cost']:,.0f}",
        xy=(best["threshold"], best["total_cost"]),
        xytext=(12, 18), textcoords="offset points", fontsize=9, color=MUTED,
    )
    ax.axvline(0.5, color=RED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=3)
    ax.text(0.5, ax.get_ylim()[1] * 0.95, "  default 0.5", color=RED, fontsize=9, va="top")

    ax.set_title("Expected cost by decision threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Net cost")
    return _save(fig, "02-cost-curve")


def plot_recall_precision_tradeoff(y_true, scores, amounts) -> Path:
    """Precision and recall against threshold, on one axis (both are 0-1)."""
    _style()
    frame = sweep(y_true, scores, amounts, n_points=150).sort_values("threshold")

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(frame["threshold"], frame["recall"], color=BLUE, linewidth=2,
            label="recall — fraud caught", zorder=4)
    ax.plot(frame["threshold"], frame["precision"], color=ORANGE, linewidth=2,
            label="precision — alerts that are real", zorder=4)

    ax.set_title("The trade-off you are actually choosing between")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.legend(loc="center right")
    return _save(fig, "03-precision-recall-tradeoff")


__all__ = [
    "score", "Scores", "confusion_at",
    "plot_pr_curves", "plot_cost_curve", "plot_recall_precision_tradeoff", "FIGURES",
]
