"""Four diagnostic figures.

Regression diagnostics, not decoration. Each answers a question a scatter of
predictions cannot:

1. **Predicted vs actual** -- is the model biased at the extremes?
2. **Residual distribution** -- are errors centred and symmetric?
3. **Error by price decile** -- which segment is the model bad at?
4. **Permutation importance** -- what is it actually using?

Colour follows role, not taste: one hue for a single series, a neutral
reference line, and a second hue reserved for the thing being contrasted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless: no display needed to render files
import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).resolve().parent.parent / "figures"

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE, "savefig.bbox": "tight", "savefig.dpi": 150,
            "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "semibold",
            "axes.titlepad": 12, "text.color": INK,
            "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
            "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
            "axes.axisbelow": True, "axes.edgecolor": GRID,
            "axes.spines.top": False, "axes.spines.right": False,
            "xtick.major.size": 0, "ytick.major.size": 0,
            "legend.frameon": False,
        }
    )


def _money(value, _pos=None) -> str:
    if abs(value) >= 1e6:
        return f"£{value / 1e6:.1f}M"
    return f"£{value / 1e3:.0f}k"


def _save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def predicted_vs_actual(y_true, y_pred) -> Path:
    _style()
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)

    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    lims = [min(y_true.min(), y_pred.min()) * 0.95, max(y_true.max(), y_pred.max()) * 1.02]

    ax.plot(lims, lims, color=MUTED, linewidth=1.4, linestyle=(0, (4, 4)),
            label="perfect prediction", zorder=3)
    ax.scatter(y_true, y_pred, s=11, alpha=0.30, color=BLUE,
               edgecolor="none", label="test houses", zorder=4)

    ax.set_title("Predicted vs actual price")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(_money); ax.yaxis.set_major_formatter(_money)
    ax.legend(loc="upper left")
    return _save(fig, "01-predicted-vs-actual")


def residual_distribution(residuals, interval=None) -> Path:
    _style()
    residuals = np.asarray(residuals, float)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.hist(residuals, bins=60, color=BLUE, alpha=0.85, zorder=3)
    ax.axvline(0, color=MUTED, linewidth=1.4, zorder=4)

    if interval is not None:
        lo, hi = interval
        for edge in (lo, hi):
            ax.axvline(edge, color=ORANGE, linewidth=1.6, linestyle=(0, (4, 3)), zorder=5)
        ax.axvspan(lo, hi, color=ORANGE, alpha=0.07, zorder=2)
        ax.text(hi, ax.get_ylim()[1] * 0.92, "  90% interval",
                color=ORANGE, fontsize=9, va="top")

    ax.set_title("Residual distribution  (actual − predicted)")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Houses")
    ax.xaxis.set_major_formatter(_money)
    ax.grid(axis="x", visible=False)
    return _save(fig, "02-residuals")


def error_by_decile(frame) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(8.6, 4.4))

    ax.bar(frame["decile"], frame["mape"], color=BLUE, width=0.66, zorder=3)
    for decile, mape in zip(frame["decile"], frame["mape"]):
        ax.text(decile, mape + 0.15, f"{mape:.1f}", ha="center", fontsize=8.5, color=MUTED)

    ax.set_title("Percentage error by price decile")
    ax.set_xlabel("Price decile  (1 = cheapest, 10 = most expensive)")
    ax.set_ylabel("Mean absolute % error")
    ax.set_xticks(frame["decile"])
    ax.set_ylim(0, frame["mape"].max() * 1.25)
    ax.grid(axis="x", visible=False)
    return _save(fig, "03-error-by-decile")


def feature_importance(frame, top: int = 12) -> Path:
    _style()
    data = frame.head(top).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.barh(data["feature"], data["importance"], xerr=data["std"],
            color=BLUE, height=0.66, zorder=3,
            error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})

    ax.set_title("Permutation importance  (measured on the test set)")
    ax.set_xlabel("Increase in MAE when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(_money)
    return _save(fig, "04-feature-importance")


def build_all(evaluation, y_test, y_pred) -> list[Path]:
    return [
        predicted_vs_actual(y_test, y_pred),
        residual_distribution(evaluation.residuals, evaluation.interval),
        error_by_decile(evaluation.by_decile),
        feature_importance(evaluation.importance),
    ]


__all__ = [
    "build_all", "predicted_vs_actual", "residual_distribution",
    "error_by_decile", "feature_importance", "FIGURES",
]
