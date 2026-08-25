"""Four figures, each answering one question.

Form follows the data's job, not habit:

| question | job | form |
|---|---|---|
| Is revenue growing? | change over time | line + rolling mean |
| Which categories carry it? | magnitude, ranked | horizontal bars, sorted |
| When does demand peak? | polarity vs a baseline | diverging bars around 100 |
| How concentrated is revenue? | cumulative share | Lorenz curve vs equality |

Deliberately *not* used: a pie chart (magnitude comparison is what bars are
for), and a dual-axis chart (two y-scales invite any conclusion you like --
revenue and order count get separate encodings instead).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from src import analyse, theme

FIGURES = Path(__file__).resolve().parent.parent / "figures"
SOURCE = "Source: transactions dataset, cleaned via src/clean.py"


def _save(fig, name: str, mode: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    suffix = "" if mode == "light" else f"-{mode}"
    path = FIGURES / f"{name}{suffix}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------- 1. over time


def revenue_over_time(sales, returns, tokens, mode="light") -> Path:
    data = analyse.monthly_revenue(sales, returns)
    fig, ax = plt.subplots(figsize=(9, 4.6))

    ax.plot(data["month"], data["net_revenue"], color=tokens.series[0],
            linewidth=2, label="Net revenue", zorder=3)
    ax.plot(data["month"], data["rolling_3m"], color=tokens.series[1],
            linewidth=2, linestyle=(0, (5, 3)), label="3-month mean", zorder=4)

    # Fill under the actual, not the smoothed line, at low alpha.
    ax.fill_between(data["month"], data["net_revenue"], color=tokens.series[0],
                    alpha=0.10, zorder=2)

    # Direct-label the peak only -- never a number on every point.
    peak = data.loc[data["net_revenue"].idxmax()]
    ax.scatter([peak["month"]], [peak["net_revenue"]], s=44,
               color=tokens.series[0], edgecolor=tokens.surface, linewidth=2, zorder=5)
    ax.annotate(
        f"peak {theme.currency(peak['net_revenue'])}\n{peak['month']:%b %Y}",
        xy=(peak["month"], peak["net_revenue"]),
        xytext=(0, 14), textcoords="offset points",
        ha="center", fontsize=9, color=tokens.text_secondary,
    )

    ax.set_title("Net revenue by month")
    ax.set_ylabel("Net revenue")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: theme.currency(v)))
    # Headroom so the peak callout cannot collide with the title.
    ax.set_ylim(0, data["net_revenue"].max() * 1.26)
    ax.legend(loc="upper left", ncols=2)
    theme.annotate_source(fig, SOURCE, tokens)
    return _save(fig, "01-revenue-over-time", mode)


# -------------------------------------------------------------- 2. categories


def category_revenue(sales, returns, tokens, mode="light") -> Path:
    data = analyse.category_performance(sales, returns).sort_values("revenue")
    fig, ax = plt.subplots(figsize=(9, 4.4))

    # One series -> one hue. Magnitude is already encoded by bar length, so a
    # per-bar rainbow would add colour without adding information.
    bars = ax.barh(data["category"], data["revenue"],
                   color=tokens.series[0], height=0.68, zorder=3)

    span = data["revenue"].max()
    for bar, (_, row) in zip(bars, data.iterrows()):
        ax.text(
            bar.get_width() + span * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{theme.currency(row['revenue'])}  ·  {row['revenue_share']:.0f}%",
            va="center", fontsize=9, color=tokens.text_secondary,
        )

    ax.set_title("Revenue by category")
    ax.set_xlabel("Revenue")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: theme.currency(v)))
    ax.set_xlim(0, span * 1.28)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    theme.annotate_source(fig, SOURCE, tokens)
    return _save(fig, "02-category-revenue", mode)


# ------------------------------------------------------------- 3. seasonality


def seasonality(sales, tokens, mode="light") -> Path:
    data = analyse.seasonality(sales)
    deviation = data["index_vs_mean"] - 100

    fig, ax = plt.subplots(figsize=(9, 4.2))

    # Genuine polarity (above/below the yearly average), so: two poles and a
    # neutral midpoint. Not a sequential ramp -- "average" must read as nothing.
    colors = [tokens.diverging[-2] if v > 0 else tokens.diverging[1] for v in deviation]
    ax.bar(data["month_name"], deviation, color=colors, width=0.66, zorder=3)
    ax.axhline(0, color=tokens.text_muted, linewidth=1, zorder=4)

    for x, v in zip(data["month_name"], deviation):
        offset = 3 if v > 0 else -12
        ax.text(x, v + offset, f"{v:+.0f}", ha="center", fontsize=8.5,
                color=tokens.text_secondary)

    ax.set_title("Seasonality: revenue index vs the yearly average")
    ax.set_ylabel("Deviation from average (%)")
    ax.set_xlabel("")
    pad = max(abs(deviation.min()), abs(deviation.max())) * 0.28
    ax.set_ylim(deviation.min() - pad, deviation.max() + pad)

    # Two colours carry meaning, so they are named rather than left to inference.
    ax.legend(
        handles=[
            plt.Line2D([], [], marker="s", linestyle="", markersize=8,
                       color=tokens.diverging[-2], label="above average"),
            plt.Line2D([], [], marker="s", linestyle="", markersize=8,
                       color=tokens.diverging[1], label="below average"),
        ],
        loc="upper left", ncols=2,
    )
    theme.annotate_source(fig, SOURCE, tokens)
    return _save(fig, "03-seasonality", mode)


# ----------------------------------------------------------- 4. concentration


def revenue_concentration(sales, tokens, mode="light") -> Path:
    rfm = analyse.customer_value(sales)
    if rfm.empty:
        raise ValueError("no identified customers to plot")

    ranked = rfm.sort_values("monetary", ascending=False)
    cum_customers = np.arange(1, len(ranked) + 1) / len(ranked) * 100
    cum_revenue = ranked["monetary"].cumsum() / ranked["monetary"].sum() * 100

    fig, ax = plt.subplots(figsize=(7.2, 5))

    ax.plot([0, 100], [0, 100], color=tokens.text_muted, linewidth=1.5,
            linestyle=(0, (4, 4)), label="Perfectly even", zorder=2)
    ax.plot(cum_customers, cum_revenue, color=tokens.series[0], linewidth=2.4,
            label="Actual", zorder=4)
    ax.fill_between(cum_customers, cum_revenue, cum_customers,
                    color=tokens.series[0], alpha=0.12, zorder=3)

    # One annotated reference point beats a labelled grid.
    at_20 = float(np.interp(20, cum_customers, cum_revenue))
    ax.scatter([20], [at_20], s=52, color=tokens.series[1],
               edgecolor=tokens.surface, linewidth=2, zorder=5)
    ax.annotate(
        f"top 20% of customers\n= {at_20:.0f}% of revenue",
        xy=(20, at_20), xytext=(14, -6), textcoords="offset points",
        fontsize=9, color=tokens.text_secondary,
    )

    ax.set_title("Revenue concentration across customers")
    ax.set_xlabel("Cumulative share of customers (%)")
    ax.set_ylabel("Cumulative share of revenue (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.grid(axis="both", visible=True)
    ax.legend(loc="lower right")
    theme.annotate_source(fig, SOURCE, tokens)
    return _save(fig, "04-revenue-concentration", mode)


# ------------------------------------------------------------------- driver


def build_all(sales: pd.DataFrame, returns: pd.DataFrame, mode: str = "light") -> list[Path]:
    """Render every figure. Dark mode is a selected set of steps, not a flip."""
    tokens = theme.apply(mode)
    return [
        revenue_over_time(sales, returns, tokens, mode),
        category_revenue(sales, returns, tokens, mode),
        seasonality(sales, tokens, mode),
        revenue_concentration(sales, tokens, mode),
    ]


__all__ = [
    "build_all", "revenue_over_time", "category_revenue",
    "seasonality", "revenue_concentration", "FIGURES",
]
