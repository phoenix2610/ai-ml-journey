"""The four questions this dataset can actually answer.

1. Is revenue growing, and how much of it is seasonal?
2. Which categories carry the business, and which just look busy?
3. Where do returns concentrate?
4. How much of revenue comes from repeat customers?

Each is one function returning a tidy frame. Nothing here plots or prints --
that keeps the numbers testable, and lets the same function feed a chart, a
report, and an assertion.
"""

from __future__ import annotations

import pandas as pd


def monthly_revenue(sales: pd.DataFrame, returns: pd.DataFrame | None = None) -> pd.DataFrame:
    """Revenue per month, netted against returns, with 3-month rolling mean."""
    gross = sales.groupby("month", as_index=False)["revenue"].sum().rename(
        columns={"revenue": "gross_revenue"}
    )

    if returns is not None and len(returns):
        refunds = (
            returns.groupby("month", as_index=False)["revenue"]
            .sum()
            .rename(columns={"revenue": "refunds"})
        )
        gross = gross.merge(refunds, on="month", how="left")
        gross["refunds"] = gross["refunds"].fillna(0.0)
    else:
        gross["refunds"] = 0.0

    # Refund revenue is already negative (negative quantity x positive price).
    gross["net_revenue"] = gross["gross_revenue"] + gross["refunds"]
    gross["orders"] = sales.groupby("month").size().reindex(gross["month"]).values
    gross["rolling_3m"] = gross["net_revenue"].rolling(3, min_periods=1).mean()
    return gross.sort_values("month").reset_index(drop=True)


def category_performance(sales: pd.DataFrame, returns: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per category: revenue, share, order count, average order value, return rate."""
    grouped = sales.groupby("category").agg(
        revenue=("revenue", "sum"),
        orders=("invoice_id", "count"),
        units=("quantity", "sum"),
        avg_price=("unit_price", "mean"),
    )
    grouped["aov"] = grouped["revenue"] / grouped["orders"]
    grouped["revenue_share"] = 100 * grouped["revenue"] / grouped["revenue"].sum()

    if returns is not None and len(returns):
        returned_units = returns.groupby("category")["quantity"].sum().abs()
        grouped["return_rate"] = (
            100 * returned_units / grouped["units"]
        ).fillna(0.0)
    else:
        grouped["return_rate"] = 0.0

    return grouped.sort_values("revenue", ascending=False).reset_index()


def top_products(sales: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    result = (
        sales.groupby(["category", "product"])
        .agg(revenue=("revenue", "sum"), units=("quantity", "sum"))
        .sort_values("revenue", ascending=False)
        .head(n)
        .reset_index()
    )
    result["revenue_share"] = 100 * result["revenue"] / sales["revenue"].sum()
    return result


def country_performance(sales: pd.DataFrame) -> pd.DataFrame:
    result = (
        sales.groupby("country")
        .agg(
            revenue=("revenue", "sum"),
            orders=("invoice_id", "count"),
            customers=("customer_id", "nunique"),
        )
        .sort_values("revenue", ascending=False)
    )
    result["aov"] = result["revenue"] / result["orders"]
    result["revenue_share"] = 100 * result["revenue"] / result["revenue"].sum()
    return result.reset_index()


def customer_value(sales: pd.DataFrame) -> pd.DataFrame:
    """Per-customer RFM. Guest checkouts are excluded -- they have no identity."""
    identified = sales[sales["customer_id"].notna()]
    if identified.empty:
        return pd.DataFrame(columns=["customer_id", "recency_days", "frequency", "monetary"])

    as_of = identified["invoice_date"].max()
    rfm = identified.groupby("customer_id").agg(
        recency_days=("invoice_date", lambda s: (as_of - s.max()).days),
        frequency=("invoice_id", "count"),
        monetary=("revenue", "sum"),
        first_seen=("invoice_date", "min"),
    )
    rfm["is_repeat"] = rfm["frequency"] > 1
    return rfm.sort_values("monetary", ascending=False).reset_index()


def repeat_customer_share(sales: pd.DataFrame) -> dict[str, float]:
    """How concentrated is revenue among people who came back?"""
    rfm = customer_value(sales)
    if rfm.empty:
        return {"repeat_customer_pct": 0.0, "repeat_revenue_pct": 0.0, "top_10pct_revenue_pct": 0.0}

    repeat = rfm[rfm["is_repeat"]]
    cutoff = max(1, int(len(rfm) * 0.10))
    top_decile = rfm.nlargest(cutoff, "monetary")

    return {
        "customers": float(len(rfm)),
        "repeat_customer_pct": 100 * len(repeat) / len(rfm),
        "repeat_revenue_pct": 100 * repeat["monetary"].sum() / rfm["monetary"].sum(),
        "top_10pct_revenue_pct": 100 * top_decile["monetary"].sum() / rfm["monetary"].sum(),
        "median_orders_per_customer": float(rfm["frequency"].median()),
    }


def seasonality(sales: pd.DataFrame) -> pd.DataFrame:
    """Average revenue by calendar month, to separate trend from season."""
    frame = sales.copy()
    frame["month_num"] = frame["invoice_date"].dt.month
    frame["month_name"] = frame["invoice_date"].dt.strftime("%b")

    result = (
        frame.groupby(["month_num", "month_name"], as_index=False)["revenue"]
        .sum()
        .sort_values("month_num")
    )
    result["index_vs_mean"] = 100 * result["revenue"] / result["revenue"].mean()
    return result.reset_index(drop=True)


def summary(sales: pd.DataFrame, returns: pd.DataFrame) -> dict[str, float | str]:
    """Headline numbers, the ones that belong at the top of a report."""
    net = sales["revenue"].sum() + (returns["revenue"].sum() if len(returns) else 0.0)
    span = f"{sales['invoice_date'].min():%b %Y} - {sales['invoice_date'].max():%b %Y}"

    return {
        "period": span,
        "orders": int(len(sales)),
        "gross_revenue": float(sales["revenue"].sum()),
        "refunds": float(returns["revenue"].sum()) if len(returns) else 0.0,
        "net_revenue": float(net),
        "aov": float(sales["revenue"].mean()),
        "units_sold": float(sales["quantity"].sum()),
        "identified_customers": int(sales["customer_id"].nunique()),
        "return_rate_pct": 100 * len(returns) / (len(sales) + len(returns)) if len(sales) else 0.0,
        **repeat_customer_share(sales),
    }


__all__ = [
    "monthly_revenue", "category_performance", "top_products", "country_performance",
    "customer_value", "repeat_customer_share", "seasonality", "summary",
]
