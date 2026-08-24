import pandas as pd
import pytest

from src.acquire import synthesise
from src.analyse import (
    category_performance,
    country_performance,
    customer_value,
    monthly_revenue,
    repeat_customer_share,
    seasonality,
    summary,
    top_products,
)
from src.clean import clean


@pytest.fixture(scope="module")
def data():
    sales, returns, _ = clean(synthesise(8000, seed=2))
    return sales, returns


# ------------------------------------------------------------------- monthly


def test_monthly_revenue_covers_every_month(data):
    sales, returns = data
    result = monthly_revenue(sales, returns)
    assert len(result) == sales["month"].nunique()


def test_monthly_revenue_is_sorted(data):
    result = monthly_revenue(*data)
    assert result["month"].is_monotonic_increasing


def test_gross_revenue_sums_to_the_total(data):
    sales, returns = data
    result = monthly_revenue(sales, returns)
    assert result["gross_revenue"].sum() == pytest.approx(sales["revenue"].sum())


def test_net_is_gross_minus_refunds(data):
    result = monthly_revenue(*data)
    # Refunds are stored negative, so net = gross + refunds.
    assert (result["net_revenue"] <= result["gross_revenue"] + 1e-6).all()


def test_rolling_mean_is_present_from_the_first_row(data):
    result = monthly_revenue(*data)
    assert result["rolling_3m"].notna().all()


def test_monthly_revenue_without_returns(data):
    sales, _ = data
    result = monthly_revenue(sales, None)
    assert (result["refunds"] == 0).all()
    assert result["net_revenue"].equals(result["gross_revenue"])


# ------------------------------------------------------------------ category


def test_category_shares_sum_to_100(data):
    result = category_performance(*data)
    assert result["revenue_share"].sum() == pytest.approx(100.0)


def test_categories_sorted_by_revenue(data):
    result = category_performance(*data)
    assert result["revenue"].is_monotonic_decreasing


def test_aov_is_revenue_over_orders(data):
    result = category_performance(*data)
    row = result.iloc[0]
    assert row["aov"] == pytest.approx(row["revenue"] / row["orders"])


def test_return_rate_is_a_percentage(data):
    result = category_performance(*data)
    assert (result["return_rate"] >= 0).all()
    assert (result["return_rate"] < 100).all()


# ------------------------------------------------------------------ products


def test_top_products_respects_n(data):
    sales, _ = data
    assert len(top_products(sales, n=5)) == 5


def test_top_products_are_ordered(data):
    sales, _ = data
    assert top_products(sales)["revenue"].is_monotonic_decreasing


# ------------------------------------------------------------------ countries


def test_country_shares_sum_to_100(data):
    sales, _ = data
    assert country_performance(sales)["revenue_share"].sum() == pytest.approx(100.0)


def test_customer_counts_are_positive(data):
    sales, _ = data
    assert (country_performance(sales)["customers"] > 0).all()


# ------------------------------------------------------------------ customers


def test_customer_value_excludes_guests(data):
    sales, _ = data
    rfm = customer_value(sales)
    assert rfm["customer_id"].notna().all()
    assert len(rfm) == sales["customer_id"].nunique()


def test_recency_is_non_negative(data):
    sales, _ = data
    assert (customer_value(sales)["recency_days"] >= 0).all()


def test_monetary_matches_grouped_revenue(data):
    sales, _ = data
    rfm = customer_value(sales)
    identified = sales[sales["customer_id"].notna()]
    assert rfm["monetary"].sum() == pytest.approx(identified["revenue"].sum())


def test_repeat_share_percentages_are_bounded(data):
    sales, _ = data
    stats = repeat_customer_share(sales)
    for key in ("repeat_customer_pct", "repeat_revenue_pct", "top_10pct_revenue_pct"):
        assert 0 <= stats[key] <= 100


def test_customer_value_on_all_guest_data():
    frame = pd.DataFrame(
        {
            "customer_id": pd.Series([pd.NA, pd.NA], dtype="Int64"),
            "invoice_date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "invoice_id": ["A", "B"],
            "revenue": [10.0, 20.0],
        }
    )
    assert customer_value(frame).empty
    assert repeat_customer_share(frame)["repeat_revenue_pct"] == 0.0


# --------------------------------------------------------------- seasonality


def test_seasonality_has_at_most_twelve_rows(data):
    sales, _ = data
    assert len(seasonality(sales)) <= 12


def test_seasonality_index_averages_100(data):
    sales, _ = data
    assert seasonality(sales)["index_vs_mean"].mean() == pytest.approx(100.0)


def test_generated_q4_lift_is_detected(data):
    """The generator boosts Nov/Dec; the analysis must surface it."""
    sales, _ = data
    season = seasonality(sales).set_index("month_name")["index_vs_mean"]
    assert season["Nov"] > 100
    assert season["Dec"] > 100


# -------------------------------------------------------------------- summary


def test_summary_headline_keys(data):
    stats = summary(*data)
    for key in ("period", "orders", "net_revenue", "aov", "return_rate_pct"):
        assert key in stats


def test_net_revenue_is_below_gross(data):
    stats = summary(*data)
    assert stats["net_revenue"] < stats["gross_revenue"]


def test_aov_matches_mean_revenue(data):
    sales, returns = data
    assert summary(sales, returns)["aov"] == pytest.approx(sales["revenue"].mean())
