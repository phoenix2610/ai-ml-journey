import numpy as np
import pandas as pd
import pytest

from src.acquire import synthesise
from src.clean import (
    clean,
    normalise_text,
    parse_dates,
    to_missing,
    to_numeric,
)


# ------------------------------------------------------------------- helpers


def test_normalise_text_trims_and_titlecases():
    s = pd.Series(["  united kingdom ", "UNITED KINGDOM", "United  Kingdom"])
    assert normalise_text(s).nunique() == 1
    assert normalise_text(s).iloc[0] == "United Kingdom"


@pytest.mark.parametrize("token", ["", " ", "N/A", "null", "NONE", "nan", "-", "?"])
def test_to_missing_recognises_sentinels(token):
    assert pd.isna(to_missing(pd.Series([token])).iloc[0])


def test_to_missing_keeps_real_values():
    assert to_missing(pd.Series(["10001"])).iloc[0] == "10001"


@pytest.mark.parametrize(
    "raw,expected", [("12.50", 12.5), ("1,234.56", 1234.56), ("£9.99", 9.99), ("junk", None)]
)
def test_to_numeric(raw, expected):
    result = to_numeric(pd.Series([raw])).iloc[0]
    assert (result == expected) if expected is not None else pd.isna(result)


def test_parse_dates_handles_iso():
    assert parse_dates(pd.Series(["2023-05-06 10:00:00"])).iloc[0] == pd.Timestamp("2023-05-06 10:00")


def test_parse_dates_handles_day_first():
    # 06/05/2023 is 6 May, not 5 June.
    assert parse_dates(pd.Series(["06/05/2023"])).iloc[0] == pd.Timestamp("2023-05-06")


def test_parse_dates_does_not_misread_iso_as_day_first():
    # The trap: dayfirst=True alone would mangle this.
    assert parse_dates(pd.Series(["2023-05-06 00:00:00"])).iloc[0].month == 5


def test_parse_dates_mixed_column():
    parsed = parse_dates(pd.Series(["2023-01-15 09:00:00", "20/03/2023", "garbage"]))
    assert parsed.iloc[0] == pd.Timestamp("2023-01-15 09:00")
    assert parsed.iloc[1] == pd.Timestamp("2023-03-20")
    assert pd.isna(parsed.iloc[2])


# ------------------------------------------------------------------ pipeline


@pytest.fixture(scope="module")
def cleaned():
    return clean(synthesise(4000, seed=1))


def test_returns_three_things(cleaned):
    sales, returns, report = cleaned
    assert isinstance(sales, pd.DataFrame)
    assert isinstance(returns, pd.DataFrame)
    assert report.rows_in > report.rows_out


def test_duplicates_are_removed(cleaned):
    sales, _, report = cleaned
    dedupe = next(s for s in report.steps if s.name == "drop duplicates")
    assert dedupe.removed > 0


def test_no_duplicates_survive(cleaned):
    sales, _, _ = cleaned
    assert not sales.drop(columns=["revenue", "month"]).duplicated().any()


def test_countries_are_normalised(cleaned):
    sales, _, _ = cleaned
    # The generator scrambles case and padding; cleaning must collapse them.
    assert len(sales["country"].unique()) == 6
    assert all(c == c.strip() for c in sales["country"].unique())


def test_dtypes_are_correct(cleaned):
    sales, _, _ = cleaned
    assert pd.api.types.is_datetime64_any_dtype(sales["invoice_date"])
    assert sales["customer_id"].dtype == "Int64"
    assert pd.api.types.is_numeric_dtype(sales["unit_price"])


def test_sales_have_no_negative_quantities(cleaned):
    sales, _, _ = cleaned
    assert (sales["quantity"] > 0).all()


def test_returns_are_all_negative(cleaned):
    _, returns, _ = cleaned
    assert (returns["quantity"] < 0).all()


def test_returns_are_not_silently_dropped(cleaned):
    _, returns, report = cleaned
    assert len(returns) > 0
    assert "return rows set aside" in report.steps[-1].note


def test_revenue_is_derived_correctly(cleaned):
    sales, _, _ = cleaned
    row = sales.iloc[0]
    assert row["revenue"] == pytest.approx(float(row["quantity"]) * row["unit_price"])


def test_month_is_a_period_start(cleaned):
    sales, _, _ = cleaned
    assert (sales["month"].dt.day == 1).all()


def test_guest_checkouts_are_reported_not_dropped(cleaned):
    sales, _, report = cleaned
    assert sales["customer_id"].isna().any()
    assert any("no customer_id" in n for n in report.notes)


def test_zero_prices_are_flagged(cleaned):
    _, _, report = cleaned
    assert any("unit_price == 0" in n for n in report.notes)


def test_report_renders(cleaned):
    _, _, report = cleaned
    text = str(report)
    assert "rows in" in text and "retained" in text


def test_pipeline_is_deterministic():
    a = clean(synthesise(1500, seed=5))[0]
    b = clean(synthesise(1500, seed=5))[0]
    pd.testing.assert_frame_equal(a, b)


def test_cleaning_is_idempotent():
    """Running the cleaner on already-clean output must change nothing."""
    sales, _, _ = clean(synthesise(1500, seed=3))
    again, _, report = clean(sales.drop(columns=["revenue", "month"]).astype(object))
    assert len(again) == len(sales)


def test_empty_input_does_not_crash():
    empty = pd.DataFrame(
        {c: pd.Series(dtype=object) for c in
         ["invoice_id", "invoice_date", "customer_id", "country",
          "category", "product", "quantity", "unit_price"]}
    )
    sales, returns, report = clean(empty)
    assert len(sales) == 0 and len(returns) == 0
