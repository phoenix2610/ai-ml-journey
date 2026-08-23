"""The cleaning pipeline.

Two rules the whole module is built around:

**Every step is logged.** `clean()` returns the frame *and* a `CleaningReport`
recording how many rows each step removed and why. A cleaning script that
silently drops 40% of the data is indistinguishable from one that works, right
up until the conclusions are wrong.

**Returns are separated, not deleted.** Negative quantities are cancellations.
Dropping them inflates revenue; keeping them mixed in corrupts "units sold".
They come back as a separate frame so the analysis can net them off
deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Sentinels that mean "missing" but survive a naive read_csv as literal text.
NULL_TOKENS = {"", " ", "na", "n/a", "null", "none", "nan", "-", "?"}


@dataclass
class Step:
    name: str
    rows_before: int
    rows_after: int
    note: str = ""

    @property
    def removed(self) -> int:
        return self.rows_before - self.rows_after

    @property
    def pct(self) -> float:
        return 100 * self.removed / self.rows_before if self.rows_before else 0.0


@dataclass
class CleaningReport:
    steps: list[Step] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def record(self, name: str, before: int, after: int, note: str = "") -> None:
        self.steps.append(Step(name, before, after, note))

    @property
    def rows_in(self) -> int:
        return self.steps[0].rows_before if self.steps else 0

    @property
    def rows_out(self) -> int:
        return self.steps[-1].rows_after if self.steps else 0

    @property
    def retained_pct(self) -> float:
        return 100 * self.rows_out / self.rows_in if self.rows_in else 0.0

    def __str__(self) -> str:
        width = max((len(s.name) for s in self.steps), default=10)
        lines = ["", f"  {'step':<{width}}  {'rows':>8}  {'removed':>8}  note", ""]
        for s in self.steps:
            drop = f"-{s.removed:,} ({s.pct:.1f}%)" if s.removed else "-"
            lines.append(f"  {s.name:<{width}}  {s.rows_after:>8,}  {drop:>8}  {s.note}")
        lines += ["", f"  {self.rows_in:,} rows in -> {self.rows_out:,} out "
                      f"({self.retained_pct:.1f}% retained)"]
        lines += [f"  ! {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


# ------------------------------------------------------------------- helpers


def normalise_text(series: pd.Series) -> pd.Series:
    """Trim, collapse inner whitespace, and title-case. 'UNITED  KINGDOM ' -> 'United Kingdom'."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
    )


def to_missing(series: pd.Series) -> pd.Series:
    """Turn textual null sentinels into real NA."""
    lowered = series.astype("string").str.strip().str.lower()
    return series.where(~lowered.isin(NULL_TOKENS) & lowered.notna(), pd.NA)


def to_numeric(series: pd.Series) -> pd.Series:
    """Coerce to float, tolerating thousands separators and currency symbols."""
    cleaned = (
        series.astype("string")
        .str.replace(r"[,\s£$€]", "", regex=True)
        .replace({"": None})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_dates(series: pd.Series) -> pd.Series:
    """Handle a column holding more than one date format.

    Tries ISO first, then day-first for whatever is left. Doing it in two
    passes beats `dayfirst=True` alone, which silently misreads '2023-05-06'.
    """
    text = series.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="%Y-%m-%d %H:%M:%S", errors="coerce")

    still_missing = parsed.isna() & text.notna()
    if still_missing.any():
        parsed = parsed.fillna(
            pd.to_datetime(text.where(still_missing), dayfirst=True, errors="coerce")
        )
    return parsed


# ------------------------------------------------------------------ pipeline


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Raw frame -> (sales, returns, report)."""
    report = CleaningReport()
    df = raw.copy()
    n = len(df)

    # 1. Exact duplicates from retried uploads.
    df = df.drop_duplicates()
    report.record("drop duplicates", n, len(df), "identical rows from batch retries")
    n = len(df)

    # 2. Normalise the categorical text before anything groups by it.
    for column in ("country", "category", "product"):
        if column in df:
            df[column] = normalise_text(df[column])
    report.record("normalise text", n, len(df), "trim, collapse spaces, title-case")

    # 3. Real NAs instead of 'N/A'/'null'/''.
    for column in df.columns:
        df[column] = to_missing(df[column])
    report.record("unify nulls", n, len(df), f"{len(NULL_TOKENS)} sentinel tokens -> NA")

    # 4. Types.
    df["invoice_date"] = parse_dates(df["invoice_date"])
    df["quantity"] = to_numeric(df["quantity"]).astype("Float64")
    df["unit_price"] = to_numeric(df["unit_price"])
    df["customer_id"] = to_numeric(df["customer_id"]).astype("Int64")

    unparsed = int(df["invoice_date"].isna().sum())
    df = df[df["invoice_date"].notna()]
    report.record("parse dates", n, len(df), f"{unparsed} unparseable dates dropped")
    n = len(df)

    # 5. Rows with no quantity or price cannot contribute revenue.
    df = df[df["quantity"].notna() & df["unit_price"].notna()]
    report.record("require numerics", n, len(df), "quantity and unit_price present")
    n = len(df)

    # 6. Free items are real but distort average price; keep them, flag them.
    zero_price = int((df["unit_price"] == 0).sum())
    if zero_price:
        report.notes.append(f"{zero_price} rows have unit_price == 0 (giveaways, kept)")

    # 7. Split returns out rather than dropping or merging them.
    is_return = df["quantity"] < 0
    returns = df[is_return].copy()
    sales = df[~is_return].copy()
    report.record("split returns", n, len(sales), f"{len(returns):,} return rows set aside")

    # 8. Derived column, computed once, after types are trustworthy.
    for frame in (sales, returns):
        frame["revenue"] = frame["quantity"].astype(float) * frame["unit_price"]
        frame["month"] = frame["invoice_date"].dt.to_period("M").dt.to_timestamp()

    guests = int(sales["customer_id"].isna().sum())
    if guests:
        report.notes.append(
            f"{guests:,} sales ({100 * guests / len(sales):.1f}%) have no customer_id "
            "-- excluded from per-customer analysis, kept for revenue"
        )

    return sales.reset_index(drop=True), returns.reset_index(drop=True), report


__all__ = [
    "clean", "CleaningReport", "Step",
    "normalise_text", "to_missing", "to_numeric", "parse_dates", "NULL_TOKENS",
]
