"""Getting the data, reproducibly.

Two sources, same schema:

* ``--url`` fetches a real CSV.
* the default *generates* one, deterministically from a seed.

The generator is not a convenience -- it is what makes this repository
runnable. A data project whose first step is "download a 300 MB file that may
have moved" is a data project nobody can reproduce, including its author six
months later. So the synthetic path injects the same six classes of mess that
the real UCI Online Retail data contains, and every downstream test asserts
against known-correct answers rather than "whatever the file happens to say".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "transactions.csv"

COUNTRIES = ["India", "United Kingdom", "Germany", "France", "Australia", "Japan"]
CATEGORIES = ["Home", "Garden", "Kitchen", "Stationery", "Toys", "Lighting"]
PRODUCTS = {
    "Home": ["Cushion Cover", "Photo Frame", "Wall Clock", "Throw Blanket"],
    "Garden": ["Plant Pot", "Watering Can", "Seed Kit", "Trowel"],
    "Kitchen": ["Ceramic Mug", "Tea Towel", "Storage Jar", "Chopping Board"],
    "Stationery": ["Notebook", "Pen Set", "Desk Pad", "Sticker Sheet"],
    "Toys": ["Wooden Puzzle", "Spinning Top", "Toy Train", "Building Blocks"],
    "Lighting": ["Table Lamp", "String Lights", "Candle Set", "Lantern"],
}


def synthesise(n_rows: int = 12_000, seed: int = 42) -> pd.DataFrame:
    """A messy but internally consistent transaction log."""
    rng = np.random.default_rng(seed)

    category = rng.choice(CATEGORIES, size=n_rows, p=[0.22, 0.14, 0.24, 0.16, 0.12, 0.12])
    product = np.array([rng.choice(PRODUCTS[c]) for c in category])

    # Two years of orders, with a Q4 lift that recurs in *each* year -- pinning
    # the boost to one fixed November would produce a single spike, which is a
    # trend artefact, not seasonality.
    dates = pd.Series(
        pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n_rows), unit="D")
    )
    q4_boost = rng.random(n_rows) < 0.18
    q4_dates = pd.to_datetime(
        pd.DataFrame({"year": dates.dt.year, "month": 11, "day": 1})
    ) + pd.to_timedelta(rng.integers(0, 59, n_rows), unit="D")
    dates = dates.where(~q4_boost, q4_dates)

    base_price = {c: p for c, p in zip(CATEGORIES, [12.5, 9.0, 7.5, 4.25, 15.0, 22.0])}
    unit_price = np.array([base_price[c] for c in category]) * rng.lognormal(0, 0.35, n_rows)
    quantity = rng.integers(1, 13, size=n_rows)

    # Customer purchase counts follow a power law in every real retail dataset:
    # a small group of regulars, a long tail who bought once. Sampling uniformly
    # would make almost everyone a "repeat customer" and leave the concentration
    # analysis with nothing to find.
    pool = 4_200
    weights = 1.0 / np.arange(1, pool + 1) ** 1.15
    customer_id = rng.choice(
        np.arange(10_000, 10_000 + pool), size=n_rows, p=weights / weights.sum()
    )

    frame = pd.DataFrame(
        {
            "invoice_id": [f"INV{100000 + i}" for i in range(n_rows)],
            "invoice_date": dates.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "customer_id": customer_id.astype(object),
            "country": rng.choice(COUNTRIES, size=n_rows, p=[0.34, 0.28, 0.12, 0.10, 0.09, 0.07]),
            "category": category,
            "product": product,
            "quantity": quantity.astype(object),
            "unit_price": np.round(unit_price, 2).astype(object),
        }
    )

    return _make_messy(frame, rng)


def _make_messy(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject the six problems that real transaction exports actually have."""
    n = len(df)

    # 1. Missing customer ids -- guest checkout. ~14%, as in the UCI data.
    missing = rng.random(n) < 0.14
    df.loc[missing, "customer_id"] = rng.choice([np.nan, "", "N/A", "null"], missing.sum())

    # 2. Inconsistent categorical text: casing and stray whitespace.
    scramble = rng.random(n) < 0.20
    df.loc[scramble, "country"] = df.loc[scramble, "country"].map(
        lambda s: rng.choice([s.upper(), s.lower(), f"  {s} ", f"{s}  "])
    )

    # 3. Mixed date formats -- the classic multi-system export artefact.
    reformat = rng.random(n) < 0.12
    df.loc[reformat, "invoice_date"] = pd.to_datetime(
        df.loc[reformat, "invoice_date"]
    ).dt.strftime("%d/%m/%Y")

    # 4. Returns as negative quantities, mixed in with sales.
    returns = rng.random(n) < 0.03
    df.loc[returns, "quantity"] = -df.loc[returns, "quantity"].astype(int)
    df.loc[returns, "invoice_id"] = "C" + df.loc[returns, "invoice_id"].astype(str)

    # 5. Bad numerics: zero-price giveaways, and a few thousand-separator strings.
    df.loc[rng.random(n) < 0.015, "unit_price"] = 0.0
    commas = rng.random(n) < 0.02
    df.loc[commas, "unit_price"] = df.loc[commas, "unit_price"].map(lambda v: f"{float(v):,.2f}")

    # 6. Duplicate rows from a retried batch upload.
    dupes = df.sample(frac=0.02, random_state=7)
    df = pd.concat([df, dupes], ignore_index=True)

    return df.sample(frac=1.0, random_state=11).reset_index(drop=True)


def load(path: Path | str | None = None, url: str | None = None) -> pd.DataFrame:
    """Read the raw data. Everything stays `object` dtype -- cleaning's job."""
    if url:
        return pd.read_csv(url, dtype=str, keep_default_na=False, na_values=[])
    path = Path(path or RAW)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m src.acquire` first, or pass --url"
        )
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch or generate the raw dataset.")
    p.add_argument("--url", help="download a real CSV instead of generating one")
    p.add_argument("--rows", type=int, default=12_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-o", "--output", type=Path, default=RAW)
    args = p.parse_args(argv)

    if args.url:
        frame = pd.read_csv(args.url, dtype=str)
        origin = args.url
    else:
        frame = synthesise(args.rows, args.seed)
        origin = f"synthetic (rows={args.rows}, seed={args.seed})"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print(f"source : {origin}")
    print(f"wrote  : {args.output}  ({len(frame):,} rows x {len(frame.columns)} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
