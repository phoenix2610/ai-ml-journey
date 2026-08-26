"""Loading house data, and splitting it honestly.

The dataset is generated from a known ground-truth price function. That is a
deliberate teaching choice: when you *know* the true effect of each feature,
you can check whether the model recovered it, rather than admiring an R² and
hoping. `TRUE_EFFECTS` documents what the generator actually did, and
`tests/test_data.py` asserts the data really behaves that way.

`--csv` swaps in a real dataset; nothing downstream cares which it got.

The split is the other thing worth getting right. Price is heavily right-skewed,
so a plain random split can hand the test set a different price distribution
than the training set. Stratifying on price deciles fixes that.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW = Path(__file__).resolve().parent.parent / "data" / "houses.csv"

TARGET = "price"

NUMERIC = ["area_sqft", "bedrooms", "bathrooms", "age_years",
           "garage_spaces", "distance_km", "lot_sqft"]
CATEGORICAL = ["neighbourhood", "condition", "heating"]

NEIGHBOURHOODS = {          # name -> price multiplier
    "Riverside": 1.42,
    "Hillcrest": 1.25,
    "Midtown": 1.08,
    "Eastgate": 0.95,
    "Northfield": 0.82,
}
CONDITIONS = {"excellent": 1.15, "good": 1.0, "fair": 0.88, "poor": 0.72}
HEATING = {"heat_pump": 1.05, "gas": 1.0, "electric": 0.97, "none": 0.90}

TRUE_EFFECTS = """\
price = 185 * area_sqft**0.92                 sublinear: doubling area does
                                              not double price
      * neighbourhood multiplier (0.82-1.42)  the single largest factor
      * condition multiplier     (0.72-1.15)
      * heating multiplier       (0.90-1.05)
      * depreciation: 1 - 0.004*age, floored at 0.72
      * (1 + 0.03*(bathrooms-2))
      * (1 - 0.012*distance_km)               distance to the centre
      + 9500 * garage_spaces
      + 4.5  * lot_sqft
      + lognormal noise (sigma 0.09)
"""


def make_houses(n_rows: int = 6_000, seed: int = 42) -> pd.DataFrame:
    """Generate houses from the price function documented in TRUE_EFFECTS."""
    rng = np.random.default_rng(seed)

    neighbourhood = rng.choice(
        list(NEIGHBOURHOODS), size=n_rows, p=[0.14, 0.18, 0.28, 0.24, 0.16]
    )
    condition = rng.choice(list(CONDITIONS), size=n_rows, p=[0.18, 0.44, 0.28, 0.10])
    heating = rng.choice(list(HEATING), size=n_rows, p=[0.22, 0.46, 0.26, 0.06])

    # Nicer areas have bigger houses -- features correlate, as they do in reality.
    prestige = np.array([NEIGHBOURHOODS[n] for n in neighbourhood])
    area = np.clip(rng.normal(1500 * prestige, 380), 420, 6500)

    bedrooms = np.clip((area / 620 + rng.normal(0, 0.6, n_rows)).round(), 1, 7)
    bathrooms = np.clip((bedrooms * 0.62 + rng.normal(0, 0.45, n_rows)).round(), 1, 5)
    age = np.clip(rng.gamma(2.2, 11, n_rows), 0, 110)
    garage = np.clip(rng.poisson(1.1 * prestige, n_rows), 0, 4)
    distance = np.clip(rng.gamma(2.0, 4.2, n_rows), 0.3, 45)
    lot = np.clip(area * rng.uniform(1.1, 3.4, n_rows), 500, 24_000)

    price = (
        185 * area**0.92
        * prestige
        * np.array([CONDITIONS[c] for c in condition])
        * np.array([HEATING[h] for h in heating])
        * np.maximum(1 - 0.004 * age, 0.72)
        * (1 + 0.03 * (bathrooms - 2))
        * (1 - 0.012 * distance)
        + 9_500 * garage
        + 4.5 * lot
    ) * rng.lognormal(0, 0.09, n_rows)

    frame = pd.DataFrame(
        {
            "area_sqft": area.round(0),
            "bedrooms": bedrooms.astype(int),
            "bathrooms": bathrooms.astype(int),
            "age_years": age.round(1),
            "garage_spaces": garage.astype(int),
            "distance_km": distance.round(2),
            "lot_sqft": lot.round(0),
            "neighbourhood": neighbourhood,
            "condition": condition,
            "heating": heating,
            TARGET: price.round(0),
        }
    )

    # Real listings have gaps. These columns are the ones people leave blank.
    for column, rate in (("age_years", 0.06), ("lot_sqft", 0.04), ("heating", 0.03)):
        frame.loc[rng.random(n_rows) < rate, column] = np.nan

    return frame


def load(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path) if path else RAW
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python -m src.data` first")
    return pd.read_csv(path)


def split(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train/test split stratified on price deciles.

    Price is right-skewed, so an unstratified split can leave the test set with
    a visibly different price distribution and make the score depend on luck.
    Binning the target and stratifying on the bins removes that.
    """
    y = frame[TARGET]
    X = frame.drop(columns=[TARGET])

    bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=bins)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate or inspect the housing dataset.")
    p.add_argument("--rows", type=int, default=6_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-o", "--output", type=Path, default=RAW)
    args = p.parse_args(argv)

    frame = make_houses(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print(f"wrote {args.output}  ({len(frame):,} rows x {len(frame.columns)} cols)")
    print(f"\nprice: median {frame[TARGET].median():,.0f}  "
          f"p10 {frame[TARGET].quantile(.1):,.0f}  p90 {frame[TARGET].quantile(.9):,.0f}")
    print(f"missing values:\n{frame.isna().sum()[lambda s: s > 0].to_string()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
