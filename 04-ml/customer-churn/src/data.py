"""Telco-style subscriber data with a realistic churn structure.

Churn is ~26% here -- imbalanced, but nothing like the fraud project's 0.3%.
The interesting problems are different ones:

* **Tenure confounding.** New customers churn far more than established ones,
  and tenure correlates with almost every other feature. Any claim about *why*
  someone churned has to survive controlling for it.
* **A leaky column.** `exit_survey_sent` is included on purpose. It is only
  populated *after* someone churns, so it predicts the target almost perfectly
  and is worthless in production. Finding and removing it is part of the work.
* **Actionability.** Predicting churn is not the goal; changing it is. Contract
  type and support-call volume can be intervened on. Tenure cannot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW = Path(__file__).resolve().parent.parent / "data" / "subscribers.csv"

TARGET = "churned"
LEAKY = ["exit_survey_sent"]        # populated only after churn -- never a feature

NUMERIC = ["tenure_months", "monthly_charges", "total_charges",
           "support_calls_90d", "outages_90d", "avg_monthly_gb"]
CATEGORICAL = ["contract", "payment_method", "internet_service", "has_dependents"]

CONTRACTS = ["month_to_month", "one_year", "two_year"]
PAYMENTS = ["electronic_check", "mailed_check", "bank_transfer", "credit_card"]
INTERNET = ["fibre", "dsl", "none"]

# Log-odds contributions. Positive pushes toward churn.
COEFFICIENTS = {
    "intercept": -2.42,
    "tenure": -1.45,            # strongest single effect, and not actionable
    "month_to_month": 1.30,
    "one_year": -0.25,
    "two_year": -0.95,
    "electronic_check": 0.45,
    "fibre": 0.40,
    "support_calls": 0.28,      # per call, capped
    "outages": 0.35,
    "high_charges": 0.30,
    "dependents": -0.35,
}


def make_subscribers(n_rows: int = 12_000, seed: int = 42) -> pd.DataFrame:
    """Generate subscribers whose churn follows a documented logistic model."""
    rng = np.random.default_rng(seed)

    tenure = np.clip(rng.gamma(1.6, 17, n_rows), 1, 72).round().astype(int)

    # Long-tenure customers skew toward longer contracts -- the confound.
    long_bias = np.clip(tenure / 72, 0, 1)
    contract_p = np.stack([
        0.75 - 0.55 * long_bias,
        0.15 + 0.15 * long_bias,
        0.10 + 0.40 * long_bias,
    ], axis=1)
    contract_p /= contract_p.sum(axis=1, keepdims=True)
    contract = np.array([rng.choice(CONTRACTS, p=row) for row in contract_p])

    internet = rng.choice(INTERNET, n_rows, p=[0.44, 0.38, 0.18])
    payment = rng.choice(PAYMENTS, n_rows, p=[0.34, 0.19, 0.23, 0.24])
    dependents = rng.choice(["yes", "no"], n_rows, p=[0.30, 0.70])

    base_charge = np.where(internet == "fibre", 74, np.where(internet == "dsl", 52, 21))
    monthly = np.round(np.clip(base_charge + rng.normal(0, 9, n_rows), 15, 130), 2)
    total = np.round(monthly * tenure * rng.uniform(0.94, 1.05, n_rows), 2)

    support_calls = rng.poisson(1.1 + 1.6 * (internet == "fibre"), n_rows)
    outages = rng.poisson(0.5 + 0.9 * (internet == "fibre"), n_rows)
    usage = np.round(np.clip(rng.gamma(2.2, 95, n_rows), 0, 1400), 1)

    logit = (
        COEFFICIENTS["intercept"]
        + COEFFICIENTS["tenure"] * (tenure / 72)
        + np.select(
            [contract == "month_to_month", contract == "one_year", contract == "two_year"],
            [COEFFICIENTS["month_to_month"], COEFFICIENTS["one_year"], COEFFICIENTS["two_year"]],
        )
        + COEFFICIENTS["electronic_check"] * (payment == "electronic_check")
        + COEFFICIENTS["fibre"] * (internet == "fibre")
        + COEFFICIENTS["support_calls"] * np.minimum(support_calls, 6)
        + COEFFICIENTS["outages"] * np.minimum(outages, 4)
        + COEFFICIENTS["high_charges"] * (monthly > 85)
        + COEFFICIENTS["dependents"] * (dependents == "yes")
    )
    churned = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(int)

    frame = pd.DataFrame(
        {
            "customer_id": [f"C{100000 + i}" for i in range(n_rows)],
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "support_calls_90d": support_calls,
            "outages_90d": outages,
            "avg_monthly_gb": usage,
            "contract": contract,
            "payment_method": payment,
            "internet_service": internet,
            "has_dependents": dependents,
            TARGET: churned,
        }
    )

    # The trap. Sent to 94% of churners and 1% of everyone else, so it predicts
    # the target almost perfectly -- and is only knowable after the fact.
    frame["exit_survey_sent"] = np.where(
        frame[TARGET] == 1, rng.random(n_rows) < 0.94, rng.random(n_rows) < 0.01
    ).astype(int)

    # A few genuinely missing values, as any real export has.
    frame.loc[rng.random(n_rows) < 0.02, "total_charges"] = np.nan
    return frame


def tenure_cohort(tenure: pd.Series) -> pd.Series:
    """Bucket tenure so the confound can be controlled for explicitly."""
    return pd.cut(
        tenure,
        bins=[0, 6, 12, 24, 48, 72],
        labels=["0-6m", "6-12m", "1-2y", "2-4y", "4y+"],
        include_lowest=True,
    )


def load(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path) if path else RAW
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python -m src.data` first")
    return pd.read_csv(path)


def split(frame: pd.DataFrame, *, test_size: float = 0.25, seed: int = 42, drop_leaky: bool = True):
    """Stratified split that drops the identifier and, by default, the leak.

    `drop_leaky=False` exists so a test can demonstrate what the leak does to
    the score -- it is never the right setting for a real model.
    """
    y = frame[TARGET]
    drop = [TARGET, "customer_id"] + (LEAKY if drop_leaky else [])
    X = frame.drop(columns=[c for c in drop if c in frame])
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate the subscriber dataset.")
    p.add_argument("--rows", type=int, default=12_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-o", "--output", type=Path, default=RAW)
    args = p.parse_args(argv)

    frame = make_subscribers(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print(f"wrote {args.output}  ({len(frame):,} rows)")
    print(f"  churn rate {frame[TARGET].mean():.1%}")
    print("\n  churn by tenure cohort:")
    by_cohort = frame.groupby(tenure_cohort(frame["tenure_months"]), observed=True)[TARGET].agg(["mean", "size"])
    for cohort, row in by_cohort.iterrows():
        print(f"    {str(cohort):<7} {row['mean']:>6.1%}  (n={int(row['size']):,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
