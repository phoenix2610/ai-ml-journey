"""Card transactions with a realistic fraud rate.

The defining property of this dataset is **imbalance**: fraud is 0.3% of rows.
That one number invalidates most of the default machine-learning workflow --
accuracy, ROC-AUC, and a 0.5 decision threshold are all misleading here, and
each has a test in this project demonstrating why.

Fraud is not simply "unusual". Fraudulent transactions here differ in specific,
learnable ways -- larger amounts, odd hours, new devices, geographic mismatch,
bursts of activity -- with heavy overlap against legitimate behaviour. A model
that separates them perfectly would mean the generator was too easy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW = Path(__file__).resolve().parent.parent / "data" / "transactions.csv"

TARGET = "is_fraud"
FRAUD_RATE = 0.003          # 3 in 1,000, in line with published card-fraud rates

NUMERIC = ["amount", "hour", "days_since_signup", "txn_count_24h",
           "distance_from_home_km", "avg_amount_30d", "amount_vs_avg"]
CATEGORICAL = ["merchant_category", "device", "entry_mode"]

MERCHANTS = ["grocery", "fuel", "restaurant", "retail", "travel", "electronics", "gambling"]
DEVICES = ["known_phone", "known_laptop", "new_device"]
ENTRY_MODES = ["chip", "contactless", "online", "manual"]


def make_transactions(n_rows: int = 60_000, seed: int = 42) -> pd.DataFrame:
    """Generate transactions with a ~0.3% fraud rate."""
    rng = np.random.default_rng(seed)
    n_fraud = max(2, int(round(n_rows * FRAUD_RATE)))
    n_legit = n_rows - n_fraud

    legit = _legit(n_legit, rng)
    fraud = _fraud(n_fraud, rng)

    legit["archetype"] = "legitimate"
    frame = pd.concat([legit, fraud], ignore_index=True)
    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Ratio feature: a £400 charge is unremarkable for some customers and a
    # screaming outlier for others. The absolute amount cannot express that.
    frame["amount_vs_avg"] = frame["amount"] / frame["avg_amount_30d"].clip(lower=1.0)
    return frame


def _legit(n: int, rng: np.random.Generator) -> pd.DataFrame:
    avg_amount = np.clip(rng.lognormal(3.4, 0.55, n), 4, 400)
    return pd.DataFrame(
        {
            "amount": np.round(np.clip(avg_amount * rng.lognormal(0, 0.45, n), 1, 4_000), 2),
            # Daytime-weighted: most legitimate spending is not at 3am.
            "hour": np.clip(rng.normal(14, 4.2, n).round(), 0, 23).astype(int),
            "days_since_signup": rng.gamma(3.0, 220, n).round().astype(int),
            "txn_count_24h": rng.poisson(2.6, n),
            "distance_from_home_km": np.round(np.clip(rng.gamma(1.4, 6.5, n), 0, 900), 1),
            "avg_amount_30d": np.round(avg_amount, 2),
            "merchant_category": rng.choice(
                MERCHANTS, n, p=[0.30, 0.16, 0.18, 0.20, 0.06, 0.08, 0.02]
            ),
            "device": rng.choice(DEVICES, n, p=[0.62, 0.33, 0.05]),
            "entry_mode": rng.choice(ENTRY_MODES, n, p=[0.34, 0.38, 0.24, 0.04]),
            TARGET: 0,
        }
    )


# Fraud archetypes: which "tells" each one exhibits, and how common it is.
# Crucially, none of them is anomalous on every axis.
ARCHETYPES = {
    "card_testing":     (["amount_low", "burst", "online"],        0.26),
    "account_takeover": (["new_device", "far", "amount_high"],     0.24),
    "night_cashout":    (["odd_hour", "amount_high"],              0.18),
    "stealth":          ([],                                       0.32),
}


def _fraud(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Fraud that mostly looks legitimate.

    Fraud rows start as draws from the *legitimate* distribution, and then only
    the tells belonging to their archetype are perturbed. That matters: a
    generator where fraud differs on every axis at once is separable by almost
    anything, which would make the whole PR-AUC and threshold discussion moot.

    ~32% of fraud is 'stealth' -- statistically indistinguishable from normal
    behaviour on these features. That is the irreducible part, and it is why
    perfect recall is not available at any threshold.
    """
    frame = _legit(n, rng)
    frame[TARGET] = 1

    names = list(ARCHETYPES)
    weights = np.array([ARCHETYPES[k][1] for k in names])
    assigned = rng.choice(names, size=n, p=weights / weights.sum())

    for i, archetype in enumerate(assigned):
        tells, _ = ARCHETYPES[archetype]
        # Even within an archetype, each tell only fires most of the time --
        # fraudsters are inconsistent, and detectors have to cope with that.
        for tell in tells:
            if rng.random() > 0.78:
                continue
            _apply_tell(frame, i, tell, rng)

    frame["archetype"] = assigned
    return frame


def _apply_tell(frame: pd.DataFrame, i: int, tell: str, rng: np.random.Generator) -> None:
    """Perturb one row on one axis. Deliberately overlapping with normal ranges."""
    if tell == "amount_low":
        frame.at[i, "amount"] = round(float(rng.uniform(0.5, 8)), 2)
    elif tell == "amount_high":
        base = frame.at[i, "avg_amount_30d"]
        frame.at[i, "amount"] = round(float(np.clip(base * rng.lognormal(1.1, 0.6), 20, 9_000)), 2)
    elif tell == "burst":
        frame.at[i, "txn_count_24h"] = int(rng.poisson(9) + 3)
    elif tell == "odd_hour":
        frame.at[i, "hour"] = int(rng.choice([0, 1, 2, 3, 4, 23]))
    elif tell == "new_device":
        frame.at[i, "device"] = "new_device"
    elif tell == "far":
        frame.at[i, "distance_from_home_km"] = round(float(rng.gamma(2.2, 120)), 1)
    elif tell == "online":
        frame.at[i, "entry_mode"] = "online"


def load(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path) if path else RAW
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python -m src.data` first")
    return pd.read_csv(path)


def split(frame: pd.DataFrame, *, test_size: float = 0.25, seed: int = 42):
    """Stratified split. Non-negotiable at a 0.3% positive rate.

    An unstratified split of 60,000 rows with 180 positives can easily hand the
    test set 60 frauds instead of 45 -- a 33% swing in the only class anyone
    cares about, which moves recall enough to change which model 'wins'.
    """
    y = frame[TARGET]
    # archetype is metadata about *why* a row is fraud -- keeping it in X would
    # be a direct label leak.
    X = frame.drop(columns=[c for c in (TARGET, "archetype") if c in frame])
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def imbalance_summary(y) -> str:
    y = np.asarray(y)
    positives = int(y.sum())
    rate = positives / len(y)
    return (
        f"{len(y):,} rows, {positives:,} fraud ({100 * rate:.3f}%), "
        f"1 in {round(1 / rate) if rate else 0:,}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate the transaction dataset.")
    p.add_argument("--rows", type=int, default=60_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-o", "--output", type=Path, default=RAW)
    args = p.parse_args(argv)

    frame = make_transactions(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print(f"wrote {args.output}")
    print(f"  {imbalance_summary(frame[TARGET])}")
    print(f"\n  a model predicting 'never fraud' scores "
          f"{100 * (1 - frame[TARGET].mean()):.2f}% accuracy and is useless.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
