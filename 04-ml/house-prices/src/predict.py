"""Inference on a single house, with an interval attached.

The saved artefact is the **whole pipeline**, not just the estimator, so
prediction applies exactly the imputation and encoding that training used.
Saving a bare estimator and re-implementing preprocessing at inference time is
the most common way a model that scored well offline produces nonsense in
production.

The stored residual quantiles travel with the model, so a prediction can say
"£412,000, and 90% of houses like this land within ±£31,000" rather than
offering a single number with no error attached.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data import CATEGORICAL, NUMERIC
from src.train import MODELS_DIR, load

INTERVAL_FILE = MODELS_DIR / "interval.json"

EXAMPLE = {
    "area_sqft": 1650, "bedrooms": 3, "bathrooms": 2, "age_years": 12,
    "garage_spaces": 1, "distance_km": 6.5, "lot_sqft": 3200,
    "neighbourhood": "Midtown", "condition": "good", "heating": "gas",
}


def save_interval(lo: float, hi: float, level: float, path: Path = INTERVAL_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"lo": lo, "hi": hi, "level": level}), encoding="utf-8")
    return path


def load_interval(path: Path = INTERVAL_FILE) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def to_frame(features: dict) -> pd.DataFrame:
    """Build a one-row frame with every training column present.

    Missing keys become NaN rather than an error -- the pipeline's imputer is
    there precisely so a partial listing can still be priced.
    """
    row = {column: features.get(column) for column in NUMERIC + CATEGORICAL}
    return pd.DataFrame([row])


def predict(features: dict, model=None) -> dict:
    model = model or load()
    frame = to_frame(features)
    point = float(model.predict(frame)[0])

    result = {"prediction": point, "inputs": {k: v for k, v in features.items() if v is not None}}

    interval = load_interval()
    if interval:
        result |= {
            "low": point + interval["lo"],
            "high": point + interval["hi"],
            "level": interval["level"],
        }
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Price a single house.")
    for column in NUMERIC:
        p.add_argument(f"--{column.replace('_', '-')}", type=float)
    for column in CATEGORICAL:
        p.add_argument(f"--{column.replace('_', '-')}", type=str)
    p.add_argument("--example", action="store_true", help="price the built-in example house")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args(argv)

    if args.example:
        features = dict(EXAMPLE)
    else:
        features = {c: getattr(args, c) for c in NUMERIC + CATEGORICAL}
        features = {k: v for k, v in features.items() if v is not None}
        if not features:
            p.error("give at least one feature, or use --example")

    try:
        result = predict(features)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print("\n  inputs")
    for key, value in sorted(result["inputs"].items()):
        print(f"    {key:<16} {value}")

    print(f"\n  predicted price   £{result['prediction']:,.0f}")
    if "low" in result:
        span = (result["high"] - result["low"]) / 2
        print(f"  {result['level']:.0%} interval     "
              f"£{result['low']:,.0f} – £{result['high']:,.0f}  (±£{span:,.0f})")
    else:
        print("  (no interval stored -- run `python run.py` to generate one)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
