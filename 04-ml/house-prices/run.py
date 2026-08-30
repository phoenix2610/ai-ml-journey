#!/usr/bin/env python3
"""The whole pipeline, one command.

    python run.py                 generate, compare, tune, evaluate, plot
    python run.py --rows 20000    more data
    python run.py --no-figures    skip plotting
    python run.py --quick         3 folds, for a fast check

Order matters here: every model decision is made by cross-validation on the
training set, and the test set is scored exactly once, at the end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import RAW, make_houses, split
from src.evaluate import full_report
from src.predict import EXAMPLE, predict, save_interval
from src.train import save, train_best

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=int, default=6_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--quick", action="store_true", help="3 folds and fewer rows")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    rows = 2_500 if args.quick else args.rows
    folds = 3 if args.quick else args.folds

    # ---------------------------------------------------------------- data
    frame = make_houses(rows, args.seed)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RAW, index=False)

    X_train, X_test, y_train, y_test = split(frame, seed=args.seed)
    print(f"\n  {len(frame):,} houses -> {len(X_train):,} train / {len(X_test):,} test")
    print(f"  price: median £{frame['price'].median():,.0f}  "
          f"p10 £{frame['price'].quantile(.1):,.0f}  p90 £{frame['price'].quantile(.9):,.0f}\n")

    # --------------------------------------------------------------- train
    model, _results = train_best(X_train, y_train, folds=folds, seed=args.seed)

    # ------------------------------------------------ evaluate (once, here)
    print("\n  held-out test set -- touched for the first time:\n")
    report = full_report(model, X_test, y_test, seed=args.seed)
    print(report.metrics)

    lo, hi = report.interval
    print(f"\n  90% prediction interval: {lo:+,.0f} to {hi:+,.0f}")
    print(f"    actual coverage on the test set: {report.coverage:.1f}%")

    print("\n  error by price decile")
    print("    decile  median price        MAE     bias    MAPE")
    for _, row in report.by_decile.iterrows():
        print(f"    {int(row['decile']):>6}  £{row['median_price']:>11,.0f}  "
              f"£{row['mae']:>8,.0f}  {row['bias']:>+8,.0f}  {row['mape']:>5.1f}%")

    print("\n  top features (permutation importance on the test set)")
    for _, row in report.importance.head(6).iterrows():
        print(f"    {row['feature']:<16} {row['importance']:>10,.0f}  ± {row['std']:,.0f}")

    # --------------------------------------------------------------- persist
    model_path = save(model)
    save_interval(lo, hi, 0.90)
    print(f"\n  saved model -> {model_path.relative_to(HERE)}")

    example = predict(EXAMPLE, model)
    print(f"  example house: £{example['prediction']:,.0f}  "
          f"(£{example['low']:,.0f} – £{example['high']:,.0f})")

    # ------------------------------------------------------------- figures
    if not args.no_figures:
        from src.plots import build_all

        print()
        for path in build_all(report, y_test, model.predict(X_test)):
            print(f"  figure: {path.relative_to(HERE)}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
