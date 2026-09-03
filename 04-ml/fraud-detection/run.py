#!/usr/bin/env python3
"""The whole pipeline, one command.

    python run.py                  60k transactions, all models
    python run.py --quick          20k rows, 3 folds
    python run.py --no-figures

Order: compare supervised models by PR-AUC on CV, fit the winner, train an
unsupervised detector on legitimate rows only, then score both on the held-out
test set and pick a threshold by cost.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.anomaly import isolation_forest
from src.data import RAW, TARGET, imbalance_summary, make_transactions, split
from src.evaluate import (
    confusion_at,
    plot_cost_curve,
    plot_pr_curves,
    plot_recall_precision_tradeoff,
    score,
)
from src.supervised import fit_best
from src.threshold import at_recall, best_by_cost, best_by_f1, cost_at

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=int, default=60_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    rows = 20_000 if args.quick else args.rows
    folds = 3 if args.quick else args.folds

    # ----------------------------------------------------------------- data
    frame = make_transactions(rows, args.seed)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RAW, index=False)

    X_train, X_test, y_train, y_test = split(frame, seed=args.seed)
    print(f"\n  {imbalance_summary(frame[TARGET])}")
    print(f"  train {imbalance_summary(y_train)}")
    print(f"  test  {imbalance_summary(y_test)}")
    print(f"\n  predicting 'never fraud' scores "
          f"{100 * (1 - frame[TARGET].mean()):.2f}% accuracy and catches nothing.\n")

    # ----------------------------------------------------------- supervised
    print("  supervised models, cross-validated on the training set:\n")
    model, winner, _ = fit_best(X_train, y_train, folds=folds, seed=args.seed)
    supervised_scores = model.predict_proba(X_test)[:, 1]

    # --------------------------------------------------------- unsupervised
    print("\n  unsupervised detector, trained on legitimate rows only:")
    detector = isolation_forest(seed=args.seed).fit_on_legitimate(X_train, y_train)
    anomaly_scores = detector.score(X_test)

    # -------------------------------------------------------------- scoring
    print("\n  held-out test set\n")
    supervised = score(y_test, supervised_scores)
    unsupervised = score(y_test, anomaly_scores)

    print(f"  {winner}:")
    print(supervised)
    print(f"\n  isolation_forest:")
    print(unsupervised)
    print(f"\n  labels are worth {supervised.average_precision / unsupervised.average_precision:.0f}x "
          f"in average precision.")

    # ------------------------------------------------------------ threshold
    amounts = X_test["amount"].to_numpy()

    by_cost = best_by_cost(y_test, supervised_scores, amounts)
    by_f1 = best_by_f1(y_test, supervised_scores, amounts)
    naive = cost_at(y_test, supervised_scores, 0.5, amounts)

    print("\n  threshold selection\n")
    print("  cost-optimal:")
    print(by_cost)
    print("\n  default 0.5:")
    print(naive)
    print(f"\n  choosing by cost instead of 0.5 saves "
          f"{naive.total_cost - by_cost.total_cost:,.0f} on this test set.")
    print(f"  F1-optimal threshold would be {by_f1.threshold:.4f} "
          f"(cost {by_f1.total_cost:,.0f}).")

    try:
        mandated = at_recall(y_test, supervised_scores, amounts, target_recall=0.80)
        print(f"\n  if recall 80% were mandated: threshold {mandated.threshold:.4f}, "
              f"precision {mandated.precision:.1%}, cost {mandated.total_cost:,.0f}")
    except ValueError as exc:
        print(f"\n  {exc}")

    print("\n  confusion matrix at the cost-optimal threshold")
    print(confusion_at(y_test, supervised_scores, by_cost.threshold).to_string())

    # -------------------------------------------------------------- figures
    if not args.no_figures:
        print()
        curves = {
            winner: (y_test, supervised_scores),
            "isolation_forest": (y_test, anomaly_scores),
        }
        paths = [
            plot_pr_curves(curves, float(np.mean(y_test))),
            plot_cost_curve(y_test, supervised_scores, amounts, by_cost.threshold),
            plot_recall_precision_tradeoff(y_test, supervised_scores, amounts),
        ]
        for path in paths:
            print(f"  figure: {path.relative_to(HERE)}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
