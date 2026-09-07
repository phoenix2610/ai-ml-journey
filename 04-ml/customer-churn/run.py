#!/usr/bin/env python3
"""The whole pipeline, one command.

    python run.py              12k subscribers
    python run.py --quick      5k rows, 3 folds
    python run.py --show-leak  demonstrate what the leaky column does

Compare models on Brier score, calibrate the winner, explain it globally and
per customer, then turn probabilities into a retention campaign.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import LEAKY, RAW, TARGET, make_subscribers, split, tenure_cohort
from src.explain import actionable_drivers, importance, reasons_for, risk_table
from src.model import candidates, fit_best, max_calibration_gap
from src.targeting import (
    build_campaign,
    campaign_at_budget,
    compare_to_probability_ranking,
    optimal_campaign,
)

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=int, default=12_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--show-leak", action="store_true")
    args = p.parse_args(argv)

    rows = 5_000 if args.quick else args.rows
    folds = 3 if args.quick else args.folds

    frame = make_subscribers(rows, args.seed)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RAW, index=False)

    X_train, X_test, y_train, y_test = split(frame, seed=args.seed)
    print(f"\n  {len(frame):,} subscribers, churn rate {frame[TARGET].mean():.1%}")
    print("\n  churn by tenure cohort")
    cohorts = frame.groupby(tenure_cohort(frame["tenure_months"]), observed=True)[TARGET].agg(["mean", "size"])
    for cohort, row in cohorts.iterrows():
        print(f"    {str(cohort):<7} {row['mean']:>6.1%}  (n={int(row['size']):,})")

    # ------------------------------------------------------------- the leak
    if args.show_leak:
        from sklearn.metrics import roc_auc_score

        Xl_tr, Xl_te, yl_tr, yl_te = split(frame, seed=args.seed, drop_leaky=False)
        leaked = candidates(args.seed, extra_numeric=tuple(LEAKY))["gradient_boosting"]
        leaked.fit(Xl_tr, yl_tr)
        auc = roc_auc_score(yl_te, leaked.predict_proba(Xl_te)[:, 1])
        print(f"\n  WITH the leaky exit_survey_sent column: ROC-AUC {auc:.4f}")
        print("  That is not a good model, it is a column that is only populated")
        print("  after the customer has already left. It is dropped everywhere else.")

    # -------------------------------------------------------------- models
    print()
    raw, calibrated, winner, _ = fit_best(X_train, y_train, folds=folds, seed=args.seed)

    raw_probability = raw.predict_proba(X_test)[:, 1]
    cal_probability = calibrated.predict_proba(X_test)[:, 1]

    print("\n  calibration on the test set")
    print(f"    worst gap, uncalibrated  {max_calibration_gap(y_test, raw_probability):.3f}")
    print(f"    worst gap, calibrated    {max_calibration_gap(y_test, cal_probability):.3f}")

    # ------------------------------------------------------------- explain
    print("\n  what drives churn (permutation importance, test set)")
    global_importance = importance(calibrated, X_test, y_test, n_repeats=5, seed=args.seed)
    for _, row in global_importance.head(6).iterrows():
        flag = "actionable" if row["actionable"] else "context only"
        print(f"    {row['feature']:<20} {row['importance']:>7.4f}  ({flag})")

    print("\n  levers a retention team can actually pull")
    for _, row in actionable_drivers(global_importance).head(4).iterrows():
        print(f"    {row['feature']:<20} -> {row['lever']}")

    # ---------------------------------------------------- one customer, why
    riskiest = int(np.argmax(cal_probability))
    print(f"\n  highest-risk customer: P(churn) {cal_probability[riskiest]:.1%}")
    for reason in reasons_for(calibrated, X_test.iloc[[riskiest]], X_train):
        print(f"    {reason['lever']:<42} -{reason['risk_reduction']:.1%} risk")

    # ------------------------------------------------------------ campaign
    print("\n  risk bands")
    bands = risk_table(calibrated, X_test)["risk_band"].value_counts().sort_index()
    for band, count in bands.items():
        print(f"    {str(band):<9} {count:>6,}")

    campaign = build_campaign(
        pd.Series(cal_probability, index=X_test.index), X_test["monthly_charges"]
    )
    plan = optimal_campaign(campaign)

    print("\n  retention campaign, sized by expected value\n")
    print(plan)

    budgeted = campaign_at_budget(campaign, budget=2_000)
    print(f"\n  under a 2,000 budget: contact {budgeted.contact_count:,}, "
          f"expected profit {budgeted.expected_profit:,.0f}")

    if plan.contact_count:
        comparison = compare_to_probability_ranking(campaign, plan.contact_count)
        print(f"\n  ranking by expected value vs by probability alone: "
              f"{comparison['uplift']:+,.0f} profit")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
