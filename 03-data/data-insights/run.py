#!/usr/bin/env python3
"""The whole pipeline, one command.

    python run.py                 generate data, clean, analyse, plot
    python run.py --rows 30000    a bigger synthetic dataset
    python run.py --input raw.csv use a real CSV instead
    python run.py --dark          also render dark-mode figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import analyse, visualise
from src.acquire import RAW, load, synthesise
from src.clean import clean
from src.theme import currency

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, help="a raw CSV to use instead of generating one")
    p.add_argument("--rows", type=int, default=12_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dark", action="store_true", help="also render dark-mode figures")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    # ---------------------------------------------------------------- acquire
    if args.input:
        raw = load(args.input)
        origin = str(args.input)
    elif RAW.exists():
        raw = load(RAW)
        origin = str(RAW)
    else:
        raw = synthesise(args.rows, args.seed)
        RAW.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(RAW, index=False)
        origin = f"synthetic (rows={args.rows}, seed={args.seed})"

    print(f"\nsource: {origin}")
    print(f"raw   : {len(raw):,} rows x {len(raw.columns)} columns")

    # ------------------------------------------------------------------ clean
    sales, returns, report = clean(raw)
    print(report)

    # ---------------------------------------------------------------- analyse
    stats = analyse.summary(sales, returns)
    print("  headline numbers")
    print(f"    period              {stats['period']}")
    print(f"    orders              {stats['orders']:,}")
    print(f"    gross revenue       {currency(stats['gross_revenue'])}")
    print(f"    refunds             {currency(stats['refunds'])}")
    print(f"    net revenue         {currency(stats['net_revenue'])}")
    print(f"    average order       {currency(stats['aov'])}")
    print(f"    return rate         {stats['return_rate_pct']:.1f}%")
    print(f"    known customers     {stats['identified_customers']:,}")
    print(f"    repeat customers    {stats['repeat_customer_pct']:.1f}% "
          f"-> {stats['repeat_revenue_pct']:.1f}% of revenue")
    print(f"    top 10% of buyers   {stats['top_10pct_revenue_pct']:.1f}% of revenue")

    print("\n  revenue by category")
    for _, row in analyse.category_performance(sales, returns).iterrows():
        print(f"    {row['category']:<12} {currency(row['revenue']):>10}  "
              f"{row['revenue_share']:>5.1f}%   AOV {currency(row['aov']):>8}   "
              f"returns {row['return_rate']:.1f}%")

    print("\n  top products")
    for _, row in analyse.top_products(sales, 5).iterrows():
        print(f"    {row['product']:<18} {currency(row['revenue']):>10}  "
              f"{int(row['units']):>6,} units")

    # --------------------------------------------------------------- visualise
    if not args.no_figures:
        print()
        for mode in ("light", "dark") if args.dark else ("light",):
            for path in visualise.build_all(sales, returns, mode):
                print(f"  figure: {path.relative_to(HERE)}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
