# Findings

Two years of transactions, Jan 2023 – Dec 2024. **11,628 orders, £836.0k net
revenue** after £25.9k of refunds. Average order £74, return rate 3.1%.

All figures below are reproducible with `python run.py`.

---

## 1. Revenue is flat. The growth is seasonal, not structural.

![Net revenue by month](figures/01-revenue-over-time.png)

Baseline months sit in a tight £25–35k band across both years. There is no
trend — 2024 is not meaningfully bigger than 2023.

What there *is* is a very sharp Q4: **November runs +87% and December +90%
against the yearly average**, and both repeat in both years.

![Seasonality](figures/03-seasonality.png)

That repetition is the point. A single spike would be a one-off event; the same
lift in two consecutive years is a pattern you can plan inventory around.
February is the trough at −30%.

> **Caveat worth stating plainly:** two years is two observations of "Q4". It is
> enough to call it recurring, not enough to forecast magnitude. A third year
> would change how much confidence this deserves.

---

## 2. Two categories carry half the business.

![Revenue by category](figures/02-category-revenue.png)

| Category | Revenue | Share | AOV | Return rate |
|---|---:|---:|---:|---:|
| Home | £219.7k | 25.5% | £85 | 3.0% |
| Lighting | £204.0k | 23.7% | £151 | 3.4% |
| Kitchen | £142.3k | 16.5% | £51 | 3.2% |
| Toys | £138.9k | 16.1% | £101 | 3.7% |
| Garden | £101.7k | 11.8% | £62 | 2.4% |
| Stationery | £55.4k | 6.4% | £30 | 3.8% |

**Home and Lighting are 49.2% of revenue** on very different mechanics: Home
sells volume at £85 an order, Lighting sells far fewer orders at £151. Lighting
is the highest-value basket in the catalogue.

**Stationery is the one to question.** It is 6.4% of revenue at a £30 average
order and the *highest* return rate at 3.8% — the least revenue for the most
handling. It may still be worth carrying as a basket-filler, but that is a
merchandising argument, not a revenue one, and the data here cannot settle it.

---

## 3. Revenue is heavily concentrated in repeat buyers.

![Revenue concentration](figures/04-revenue-concentration.png)

Of 1,459 identified customers:

- **43.2% bought more than once — and they account for 91.6% of revenue.**
- **The top 10% of customers account for 77.3% of revenue.**

That is a steeper curve than the usual 80/20 shorthand, and it points retention
spend at a small, identifiable group rather than at broad acquisition.

**The honest limitation:** 13.7% of orders have no `customer_id` at all — guest
checkout. Those orders are counted in revenue but are invisible to every
per-customer number above. If guests skew toward one-time buyers (they almost
certainly do), the true repeat-customer *share* is lower than 43.2%. The revenue
concentration figure is more robust, since the missing orders are small ones.

---

## What this analysis cannot tell you

Worth being explicit, because these are the questions someone will ask next:

- **No margin data.** Every ranking here is revenue, not profit. Lighting's £151
  basket could be the least profitable line in the catalogue and nothing in this
  dataset would show it.
- **No acquisition channel.** "Retention beats acquisition" follows from the
  concentration curve, but the cost side is absent.
- **No causality.** Q4 lift is a description. Whether it is demand, discounting,
  or a campaign is not in the data.
- **Returns are counted, not explained.** Stationery's 3.8% return rate has no
  reason attached to it.

## Reproducing

```bash
pip install -r requirements.txt
python run.py --dark          # numbers to stdout, figures to figures/
pytest -q                     # 59 tests
```

The dataset is generated deterministically from a seed, so every number in this
document reproduces exactly. See the README for why that choice was made.
