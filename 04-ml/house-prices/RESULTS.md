# Results

6,000 houses, 4,800 train / 1,200 test. Every number below reproduces with
`python run.py`.

## Model selection (cross-validated on the training set only)

| Model | CV MAE | RMSE | R² | Fit |
|---|---:|---:|---:|---:|
| **gradient_boosting** | **14,778** ±363 | 20,235 | 0.924 | 1.8s |
| ridge | 15,894 ±305 | 21,313 | 0.916 | 0.1s |
| random_forest | 17,631 ±278 | 23,599 | 0.897 | 4.5s |
| baseline_median | 57,423 ±1,009 | 74,444 | −0.027 | 0.1s |

Gradient boosting wins with **74% lower MAE than predicting the median**. That
comparison is the point of keeping a dummy in the lineup — R² = 0.92 sounds
impressive in isolation and means nothing without a floor to measure from.

Ridge is within 8% of the winner at 1/18th the fit time. On a real system that
trade is worth considering: the boosted model buys ~£1,100 of average accuracy
for a large increase in complexity and inference cost.

Tuned: `learning_rate=0.05, max_leaf_nodes=15` → CV MAE **14,482**.

## Held-out test set (touched once, after every decision was final)

```
MAE                      13,772
RMSE                     18,628
median abs error         10,571
MAPE                       8.1%
R²                        0.934
within 10% of true        66.8%
test rows                  1,200
```

Test MAE (13,772) came in slightly **below** CV MAE (14,482), which is the
expected direction — the final model trains on all 4,800 rows rather than 3,840
per fold. A test score far *above* CV would be the warning sign.

Median absolute error (10,571) sits well below MAE (13,772): the typical house
is priced better than the average implies, and a tail of hard cases drags the
mean. That tail is the top decile — see below.

## Prediction intervals are calibrated

```
90% interval:  −27,176 to +32,601
actual coverage on the test set: 90.0%
```

Exactly the nominal rate. Worth noting the interval is **asymmetric**
(−27k/+33k) — residuals are right-skewed, so a symmetric ±1.645σ interval would
be wrong in both tails simultaneously. This is why the interval is computed from
empirical residual quantiles rather than assuming normality.

## Where the model is weak

| Decile | Median price | MAE | Bias | MAPE |
|---:|---:|---:|---:|---:|
| 1 | £75,260 | £8,001 | **+5,521** | 11.5% |
| 2 | £100,800 | £8,604 | +3,385 | 8.4% |
| 3 | £122,040 | £9,158 | +2,611 | 7.5% |
| 4 | £136,843 | £10,474 | +1,445 | 7.6% |
| 5 | £152,882 | £11,413 | +2,103 | 7.4% |
| 6 | £173,282 | £13,736 | −144 | 7.9% |
| 7 | £192,818 | £16,403 | +208 | 8.5% |
| 8 | £217,819 | £15,940 | −2,915 | 7.4% |
| 9 | £248,857 | £17,247 | −1,655 | 6.9% |
| 10 | £310,080 | £26,739 | **−13,126** | 8.1% |

**This is the finding that matters, and no headline metric shows it.**

The bias column runs monotonically from **+5,521 at the bottom to −13,126 at
the top**: the model systematically overprices cheap houses and underprices
expensive ones. That is textbook regression to the mean — squared-error loss
pulls predictions toward the centre, and the tails have the fewest neighbours to
learn from.

Practical consequence: on a £310k house the model is off by £26,739 on average
and **biased low**. If this priced mortgage collateral, the top decile would be
systematically under-reserved. R² = 0.934 tells you none of that.

MAPE stays flat at 7–8% across deciles 2–10, so in *relative* terms the model is
consistent. Decile 1 is the exception at 11.5% — small denominators make
percentage error unforgiving.

## What the model learned

| Feature | Δ MAE when shuffled |
|---|---:|
| area_sqft | 35,935 ±735 |
| neighbourhood | 15,342 ±411 |
| condition | 10,446 ±414 |
| distance_km | 4,201 ±217 |
| garage_spaces | 3,913 ±184 |
| age_years | 3,229 ±120 |

This ranking **matches the generator's true structure** — `TRUE_EFFECTS` makes
area the dominant term, then the neighbourhood multiplier (0.82–1.42), then
condition (0.72–1.15). The model recovered the data-generating process rather
than latching onto an artefact.

Measured on the test set, since importance on training data rewards
memorisation. Correlated features share credit, so read this as a ranking, not
an attribution — `area_sqft`, `bedrooms`, and `lot_sqft` all move together, and
area absorbs credit the others could equally claim.

## Honest limitations

- **Synthetic data.** The model recovering the generator proves the *pipeline*
  is sound; it does not prove anything about real housing markets. Real data
  has regime changes, spatial autocorrelation, and mispriced listings that no
  i.i.d. generator reproduces.
- **No temporal split.** Rows are shuffled at random. Real price prediction is a
  forecasting problem, and a random split leaks the future into the past. A
  production version must split by date.
- **The top-decile bias is unfixed.** Options: train on `log(price)`, use a
  quantile loss, or model the tail separately. Each has costs and none is free.
- **Intervals assume the residual distribution is stationary.** They are
  calibrated for the test set's price mix, and would need re-estimating on a
  materially different portfolio.
