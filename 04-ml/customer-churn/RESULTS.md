# Results

12,000 subscribers, 26.3% churn. 9,000 train / 3,000 test.
Reproduce with `python run.py`.

## Churn is concentrated in new customers

| Cohort | Churn | n |
|---|---:|---:|
| 0–6m | 38.8% | 1,399 |
| 6–12m | 35.2% | 1,864 |
| 1–2y | 29.1% | 3,350 |
| 2–4y | 21.3% | 3,628 |
| 4y+ | 11.5% | 1,759 |

A subscriber in their first six months is **3.4× more likely to leave** than one
past four years. This is also the confound that contaminates every other
comparison: tenure correlates with contract type, with total charges, and with
service mix. Any claim about *why* someone churned has to survive controlling
for it — and `test_contract_effect_survives_controlling_for_tenure` checks that
the contract effect still holds within the 0–6m cohort alone.

## Model selection, ranked by calibration

| Model | ROC-AUC | Brier | Log-loss |
|---|---:|---:|---:|
| **logistic** | **0.809** | **0.1471** | 0.4495 |
| gradient_boosting | 0.795 | 0.1516 | 0.4628 |
| random_forest | 0.793 | 0.1524 | 0.4653 |
| baseline_majority | 0.500 | 0.1936 | 0.5757 |

**Logistic regression wins** — on both discrimination and calibration. That is
not the expected result for tabular data, and it is worth taking at face value
rather than tuning until the boosted model wins: the generator's churn process
*is* a logistic function, so a linear model in log-odds is correctly specified.
Gradient boosting spends capacity approximating a shape logistic regression
already has exactly.

Models are ranked by **Brier score**, not ROC-AUC, because the probabilities get
multiplied by money downstream.

## Calibration

```
worst gap, uncalibrated  0.020
worst gap, calibrated    0.026
```

Logistic regression is already well calibrated — it optimises log loss directly
— so isotonic calibration has nothing to fix and adds a little noise. **The
honest read is that calibration was unnecessary here**, and the check is what
established that. Had gradient boosting won, the gap would have been much wider;
boosting pushes probabilities toward 0 and 1.

This matters because retention budgeting computes
`P(churn) × save_rate × customer_value`. If the model says 0.30 where the true
rate is 0.55, every spending decision inherits the error.

## What drives churn

| Feature | Importance | |
|---|---:|---|
| contract | 0.1682 | **actionable** |
| support_calls_90d | 0.0331 | **actionable** |
| tenure_months | 0.0261 | context only |
| outages_90d | 0.0179 | **actionable** |
| payment_method | 0.0063 | **actionable** |
| internet_service | 0.0049 | **actionable** |

`contract` dominates by 5×, which matches the generator: month-to-month carries
a +1.30 log-odds coefficient against −0.95 for two-year.

The important distinction is the right-hand column. **Importance is not
actionability.** Tenure is a strong predictor and completely useless as an
intervention — you cannot make a customer older. Sorting drivers by importance
alone produces a report that reads well and cannot be acted on, so
`actionable_drivers()` filters to features a retention team can actually change.

## Per-customer reasons

For the highest-risk account (P = 95.0%), the counterfactual analysis gives:

```
offer a longer contract                  -32.5% risk
proactive support outreach               -26.2% risk
fix service reliability                   -7.2% risk
```

Each number is measured by replacing one feature with the population's typical
value and re-scoring. That is what a retention call needs — "the model said
0.95" is not a conversation.

## The campaign

Economics: £45 per contact (staff time **plus the incentive actually given**),
25% save rate on genuinely at-risk customers, 12-month value horizon.

```
contact          1,163 of 3,000 (38.8%)
campaign cost          52,335
expected saves          140.4 customers
expected profit        67,234
cutoff P(churn)         0.177
```

The cutoff is the output, not an input. Everyone above a **17.7%** churn
probability is worth contacting; below it the expected save is worth less than
the offer. Under a £2,000 budget instead, contact the top 44 for £7,049 profit.

**Ranking by expected value rather than by probability is worth +£4,449** on the
same 1,163 contacts. A high-risk cheap customer can be worth less than a
medium-risk expensive one, and ranking on probability alone misses that.

An earlier version of this model used £8 per contact — pricing only the phone
call. It recommended contacting **86% of the book** for a 12× return, which is
the classic way a churn model "proves" you should call everybody. Pricing the
incentive is what makes the answer sane.

## Limitations

- **Synthetic data from a logistic process.** Logistic regression winning is
  partly a fixed fight. On real data the ranking would likely differ.
- **No temporal validation.** Churn is a time-to-event problem and this treats
  it as a static snapshot. A production model needs a defined prediction window
  and a time-based split.
- **`save_rate` is assumed, not measured.** The 25% is the single most load-
  bearing number in the campaign, and the only way to learn it is a randomised
  holdout — contact a random subset and compare. Without that, the profit figure
  is a projection, not a result.
- **Counterfactual reasons are not causal.** "Moving to a two-year contract cuts
  risk 32.5%" is what the *model* predicts, not what would happen. Customers who
  accept a longer contract differ from those who don't.
