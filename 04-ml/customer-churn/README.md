# Customer Churn

Predicting churn is the easy half. This project is mostly about the other
half — **calibrated probabilities, explanations someone can act on, and a
campaign that pays for itself.**

**[See the results →](RESULTS.md)**

```bash
pip install -r requirements.txt
python run.py                # 12k subscribers, end to end
python run.py --show-leak    # what the leaky column does to the score
python run.py --quick        # 5k rows, ~30 seconds
pytest -q                    # 58 tests
```

## Ranking is not enough here

In the [fraud project](../fraud-detection/) scores only had to *rank* — a
threshold turned them into decisions and the absolute value never mattered.
Churn is different, because the probability gets multiplied by money:

```
expected_loss = P(churn) × monthly_charges × remaining_months
```

If the model says 0.30 where the true rate is 0.55, that arithmetic is wrong and
every budgeting decision inherits the error. So models are ranked by **Brier
score**, not ROC-AUC, and the pipeline ends with a calibration check.

Result: logistic regression won on both, and its calibration gap was already
0.020 — **calibration turned out to be unnecessary**, and running the check is
what established that rather than assuming it either way.

## The deliberate trap

`exit_survey_sent` is in the dataset on purpose. It agrees with the target 94%
of the time — and is only populated *after* someone churns. Include it and:

```
without leak   ROC-AUC 0.8035
WITH leak      ROC-AUC 0.9790
```

A model that looks 22% better and is worth nothing, because at prediction time
the column is always zero.

**Writing this exposed a subtler bug.** The first version of `--show-leak`
reported *identical* scores with and without the column. The reason was that
`build_preprocessor` names its columns explicitly and sets `remainder="drop"` —
so the stray column was silently discarded and the demo was demonstrating
nothing.

That is worth knowing in both directions: naming columns explicitly is a
**structural defence against leakage** (a column added upstream cannot sneak
into the model), and it will also quietly swallow a column you *meant* to
include. Both behaviours now have tests.

## Importance is not actionability

| Feature | Importance | |
|---|---:|---|
| contract | 0.1682 | **actionable** |
| support_calls_90d | 0.0331 | **actionable** |
| tenure_months | 0.0261 | context only |
| outages_90d | 0.0179 | **actionable** |

Tenure is a strong predictor and **useless as an intervention** — you cannot
make a customer older. A driver report sorted by importance alone reads well and
cannot be acted on, so `ACTIONABLE` maps each usable feature to a concrete lever
and `actionable_drivers()` filters to those.

Per customer, `reasons_for()` runs a counterfactual — replace one feature with
the population's typical value, re-score, and report the drop:

```
highest-risk customer: P(churn) 95.0%
  offer a longer contract        -32.5% risk
  proactive support outreach     -26.2% risk
  fix service reliability         -7.2% risk
```

These are what the *model* predicts, not causal effects. Customers who accept a
longer contract differ from those who don't.

## The tenure confound

New customers churn 3.4× more than long-tenured ones — and tenure correlates
with contract type, total charges, and service mix. Any claim about *why*
someone churned has to survive controlling for it, so the contract effect is
verified **within the 0–6m cohort alone**, not just across the whole book.

## Sizing the campaign

The step most churn projects skip. A ranked risk list is not a decision; you
still have to choose how many people to call, and that has a computable optimum.

```
expected_gain = P(churn) × save_rate × customer_value − contact_cost
```

```
contact          1,163 of 3,000 (38.8%)
expected profit        67,234
cutoff P(churn)         0.177
```

The 17.7% cutoff is an **output**, not a chosen threshold.

Two things the naive version gets wrong, both pinned by tests:

**Ranking by probability instead of expected value** costs £4,449 on the same
number of contacts — a high-risk cheap customer can be worth less than a
medium-risk expensive one.

**Pricing only the phone call.** An earlier version used £8 per contact and
recommended contacting **86% of the book** for a 12× return. Including the
retention *incentive* (£45) brings it to a sane 38.8%. This is the standard way
a churn model "proves" you should call everybody.

## Layout

```
src/
├── data.py       generation from a documented logistic model; the leaky column
├── model.py      candidates ranked by Brier; calibration
├── explain.py    global importance, actionable filter, per-customer reasons
└── targeting.py  campaign economics and sizing
run.py            the whole thing
```

## Tests

58 tests. The load-bearing ones: that the contract effect survives controlling
for tenure, that a badly-calibrated model is actually detected (halving every
probability must show up), that the leaky column inflates ROC-AUC by >0.10, and
that an unlisted column cannot become a feature.
