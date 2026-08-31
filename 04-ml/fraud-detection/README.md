# Fraud Detection

Binary classification where **0.3% of rows are positive**. That single number
invalidates most of the default workflow — accuracy, ROC-AUC, and a 0.5
threshold are all misleading here, and this project demonstrates each one
rather than asserting it.

**[See the results →](RESULTS.md)**

```bash
pip install -r requirements.txt
python run.py              # 60k transactions, supervised + unsupervised
python run.py --quick      # 20k rows, ~1 minute
pytest -q                  # 54 tests
```

## Three things that go wrong at 0.3%

**1. Accuracy is meaningless.** Predicting "never fraud" scores 99.70%. It is
not reported anywhere in this project.

**2. ROC-AUC is flattering.** Measured on real output:

| Model | Average precision | ROC-AUC |
|---|---:|---:|
| random_forest | **0.3516** | 0.8017 |
| isolation_forest | **0.0777** | 0.7585 |

By ROC-AUC these look comparable. By average precision one is **4.5× better**.
ROC's x-axis is false-positive *rate* over 14,955 negatives — going from 23
false alarms to 230 barely moves it, while tenfold-ing the review queue.
Precision divides by the number of alerts, so it notices.

**3. The 0.5 threshold is arbitrary.** At 0.5 this model scores **100%
precision and 13.3% recall** — it never raises a false alarm and misses 39 of
45 frauds. Maximal precision, barely switched on. Choosing by cost instead
saves 24%.

## Choosing the threshold by cost

State what the errors cost, sweep, pick the cheapest:

- **False negative** — fraud goes through. Costs the transaction amount, which
  the data knows per row.
- **False positive** — a customer is declined. Fixed review cost.

```
cost-optimal   threshold 0.1141   precision 39.5%   recall 33.3%   cost   995
default 0.5    threshold 0.5000   precision 100.0%  recall 13.3%   cost 1,312
```

The tests pin the behaviour that makes this principled rather than a fitted
curiosity: expensive reviews push the threshold **up**, valuable recoveries
push it **down**. `at_recall()` handles the case where a risk committee mandates
recall and the job becomes minimising cost subject to it — at 80% recall, cost
rises **65×**, which is worth knowing before agreeing to it.

## The data had to be made harder

The first version of the generator produced fraud that differed from legitimate
traffic on **seven dimensions at once**. An isolation forest scored **AP 0.948**
on it — near-perfect, and completely unrealistic (real unsupervised fraud
detection lands around 0.05–0.15).

Separable data would have made this entire project meaningless: PR-AUC,
threshold tuning, and the cost model all only matter when the problem is hard.

The fix was structural. Fraud rows now start as draws from the **legitimate**
distribution, and only the tells belonging to their archetype get perturbed —
and each tell only fires ~78% of the time:

| Archetype | Tells | Share |
|---|---|---:|
| card_testing | tiny amount, burst, online | 26% |
| account_takeover | new device, far from home, high amount | 24% |
| night_cashout | odd hour, high amount | 18% |
| **stealth** | **none** | **32%** |

AP fell to **0.040** — realistic. The stealth third is the important part: it is
statistically indistinguishable from normal behaviour on these features, which
is why recall tops out around 33% no matter what threshold you pick. That is not
a modelling failure, it is the information limit of the feature set.

## Unsupervised, and why it's here

Real fraud teams rarely start with labels — chargebacks arrive weeks late, most
fraud is never reported, and a new attack has no examples by definition.

`IsolationForest` is trained on **legitimate transactions only** (`fit_on_legitimate`).
Training it on everything teaches it that fraud is normal too, which is exactly
what it is supposed to find surprising; there is a test comparing both ways.

It reaches a **26× lift over random with zero labels** — genuinely useful, and
5× worse than the supervised model. That gap *quantifies* what labels are worth
instead of assuming it.

## No resampling

SMOTE and random oversampling are conspicuously absent. They distort the base
rate, so predicted probabilities stop meaning what they say, and a threshold
chosen on resampled data does not transfer to production traffic.
`class_weight='balanced'` achieves the same re-weighting without inventing rows
or wrecking calibration.

## Layout

```
src/
├── data.py        archetype-based generation; stratified split
├── features.py    log-compress skewed money columns; no resampling
├── supervised.py  the labelled path, compared on average precision
├── anomaly.py     the unlabelled path, fitted on negatives only
├── threshold.py   cost model and threshold sweep
└── evaluate.py    PR curves, cost curve, trade-off plot
run.py             the whole thing
```

## Tests

54 tests. The ones carrying weight: that a stratified split preserves the
positive rate (an unstratified one swings test-set fraud by ±33%), that
`archetype` never leaks into features, that anomaly scores are oriented so
"higher = more suspicious" matches `predict_proba[:, 1]`, and that the cost
model responds correctly to changes in review cost and recovery rate.
