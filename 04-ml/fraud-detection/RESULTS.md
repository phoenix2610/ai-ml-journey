# Results

60,000 transactions, 180 fraud (0.300%). 45,000 train / 15,000 test.
Reproduce with `python run.py`.

## Why accuracy is not reported anywhere

A model that predicts "never fraud" scores **99.70% accuracy** and catches zero
fraud. That number is not a weak result — it is a meaningless one, so it does
not appear in the output at all.

## ROC-AUC is flattering. Here is the proof.

| Model | Average precision | ROC-AUC |
|---|---:|---:|
| random_forest (supervised) | **0.3516** | 0.8017 |
| isolation_forest (unsupervised) | **0.0777** | 0.7585 |
| random baseline | 0.0030 | 0.500 |

Read the two columns against each other. By **ROC-AUC** these models look
comparable — 0.80 against 0.76, a 5% gap you might shrug at. By **average
precision** the supervised model is **4.5× better**.

The reason is the denominator. ROC's x-axis is false-positive *rate*, measured
against 14,955 legitimate transactions. Going from 23 false alarms to 230 moves
FPR from 0.0015 to 0.0154 — invisible on an ROC curve. To the fraud team it is
ten times the review queue. Precision divides by the number of *alerts*, so it
registers that change immediately.

**If you pick a model on ROC-AUC with a 0.3% positive rate, you will pick the
wrong model.**

## What labels are worth

Supervised beats unsupervised by **5× in average precision** (0.352 vs 0.078).

That is the number worth having, because it is rarely stated. The isolation
forest still achieves a **26× lift over random** while never seeing a single
label — genuinely useful when a new attack pattern has no examples yet, or
while you are waiting weeks for chargebacks to confirm. It is not a substitute
for labels; it is what you run before you have them.

## The threshold is a business decision

Cost model: a missed fraud costs the transaction amount; a caught fraud recovers
35% of it; a false alarm costs £12 of analyst review.

| | Threshold | Precision | Recall | Caught/missed | False alarms | Net cost |
|---|---:|---:|---:|---:|---:|---:|
| **Cost-optimal** | **0.1141** | 39.5% | 33.3% | 15 / 30 | 23 | **995** |
| Default 0.5 | 0.5000 | 100.0% | 13.3% | 6 / 39 | 0 | 1,312 |

Moving off the default saves **317 (24%)** on this test set alone.

Look at what the default actually does: **100% precision and 13.3% recall.** It
never raises a false alarm and it misses 39 of 45 frauds. On paper that
precision looks like the model working perfectly. In reality it is a threshold
so conservative the system is barely switched on — and no aggregate metric
flags it, because precision is *maximal*.

`class_weight='balanced'` is part of why: it deliberately re-weights the loss,
so a score of 0.5 does not mean "50% likely to be fraud". The default was never
meaningful here.

## Mandated recall is expensive, and worth pricing

If a risk committee mandated 80% recall:

```
threshold 0.0033   precision 0.7%   cost 65,012
```

Cost rises **65×**. At 0.7% precision, 99.3% of alerts are false — roughly 1,700
legitimate customers declined to catch 36 frauds. That may still be the right
call for reputational or regulatory reasons, but this makes the price explicit
instead of leaving it as an assumption.

## Confusion matrix at the cost-optimal threshold

```
                passed  flagged
actually legit   14932       23
actually fraud      30       15
```

33% recall is not a good detector, and it should not be dressed up as one. It
reflects the data honestly: **32% of the fraud in this dataset is "stealth"** —
generated with no distinguishing tells on the available features. No threshold
recovers it, because the signal is not there. Catching it needs *different
features* (device fingerprints, velocity across merchants, graph links between
accounts), not a better model on these ones.

## Limitations

- **Synthetic data.** The archetype mix is a modelling assumption, not an
  observation. Real fraud adapts adversarially; a generator does not.
- **No temporal split.** Rows are shuffled at random. Real fraud models must be
  trained on the past and tested on the future — fraud patterns drift within
  weeks, and a random split hides that completely.
- **Cost parameters are invented.** £12 review, 35% recovery. In a real
  deployment finance supplies these, and the optimal threshold moves with them.
- **No calibration check.** Scores are used for ranking and thresholding only.
  Using them as probabilities would require a reliability curve first.
