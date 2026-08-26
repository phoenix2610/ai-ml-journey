# House Price Prediction

Regression done the way it has to be done if the number is going to be used for
anything: **leak-free pipelines, one look at the test set, and error bars on
every prediction.**

**[See the results →](RESULTS.md)**

```bash
pip install -r requirements.txt
python run.py                       # generate, compare, tune, evaluate, plot
python run.py --quick               # 3 folds, ~1 minute
python -m src.predict --example     # price one house
pytest -q                           # 40 tests
```

## The dataset is generated from a known price function

`data.py` builds houses from a documented ground truth — `TRUE_EFFECTS` states
exactly what the generator did:

```
price = 185 * area_sqft**0.92        sublinear in area
      * neighbourhood multiplier     0.82 – 1.42, the largest single factor
      * condition, heating multipliers
      * depreciation, floored at 0.72
      + 9500 * garage_spaces + 4.5 * lot_sqft
      + lognormal noise
```

That is a teaching choice, not a shortcut. When you *know* the true effect of
each feature you can check whether the model **recovered** it, instead of
admiring an R² and hoping. It does: permutation importance ranks
`area_sqft > neighbourhood > condition > distance_km`, which is the generator's
own ordering. `--csv` swaps in a real dataset; nothing downstream cares.

## The one bug this project exists to prevent

```python
X[num] = SimpleImputer().fit_transform(X[num])      # WRONG
X[num] = StandardScaler().fit_transform(X[num])
scores = cross_val_score(model, X, y, cv=5)
```

The imputer's medians and the scaler's means were computed from **all** rows —
including the ones about to become each validation fold. Every fold is scored
on data whose statistics it already absorbed, and CV reports a number better
than the model will ever achieve in production.

The fix is structural, not disciplinary: put preprocessing **inside** a
`Pipeline`, and `cross_val_score` re-fits it within each fold.

```python
Pipeline([("pre", build_preprocessor()), ("model", Ridge())])
```

There is a test asserting the honest pipeline never scores better than the
leaky version — so the guarantee is checked, not just documented.

## A confound worth seeing

Bigger houses cluster in pricier neighbourhoods:

```
Northfield  1,229 sqft        Riverside  2,122 sqft
```

So a raw "big vs small" price comparison measures **area and prestige at
once**, and the generator's sublinear area effect looks super-linear. Hold
neighbourhood and condition fixed and it reappears — price per sqft falls as
houses grow. Two tests pin this: one asserting the confound exists, one
asserting the real effect shows up once you control for it.

This is why `test_area_effect_is_sublinear` failed on the first run. The
generator was right; the test was measuring the wrong thing.

## Evaluation reports how wrong it is

A single R² is not an evaluation. What anyone actually needs:

| | Why |
|---|---|
| **MAE** in currency | "£15k out" is a sentence someone can act on |
| **MAPE** | £40k on a £900k mansion ≠ £40k on a £120k flat |
| **Error by price decile** | aggregates hide the segment you cared about |
| **Empirical prediction intervals** | residuals are not Gaussian; ±1.645σ would be confidently wrong in both tails |

The decile table is the interesting one — it shows the model **underpredicting
the top decile** by a wide margin. Regression to the mean, and it is invisible
in any headline metric. Details in [RESULTS.md](RESULTS.md).

## Model selection

Every candidate is a full pipeline, compared on CV over the training set only.
`DummyRegressor` is in the lineup deliberately: R² means nothing until you know
what predicting the median scores.

```
gradient_boosting  MAE  14,888   R² 0.926
ridge              MAE  15,363   R² 0.923
random_forest      MAE  18,412   R² 0.890
baseline_median    MAE  58,530   R² -0.040
```

**The test set is touched exactly once**, by `evaluate.py`, after every
decision is final. Selecting on test scores turns the test score into a
training score wearing a disguise.

## What gets saved

The **whole pipeline**, not the bare estimator — so inference applies exactly
the imputation and encoding that training used. Saving an estimator alone and
re-implementing preprocessing at inference is the most common way a model that
scored well offline produces nonsense in production.

Residual quantiles are saved alongside it, so a prediction carries an interval:

```
$ python -m src.predict --example
  predicted price   £185,248
  90% interval      £153,656 – £223,271  (±£34,807)
```

## Layout

```
src/
├── data.py       generation from a known price function; stratified split
├── features.py   ColumnTransformer; the leakage guarantee
├── models.py     the zoo, including the dummy baseline
├── train.py      CV comparison, then tuning the winner
├── evaluate.py   metrics, decile breakdown, prediction intervals
├── plots.py      four diagnostics
└── predict.py    single-house inference with an interval
run.py            the whole thing
```

## Tests

40 tests. The ones that matter are the leakage guarantee, the confound pair,
and the data assertions that check the generator produces the relationships
`TRUE_EFFECTS` claims — because if the data is wrong, every downstream number
is measuring nothing.
