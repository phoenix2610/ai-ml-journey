import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.data import CATEGORICAL, NUMERIC, make_houses, split
from src.features import (
    DERIVED,
    add_derived,
    build_preprocessor,
    build_preprocessor_with_derived,
    feature_names,
)


@pytest.fixture(scope="module")
def data():
    return split(make_houses(1200, seed=11), seed=11)


def test_output_has_no_missing_values(data):
    X_tr, _, _, _ = data
    out = build_preprocessor().fit_transform(X_tr)
    assert not np.isnan(out).any()


def test_row_count_is_preserved(data):
    X_tr, _, _, _ = data
    assert build_preprocessor().fit_transform(X_tr).shape[0] == len(X_tr)


def test_categoricals_are_expanded(data):
    X_tr, _, _, _ = data
    out = build_preprocessor().fit_transform(X_tr)
    # One-hot expansion means more columns out than raw features in.
    assert out.shape[1] > len(NUMERIC) + len(CATEGORICAL)


def test_scaling_standardises_numeric_columns(data):
    X_tr, _, _, _ = data
    pre = build_preprocessor(scale=True).fit(X_tr)
    out = pre.transform(X_tr)
    numeric_block = out[:, : len(NUMERIC)]
    assert np.allclose(numeric_block.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(numeric_block.std(axis=0), 1, atol=1e-6)


def test_scaling_can_be_disabled(data):
    X_tr, _, _, _ = data
    out = build_preprocessor(scale=False).fit_transform(X_tr)
    assert out[:, : len(NUMERIC)].std(axis=0).max() > 5


def test_feature_names_match_column_count(data):
    X_tr, _, _, _ = data
    pre = build_preprocessor().fit(X_tr)
    assert len(feature_names(pre)) == pre.transform(X_tr).shape[1]


def test_missing_becomes_its_own_category(data):
    X_tr, _, _, _ = data
    pre = build_preprocessor().fit(X_tr)
    assert any("missing" in name for name in feature_names(pre))


def test_unseen_category_does_not_raise(data):
    """handle_unknown='ignore' -- a new neighbourhood must not crash inference."""
    X_tr, X_te, _, _ = data
    pre = build_preprocessor().fit(X_tr)
    novel = X_te.head(3).copy()
    novel["neighbourhood"] = "Atlantis"
    out = pre.transform(novel)
    assert out.shape == (3, pre.transform(X_te.head(3)).shape[1])


def test_transform_is_deterministic(data):
    X_tr, _, _, _ = data
    pre = build_preprocessor().fit(X_tr)
    assert np.array_equal(pre.transform(X_tr), pre.transform(X_tr))


# ------------------------------------------------------- the leakage guarantee


def test_preprocessing_is_refit_inside_each_cv_fold(data):
    """The whole reason preprocessing lives in a Pipeline.

    Fitting the imputer/scaler once on all of X leaks validation-fold
    statistics into training and inflates the CV score. A Pipeline is re-fit
    per fold, so its score must be no better than the leaky version.
    """
    X_tr, _, y_tr, _ = data

    pipeline = Pipeline([("pre", build_preprocessor()), ("model", Ridge(alpha=1.0))])
    honest = cross_val_score(pipeline, X_tr, y_tr, cv=5, scoring="r2").mean()

    leaked_X = build_preprocessor().fit_transform(X_tr)   # fit on everything
    leaky = cross_val_score(Ridge(alpha=1.0), leaked_X, y_tr, cv=5, scoring="r2").mean()

    assert honest <= leaky + 1e-9


def test_pipeline_never_sees_the_target(data):
    X_tr, _, _, _ = data
    from src.data import TARGET
    assert TARGET not in X_tr.columns
    assert not any(TARGET in name for name in feature_names(build_preprocessor().fit(X_tr)))


# ------------------------------------------------------------------- derived


def test_add_derived_creates_every_column(data):
    X_tr, _, _, _ = data
    out = add_derived(X_tr)
    assert all(column in out.columns for column in DERIVED)


def test_add_derived_does_not_mutate_its_input(data):
    X_tr, _, _, _ = data
    before = X_tr.copy()
    add_derived(X_tr)
    pd.testing.assert_frame_equal(X_tr, before)


def test_rooms_total_is_the_sum(data):
    X_tr, _, _, _ = data
    out = add_derived(X_tr)
    assert (out["rooms_total"] == out["bedrooms"] + out["bathrooms"]).all()


def test_area_per_room_survives_zero_rooms():
    frame = pd.DataFrame(
        {
            "bedrooms": [0], "bathrooms": [0], "area_sqft": [900.0],
            "lot_sqft": [1800.0], "age_years": [10.0],
        }
    )
    assert np.isnan(add_derived(frame)["area_per_room"].iloc[0])


def test_derived_preprocessor_produces_more_columns(data):
    X_tr, _, _, _ = data
    plain = build_preprocessor().fit_transform(X_tr).shape[1]
    rich = build_preprocessor_with_derived().fit_transform(add_derived(X_tr)).shape[1]
    assert rich == plain + len(DERIVED)


def test_derived_pipeline_handles_the_nan_it_can_create():
    """area_per_room is NaN when rooms are 0; the imputer must absorb it."""
    frame = make_houses(200, seed=4).drop(columns=["price"])
    frame.loc[frame.index[:5], ["bedrooms", "bathrooms"]] = 0
    out = build_preprocessor_with_derived().fit_transform(add_derived(frame))
    assert not np.isnan(out).any()
