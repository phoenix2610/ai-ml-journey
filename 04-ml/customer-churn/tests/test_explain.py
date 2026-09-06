import numpy as np
import pandas as pd
import pytest

from src.data import make_subscribers, split
from src.explain import (
    ACTIONABLE,
    IMMUTABLE,
    actionable_drivers,
    importance,
    reasons_for,
    risk_table,
)
from src.model import candidates


@pytest.fixture(scope="module")
def data():
    return split(make_subscribers(4_000, seed=31), seed=31)


@pytest.fixture(scope="module")
def fitted(data):
    X_tr, _, y_tr, _ = data
    pipeline = candidates(31)["gradient_boosting"]
    pipeline.fit(X_tr, y_tr)
    return pipeline


@pytest.fixture(scope="module")
def global_importance(fitted, data):
    _, X_te, _, y_te = data
    return importance(fitted, X_te, y_te, n_repeats=4, seed=31)


# ------------------------------------------------------------------- global


def test_importance_covers_every_feature(global_importance, data):
    _, X_te, _, _ = data
    assert len(global_importance) == X_te.shape[1]


def test_importance_is_sorted(global_importance):
    assert global_importance["importance"].is_monotonic_decreasing


def test_contract_is_a_top_driver(global_importance):
    """It carries the largest coefficient in the generator, so it must surface."""
    assert "contract" in set(global_importance.head(4)["feature"])


def test_actionable_flag_matches_the_registry(global_importance):
    for _, row in global_importance.iterrows():
        assert row["actionable"] == (row["feature"] in ACTIONABLE)


def test_tenure_is_important_but_not_actionable(global_importance):
    tenure = global_importance[global_importance["feature"] == "tenure_months"].iloc[0]
    assert tenure["importance"] > 0
    assert not tenure["actionable"]
    assert "tenure_months" in IMMUTABLE


def test_actionable_drivers_filters_and_labels(global_importance):
    drivers = actionable_drivers(global_importance)
    assert len(drivers) < len(global_importance)
    assert drivers["actionable"].all()
    assert drivers["lever"].notna().all()


def test_actionable_drivers_keeps_the_ranking(global_importance):
    assert actionable_drivers(global_importance)["importance"].is_monotonic_decreasing


# -------------------------------------------------------------- per customer


def test_reasons_are_returned_for_a_risky_customer(fitted, data):
    X_tr, X_te, _, _ = data
    risky = X_te.loc[fitted.predict_proba(X_te)[:, 1].argmax()]
    reasons = reasons_for(fitted, X_te.iloc[[fitted.predict_proba(X_te)[:, 1].argmax()]], X_tr)
    assert isinstance(reasons, list)
    for reason in reasons:
        assert {"feature", "current", "suggested", "risk_reduction", "lever"} <= set(reason)


def test_reasons_are_capped(fitted, data):
    X_tr, X_te, _, _ = data
    assert len(reasons_for(fitted, X_te.iloc[[0]], X_tr, top=2)) <= 2


def test_reasons_only_cite_actionable_features(fitted, data):
    X_tr, X_te, _, _ = data
    for i in range(6):
        for reason in reasons_for(fitted, X_te.iloc[[i]], X_tr):
            assert reason["feature"] in ACTIONABLE


def test_reasons_are_ordered_by_impact(fitted, data):
    X_tr, X_te, _, _ = data
    scores = fitted.predict_proba(X_te)[:, 1]
    reasons = reasons_for(fitted, X_te.iloc[[int(scores.argmax())]], X_tr)
    reductions = [r["risk_reduction"] for r in reasons]
    assert reductions == sorted(reductions, reverse=True)


def test_risk_reductions_are_positive(fitted, data):
    X_tr, X_te, _, _ = data
    scores = fitted.predict_proba(X_te)[:, 1]
    for reason in reasons_for(fitted, X_te.iloc[[int(scores.argmax())]], X_tr):
        assert reason["risk_reduction"] > 0


def test_a_low_risk_customer_may_have_no_levers(fitted, data):
    X_tr, X_te, _, _ = data
    scores = fitted.predict_proba(X_te)[:, 1]
    safest = X_te.iloc[[int(scores.argmin())]]
    assert isinstance(reasons_for(fitted, safest, X_tr), list)   # never raises


def test_reasons_accepts_a_series(fitted, data):
    X_tr, X_te, _, _ = data
    assert isinstance(reasons_for(fitted, X_te.iloc[0], X_tr), list)


# --------------------------------------------------------------- risk table


def test_risk_table_has_a_row_per_customer(fitted, data):
    _, X_te, _, _ = data
    assert len(risk_table(fitted, X_te)) == len(X_te)


def test_risk_table_is_sorted_by_risk(fitted, data):
    _, X_te, _, _ = data
    assert risk_table(fitted, X_te)["churn_probability"].is_monotonic_decreasing


def test_risk_bands_are_assigned(fitted, data):
    _, X_te, _, _ = data
    bands = set(risk_table(fitted, X_te)["risk_band"].dropna().unique())
    assert bands <= {"low", "medium", "high", "critical"}


def test_risk_table_can_carry_ids(fitted, data):
    _, X_te, _, _ = data
    ids = pd.Series([f"C{i}" for i in range(len(X_te))], index=X_te.index)
    assert "customer_id" in risk_table(fitted, X_te, ids=ids).columns
