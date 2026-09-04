import numpy as np
import pandas as pd
import pytest

from src.data import (
    CATEGORICAL,
    LEAKY,
    NUMERIC,
    TARGET,
    make_subscribers,
    split,
    tenure_cohort,
)


@pytest.fixture(scope="module")
def subs():
    return make_subscribers(8_000, seed=13)


# --------------------------------------------------------------------- shape


def test_row_count(subs):
    assert len(subs) == 8_000


def test_expected_columns(subs):
    expected = set(NUMERIC + CATEGORICAL + LEAKY + [TARGET, "customer_id"])
    assert set(subs.columns) == expected


def test_target_is_binary(subs):
    assert set(subs[TARGET].unique()) == {0, 1}


def test_churn_rate_is_plausible(subs):
    assert 0.15 < subs[TARGET].mean() < 0.40


def test_generation_is_deterministic():
    pd.testing.assert_frame_equal(make_subscribers(500, 3), make_subscribers(500, 3))


def test_customer_ids_are_unique(subs):
    assert subs["customer_id"].is_unique


def test_some_values_are_missing(subs):
    assert subs["total_charges"].isna().sum() > 0


# ------------------------------------------------- the documented effects hold


def test_tenure_reduces_churn(subs):
    new = subs[subs["tenure_months"] <= 6][TARGET].mean()
    old = subs[subs["tenure_months"] >= 48][TARGET].mean()
    assert new > old * 2


def test_month_to_month_churns_most(subs):
    by_contract = subs.groupby("contract")[TARGET].mean()
    assert by_contract["month_to_month"] > by_contract["one_year"] > by_contract["two_year"]


def test_support_calls_increase_churn(subs):
    quiet = subs[subs["support_calls_90d"] == 0][TARGET].mean()
    noisy = subs[subs["support_calls_90d"] >= 4][TARGET].mean()
    assert noisy > quiet


def test_outages_increase_churn(subs):
    none = subs[subs["outages_90d"] == 0][TARGET].mean()
    several = subs[subs["outages_90d"] >= 2][TARGET].mean()
    assert several > none


def test_dependents_reduce_churn(subs):
    by_dependents = subs.groupby("has_dependents")[TARGET].mean()
    assert by_dependents["yes"] < by_dependents["no"]


# ---------------------------------------------------------- the tenure confound


def test_long_tenure_customers_hold_longer_contracts(subs):
    """The confound: contract type and tenure are entangled by construction."""
    share_m2m = subs.groupby(tenure_cohort(subs["tenure_months"]), observed=True)[
        "contract"
    ].apply(lambda s: (s == "month_to_month").mean())
    assert share_m2m.iloc[0] > share_m2m.iloc[-1]


def test_contract_effect_survives_controlling_for_tenure(subs):
    """The effect is real, not just tenure in disguise -- within one cohort too."""
    cohort = subs[tenure_cohort(subs["tenure_months"]) == "0-6m"]
    by_contract = cohort.groupby("contract")[TARGET].mean()
    assert by_contract["month_to_month"] > by_contract["two_year"]


def test_tenure_cohort_labels(subs):
    cohorts = tenure_cohort(subs["tenure_months"])
    assert set(cohorts.dropna().unique()) <= {"0-6m", "6-12m", "1-2y", "2-4y", "4y+"}


def test_tenure_cohort_includes_the_lowest_value():
    assert tenure_cohort(pd.Series([1])).iloc[0] == "0-6m"


# -------------------------------------------------------------------- the leak


def test_leaky_column_almost_perfectly_predicts_churn(subs):
    """Which is exactly why it must never be a feature."""
    agreement = (subs["exit_survey_sent"] == subs[TARGET]).mean()
    assert agreement > 0.93


def test_split_drops_the_leak_by_default(subs):
    X_tr, X_te, _, _ = split(subs)
    for column in LEAKY:
        assert column not in X_tr.columns
        assert column not in X_te.columns


def test_split_drops_the_identifier(subs):
    X_tr, _, _, _ = split(subs)
    assert "customer_id" not in X_tr.columns


def test_leak_can_be_kept_for_demonstration(subs):
    X_tr, _, _, _ = split(subs, drop_leaky=False)
    assert "exit_survey_sent" in X_tr.columns


# -------------------------------------------------------------------- split


def test_split_preserves_the_churn_rate(subs):
    _, _, y_tr, y_te = split(subs, seed=2)
    assert y_te.mean() == pytest.approx(y_tr.mean(), rel=0.06)


def test_split_sizes(subs):
    X_tr, X_te, _, _ = split(subs, test_size=0.25)
    assert len(X_te) == pytest.approx(len(subs) * 0.25, abs=2)


def test_split_is_reproducible(subs):
    a = split(subs, seed=8)[0]
    b = split(subs, seed=8)[0]
    pd.testing.assert_frame_equal(a, b)
