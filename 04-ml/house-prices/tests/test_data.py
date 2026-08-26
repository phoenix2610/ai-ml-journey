import numpy as np
import pandas as pd
import pytest

from src.data import (
    CATEGORICAL,
    NEIGHBOURHOODS,
    NUMERIC,
    TARGET,
    make_houses,
    split,
)


@pytest.fixture(scope="module")
def houses():
    return make_houses(3000, seed=7)


# ------------------------------------------------------------------- shape


def test_row_count(houses):
    assert len(houses) == 3000


def test_all_expected_columns(houses):
    assert set(houses.columns) == set(NUMERIC + CATEGORICAL + [TARGET])


def test_target_is_positive(houses):
    assert (houses[TARGET] > 0).all()


def test_generation_is_deterministic():
    pd.testing.assert_frame_equal(make_houses(500, 3), make_houses(500, 3))


def test_different_seeds_differ():
    assert not make_houses(500, 1)[TARGET].equals(make_houses(500, 2)[TARGET])


# -------------------------------------------------------------- plausibility


def test_bedrooms_are_within_a_sane_range(houses):
    assert houses["bedrooms"].between(1, 7).all()


def test_no_negative_ages(houses):
    assert (houses["age_years"].dropna() >= 0).all()


def test_price_is_right_skewed(houses):
    # Which is exactly why the split stratifies on deciles.
    assert houses[TARGET].mean() > houses[TARGET].median()


def test_missingness_is_injected(houses):
    missing = houses.isna().sum()
    assert missing["age_years"] > 0
    assert missing["lot_sqft"] > 0


def test_target_is_never_missing(houses):
    assert houses[TARGET].notna().all()


# ------------------------------- the model should be able to recover these ---


def test_area_and_price_move_together(houses):
    assert houses["area_sqft"].corr(houses[TARGET]) > 0.6


def test_neighbourhood_ordering_matches_its_multiplier(houses):
    """Median price per neighbourhood must rank the same as the generator's multiplier."""
    actual = houses.groupby("neighbourhood")[TARGET].median().sort_values()
    expected = sorted(NEIGHBOURHOODS, key=NEIGHBOURHOODS.get)
    assert list(actual.index) == expected


def test_condition_is_monotonic_in_price(houses):
    medians = houses.groupby("condition")[TARGET].median()
    assert medians["excellent"] > medians["good"] > medians["fair"] > medians["poor"]


def test_older_houses_are_cheaper_holding_area_roughly_fixed(houses):
    mid = houses[houses["area_sqft"].between(1200, 1800)].dropna(subset=["age_years"])
    young = mid[mid["age_years"] < 15][TARGET].median()
    old = mid[mid["age_years"] > 45][TARGET].median()
    assert young > old


def test_distance_from_centre_reduces_price(houses):
    near = houses[houses["distance_km"] < 5][TARGET].median()
    far = houses[houses["distance_km"] > 20][TARGET].median()
    assert near > far


def test_bigger_houses_sit_in_pricier_neighbourhoods(houses):
    """The confound that makes naive feature interpretation wrong.

    Area is correlated with neighbourhood prestige, so a raw 'big vs small'
    price comparison measures both effects at once. Any conclusion about area
    has to hold prestige fixed -- see the next test.
    """
    median_area = houses.groupby("neighbourhood")["area_sqft"].median()
    ranked_by_prestige = sorted(NEIGHBOURHOODS, key=NEIGHBOURHOODS.get)
    assert list(median_area.sort_values().index) == ranked_by_prestige


def test_area_effect_is_sublinear_once_prestige_is_held_fixed(houses):
    """Price per sqft falls as houses get bigger -- the 0.92 exponent showing.

    Only visible within a single neighbourhood and condition. Across the whole
    dataset the prestige confound swamps it and price looks super-linear in area.
    """
    subset = houses[
        (houses["neighbourhood"] == "Midtown") & (houses["condition"] == "good")
    ].copy()
    subset["price_per_sqft"] = subset[TARGET] / subset["area_sqft"]

    # Correlation rather than binned medians: binning into four groups leaves
    # ~125 rows per band, where noise flips adjacent pairs often enough to make
    # a strict-monotonicity assertion flaky.
    assert subset["area_sqft"].corr(subset["price_per_sqft"]) < -0.10


# ------------------------------------------------------------------- split


def test_split_sizes(houses):
    X_tr, X_te, y_tr, y_te = split(houses, test_size=0.2, seed=1)
    assert len(X_te) == pytest.approx(len(houses) * 0.2, abs=2)
    assert len(X_tr) + len(X_te) == len(houses)


def test_split_drops_the_target_from_features(houses):
    X_tr, _, _, _ = split(houses)
    assert TARGET not in X_tr.columns


def test_split_is_reproducible(houses):
    a = split(houses, seed=5)[0]
    b = split(houses, seed=5)[0]
    pd.testing.assert_frame_equal(a, b)


def test_split_has_no_overlap(houses):
    X_tr, X_te, _, _ = split(houses)
    assert set(X_tr.index).isdisjoint(set(X_te.index))


def test_stratification_matches_the_price_distribution(houses):
    """The point of stratifying: train and test medians should track closely."""
    _, _, y_tr, y_te = split(houses, seed=3)
    assert abs(y_tr.median() - y_te.median()) / y_tr.median() < 0.05


def test_features_and_target_stay_aligned(houses):
    X_tr, _, y_tr, _ = split(houses)
    assert (X_tr.index == y_tr.index).all()
