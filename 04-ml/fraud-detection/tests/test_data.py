import numpy as np
import pandas as pd
import pytest

from src.data import (
    CATEGORICAL,
    FRAUD_RATE,
    NUMERIC,
    TARGET,
    imbalance_summary,
    make_transactions,
    split,
)


@pytest.fixture(scope="module")
def txns():
    return make_transactions(40_000, seed=7)


# --------------------------------------------------------------------- shape


def test_row_count(txns):
    assert len(txns) == 40_000


def test_all_columns_present(txns):
    # 'archetype' is metadata explaining *why* a row is fraud. It is deliberately
    # not a feature -- src.data.split drops it, and test_archetype_is_not_a_feature
    # pins that.
    assert set(txns.columns) == set(NUMERIC + CATEGORICAL + [TARGET, "archetype"])


def test_archetype_is_not_a_feature(txns):
    """It encodes the label directly; leaving it in X would be a perfect leak."""
    X_tr, X_te, _, _ = split(txns)
    assert "archetype" not in X_tr.columns
    assert "archetype" not in X_te.columns


def test_stealth_fraud_is_a_large_share(txns):
    """The irreducible part: fraud with no tells at all on these features."""
    share = txns[txns[TARGET] == 1]["archetype"].eq("stealth").mean()
    assert 0.20 < share < 0.45


def test_target_is_binary(txns):
    assert set(txns[TARGET].unique()) == {0, 1}


def test_fraud_rate_is_realistic(txns):
    assert txns[TARGET].mean() == pytest.approx(FRAUD_RATE, rel=0.02)


def test_generation_is_deterministic():
    pd.testing.assert_frame_equal(make_transactions(5_000, 3), make_transactions(5_000, 3))


def test_tiny_dataset_still_has_fraud():
    """Guard the rounding path -- 200 * 0.003 rounds to 1, and 0 would break stratify."""
    assert make_transactions(200, seed=1)[TARGET].sum() >= 2


# ---------------------------------------------------- fraud must be learnable


def test_fraud_amounts_skew_higher(txns):
    fraud = txns[txns[TARGET] == 1]["amount"]
    legit = txns[txns[TARGET] == 0]["amount"]
    assert fraud.median() > legit.median()


def test_fraud_happens_at_odd_hours(txns):
    fraud = txns[txns[TARGET] == 1]
    legit = txns[txns[TARGET] == 0]
    night = lambda f: ((f["hour"] < 6) | (f["hour"] >= 23)).mean()
    assert night(fraud) > night(legit) * 2


def test_fraud_favours_new_devices(txns):
    # Elevated, not overwhelming: only the account_takeover archetype sets this
    # tell, and only ~78% of the time, so most fraud is on a known device.
    fraud = txns[txns[TARGET] == 1]["device"].eq("new_device").mean()
    legit = txns[txns[TARGET] == 0]["device"].eq("new_device").mean()
    assert fraud > legit * 2.5
    assert fraud < 0.5


def test_fraud_is_further_from_home(txns):
    fraud = txns[txns[TARGET] == 1]["distance_from_home_km"].median()
    legit = txns[txns[TARGET] == 0]["distance_from_home_km"].median()
    assert fraud > legit


def test_fraud_comes_in_bursts(txns):
    assert (
        txns[txns[TARGET] == 1]["txn_count_24h"].mean()
        > txns[txns[TARGET] == 0]["txn_count_24h"].mean()
    )


def test_amount_vs_avg_is_the_ratio(txns):
    row = txns.iloc[0]
    assert row["amount_vs_avg"] == pytest.approx(row["amount"] / max(row["avg_amount_30d"], 1.0))


def test_classes_overlap(txns):
    """If a single feature separated the classes cleanly the task would be fake."""
    threshold = txns[txns[TARGET] == 1]["amount"].median()
    legit_above = (txns[txns[TARGET] == 0]["amount"] > threshold).mean()
    assert legit_above > 0.02


def test_card_testing_micro_charges_exist(txns):
    """The bimodal fraud pattern: tiny probe charges as well as big cash-outs."""
    fraud = txns[txns[TARGET] == 1]["amount"]
    assert (fraud < 10).mean() > 0.10


# --------------------------------------------------------------------- split


def test_split_preserves_the_fraud_rate(txns):
    _, _, y_tr, y_te = split(txns, test_size=0.25, seed=1)
    assert y_te.mean() == pytest.approx(y_tr.mean(), rel=0.12)


def test_both_sides_contain_fraud(txns):
    _, _, y_tr, y_te = split(txns)
    assert y_tr.sum() > 0 and y_te.sum() > 0


def test_split_sizes(txns):
    X_tr, X_te, _, _ = split(txns, test_size=0.25)
    assert len(X_te) == pytest.approx(len(txns) * 0.25, abs=2)


def test_split_removes_the_target(txns):
    X_tr, _, _, _ = split(txns)
    assert TARGET not in X_tr.columns


def test_split_is_reproducible(txns):
    a = split(txns, seed=4)[0]
    b = split(txns, seed=4)[0]
    pd.testing.assert_frame_equal(a, b)


# ------------------------------------------------------------------ reporting


def test_imbalance_summary_mentions_the_ratio(txns):
    text = imbalance_summary(txns[TARGET])
    assert "fraud" in text and "1 in" in text


def test_accuracy_of_predicting_never_fraud_is_absurdly_high(txns):
    """The number that makes accuracy a useless metric here."""
    assert (1 - txns[TARGET].mean()) > 0.99
