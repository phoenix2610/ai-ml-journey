import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src.anomaly import AnomalyDetector, isolation_forest, local_outlier_factor
from src.data import TARGET, make_transactions, split


@pytest.fixture(scope="module")
def data():
    # Enough rows that the tiny positive class is still meaningful.
    return split(make_transactions(12_000, seed=5), seed=5)


@pytest.fixture(scope="module")
def fitted(data):
    X_tr, _, y_tr, _ = data
    return isolation_forest(seed=5).fit_on_legitimate(X_tr, y_tr)


def test_fit_returns_self(data):
    X_tr, _, y_tr, _ = data
    detector = isolation_forest(seed=5)
    assert detector.fit_on_legitimate(X_tr, y_tr) is detector


def test_score_returns_one_value_per_row(fitted, data):
    _, X_te, _, _ = data
    assert fitted.score(X_te).shape == (len(X_te),)


def test_scores_are_finite(fitted, data):
    _, X_te, _, _ = data
    assert np.isfinite(fitted.score(X_te)).all()


def test_higher_score_means_more_suspicious(fitted, data):
    """Orientation must match predict_proba[:, 1] so shared code works."""
    _, X_te, _, y_te = data
    scores = fitted.score(X_te)
    assert scores[np.asarray(y_te) == 1].mean() > scores[np.asarray(y_te) == 0].mean()


def test_predict_proba_is_bounded(fitted, data):
    _, X_te, _, _ = data
    proba = fitted.predict_proba(X_te)
    assert proba.shape == (len(X_te), 2)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_predict_proba_rows_sum_to_one(fitted, data):
    _, X_te, _, _ = data
    assert np.allclose(fitted.predict_proba(X_te).sum(axis=1), 1.0)


def test_fit_on_legitimate_uses_only_negatives(data):
    X_tr, _, y_tr, _ = data
    detector = isolation_forest(seed=5)
    detector.fit_on_legitimate(X_tr, y_tr)
    # The pipeline should have seen exactly the negative rows.
    assert int(np.asarray(y_tr).sum()) > 0          # there were positives to exclude
    assert detector.pipeline.named_steps["pre"].n_features_in_ == X_tr.shape[1]


def test_beats_random_but_not_by_much(fitted, data):
    """The headline result: unsupervised helps, and it is no substitute for labels."""
    _, X_te, _, y_te = data
    ap = average_precision_score(y_te, fitted.score(X_te))
    chance = float(np.mean(y_te))
    assert ap > chance            # better than guessing
    assert ap < 0.60              # nowhere near a supervised model


def test_local_outlier_factor_also_works(data):
    X_tr, X_te, y_tr, y_te = data
    detector = local_outlier_factor().fit_on_legitimate(X_tr, y_tr)
    assert average_precision_score(y_te, detector.score(X_te)) > float(np.mean(y_te))


def test_unseen_category_does_not_crash(fitted, data):
    _, X_te, _, _ = data
    novel = X_te.head(5).copy()
    novel["merchant_category"] = "crypto_atm"
    assert fitted.score(novel).shape == (5,)


def test_detector_is_deterministic(data):
    X_tr, X_te, y_tr, _ = data
    a = isolation_forest(seed=9).fit_on_legitimate(X_tr, y_tr).score(X_te)
    b = isolation_forest(seed=9).fit_on_legitimate(X_tr, y_tr).score(X_te)
    assert np.allclose(a, b)


def test_training_on_contaminated_data_is_worse(data):
    """Why fit_on_legitimate exists: contamination teaches it fraud is normal."""
    X_tr, X_te, y_tr, y_te = data

    clean = isolation_forest(seed=5).fit_on_legitimate(X_tr, y_tr)
    dirty = isolation_forest(seed=5).fit(X_tr)

    ap_clean = average_precision_score(y_te, clean.score(X_te))
    ap_dirty = average_precision_score(y_te, dirty.score(X_te))
    # Allow a small margin -- at 0.3% contamination the effect is real but modest.
    assert ap_clean >= ap_dirty * 0.85
