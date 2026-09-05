import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.data import make_subscribers, split
from src.model import (
    build_preprocessor,
    calibrate,
    calibration_report,
    candidates,
    compare,
    make_pipeline,
    max_calibration_gap,
)


@pytest.fixture(scope="module")
def data():
    return split(make_subscribers(6_000, seed=21), seed=21)


@pytest.fixture(scope="module")
def fitted(data):
    X_tr, _, y_tr, _ = data
    pipeline = candidates(21)["gradient_boosting"]
    pipeline.fit(X_tr, y_tr)
    return pipeline


# ------------------------------------------------------------- preprocessing


def test_preprocessor_removes_missing_values(data):
    X_tr, _, _, _ = data
    assert not np.isnan(build_preprocessor().fit_transform(X_tr)).any()


def test_preprocessor_expands_categoricals(data):
    X_tr, _, _, _ = data
    assert build_preprocessor().fit_transform(X_tr).shape[1] > X_tr.shape[1]


def test_unseen_category_is_handled(data):
    X_tr, X_te, _, _ = data
    pre = build_preprocessor().fit(X_tr)
    novel = X_te.head(3).copy()
    novel["contract"] = "lifetime"
    assert pre.transform(novel).shape[0] == 3


# -------------------------------------------------------------------- models


def test_model_beats_the_baseline(data):
    X_tr, _, y_tr, _ = data
    results = {r.name: r for r in compare(X_tr, y_tr, folds=3, seed=21)}
    assert results["gradient_boosting"].roc_auc > results["baseline_majority"].roc_auc + 0.10


def test_baseline_roc_auc_is_chance(data):
    X_tr, _, y_tr, _ = data
    results = {r.name: r for r in compare(X_tr, y_tr, folds=3, seed=21)}
    assert results["baseline_majority"].roc_auc == pytest.approx(0.5, abs=0.02)


def test_compare_is_sorted_by_brier(data):
    X_tr, _, y_tr, _ = data
    briers = [r.brier for r in compare(X_tr, y_tr, folds=3, seed=21)]
    assert briers == sorted(briers)


def test_predictions_are_probabilities(fitted, data):
    _, X_te, _, _ = data
    proba = fitted.predict_proba(X_te)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_model_has_real_signal(fitted, data):
    _, X_te, _, y_te = data
    assert roc_auc_score(y_te, fitted.predict_proba(X_te)[:, 1]) > 0.70


# --------------------------------------------------------------- calibration


def test_calibration_report_shape(fitted, data):
    _, X_te, _, y_te = data
    report = calibration_report(y_te, fitted.predict_proba(X_te)[:, 1], bins=8)
    assert set(report.columns) == {"predicted", "observed", "gap"}
    assert len(report) <= 8


def test_gap_is_observed_minus_predicted(fitted, data):
    _, X_te, _, y_te = data
    report = calibration_report(y_te, fitted.predict_proba(X_te)[:, 1])
    assert np.allclose(report["gap"], report["observed"] - report["predicted"])


def test_calibrated_model_still_predicts_probabilities(data):
    X_tr, X_te, y_tr, _ = data
    calibrated = calibrate(candidates(21)["gradient_boosting"], X_tr, y_tr, folds=3)
    proba = calibrated.predict_proba(X_te)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_calibration_preserves_ranking_quality(data):
    """Calibration must not destroy discrimination -- it remaps, monotonically."""
    X_tr, X_te, y_tr, y_te = data
    raw = candidates(21)["gradient_boosting"]
    raw.fit(X_tr, y_tr)
    calibrated = calibrate(candidates(21)["gradient_boosting"], X_tr, y_tr, folds=3)

    raw_auc = roc_auc_score(y_te, raw.predict_proba(X_te)[:, 1])
    cal_auc = roc_auc_score(y_te, calibrated.predict_proba(X_te)[:, 1])
    assert cal_auc > raw_auc - 0.05


def test_calibration_does_not_make_the_gap_worse(data):
    X_tr, X_te, y_tr, y_te = data
    raw = candidates(21)["gradient_boosting"]
    raw.fit(X_tr, y_tr)
    calibrated = calibrate(candidates(21)["gradient_boosting"], X_tr, y_tr, folds=3)

    raw_gap = max_calibration_gap(y_te, raw.predict_proba(X_te)[:, 1], bins=8)
    cal_gap = max_calibration_gap(y_te, calibrated.predict_proba(X_te)[:, 1], bins=8)
    assert cal_gap <= raw_gap + 0.05


def test_a_perfectly_calibrated_model_has_no_gap():
    rng = np.random.default_rng(0)
    truth = rng.random(4000)
    outcomes = (rng.random(4000) < truth).astype(int)
    assert max_calibration_gap(outcomes, truth, bins=10) < 0.06


def test_a_badly_calibrated_model_is_detected():
    """Halving every probability must show up as a large gap."""
    rng = np.random.default_rng(1)
    truth = rng.random(4000)
    outcomes = (rng.random(4000) < truth).astype(int)
    assert max_calibration_gap(outcomes, truth * 0.5, bins=10) > 0.15


# --------------------------------------------------------------------- leakage


def test_explicit_columns_block_a_stray_feature(data):
    """remainder='drop' means an unlisted column cannot silently become a feature."""
    X_tr, _, _, _ = data
    contaminated = X_tr.copy()
    contaminated["leaked_nonsense"] = 1.0
    baseline = build_preprocessor().fit_transform(X_tr).shape[1]
    assert build_preprocessor().fit_transform(contaminated).shape[1] == baseline


def test_the_leaky_column_inflates_the_score():
    """Why exit_survey_sent is dropped: it is only knowable after churn."""
    from sklearn.metrics import roc_auc_score

    from src.data import LEAKY, make_subscribers, split

    frame = make_subscribers(4_000, seed=17)

    X_tr, X_te, y_tr, y_te = split(frame, seed=17)
    honest = candidates(17)["gradient_boosting"]
    honest.fit(X_tr, y_tr)
    honest_auc = roc_auc_score(y_te, honest.predict_proba(X_te)[:, 1])

    Xl_tr, Xl_te, yl_tr, yl_te = split(frame, seed=17, drop_leaky=False)
    leaked = candidates(17, extra_numeric=tuple(LEAKY))["gradient_boosting"]
    leaked.fit(Xl_tr, yl_tr)
    leaked_auc = roc_auc_score(yl_te, leaked.predict_proba(Xl_te)[:, 1])

    assert leaked_auc > honest_auc + 0.10
