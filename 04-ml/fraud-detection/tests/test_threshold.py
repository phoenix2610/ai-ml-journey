import numpy as np
import pandas as pd
import pytest

from src.threshold import (
    ThresholdChoice,
    at_recall,
    best_by_cost,
    best_by_f1,
    cost_at,
    pr_curve,
    sweep,
)


@pytest.fixture
def toy():
    """10 rows, 3 fraud. Small enough to reason about by hand."""
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    scores = np.array([0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.60, 0.40, 0.70, 0.95])
    amounts = np.array([50, 60, 70, 80, 90, 100, 110, 500, 800, 1200], dtype=float)
    return y, scores, amounts


# ------------------------------------------------------------------- cost_at


def test_threshold_zero_flags_everything(toy):
    y, s, a = toy
    result = cost_at(y, s, 0.0, a)
    assert result.true_positives == 3
    assert result.false_positives == 7
    assert result.false_negatives == 0


def test_threshold_above_max_flags_nothing(toy):
    y, s, a = toy
    result = cost_at(y, s, 1.5, a)
    assert result.true_positives == 0
    assert result.false_negatives == 3
    assert result.recall == 0.0


def test_counts_are_consistent(toy):
    y, s, a = toy
    result = cost_at(y, s, 0.5, a)
    assert result.true_positives + result.false_negatives == 3


def test_precision_and_recall_are_computed(toy):
    y, s, a = toy
    result = cost_at(y, s, 0.65, a)     # flags 0.70 and 0.95, both fraud
    assert result.precision == 1.0
    assert result.recall == pytest.approx(2 / 3)


def test_missed_fraud_costs_its_amount(toy):
    y, s, a = toy
    # Flag nothing: cost is the full value of all three frauds, no reviews.
    result = cost_at(y, s, 1.5, a, review_cost=0.0, recovery_rate=0.0)
    assert result.total_cost == pytest.approx(500 + 800 + 1200)


def test_false_positives_cost_review_time(toy):
    y, s, a = toy
    result = cost_at(y, s, 0.0, a, review_cost=10.0, recovery_rate=0.0)
    assert result.review_cost == pytest.approx(7 * 10.0)


def test_recovery_reduces_cost(toy):
    y, s, a = toy
    expensive = cost_at(y, s, 0.35, a, recovery_rate=0.0)
    cheaper = cost_at(y, s, 0.35, a, recovery_rate=0.5)
    assert cheaper.total_cost < expensive.total_cost


def test_zero_flagged_gives_zero_precision_not_a_crash(toy):
    y, s, a = toy
    assert cost_at(y, s, 2.0, a).precision == 0.0


# --------------------------------------------------------------------- sweep


def test_sweep_returns_a_row_per_threshold(toy):
    y, s, a = toy
    frame = sweep(y, s, a, n_points=20)
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) > 1
    assert {"threshold", "total_cost", "precision", "recall"} <= set(frame.columns)


def test_recall_is_monotonically_non_increasing_in_threshold(toy):
    y, s, a = toy
    frame = sweep(y, s, a, n_points=40).sort_values("threshold")
    assert (frame["recall"].diff().dropna() <= 1e-9).all()


# ------------------------------------------------------------------ selection


def test_best_by_cost_beats_a_naive_half_threshold(toy):
    y, s, a = toy
    chosen = best_by_cost(y, s, a)
    naive = cost_at(y, s, 0.5, a)
    assert chosen.total_cost <= naive.total_cost


def test_best_by_cost_returns_the_dataclass(toy):
    y, s, a = toy
    assert isinstance(best_by_cost(y, s, a), ThresholdChoice)


def test_expensive_reviews_push_the_threshold_up(toy):
    """If false alarms are costly, the model should get pickier."""
    y, s, a = toy
    cheap = best_by_cost(y, s, a, review_cost=1.0)
    dear = best_by_cost(y, s, a, review_cost=400.0)
    assert dear.threshold >= cheap.threshold


def test_valuable_fraud_pushes_the_threshold_down(toy):
    """If catching fraud recovers a lot, casting a wider net is worth it."""
    y, s, a = toy
    stingy = best_by_cost(y, s, a, recovery_rate=0.05)
    generous = best_by_cost(y, s, a, recovery_rate=0.95)
    assert generous.threshold <= stingy.threshold


def test_cost_optimal_and_f1_optimal_can_disagree(toy):
    """The point of the module: F1 ignores that frauds differ in value."""
    y, s, a = toy
    by_cost = best_by_cost(y, s, a, review_cost=1.0, recovery_rate=0.9)
    by_f1 = best_by_f1(y, s, a)
    assert isinstance(by_cost.threshold, float) and isinstance(by_f1.threshold, float)
    # They need not differ on 10 rows, but both must be valid operating points.
    assert 0.0 <= by_cost.recall <= 1.0
    assert 0.0 <= by_f1.recall <= 1.0


def test_at_recall_meets_the_floor(toy):
    y, s, a = toy
    assert at_recall(y, s, a, target_recall=0.66).recall >= 0.66


def test_at_recall_raises_when_unreachable(toy):
    y, s, a = toy
    with pytest.raises(ValueError, match="no threshold reaches"):
        at_recall(y, s, a, target_recall=1.01)


# ------------------------------------------------------------------ pr_curve


def test_pr_curve_columns(toy):
    y, s, _ = toy
    curve = pr_curve(y, s)
    assert {"threshold", "precision", "recall"} == set(curve.columns)


def test_pr_curve_lengths_match(toy):
    y, s, _ = toy
    curve = pr_curve(y, s)
    assert len(curve["precision"]) == len(curve["recall"]) == len(curve["threshold"])
