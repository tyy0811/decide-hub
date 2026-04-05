"""Tests for action distribution comparison utilities."""

from src.evaluation.comparison import compute_action_deltas, total_variation_distance


def test_identical_distributions_zero_tvd():
    """Identical distributions produce zero TVD and zero deltas."""
    dist = {"priority_outreach": 10, "standard_sequence": 20}
    assert total_variation_distance(dist, dist) == 0.0
    deltas = compute_action_deltas(dist, dist)
    assert all(v == 0.0 for v in deltas.values())


def test_completely_different_distributions():
    """Disjoint distributions produce TVD = 1.0."""
    baseline = {"a": 10, "b": 0}
    candidate = {"a": 0, "b": 10}
    assert total_variation_distance(baseline, candidate) == 1.0


def test_partial_shift():
    """Known partial shift produces expected TVD."""
    # Baseline: 50% a, 50% b. Candidate: 70% a, 30% b. Shift = 20% each side.
    baseline = {"a": 50, "b": 50}
    candidate = {"a": 70, "b": 30}
    tvd = total_variation_distance(baseline, candidate)
    assert abs(tvd - 0.2) < 1e-9


def test_action_deltas_signs():
    """Positive delta = candidate does this action more."""
    baseline = {"a": 30, "b": 70}
    candidate = {"a": 60, "b": 40}
    deltas = compute_action_deltas(baseline, candidate)
    assert deltas["a"] > 0  # candidate does more a
    assert deltas["b"] < 0  # candidate does less b


def test_missing_actions_handled():
    """Actions present in one distribution but not the other are handled."""
    baseline = {"a": 10, "b": 5}
    candidate = {"a": 10, "c": 5}
    deltas = compute_action_deltas(baseline, candidate)
    assert "b" in deltas and deltas["b"] < 0  # b disappeared
    assert "c" in deltas and deltas["c"] > 0  # c appeared
    tvd = total_variation_distance(baseline, candidate)
    assert tvd > 0


def test_empty_distributions():
    """Empty distributions produce zero TVD."""
    assert total_variation_distance({}, {}) == 0.0
    assert compute_action_deltas({}, {}) == {}
