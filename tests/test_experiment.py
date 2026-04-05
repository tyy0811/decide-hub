"""Tests for experiment engine — bootstrap CIs and treatment effects."""

import numpy as np
import pytest

from src.evaluation.experiment import (
    run_experiment,
    bootstrap_ci,
    minimum_detectable_effect,
)


def test_bootstrap_ci_contains_true_mean():
    """CI from a normal sample should contain the true mean."""
    rng = np.random.default_rng(42)
    data = rng.normal(loc=5.0, scale=1.0, size=1000)
    ci_lower, ci_upper = bootstrap_ci(data, n_resamples=5000, seed=42)
    assert ci_lower < 5.0 < ci_upper


def test_bootstrap_ci_wider_with_fewer_samples():
    """Smaller sample → wider CI."""
    rng = np.random.default_rng(42)
    small = rng.normal(loc=0, scale=1, size=20)
    large = rng.normal(loc=0, scale=1, size=2000)
    ci_small = bootstrap_ci(small, seed=42)
    ci_large = bootstrap_ci(large, seed=42)
    width_small = ci_small[1] - ci_small[0]
    width_large = ci_large[1] - ci_large[0]
    assert width_small > width_large


def test_bootstrap_ci_coverage():
    """95% CI covers the true effect ~95% of the time (±4pp tolerance).

    Generate data with known true effect (+0.05), run 200 experiments,
    check that the CI contains the true effect ~95% of the time.
    """
    true_effect = 0.05
    covered = 0
    n_trials = 200

    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        control = rng.normal(loc=0.5, scale=0.3, size=500)
        treatment = rng.normal(loc=0.5 + true_effect, scale=0.3, size=500)
        result = run_experiment(control, treatment, n_resamples=2000, seed=seed)
        if result["ci_lower"] <= true_effect <= result["ci_upper"]:
            covered += 1

    coverage = covered / n_trials
    assert 0.91 <= coverage <= 0.99, f"Coverage {coverage:.2%} outside [91%, 99%]"


def test_run_experiment_structure():
    """Experiment result has all expected fields."""
    rng = np.random.default_rng(42)
    control = rng.normal(loc=0.5, scale=0.1, size=100)
    treatment = rng.normal(loc=0.55, scale=0.1, size=100)
    result = run_experiment(control, treatment, seed=42)

    assert "baseline_mean" in result
    assert "treatment_mean" in result
    assert "lift" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert "sample_size_control" in result
    assert "sample_size_treatment" in result
    assert result["confidence"] == 0.95


def test_run_experiment_detects_positive_effect():
    """Clear positive effect: CI should be above zero."""
    rng = np.random.default_rng(42)
    control = rng.normal(loc=0.5, scale=0.1, size=500)
    treatment = rng.normal(loc=0.7, scale=0.1, size=500)
    result = run_experiment(control, treatment, seed=42)
    assert result["ci_lower"] > 0


def test_run_experiment_null_effect():
    """No effect: CI should contain zero."""
    rng = np.random.default_rng(42)
    control = rng.normal(loc=0.5, scale=0.1, size=500)
    treatment = rng.normal(loc=0.5, scale=0.1, size=500)
    result = run_experiment(control, treatment, seed=42)
    assert result["ci_lower"] < 0 < result["ci_upper"]


def test_run_experiment_with_segments():
    """Segment-wise breakdown produces per-segment results."""
    rng = np.random.default_rng(42)
    control = rng.normal(loc=0.5, scale=0.1, size=200)
    treatment = rng.normal(loc=0.6, scale=0.1, size=200)
    segments_control = ["A"] * 100 + ["B"] * 100
    segments_treatment = ["A"] * 100 + ["B"] * 100

    result = run_experiment(
        control, treatment,
        segments_control=segments_control,
        segments_treatment=segments_treatment,
        seed=42,
    )

    assert "segments" in result
    assert "A" in result["segments"]
    assert "B" in result["segments"]
    assert "lift" in result["segments"]["A"]


def test_bootstrap_ci_empty_raises():
    """Empty data raises ValueError, not silent nan."""
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci(np.array([]))


def test_run_experiment_empty_control_raises():
    """Empty control group raises ValueError."""
    with pytest.raises(ValueError, match="Control group is empty"):
        run_experiment(np.array([]), np.array([1.0, 2.0]))


def test_run_experiment_empty_treatment_raises():
    """Empty treatment group raises ValueError."""
    with pytest.raises(ValueError, match="Treatment group is empty"):
        run_experiment(np.array([1.0, 2.0]), np.array([]))


def test_run_experiment_segment_length_mismatch_raises():
    """Segment labels must match data length."""
    control = np.array([1.0, 2.0, 3.0])
    treatment = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="segments_control length"):
        run_experiment(
            control, treatment,
            segments_control=["A"],  # wrong length
            segments_treatment=["A", "B", "A"],
        )


def test_mde_decreases_with_larger_sample():
    """Larger sample → smaller minimum detectable effect."""
    mde_small = minimum_detectable_effect(n=100, baseline_std=0.3)
    mde_large = minimum_detectable_effect(n=10000, baseline_std=0.3)
    assert mde_large < mde_small


def test_mde_increases_with_higher_variance():
    """Higher variance → larger MDE."""
    mde_low = minimum_detectable_effect(n=1000, baseline_std=0.1)
    mde_high = minimum_detectable_effect(n=1000, baseline_std=1.0)
    assert mde_high > mde_low
