"""Tests for Doubly Robust off-policy estimator."""

import numpy as np
import pytest

from src.evaluation.doubly_robust import dr_estimate
from src.evaluation.counterfactual import ips_estimate


def test_dr_matches_ips_with_uninformative_model():
    """When reward model is constant, DR reduces to IPS."""
    rng = np.random.default_rng(42)
    rewards = rng.binomial(1, 0.5, size=1000).astype(float).tolist()
    propensities = [0.5] * 1000
    target_probs = [0.3] * 1000
    # Zero model: DR reduces to IPS when model_pred = 0
    model_preds = [0.0] * 1000

    dr_val = dr_estimate(rewards, propensities, target_probs, model_preds)
    ips_val = ips_estimate(rewards, propensities, target_probs)
    assert abs(dr_val - ips_val) < 0.05


def test_dr_lower_variance_than_ips():
    """DR should have lower variance across multiple seeds."""
    dr_vals = []
    ips_vals = []

    for seed in range(100):
        rng = np.random.default_rng(seed)
        n = 500
        rewards = rng.binomial(1, 0.6, size=n).astype(float).tolist()
        propensities = (rng.uniform(0.2, 0.8, size=n)).tolist()
        target_probs = (rng.uniform(0.1, 0.9, size=n)).tolist()
        # Reasonable model: predict 0.6 (the true mean) + noise
        model_preds = (0.6 + rng.normal(0, 0.1, size=n)).clip(0, 1).tolist()

        dr_vals.append(dr_estimate(rewards, propensities, target_probs, model_preds))
        ips_vals.append(ips_estimate(rewards, propensities, target_probs))

    assert np.std(dr_vals) < np.std(ips_vals)


def test_dr_correct_with_wrong_propensities_right_model():
    """Doubly robust: correct when propensities are wrong but model is right."""
    rng = np.random.default_rng(42)
    n = 5000
    true_reward_prob = 0.7
    rewards = rng.binomial(1, true_reward_prob, size=n).astype(float).tolist()
    # Wrong propensities (uniform instead of true)
    wrong_propensities = [0.5] * n
    target_probs = [0.5] * n
    # Perfect model knows true reward probability
    perfect_model = [true_reward_prob] * n

    dr_val = dr_estimate(rewards, wrong_propensities, target_probs, perfect_model)
    # Should be close to true_reward_prob
    assert abs(dr_val - true_reward_prob) < 0.05


def test_dr_correct_with_right_propensities_wrong_model():
    """Doubly robust: correct when model is wrong but propensities are right."""
    rng = np.random.default_rng(42)
    n = 5000
    true_reward_prob = 0.7
    rewards = rng.binomial(1, true_reward_prob, size=n).astype(float).tolist()
    # Correct propensities
    correct_propensities = [0.5] * n
    target_probs = [0.5] * n
    # Wrong model (predicts 0.3 instead of 0.7)
    wrong_model = [0.3] * n

    dr_val = dr_estimate(rewards, correct_propensities, target_probs, wrong_model)
    # When propensities are correct, DR corrects for model bias via IPS term
    # Should still be reasonably close to true value
    assert abs(dr_val - true_reward_prob) < 0.1


def test_dr_empty_returns_zero():
    assert dr_estimate([], [], [], []) == 0.0


def test_dr_zero_propensity_raises():
    with pytest.raises(ValueError, match="Propensity must be > 0"):
        dr_estimate([1.0], [0.0], [0.5], [0.5])
