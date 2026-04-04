"""Tests for bandit vs static policy comparison."""

import numpy as np
from src.evaluation.bandit_comparison import run_bandit_comparison


def test_comparison_returns_expected_keys():
    """Comparison result has all required fields."""
    result = run_bandit_comparison(n_rounds=100, seed=42)
    assert "cumulative_reward_static" in result
    assert "cumulative_reward_bandit" in result
    assert "n_rounds" in result
    assert "final_reward_static" in result
    assert "final_reward_bandit" in result
    assert len(result["cumulative_reward_static"]) == 100
    assert len(result["cumulative_reward_bandit"]) == 100


def test_cumulative_rewards_are_monotonically_nondecreasing():
    """Cumulative reward curves never decrease."""
    result = run_bandit_comparison(n_rounds=500, seed=42)
    for curve in ["cumulative_reward_static", "cumulative_reward_bandit"]:
        values = result[curve]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]


def test_static_policy_reward_rate_is_constant():
    """Static policy picks the same arm every round — reward rate is ~constant."""
    result = run_bandit_comparison(n_rounds=2000, seed=42)
    static = result["cumulative_reward_static"]
    # Reward rate in first half vs second half should be similar
    first_half_rate = static[999] / 1000
    second_half_rate = (static[1999] - static[999]) / 1000
    assert abs(first_half_rate - second_half_rate) < 0.1


def test_bandit_learns_over_time():
    """Bandit's reward rate improves as it learns arm estimates.

    With per-arm bias (linspace -1.5 to +1.5), the best arm has ~82%
    expected reward. The bandit should converge toward it, showing
    strictly higher late reward rate than early rate.
    """
    result = run_bandit_comparison(
        n_rounds=5000, n_items=5, epsilon=0.1, seed=42,
    )
    bandit = result["cumulative_reward_bandit"]
    # Reward rate in last 1000 rounds must strictly exceed first 1000
    early_rate = bandit[999] / 1000
    late_rate = (bandit[4999] - bandit[3999]) / 1000
    assert late_rate > early_rate


def test_zero_epsilon_bandit_is_deterministic():
    """With epsilon=0 and same seed, bandit produces identical results."""
    r1 = run_bandit_comparison(n_rounds=200, epsilon=0.0, seed=99)
    r2 = run_bandit_comparison(n_rounds=200, epsilon=0.0, seed=99)
    assert r1["cumulative_reward_bandit"] == r2["cumulative_reward_bandit"]
    assert r1["cumulative_reward_static"] == r2["cumulative_reward_static"]
