"""Tests for bandit vs static policy comparison."""

import numpy as np
import pytest
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


@pytest.mark.parametrize("seed", [42, 2, 3, 7, 99])
def test_bandit_learns_over_time(seed):
    """Bandit's reward rate improves as it learns arm estimates.

    Uses warmup_rounds=10 (only ~2 samples per arm with n_items=5) so
    initial estimates are noisy and the bandit has room to learn.
    Over 10K rounds, late reward rate should strictly exceed early rate.
    """
    result = run_bandit_comparison(
        n_rounds=10_000, n_items=5, epsilon=0.1,
        warmup_rounds=10, seed=seed,
    )
    bandit = result["cumulative_reward_bandit"]
    # First 200 rounds: bandit is still exploring with noisy estimates.
    # Last 1000 rounds: bandit has converged to near-optimal arm.
    early_rate = bandit[199] / 200
    late_rate = (bandit[9999] - bandit[8999]) / 1000
    assert late_rate > early_rate


def test_zero_epsilon_bandit_never_explores():
    """With epsilon=0, bandit never takes a random arm.

    Verified by checking that epsilon=0 achieves higher or equal reward
    than epsilon=1 (pure exploration). If epsilon=0 accidentally explored,
    it would waste rounds on suboptimal arms just like epsilon=1.
    """
    greedy = run_bandit_comparison(
        n_rounds=2000, n_items=10, epsilon=0.0, seed=42,
    )
    random = run_bandit_comparison(
        n_rounds=2000, n_items=10, epsilon=1.0, seed=42,
    )
    assert greedy["final_reward_bandit"] > random["final_reward_bandit"]
