"""Tests for online simulation environment."""

import numpy as np
import pytest

from src.evaluation.online_sim import OnlineEnvironment, run_simulation


def test_environment_produces_contexts():
    env = OnlineEnvironment(n_items=5, n_features=3, seed=42)
    ctx = env.get_context()
    assert ctx.shape == (3,)


def test_environment_step_returns_reward():
    env = OnlineEnvironment(n_items=5, n_features=3, seed=42)
    env.get_context()
    reward = env.step(0)
    assert 0.0 <= reward <= 1.0


def test_environment_same_seed_same_sequence():
    """Same seed produces identical context sequences."""
    env1 = OnlineEnvironment(n_items=5, n_features=3, seed=42)
    env2 = OnlineEnvironment(n_items=5, n_features=3, seed=42)
    for _ in range(10):
        np.testing.assert_array_equal(env1.get_context(), env2.get_context())
        assert env1.step(0) == env2.step(0)


def test_optimal_reward_exists():
    """Environment knows the optimal action per context."""
    env = OnlineEnvironment(n_items=5, n_features=3, seed=42)
    env.get_context()
    optimal = env.optimal_reward()
    assert 0.0 <= optimal <= 1.0


def test_run_simulation_structure():
    """Simulation returns per-policy regret curves."""
    def random_policy(ctx, n_items, rng):
        return rng.integers(n_items)

    def greedy_policy(ctx, n_items, rng):
        return 0  # always pick first item

    result = run_simulation(
        policies={"random": random_policy, "greedy": greedy_policy},
        n_rounds=100,
        n_items=5,
        seed=42,
    )

    assert "random" in result
    assert "greedy" in result
    assert len(result["random"]["cumulative_regret"]) == 100
    assert len(result["greedy"]["cumulative_regret"]) == 100


def test_random_policy_has_positive_final_regret():
    """Random policy accumulates positive regret over many rounds."""
    def random_policy(ctx, n_items, rng):
        return rng.integers(n_items)

    result = run_simulation(
        policies={"random": random_policy},
        n_rounds=500,
        n_items=5,
        seed=42,
    )

    # Final cumulative regret should be positive for a random policy
    assert result["random"]["final_regret"] > 0
