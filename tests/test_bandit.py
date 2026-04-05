"""Tests for epsilon-greedy bandit policy."""

import numpy as np
import pytest

from src.policies.bandit import EpsilonGreedyPolicy


def test_bandit_is_base_policy():
    """EpsilonGreedyPolicy extends BasePolicy."""
    from src.policies.base import BasePolicy
    assert issubclass(EpsilonGreedyPolicy, BasePolicy)


def test_epsilon_cap_enforced():
    """Cannot create bandit with epsilon > max_epsilon."""
    with pytest.raises(ValueError, match="exceeds max_epsilon"):
        EpsilonGreedyPolicy(epsilon=0.5, max_epsilon=0.10)


def test_score_exploit_mode():
    """With epsilon=0, score returns items ranked by arm estimate."""
    bandit = EpsilonGreedyPolicy(epsilon=0.0)
    # Manually set arm estimates
    bandit.arm_rewards = {1: 10.0, 2: 30.0, 3: 20.0}
    bandit.arm_counts = {1: 5, 2: 5, 3: 5}

    scored = bandit.score([1, 2, 3])
    item_ids = [item_id for item_id, _ in scored]
    assert item_ids == [2, 3, 1]  # 6.0, 4.0, 2.0


def test_score_returns_sorted_descending():
    """Score output is always sorted descending by score."""
    bandit = EpsilonGreedyPolicy(epsilon=0.0, seed=42)
    bandit.arm_rewards = {1: 50.0, 2: 10.0, 3: 30.0, 4: 20.0}
    bandit.arm_counts = {1: 10, 2: 10, 3: 10, 4: 10}

    scored = bandit.score([1, 2, 3, 4])
    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True)


def test_score_unknown_items_get_zero():
    """Items not in arm estimates get score 0."""
    bandit = EpsilonGreedyPolicy(epsilon=0.0)
    bandit.arm_rewards = {1: 10.0}
    bandit.arm_counts = {1: 5}

    scored = bandit.score([1, 99])
    scores_dict = dict(scored)
    assert scores_dict[1] == 2.0  # 10/5
    assert scores_dict[99] == 0.0


def test_score_explore_mode_is_random():
    """With epsilon=1, score returns all items (in random order)."""
    bandit = EpsilonGreedyPolicy(epsilon=1.0, max_epsilon=1.0, seed=42)
    bandit.arm_rewards = {1: 100.0, 2: 0.0, 3: 0.0}
    bandit.arm_counts = {1: 1, 2: 1, 3: 1}

    # Run many times — if truly random, item 1 won't always be first
    first_items = []
    for _ in range(50):
        scored = bandit.score([1, 2, 3])
        first_items.append(scored[0][0])

    # With random ordering, we should see variation in first item
    assert len(set(first_items)) > 1


def test_update_changes_estimates():
    """update() modifies arm reward estimates."""
    bandit = EpsilonGreedyPolicy(epsilon=0.0)
    bandit.arm_rewards = {1: 0.3, 2: 0.3}
    bandit.arm_counts = {1: 10, 2: 10}

    # Item 2 starts at 0.03 avg, update with high rewards
    bandit.update(2, 1.0)
    bandit.update(2, 1.0)

    scored = bandit.score([1, 2])
    # Item 2 now: (0.3 + 1.0 + 1.0) / 12 ≈ 0.19, Item 1: 0.3/10 = 0.03
    assert scored[0][0] == 2


def test_update_new_item():
    """update() works for items not seen during fit."""
    bandit = EpsilonGreedyPolicy(epsilon=0.0)
    bandit.update(99, 0.8)
    bandit.update(99, 0.6)

    scored = bandit.score([99])
    assert scored[0] == (99, pytest.approx(0.7))  # (0.8+0.6)/2


def test_score_exploit_is_stable_across_calls():
    """With epsilon=0, repeated score() calls return identical rankings.

    This is the direct no-exploration property: the exploit branch is
    deterministic for fixed arm estimates, regardless of RNG state.
    """
    bandit = EpsilonGreedyPolicy(epsilon=0.0, seed=42)
    bandit.arm_rewards = {1: 0.9, 2: 0.3, 3: 0.6}
    bandit.arm_counts = {1: 3, 2: 3, 3: 3}

    results = [bandit.score([1, 2, 3]) for _ in range(20)]
    assert all(r == results[0] for r in results)


def test_update_rejects_out_of_range_reward():
    """update() raises ValueError for rewards outside [0, 1]."""
    bandit = EpsilonGreedyPolicy(epsilon=0.0)
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        bandit.update(1, 5.0)
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        bandit.update(1, -0.1)


def test_evaluate_does_not_mutate_epsilon():
    """evaluate() must not change self.epsilon (concurrency-safe)."""
    import polars as pl

    bandit = EpsilonGreedyPolicy(epsilon=0.08)
    # Minimal training data for fit()
    train = pl.DataFrame({
        "user_id": [1, 1, 2],
        "movie_id": [10, 20, 10],
        "rating": [4.0, 3.0, 5.0],
    })
    test = pl.DataFrame({
        "user_id": [1],
        "movie_id": [20],
        "rating": [3.0],
    })
    bandit.fit(train)

    assert bandit.epsilon == 0.08
    bandit.evaluate(test, k=2)
    assert bandit.epsilon == 0.08  # must be unchanged
