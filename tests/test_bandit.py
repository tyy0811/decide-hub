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
    bandit.arm_rewards = {1: 10.0, 2: 10.0}
    bandit.arm_counts = {1: 10, 2: 10}

    # Item 2 starts at 1.0 avg, update with high rewards
    bandit.update(2, 5.0)
    bandit.update(2, 5.0)

    scored = bandit.score([1, 2])
    # Item 2 now: (10 + 5 + 5) / 12 = 1.67, Item 1: 10/10 = 1.0
    assert scored[0][0] == 2


def test_update_new_item():
    """update() works for items not seen during fit."""
    bandit = EpsilonGreedyPolicy(epsilon=0.0)
    bandit.update(99, 5.0)
    bandit.update(99, 3.0)

    scored = bandit.score([99])
    assert scored[0] == (99, 4.0)  # (5+3)/2
