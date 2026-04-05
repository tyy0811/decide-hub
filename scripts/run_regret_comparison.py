"""Run regret comparison across all available policy types.

Usage: python scripts/run_regret_comparison.py
"""

import numpy as np

from src.evaluation.online_sim import run_simulation


def random_policy(ctx, n_items, rng):
    return rng.integers(n_items)


def greedy_policy(ctx, n_items, rng):
    """Always pick item 0 (static, non-adaptive)."""
    return 0


class BanditPolicy:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.rewards = None
        self.counts = None

    def __call__(self, ctx, n_items, rng):
        if self.rewards is None:
            self.rewards = np.zeros(n_items)
            self.counts = np.zeros(n_items)

        if rng.random() < self.epsilon:
            action = rng.integers(n_items)
        else:
            estimates = np.where(
                self.counts > 0, self.rewards / self.counts, 0.0,
            )
            action = int(np.argmax(estimates))
        return action

    def update(self, action, reward):
        self.rewards[action] += reward
        self.counts[action] += 1


def main():
    bandit = BanditPolicy(epsilon=0.1)

    # Wrap bandit to update after each step
    class BanditWithUpdate:
        def __init__(self, bandit):
            self.bandit = bandit
            self.last_action = None

        def __call__(self, ctx, n_items, rng):
            action = self.bandit(ctx, n_items, rng)
            self.last_action = action
            return action

    bandit_wrapper = BanditWithUpdate(bandit)

    result = run_simulation(
        policies={
            "random": random_policy,
            "greedy (item 0)": greedy_policy,
            "epsilon-greedy (e=0.1)": bandit_wrapper,
        },
        n_rounds=10_000,
        n_items=20,
        seed=42,
    )

    print("=== Regret Comparison (10K rounds) ===\n")
    for name, data in result.items():
        print(f"{name}:")
        print(f"  Final regret:   {data['final_regret']:.0f}")
        print(f"  Final reward:   {data['final_reward']:.0f}")
        print(f"  Avg reward:     {data['avg_reward']:.4f}")
        print()


if __name__ == "__main__":
    main()
