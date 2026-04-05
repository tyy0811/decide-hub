"""Online simulation environment for multi-policy regret comparison.

Generalizes bandit_comparison.py: any policy can participate, configurable
environment, regret curves for all policies.
"""

import numpy as np
from typing import Callable


class OnlineEnvironment:
    """Simulated online environment with context-dependent rewards.

    Same reward model as simulator.py: reward = Bernoulli(sigmoid(ctx @ item_features)).
    Fixed seed ensures all policies see the same context sequence.
    """

    def __init__(
        self,
        n_items: int = 20,
        n_features: int = 5,
        seed: int = 42,
    ):
        self.n_items = n_items
        self.n_features = n_features
        self._rng = np.random.default_rng(seed)
        self._item_features = self._rng.standard_normal((n_items, n_features))
        self._current_ctx: np.ndarray | None = None
        self._current_reward_probs: np.ndarray | None = None

    def get_context(self) -> np.ndarray:
        """Generate next context vector."""
        self._current_ctx = self._rng.standard_normal(self.n_features)
        scores = self._current_ctx @ self._item_features.T
        self._current_reward_probs = 1.0 / (1.0 + np.exp(-scores))
        return self._current_ctx.copy()

    def step(self, action: int) -> float:
        """Take action, observe reward."""
        if self._current_reward_probs is None:
            raise RuntimeError("Call get_context() before step()")
        prob = self._current_reward_probs[action]
        reward = float(self._rng.binomial(1, prob))
        return reward

    def optimal_reward(self) -> float:
        """Expected reward of the optimal action for current context."""
        if self._current_reward_probs is None:
            raise RuntimeError("Call get_context() before optimal_reward()")
        return float(self._current_reward_probs.max())

    def reset_rng(self, seed: int) -> None:
        """Reset RNG for a new comparison run."""
        self._rng = np.random.default_rng(seed)


# Policy type: (context, n_items, rng) -> action
PolicyFn = Callable[[np.ndarray, int, np.random.Generator], int]


def run_simulation(
    policies: dict[str, PolicyFn],
    n_rounds: int = 10_000,
    n_items: int = 20,
    n_features: int = 5,
    seed: int = 42,
) -> dict[str, dict]:
    """Run multi-policy simulation with shared context sequences.

    Each policy sees the same context sequence (environment re-seeded per policy).

    Args:
        policies: Dict of {name: policy_fn}. Each fn takes (context, n_items, rng)
                  and returns an action (int).
        n_rounds: Interaction rounds per policy.
        n_items: Number of arms.
        n_features: Context vector dimension.
        seed: Base seed for environment.

    Returns:
        Dict of {policy_name: {cumulative_regret, cumulative_reward, final_regret}}.
    """
    results = {}

    for name, policy_fn in policies.items():
        env = OnlineEnvironment(n_items=n_items, n_features=n_features, seed=seed)
        policy_rng = np.random.default_rng(seed + hash(name) % (2**31))

        cumulative_regret = []
        cumulative_reward = []
        total_regret = 0.0
        total_reward = 0.0

        for _ in range(n_rounds):
            ctx = env.get_context()
            optimal = env.optimal_reward()
            action = policy_fn(ctx, n_items, policy_rng)
            reward = env.step(action)

            total_reward += reward
            total_regret += optimal - reward
            cumulative_regret.append(total_regret)
            cumulative_reward.append(total_reward)

        results[name] = {
            "cumulative_regret": cumulative_regret,
            "cumulative_reward": cumulative_reward,
            "final_regret": total_regret,
            "final_reward": total_reward,
            "avg_reward": total_reward / n_rounds,
        }

    return results
