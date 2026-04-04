"""Bandit vs static policy comparison via simulated online interaction.

Simulates a sequence of interaction rounds where:
- A context vector is generated each round
- Each policy selects an item
- A reward is observed (Bernoulli with sigmoid probability)
- The bandit updates its arm estimates; the static policy does not

Output: cumulative reward curves for both policies across all rounds.
"""

import numpy as np


def run_bandit_comparison(
    n_rounds: int = 10_000,
    n_items: int = 20,
    n_features: int = 5,
    epsilon: float = 0.1,
    warmup_rounds: int = 100,
    seed: int = 42,
) -> dict:
    """Run head-to-head: static best-arm vs epsilon-greedy bandit.

    The static policy picks the single item with highest marginal expected
    reward (estimated from warmup rounds). This is the multi-armed bandit
    equivalent of PopularityPolicy — it ignores context.

    The bandit explores with probability epsilon and otherwise picks the
    arm with highest estimated reward, learning from every interaction.

    Args:
        n_rounds: Total interaction rounds.
        n_items: Number of items (arms).
        n_features: Context vector dimension.
        epsilon: Exploration probability for the bandit.
        warmup_rounds: Random rounds to estimate static policy's best arm.
        seed: Random seed for reproducibility.

    Returns:
        Dict with cumulative_reward_static, cumulative_reward_bandit (lists),
        n_rounds, epsilon, final_reward_static, final_reward_bandit.
    """
    rng = np.random.default_rng(seed)

    # Environment: per-arm bias so arms have different marginal expected rewards.
    # bias[i] = evenly spaced from -1.5 to +1.5, giving the best arm ~82%
    # reward probability and the worst ~18%. Without bias, ctx ~ N(0,I) makes
    # every arm average ~0.5 by symmetry, leaving nothing for the bandit to learn.
    arm_bias = np.linspace(-1.5, 1.5, n_items)
    item_features = rng.standard_normal((n_items, n_features))

    # --- Warmup: estimate best static arm ---
    warmup_rewards = np.zeros(n_items)
    warmup_counts = np.zeros(n_items)
    for _ in range(warmup_rounds):
        ctx = rng.standard_normal(n_features)
        scores = ctx @ item_features.T + arm_bias
        reward_probs = 1.0 / (1.0 + np.exp(-scores))
        arm = rng.integers(n_items)
        reward = float(rng.binomial(1, reward_probs[arm]))
        warmup_rewards[arm] += reward
        warmup_counts[arm] += 1
    # Static policy: pick arm with highest estimated reward
    warmup_estimates = np.where(
        warmup_counts > 0, warmup_rewards / warmup_counts, 0.0,
    )
    static_arm = int(np.argmax(warmup_estimates))

    # --- Bandit: start with warmup knowledge ---
    bandit_rewards = warmup_rewards.copy()
    bandit_counts = warmup_counts.copy()

    cumulative_static = []
    cumulative_bandit = []
    total_static = 0.0
    total_bandit = 0.0

    for _ in range(n_rounds):
        ctx = rng.standard_normal(n_features)
        scores = ctx @ item_features.T + arm_bias
        reward_probs = 1.0 / (1.0 + np.exp(-scores))

        # Static: always pick static_arm
        static_reward = float(rng.binomial(1, reward_probs[static_arm]))
        total_static += static_reward
        cumulative_static.append(total_static)

        # Bandit: epsilon-greedy
        if rng.random() < epsilon:
            bandit_arm = rng.integers(n_items)
        else:
            estimates = np.where(
                bandit_counts > 0, bandit_rewards / bandit_counts, 0.0,
            )
            bandit_arm = int(np.argmax(estimates))

        bandit_reward = float(rng.binomial(1, reward_probs[bandit_arm]))
        total_bandit += bandit_reward
        cumulative_bandit.append(total_bandit)

        # Update bandit state
        bandit_rewards[bandit_arm] += bandit_reward
        bandit_counts[bandit_arm] += 1

    return {
        "cumulative_reward_static": cumulative_static,
        "cumulative_reward_bandit": cumulative_bandit,
        "n_rounds": n_rounds,
        "epsilon": epsilon,
        "final_reward_static": total_static,
        "final_reward_bandit": total_bandit,
    }
