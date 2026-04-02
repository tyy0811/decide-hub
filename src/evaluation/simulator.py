"""Synthetic logged-policy data generator.

Generates context-action-reward tuples where the logging policy is known
by construction. True propensities are recorded — IPS is exact, not estimated.

This is the ONLY dataset used for counterfactual evaluation. MovieLens is
used for naive ranking metrics only.
"""

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax with temperature scaling."""
    scaled = logits / temperature
    shifted = scaled - np.max(scaled)  # numerical stability
    exp = np.exp(shifted)
    return exp / exp.sum()


def generate_logged_data(
    n_samples: int = 50_000,
    n_items: int = 20,
    n_features: int = 5,
    temperature: float = 1.0,
    seed: int = 42,
) -> dict:
    """Generate synthetic logged-policy data.

    Logging policy: softmax over (context @ item_features.T) / temperature.
    Reward: Bernoulli(sigmoid(context @ item_features[chosen_item])).
    Propensities: exact softmax probabilities (known by construction).

    Returns dict with keys: contexts, actions, rewards, propensities,
    item_features, n_items, n_features.
    """
    rng = np.random.default_rng(seed)

    # Fixed item feature matrix (shared across all samples)
    item_features = rng.standard_normal((n_items, n_features))

    contexts = []
    actions = []
    rewards = []
    propensities = []

    for _ in range(n_samples):
        # Random context vector
        ctx = rng.standard_normal(n_features)

        # Logging policy: softmax over scores
        scores = ctx @ item_features.T
        probs = softmax(scores, temperature=temperature)

        # Sample action from logging policy
        action = rng.choice(n_items, p=probs)

        # Reward: Bernoulli(sigmoid(score for chosen item))
        score = scores[action]
        reward_prob = 1.0 / (1.0 + np.exp(-score))  # sigmoid
        reward = float(rng.binomial(1, reward_prob))

        contexts.append(ctx)
        actions.append(int(action))
        rewards.append(reward)
        propensities.append(float(probs[action]))

    return {
        "contexts": contexts,
        "actions": actions,
        "rewards": rewards,
        "propensities": propensities,
        "item_features": item_features,
        "n_items": n_items,
        "n_features": n_features,
    }
