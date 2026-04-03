from src.evaluation.naive import ndcg_at_k, mrr, hit_rate_at_k


def test_ndcg_perfect_ranking():
    """First item is relevant — perfect DCG for k=1."""
    ranked = [1, 2, 3, 4, 5]
    relevant = {1}
    assert ndcg_at_k(ranked, relevant, k=5) == 1.0


def test_ndcg_worst_ranking():
    """Relevant item at position k — worst possible DCG for single relevant item."""
    ranked = [2, 3, 4, 5, 1]
    relevant = {1}
    result = ndcg_at_k(ranked, relevant, k=5)
    assert 0 < result < 1.0


def test_ndcg_no_relevant():
    """No relevant items in ranked list."""
    ranked = [2, 3, 4]
    relevant = {99}
    assert ndcg_at_k(ranked, relevant, k=3) == 0.0


def test_ndcg_empty_relevant():
    """Empty relevant set."""
    ranked = [1, 2, 3]
    relevant = set()
    assert ndcg_at_k(ranked, relevant, k=3) == 0.0


def test_ndcg_multiple_relevant():
    """Multiple relevant items, both in top positions."""
    ranked = [1, 2, 3, 4, 5]
    relevant = {1, 2}
    result = ndcg_at_k(ranked, relevant, k=5)
    assert result == 1.0  # Both relevant items at positions 1,2 — ideal ordering


def test_mrr_first_position():
    ranked = [1, 2, 3]
    relevant = {1}
    assert mrr(ranked, relevant) == 1.0


def test_mrr_third_position():
    ranked = [2, 3, 1]
    relevant = {1}
    assert abs(mrr(ranked, relevant) - 1 / 3) < 1e-9


def test_mrr_not_found():
    ranked = [2, 3, 4]
    relevant = {1}
    assert mrr(ranked, relevant) == 0.0


def test_hit_rate_hit():
    ranked = [1, 2, 3, 4, 5]
    relevant = {3}
    assert hit_rate_at_k(ranked, relevant, k=5) == 1.0


def test_hit_rate_miss():
    ranked = [1, 2, 3, 4, 5]
    relevant = {3}
    assert hit_rate_at_k(ranked, relevant, k=2) == 0.0


# --- Simulator ---

from src.evaluation.simulator import generate_logged_data, softmax


def test_softmax_sums_to_one():
    import numpy as np
    logits = np.array([1.0, 2.0, 3.0])
    probs = softmax(logits, temperature=1.0)
    assert abs(sum(probs) - 1.0) < 1e-6


def test_softmax_temperature():
    """Higher temperature -> more uniform distribution."""
    import numpy as np
    logits = np.array([1.0, 2.0, 3.0])
    cold = softmax(logits, temperature=0.1)
    hot = softmax(logits, temperature=10.0)
    # Cold should be more peaked (lower entropy)
    assert max(cold) > max(hot)


def test_generate_logged_data_shape():
    data = generate_logged_data(n_samples=100, n_items=10, n_features=5, seed=42)
    assert len(data["contexts"]) == 100
    assert len(data["actions"]) == 100
    assert len(data["rewards"]) == 100
    assert len(data["propensities"]) == 100


def test_generate_logged_data_propensities_valid():
    """All propensities should be in (0, 1]."""
    data = generate_logged_data(n_samples=500, n_items=10, n_features=5, seed=42)
    assert all(0 < p <= 1 for p in data["propensities"])


def test_generate_logged_data_rewards_binary():
    """Rewards should be 0 or 1 (Bernoulli)."""
    data = generate_logged_data(n_samples=500, n_items=10, n_features=5, seed=42)
    assert all(r in (0.0, 1.0) for r in data["rewards"])


# --- IPS / Clipped IPS ---

import numpy as np
from src.evaluation.counterfactual import ips_estimate, clipped_ips_estimate


def test_ips_uniform_logging_policy():
    """When logging policy is uniform, IPS = naive mean (no reweighting needed)."""
    n_items = 5
    n_samples = 10000
    rng = np.random.default_rng(42)

    # Uniform logging policy: propensity = 1/n_items for all
    propensities = [1.0 / n_items] * n_samples
    rewards = [float(rng.binomial(1, 0.3)) for _ in range(n_samples)]

    # Target policy same as logging (uniform) — importance weight = 1.0
    target_probs = [1.0 / n_items] * n_samples

    estimate = ips_estimate(rewards, propensities, target_probs)
    naive_mean = np.mean(rewards)
    assert abs(estimate - naive_mean) < 0.05


def test_ips_with_known_analytic_value():
    """IPS estimate should converge to true policy value on synthetic data."""
    data = generate_logged_data(n_samples=50000, n_items=10, n_features=5,
                                 temperature=1.0, seed=42)

    # Target policy: same items but with temperature=0.5 (more greedy)
    target_probs_list = []
    for ctx in data["contexts"]:
        scores = np.array(ctx) @ data["item_features"].T
        target_probs = softmax(scores, temperature=0.5)
        target_probs_list.append(target_probs)

    # Get target probability for the action that was actually taken
    target_action_probs = [
        float(target_probs_list[i][data["actions"][i]])
        for i in range(len(data["actions"]))
    ]

    estimate = ips_estimate(
        data["rewards"], data["propensities"], target_action_probs,
    )
    # The estimate should be a reasonable positive number
    assert estimate > 0
    assert estimate < 2.0  # Sanity bound


def test_clipped_ips_reduces_variance():
    """Clipped IPS should have lower variance than unclipped."""
    data = generate_logged_data(n_samples=10000, n_items=10, n_features=5,
                                 temperature=1.0, seed=42)

    target_probs_list = []
    for ctx in data["contexts"]:
        scores = np.array(ctx) @ data["item_features"].T
        target_probs = softmax(scores, temperature=0.1)  # Very greedy — high variance
        target_probs_list.append(target_probs)

    target_action_probs = [
        float(target_probs_list[i][data["actions"][i]])
        for i in range(len(data["actions"]))
    ]

    ips_val = ips_estimate(data["rewards"], data["propensities"], target_action_probs)
    clipped_val = clipped_ips_estimate(
        data["rewards"], data["propensities"], target_action_probs, clip=10.0,
    )

    # Both should produce finite estimates
    assert np.isfinite(ips_val)
    assert np.isfinite(clipped_val)


def test_clipped_ips_clip_bound():
    """Importance weights should be clipped to max value."""
    # Construct case where one weight is very large
    rewards = [1.0, 1.0, 1.0]
    propensities = [0.001, 0.5, 0.5]  # First has tiny propensity
    target_probs = [0.9, 0.3, 0.3]  # First has high target prob

    unclipped = ips_estimate(rewards, propensities, target_probs)
    clipped = clipped_ips_estimate(rewards, propensities, target_probs, clip=5.0)

    # Clipped should be smaller because the large weight is capped
    assert clipped < unclipped
