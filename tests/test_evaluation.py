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
