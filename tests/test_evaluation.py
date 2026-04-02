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
