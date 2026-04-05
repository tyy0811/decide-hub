"""Tests for constrained ranking policy wrapper."""

import numpy as np
import pytest

from src.policies.constrained import ConstrainedPolicy, compute_item_clusters


def _mock_score(items, context=None):
    """Deterministic scoring: item ID = score."""
    return [(item, float(item)) for item in sorted(items, reverse=True)]


class MockPolicy:
    def score(self, items, context=None):
        return _mock_score(items, context)

    _item_ids = list(range(1, 31))


def test_unconstrained_passthrough():
    """No constraints = same ranking as base policy."""
    policy = ConstrainedPolicy(MockPolicy(), clusters={})
    scored = policy.score(list(range(1, 11)))
    ids = [i for i, _ in scored]
    assert ids == list(range(10, 0, -1))


def test_diversity_constraint_enforced():
    """Top-K contains items from at least M clusters."""
    # All items in same cluster -> diversity forces swaps
    clusters = {i: 0 for i in range(1, 31)}
    clusters[25] = 1  # Only item 25 is in cluster 1
    clusters[26] = 2  # Only item 26 is in cluster 2

    policy = ConstrainedPolicy(
        MockPolicy(), clusters=clusters,
        min_categories=3, k=10,
    )
    scored = policy.score(list(range(1, 31)))
    top_10_ids = [i for i, _ in scored[:10]]
    # Items 25 and 26 should be promoted into top 10
    top_10_clusters = {clusters[i] for i in top_10_ids}
    assert len(top_10_clusters) >= 3


def test_fairness_cap_enforced():
    """No cluster exceeds max_share of top-K."""
    clusters = {i: i % 2 for i in range(1, 31)}  # 2 clusters: even/odd

    policy = ConstrainedPolicy(
        MockPolicy(), clusters=clusters,
        max_category_share=0.5, k=10,
    )
    scored = policy.score(list(range(1, 31)))
    top_10_ids = [i for i, _ in scored[:10]]
    cluster_counts = {}
    for i in top_10_ids:
        c = clusters[i]
        cluster_counts[c] = cluster_counts.get(c, 0) + 1
    for count in cluster_counts.values():
        assert count <= 5  # 50% of 10


def test_score_with_metadata():
    """score_with_metadata returns ranking + constraint info."""
    clusters = {i: i % 3 for i in range(1, 31)}
    policy = ConstrainedPolicy(
        MockPolicy(), clusters=clusters,
        min_categories=2, k=10,
    )
    scored, meta = policy.score_with_metadata(list(range(1, 31)))
    assert "categories_in_topk" in meta
    assert "max_category_share" in meta
    assert "items_swapped" in meta
    assert len(scored) == 30


def test_compute_item_clusters():
    """K-means clustering produces expected number of clusters."""
    item_features = np.random.default_rng(42).standard_normal((100, 3))
    item_ids = list(range(100))
    clusters = compute_item_clusters(item_ids, item_features, n_clusters=5)
    assert len(clusters) == 100
    assert len(set(clusters.values())) <= 5
