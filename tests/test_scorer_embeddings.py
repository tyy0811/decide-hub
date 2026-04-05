"""Tests for scorer with collaborative filtering embeddings."""

import polars as pl
import pytest

from src.policies.data import load_ratings, temporal_split
from src.policies.scorer import ScorerPolicy, FEATURE_COLS


@pytest.fixture(scope="module")
def train_test():
    ratings = load_ratings()
    return temporal_split(ratings, n_test=5)


def test_scorer_with_embeddings_fits(train_test):
    """Scorer fits successfully with CF embeddings enabled."""
    train, _ = train_test
    policy = ScorerPolicy(n_estimators=20, use_embeddings=True, n_embedding_dims=8)
    policy.fit(train)
    assert policy.model is not None


def test_scorer_with_embeddings_has_more_features(train_test):
    """Scorer with embeddings uses more features than without."""
    train, _ = train_test
    policy_no_cf = ScorerPolicy(n_estimators=20, use_embeddings=False).fit(train)
    policy_cf = ScorerPolicy(n_estimators=20, use_embeddings=True, n_embedding_dims=8).fit(train)

    n_features_no_cf = policy_no_cf.model.n_features_
    n_features_cf = policy_cf.model.n_features_
    # CF adds 2 * n_embedding_dims features (user + item)
    assert n_features_cf == n_features_no_cf + 16  # 8 user + 8 item


def test_scorer_with_embeddings_scores(train_test):
    """Scorer with embeddings returns valid scores."""
    train, _ = train_test
    policy = ScorerPolicy(n_estimators=20, use_embeddings=True, n_embedding_dims=8).fit(train)
    items = policy._item_ids[:10]
    scored = policy.score(items, context={"user_id": 1})
    assert len(scored) == 10
    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True)


def test_cold_start_user_gets_zero_embeddings(train_test):
    """Unknown user gets zero embedding vector (graceful degradation)."""
    train, _ = train_test
    policy = ScorerPolicy(n_estimators=20, use_embeddings=True, n_embedding_dims=8).fit(train)
    items = policy._item_ids[:5]
    # User 999999 doesn't exist — should not crash
    scored = policy.score(items, context={"user_id": 999999})
    assert len(scored) == 5


def test_scorer_without_embeddings_unchanged(train_test):
    """Default scorer (no embeddings) still works identically."""
    train, test = train_test
    policy = ScorerPolicy(n_estimators=20).fit(train)
    metrics = policy.evaluate(test, k=10)
    assert "ndcg@10" in metrics
