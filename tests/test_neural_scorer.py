"""Tests for neural two-tower ranker."""

import polars as pl
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytestmark = pytest.mark.slow

from src.policies.neural_scorer import NeuralScorerPolicy
from src.policies.base import BasePolicy
from src.policies.data import load_ratings, temporal_split


def test_neural_is_base_policy():
    assert issubclass(NeuralScorerPolicy, BasePolicy)


@pytest.fixture(scope="module")
def train_test():
    """Subsample to 10K rows for fast neural training in tests."""
    ratings = load_ratings()
    train, test = temporal_split(ratings, n_test=5)
    # Subsample training data — neural BPR is per-sample, full dataset is too slow
    train_sub = train.sample(n=min(10_000, len(train)), seed=42)
    test_sub = test.filter(pl.col("user_id").is_in(train_sub["user_id"].unique().to_list()))
    return train_sub, test_sub


def test_neural_fits(train_test):
    train, _ = train_test
    policy = NeuralScorerPolicy(epochs=2, embed_dim=8).fit(train)
    assert policy._user_tower is not None
    assert policy._item_tower is not None


def test_neural_score_returns_sorted(train_test):
    train, _ = train_test
    policy = NeuralScorerPolicy(epochs=2, embed_dim=8).fit(train)
    items = policy._item_ids[:20]
    scored = policy.score(items, context={"user_id": 1})
    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True)


def test_neural_score_unknown_user(train_test):
    train, _ = train_test
    policy = NeuralScorerPolicy(epochs=2, embed_dim=8).fit(train)
    items = policy._item_ids[:5]
    scored = policy.score(items, context={"user_id": 999999})
    assert len(scored) == 5


def test_neural_evaluate(train_test):
    train, test = train_test
    policy = NeuralScorerPolicy(epochs=2, embed_dim=8).fit(train)
    metrics = policy.evaluate(test, k=10)
    assert "ndcg@10" in metrics
    assert "mrr" in metrics


def test_neural_with_embeddings(train_test):
    """Neural scorer accepts SVD embedding input."""
    train, _ = train_test
    policy = NeuralScorerPolicy(epochs=2, embed_dim=8, use_embeddings=True).fit(train)
    assert policy._user_tower is not None
    items = policy._item_ids[:5]
    scored = policy.score(items, context={"user_id": 1})
    assert len(scored) == 5
