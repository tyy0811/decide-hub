"""Tests for pointwise LTR scorer (LGBMRegressor)."""

import polars as pl
import pytest

from src.policies.ltr_scorer import PointwiseScorerPolicy
from src.policies.base import BasePolicy
from src.policies.data import load_ratings, temporal_split


def test_pointwise_is_base_policy():
    assert issubclass(PointwiseScorerPolicy, BasePolicy)


@pytest.fixture(scope="module")
def train_test():
    ratings = load_ratings()
    return temporal_split(ratings, n_test=5)


def test_pointwise_fits(train_test):
    train, _ = train_test
    policy = PointwiseScorerPolicy(n_estimators=20).fit(train)
    assert policy.model is not None


def test_pointwise_score_returns_sorted(train_test):
    train, _ = train_test
    policy = PointwiseScorerPolicy(n_estimators=20).fit(train)
    items = policy._item_ids[:20]
    scored = policy.score(items, context={"user_id": 1})
    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True)


def test_pointwise_score_unknown_user(train_test):
    train, _ = train_test
    policy = PointwiseScorerPolicy(n_estimators=20).fit(train)
    items = policy._item_ids[:5]
    scored = policy.score(items, context={"user_id": 999999})
    assert len(scored) == 5


def test_pointwise_evaluate(train_test):
    train, test = train_test
    policy = PointwiseScorerPolicy(n_estimators=20).fit(train)
    metrics = policy.evaluate(test, k=10)
    assert "ndcg@10" in metrics
    assert "mrr" in metrics
    assert "hit_rate@10" in metrics


def test_pointwise_uses_regressor_not_ranker(train_test):
    """Verify the model is a regressor (pointwise), not a ranker (pairwise)."""
    import lightgbm as lgb
    train, _ = train_test
    policy = PointwiseScorerPolicy(n_estimators=20).fit(train)
    assert isinstance(policy.model, lgb.LGBMRegressor)
