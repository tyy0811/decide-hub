import polars as pl
import pytest
from src.policies.base import BasePolicy
from src.policies.popularity import PopularityPolicy


def make_ratings() -> pl.DataFrame:
    """Small synthetic ratings for testing."""
    return pl.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 2, 3, 3],
        "movie_id": [10, 20, 30, 10, 20, 40, 10, 30],
        "rating": [5.0, 4.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0],
        "timestamp": [1, 2, 3, 1, 2, 3, 1, 2],
    })


def test_base_policy_is_abstract():
    """Cannot instantiate BasePolicy directly."""
    with pytest.raises(TypeError):
        BasePolicy()


def test_popularity_fit_counts():
    """PopularityPolicy counts interactions per item."""
    ratings = make_ratings()
    policy = PopularityPolicy().fit(ratings)
    # movie_id 10 appears 3 times (most popular)
    # movie_id 20 appears 2 times
    # movie_id 30 appears 2 times
    # movie_id 40 appears 1 time
    assert policy.item_counts[10] == 3
    assert policy.item_counts[20] == 2
    assert policy.item_counts[40] == 1


def test_popularity_score_ranking():
    """Most popular items ranked first."""
    ratings = make_ratings()
    policy = PopularityPolicy().fit(ratings)
    scored = policy.score([10, 20, 30, 40])
    item_ids = [item_id for item_id, _ in scored]
    assert item_ids[0] == 10  # 3 interactions — most popular


def test_popularity_score_unknown_item():
    """Unknown items get score 0."""
    ratings = make_ratings()
    policy = PopularityPolicy().fit(ratings)
    scored = policy.score([10, 999])
    scores_dict = dict(scored)
    assert scores_dict[999] == 0


def test_popularity_evaluate():
    """Evaluate returns metric dict with expected keys."""
    ratings = make_ratings()
    train = ratings.filter(pl.col("timestamp") <= 2)
    test = ratings.filter(pl.col("timestamp") > 2)

    policy = PopularityPolicy().fit(train)
    metrics = policy.evaluate(test, k=2)

    assert "ndcg@2" in metrics
    assert "mrr" in metrics
    assert "hit_rate@2" in metrics
    assert all(0 <= v <= 1 for v in metrics.values())
