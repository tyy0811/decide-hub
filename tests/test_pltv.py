"""Tests for pLTV label construction and scorer."""

import polars as pl
import pytest

from src.policies.labels import compute_pltv_labels
from src.policies.pltv_scorer import PLTVScorerPolicy
from src.policies.data import load_ratings, temporal_split


def _make_ratings():
    return pl.DataFrame({
        "user_id": [1, 1, 1, 1, 2, 2],
        "movie_id": [10, 20, 30, 40, 10, 20],
        "rating": [5.0, 3.0, 4.0, 2.0, 4.0, 5.0],
        "timestamp": [100, 200, 300, 400, 100, 500],
    })


def test_pltv_labels_shape():
    """pLTV labels include only rows with complete future windows."""
    ratings = _make_ratings()
    # n_days=100, max_timestamp=600: all rows pass (max ts 500 + 100 = 600)
    labels = compute_pltv_labels(ratings, n_days=100, max_timestamp=600)
    assert len(labels) == len(ratings)


def test_pltv_labels_early_interactions_have_higher_value():
    """Earlier interactions should have higher future value."""
    ratings = _make_ratings()
    # Use wide window so all user 1 rows are included
    labels = compute_pltv_labels(ratings, n_days=500, max_timestamp=900)
    user1_labels = labels.filter(pl.col("user_id") == 1)
    first_label = user1_labels.sort("timestamp")["pltv"][0]
    last_label = user1_labels.sort("timestamp")["pltv"][-1]
    # t=100 has 3 future interactions (t=200,300,400), t=400 has 0
    assert first_label > last_label


def test_pltv_labels_respect_max_timestamp():
    """Rows where future window exceeds max_timestamp are excluded."""
    ratings = _make_ratings()
    # max_timestamp=250, n_days=100: only timestamps <= 150 pass
    labels = compute_pltv_labels(ratings, n_days=100, max_timestamp=250)
    assert len(labels) < len(ratings)
    # Only t=100 rows should survive (100 + 100 = 200 <= 250)
    assert all(ts <= 150 for ts in labels["timestamp"].to_list())


@pytest.fixture(scope="module")
def train_test():
    ratings = load_ratings()
    return temporal_split(ratings, n_test=5)


def test_pltv_scorer_fits(train_test):
    train, _ = train_test
    # Subsample for pLTV — per-row label computation is O(n^2)
    train_sub = train.sample(n=min(5_000, len(train)), seed=42)
    policy = PLTVScorerPolicy(n_estimators=20, n_days=86400).fit(train_sub)
    assert policy.model is not None


def test_pltv_scorer_evaluate(train_test):
    train, test = train_test
    train_sub = train.sample(n=min(5_000, len(train)), seed=42)
    policy = PLTVScorerPolicy(n_estimators=20, n_days=86400).fit(train_sub)
    # Use subset of test users for speed
    test_sub = test.head(500)
    metrics = policy.evaluate(test_sub, k=10)
    assert "ndcg@10" in metrics
