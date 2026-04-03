import polars as pl
import pytest
from pathlib import Path
from src.policies.base import BasePolicy
from src.policies.popularity import PopularityPolicy
from src.policies.data import _read_dat, temporal_split


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


# --- Data loading ---


def test_read_dat_parses_double_colon(tmp_path):
    """_read_dat correctly parses :: separated files."""
    dat_file = tmp_path / "test.dat"
    dat_file.write_text("1::Alice::30\n2::Bob::25\n")
    df = _read_dat(
        dat_file,
        columns=["id", "name", "age"],
        dtypes={"id": int, "name": str, "age": int},
    )
    assert len(df) == 2
    assert df["id"].to_list() == [1, 2]
    assert df["name"].to_list() == ["Alice", "Bob"]
    assert df["age"].to_list() == [30, 25]


def test_read_dat_handles_latin1(tmp_path):
    """_read_dat handles latin-1 encoded characters."""
    dat_file = tmp_path / "test.dat"
    dat_file.write_bytes("1::caf\xe9::5\n".encode("latin-1"))
    df = _read_dat(
        dat_file,
        columns=["id", "name", "val"],
        dtypes={"id": int, "name": str, "val": int},
    )
    assert df["name"].to_list() == ["caf\xe9"]


def test_temporal_split_last_n():
    """temporal_split puts last n interactions per user in test."""
    ratings = make_ratings()
    train, test = temporal_split(ratings, n_test=1)
    # Each user's highest-timestamp interaction goes to test
    for uid in [1, 2, 3]:
        user_test = test.filter(pl.col("user_id") == uid)
        user_train = train.filter(pl.col("user_id") == uid)
        assert len(user_test) == 1
        # Test item should have highest timestamp for that user
        assert user_test["timestamp"].max() >= user_train["timestamp"].max()


def test_temporal_split_no_overlap():
    """Train and test sets don't share rows."""
    ratings = make_ratings()
    train, test = temporal_split(ratings, n_test=2)
    assert len(train) + len(test) == len(ratings)


# --- Feature engineering ---

from src.policies.features import build_features


def test_build_features_shape():
    """Feature builder returns user and item feature matrices."""
    ratings = make_ratings()
    features = build_features(ratings)
    assert "user_features" in features
    assert "item_features" in features
    # Each user/item should have features
    assert len(features["user_features"]) > 0
    assert len(features["item_features"]) > 0


def test_build_features_columns():
    """Feature matrix has expected columns."""
    ratings = make_ratings()
    features = build_features(ratings)
    # User features should include interaction stats
    assert "user_avg_rating" in features["user_features"].columns
    assert "user_rating_count" in features["user_features"].columns
    # Item features should include popularity stats
    assert "item_avg_rating" in features["item_features"].columns
    assert "item_popularity" in features["item_features"].columns


# --- ScorerPolicy ---

from src.policies.scorer import ScorerPolicy
import numpy as np


def make_scorer_ratings() -> pl.DataFrame:
    """Larger synthetic ratings with enough signal for LambdaRank.

    10 users, 20 items, ~100 interactions with varied ratings.
    Users have distinct preferences so the ranker can learn personalization.
    """
    rng = np.random.default_rng(42)
    rows = []
    for uid in range(1, 11):
        # Each user rates 8-12 items
        n_items = rng.integers(8, 13)
        items = rng.choice(range(1, 21), size=n_items, replace=False)
        for i, mid in enumerate(items):
            # Users with even IDs prefer low movie IDs; odd prefer high
            if uid % 2 == 0:
                base = 5.0 - (mid / 20.0) * 3.0
            else:
                base = 2.0 + (mid / 20.0) * 3.0
            rating = float(np.clip(base + rng.normal(0, 0.5), 1, 5))
            rows.append({"user_id": uid, "movie_id": int(mid), "rating": round(rating, 1), "timestamp": i + 1})
    return pl.DataFrame(rows)


def test_scorer_fit_and_score():
    """ScorerPolicy trains on ratings and scores items."""
    ratings = make_scorer_ratings()
    policy = ScorerPolicy(num_leaves=8, n_estimators=50).fit(ratings)

    policy.observe({"user_id": 1})
    scored = policy.score(list(range(1, 21)))
    assert len(scored) == 20
    assert all(isinstance(s, float) for _, s in scored)
    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True)


def test_scorer_predictions_non_constant():
    """Scorer must produce distinct scores — not a degenerate model."""
    ratings = make_scorer_ratings()
    policy = ScorerPolicy(num_leaves=8, n_estimators=50).fit(ratings)

    policy.observe({"user_id": 1})
    scored = policy.score(list(range(1, 21)))
    scores = [s for _, s in scored]
    assert len(set(scores)) > 1, f"All scores identical: {scores}"


def test_scorer_distinguishes_users():
    """Different users should get different rankings (personalization signal)."""
    ratings = make_scorer_ratings()
    policy = ScorerPolicy(num_leaves=8, n_estimators=50).fit(ratings)

    items = list(range(1, 21))

    policy.observe({"user_id": 2})  # even user — prefers low movie IDs
    scores_even = [s for _, s in policy.score(items)]

    policy.observe({"user_id": 3})  # odd user — prefers high movie IDs
    scores_odd = [s for _, s in policy.score(items)]

    assert scores_even != scores_odd, "Scorer produces identical scores for different users"


def test_scorer_evaluate():
    """ScorerPolicy evaluate returns metrics dict."""
    ratings = make_scorer_ratings()
    train = ratings.filter(pl.col("timestamp") <= 6)
    test = ratings.filter(pl.col("timestamp") > 6)

    policy = ScorerPolicy(num_leaves=8, n_estimators=50).fit(train)
    metrics = policy.evaluate(test, k=5)

    assert "ndcg@5" in metrics
    assert "mrr" in metrics
    assert "hit_rate@5" in metrics
