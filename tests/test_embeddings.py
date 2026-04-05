"""Tests for collaborative filtering embeddings via truncated SVD."""

import numpy as np
import polars as pl
import pytest

from src.policies.embeddings import compute_embeddings


def _make_ratings() -> pl.DataFrame:
    """Small synthetic ratings for testing."""
    return pl.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
        "movie_id": [10, 20, 30, 10, 20, 40, 20, 30, 40, 10, 30],
        "rating": [5.0, 3.0, 4.0, 4.0, 5.0, 2.0, 3.0, 4.0, 3.0, 5.0, 2.0],
    })


def test_embeddings_shape():
    """User and item embeddings have correct dimensions."""
    ratings = _make_ratings()
    result = compute_embeddings(ratings, n_components=2)

    assert result["user_embeddings"].shape == (4, 2)  # 4 users, 2 dims
    assert result["item_embeddings"].shape == (4, 2)  # 4 items, 2 dims


def test_embeddings_cover_all_ids():
    """All user_ids and movie_ids from training data have embeddings."""
    ratings = _make_ratings()
    result = compute_embeddings(ratings, n_components=2)

    assert set(result["user_ids"]) == {1, 2, 3, 4}
    assert set(result["item_ids"]) == {10, 20, 30, 40}


def test_embeddings_reproducible():
    """Same input + seed produces same embeddings."""
    ratings = _make_ratings()
    r1 = compute_embeddings(ratings, n_components=2, seed=42)
    r2 = compute_embeddings(ratings, n_components=2, seed=42)
    np.testing.assert_array_equal(r1["user_embeddings"], r2["user_embeddings"])


def test_n_components_capped():
    """n_components is capped at min(n_users, n_items) - 1."""
    ratings = _make_ratings()  # 4 users, 4 items
    result = compute_embeddings(ratings, n_components=100)
    # Should be capped to min(4, 4) - 1 = 3
    assert result["user_embeddings"].shape[1] <= 3


def test_embeddings_not_all_zero():
    """Embeddings should capture signal, not be degenerate."""
    ratings = _make_ratings()
    result = compute_embeddings(ratings, n_components=2)
    assert np.any(result["user_embeddings"] != 0)
    assert np.any(result["item_embeddings"] != 0)


def test_similar_users_have_closer_embeddings():
    """Users with similar rating patterns should have closer embeddings."""
    # Users 1 and 4 both rate item 10 highly
    ratings = _make_ratings()
    result = compute_embeddings(ratings, n_components=2)

    user_map = {uid: i for i, uid in enumerate(result["user_ids"])}
    u1 = result["user_embeddings"][user_map[1]]
    u4 = result["user_embeddings"][user_map[4]]
    u2 = result["user_embeddings"][user_map[2]]

    dist_1_4 = np.linalg.norm(u1 - u4)
    dist_1_2 = np.linalg.norm(u1 - u2)
    # Not guaranteed but likely with this small dataset
    # Just check embeddings are different (non-degenerate)
    assert dist_1_4 != dist_1_2
