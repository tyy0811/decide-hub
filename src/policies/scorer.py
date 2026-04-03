"""Learned scorer: LightGBM LambdaRank on user-item features."""

import lightgbm as lgb
import numpy as np
import polars as pl
from src.policies.base import BasePolicy
from src.policies.features import build_features, build_training_pairs
from src.evaluation.naive import ndcg_at_k, mrr, hit_rate_at_k


FEATURE_COLS = [
    "user_avg_rating", "user_rating_count", "user_rating_std",
    "item_avg_rating", "item_popularity", "item_rating_std",
]

# Columns used for item features only (for batch scoring)
_ITEM_FEATURE_COLS = ["item_avg_rating", "item_popularity", "item_rating_std"]
_USER_FEATURE_COLS = ["user_avg_rating", "user_rating_count", "user_rating_std"]


class ScorerPolicy(BasePolicy):
    def __init__(self, num_leaves: int = 31, n_estimators: int = 100):
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.model: lgb.LGBMRanker | None = None
        self.user_features: pl.DataFrame | None = None
        self.item_features: pl.DataFrame | None = None
        self._item_feature_matrix: np.ndarray | None = None
        self._item_ids: list[int] | None = None
        self._context: dict = {}

    def fit(self, train_data: pl.DataFrame) -> "ScorerPolicy":
        features = build_features(train_data)
        self.user_features = features["user_features"]
        self.item_features = features["item_features"]

        # Pre-compute item feature matrix for batch scoring
        sorted_items = self.item_features.sort("movie_id")
        self._item_ids = sorted_items["movie_id"].to_list()
        self._item_feature_matrix = sorted_items.select(_ITEM_FEATURE_COLS).to_numpy()

        pairs = build_training_pairs(
            train_data, self.user_features, self.item_features,
        )

        # Sort by user_id so groups are contiguous
        pairs = pairs.sort("user_id")
        X = pairs.select(FEATURE_COLS).to_numpy()
        # LambdaRank needs integer relevance labels — round ratings to int grades
        y = pairs["rating"].cast(pl.Int32).to_numpy()

        # Group sizes: number of interactions per user (for LambdaRank)
        group_sizes = (
            pairs.group_by("user_id", maintain_order=True)
            .agg(pl.len().alias("count"))["count"]
            .to_list()
        )

        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            num_leaves=self.num_leaves,
            n_estimators=self.n_estimators,
            verbose=-1,
        )
        self.model.fit(X, y, group=group_sizes)
        return self

    def observe(self, context: dict) -> None:
        self._context = context

    def score(self, items: list[int]) -> list[tuple[int, float]]:
        """Score candidate items via batch prediction.

        Builds one feature matrix for all items and calls predict once.
        """
        if self.model is None:
            raise RuntimeError("Policy not fitted. Call fit() first.")

        user_id = self._context.get("user_id")
        user_row = self.user_features.filter(pl.col("user_id") == user_id)

        if len(user_row) == 0:
            user_vals = np.zeros(len(_USER_FEATURE_COLS))
        else:
            user_vals = user_row.select(_USER_FEATURE_COLS).to_numpy()[0]

        # Build item feature matrix for requested items
        item_set = set(items)
        item_indices = [i for i, iid in enumerate(self._item_ids) if iid in item_set]
        known_ids = [self._item_ids[i] for i in item_indices]
        known_features = self._item_feature_matrix[item_indices]

        # Unknown items get zero features
        unknown_ids = [iid for iid in items if iid not in item_set or iid not in set(known_ids)]

        # Broadcast user features across all items: shape (n_items, n_user_features + n_item_features)
        n_known = len(known_ids)
        if n_known > 0:
            user_block = np.tile(user_vals, (n_known, 1))
            X = np.hstack([user_block, known_features])
            preds = self.model.predict(X)
            results = list(zip(known_ids, [float(p) for p in preds]))
        else:
            results = []

        # Unknown items get score -inf so they sort last
        for iid in unknown_ids:
            results.append((iid, float("-inf")))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def evaluate(self, test_data: pl.DataFrame, k: int = 10) -> dict[str, float]:
        all_items = self._item_ids
        users = test_data["user_id"].unique().to_list()

        ndcg_scores = []
        mrr_scores = []
        hit_scores = []

        for user_id in users:
            user_test = test_data.filter(pl.col("user_id") == user_id)
            relevant = set(user_test["movie_id"].to_list())

            self.observe({"user_id": user_id})
            ranked = self.score(all_items)
            ranked_ids = [item_id for item_id, _ in ranked]

            ndcg_scores.append(ndcg_at_k(ranked_ids, relevant, k))
            mrr_scores.append(mrr(ranked_ids, relevant))
            hit_scores.append(hit_rate_at_k(ranked_ids, relevant, k))

        return {
            f"ndcg@{k}": sum(ndcg_scores) / len(ndcg_scores),
            "mrr": sum(mrr_scores) / len(mrr_scores),
            f"hit_rate@{k}": sum(hit_scores) / len(hit_scores),
        }
