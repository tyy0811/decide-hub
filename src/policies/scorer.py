"""Learned scorer: LightGBM pointwise ranker on user-item features."""

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


class ScorerPolicy(BasePolicy):
    def __init__(self, num_leaves: int = 31, n_estimators: int = 100):
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.model: lgb.LGBMRegressor | None = None
        self.user_features: pl.DataFrame | None = None
        self.item_features: pl.DataFrame | None = None
        self._context: dict = {}

    def fit(self, train_data: pl.DataFrame) -> "ScorerPolicy":
        features = build_features(train_data)
        self.user_features = features["user_features"]
        self.item_features = features["item_features"]

        pairs = build_training_pairs(
            train_data, self.user_features, self.item_features,
        )

        X = pairs.select(FEATURE_COLS).to_numpy()
        y = pairs["rating"].to_numpy()

        self.model = lgb.LGBMRegressor(
            num_leaves=self.num_leaves,
            n_estimators=self.n_estimators,
            verbose=-1,
        )
        self.model.fit(X, y)
        return self

    def observe(self, context: dict) -> None:
        self._context = context

    def score(self, items: list[int]) -> list[tuple[int, float]]:
        if self.model is None:
            raise RuntimeError("Policy not fitted. Call fit() first.")

        user_id = self._context.get("user_id")
        user_row = self.user_features.filter(
            pl.col("user_id") == user_id
        )

        if len(user_row) == 0:
            # Unknown user — fall back to item features only, zero user features
            user_vals = [0.0] * 3  # avg, count, std
        else:
            user_vals = user_row.select(
                "user_avg_rating", "user_rating_count", "user_rating_std",
            ).row(0)

        item_df = self.item_features.filter(
            pl.col("movie_id").is_in(items)
        )

        results = []
        for item_id in items:
            item_row = item_df.filter(pl.col("movie_id") == item_id)
            if len(item_row) == 0:
                item_vals = [0.0] * 3
            else:
                item_vals = item_row.select(
                    "item_avg_rating", "item_popularity", "item_rating_std",
                ).row(0)

            feature_vec = np.array([list(user_vals) + list(item_vals)])
            pred = float(self.model.predict(feature_vec)[0])
            results.append((item_id, pred))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def evaluate(self, test_data: pl.DataFrame, k: int = 10) -> dict[str, float]:
        all_items = self.item_features["movie_id"].to_list()
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
