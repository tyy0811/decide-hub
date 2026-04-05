"""pLTV Scorer — LightGBM trained on cumulative future user value.

Same model and features as the rating scorer, but trained to predict
how much total engagement a user will have after this interaction.
Items rated by high-future-value users get higher labels.
"""

import lightgbm as lgb
import numpy as np
import polars as pl

from src.policies.base import BasePolicy
from src.policies.features import build_features, build_training_pairs
from src.policies.labels import compute_pltv_labels
from src.evaluation.naive import ndcg_at_k, mrr, hit_rate_at_k


FEATURE_COLS = [
    "user_avg_rating", "user_rating_count", "user_rating_std",
    "item_avg_rating", "item_popularity", "item_rating_std",
]

_ITEM_FEATURE_COLS = ["item_avg_rating", "item_popularity", "item_rating_std"]
_USER_FEATURE_COLS = ["user_avg_rating", "user_rating_count", "user_rating_std"]


class PLTVScorerPolicy(BasePolicy):
    """Predicts cumulative future user value, ranks by prediction."""

    def __init__(self, num_leaves: int = 31, n_estimators: int = 100, n_days: int = 30):
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.n_days = n_days
        self.model: lgb.LGBMRegressor | None = None
        self.user_features: pl.DataFrame | None = None
        self.item_features: pl.DataFrame | None = None
        self._item_feature_matrix: np.ndarray | None = None
        self._item_ids: list[int] | None = None

    def fit(self, train_data: pl.DataFrame) -> "PLTVScorerPolicy":
        features = build_features(train_data)
        self.user_features = features["user_features"]
        self.item_features = features["item_features"]

        sorted_items = self.item_features.sort("movie_id")
        self._item_ids = sorted_items["movie_id"].to_list()
        self._item_feature_matrix = sorted_items.select(_ITEM_FEATURE_COLS).to_numpy()

        # Compute pLTV labels (respecting temporal boundary)
        max_ts = int(train_data["timestamp"].max())
        labeled = compute_pltv_labels(train_data, n_days=self.n_days, max_timestamp=max_ts)

        pairs = build_training_pairs(
            labeled, self.user_features, self.item_features,
        )

        X = pairs.select(FEATURE_COLS).to_numpy()
        y = pairs["pltv"].to_numpy()

        self.model = lgb.LGBMRegressor(
            num_leaves=self.num_leaves,
            n_estimators=self.n_estimators,
            verbose=-1,
        )
        self.model.fit(X, y)
        return self

    def score(
        self, items: list[int], context: dict | None = None,
    ) -> list[tuple[int, float]]:
        if self.model is None:
            raise RuntimeError("Policy not fitted.")

        user_id = (context or {}).get("user_id")
        user_row = self.user_features.filter(pl.col("user_id") == user_id)
        user_vals = user_row.select(_USER_FEATURE_COLS).to_numpy()[0] if len(user_row) > 0 else np.zeros(len(_USER_FEATURE_COLS))

        request_set = set(items)
        item_indices = [i for i, iid in enumerate(self._item_ids) if iid in request_set]
        known_ids = [self._item_ids[i] for i in item_indices]
        known_features = self._item_feature_matrix[item_indices]
        unknown_ids = [iid for iid in items if iid not in set(known_ids)]

        results = []
        if known_ids:
            user_block = np.tile(user_vals, (len(known_ids), 1))
            X = np.hstack([user_block, known_features])
            preds = self.model.predict(X)
            results = list(zip(known_ids, [float(p) for p in preds]))

        for iid in unknown_ids:
            results.append((iid, float("-inf")))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def evaluate(self, test_data: pl.DataFrame, k: int = 10) -> dict[str, float]:
        all_items = self._item_ids
        users = test_data["user_id"].unique().to_list()

        ndcg_scores, mrr_scores, hit_scores = [], [], []
        for user_id in users:
            user_test = test_data.filter(pl.col("user_id") == user_id)
            relevant = set(user_test["movie_id"].to_list())
            ranked = self.score(all_items, context={"user_id": user_id})
            ranked_ids = [item_id for item_id, _ in ranked]

            ndcg_scores.append(ndcg_at_k(ranked_ids, relevant, k))
            mrr_scores.append(mrr(ranked_ids, relevant))
            hit_scores.append(hit_rate_at_k(ranked_ids, relevant, k))

        return {
            f"ndcg@{k}": sum(ndcg_scores) / len(ndcg_scores),
            "mrr": sum(mrr_scores) / len(mrr_scores),
            f"hit_rate@{k}": sum(hit_scores) / len(hit_scores),
        }
