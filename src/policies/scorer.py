"""Learned scorer: LightGBM LambdaRank on user-item features."""

import lightgbm as lgb
import numpy as np
import polars as pl
from src.policies.base import BasePolicy
from src.policies.features import build_features, build_training_pairs
from src.policies.embeddings import compute_embeddings
from src.evaluation.naive import ndcg_at_k, mrr, hit_rate_at_k


FEATURE_COLS = [
    "user_avg_rating", "user_rating_count", "user_rating_std",
    "item_avg_rating", "item_popularity", "item_rating_std",
]

# Columns used for item features only (for batch scoring)
_ITEM_FEATURE_COLS = ["item_avg_rating", "item_popularity", "item_rating_std"]
_USER_FEATURE_COLS = ["user_avg_rating", "user_rating_count", "user_rating_std"]


class ScorerPolicy(BasePolicy):
    def __init__(
        self,
        num_leaves: int = 31,
        n_estimators: int = 100,
        use_embeddings: bool = False,
        n_embedding_dims: int = 16,
    ):
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.use_embeddings = use_embeddings
        self.n_embedding_dims = n_embedding_dims
        self.model: lgb.LGBMRanker | None = None
        self.user_features: pl.DataFrame | None = None
        self.item_features: pl.DataFrame | None = None
        self._item_feature_matrix: np.ndarray | None = None
        self._item_ids: list[int] | None = None
        self._embeddings: dict | None = None
        self._feature_cols: list[str] = list(FEATURE_COLS)

    def fit(self, train_data: pl.DataFrame) -> "ScorerPolicy":
        features = build_features(train_data)
        self.user_features = features["user_features"]
        self.item_features = features["item_features"]

        # Pre-compute item feature matrix for batch scoring
        sorted_items = self.item_features.sort("movie_id")
        self._item_ids = sorted_items["movie_id"].to_list()
        self._item_feature_matrix = sorted_items.select(_ITEM_FEATURE_COLS).to_numpy()

        # Compute CF embeddings if enabled (training data only — no leakage)
        if self.use_embeddings:
            self._embeddings = compute_embeddings(
                train_data, n_components=self.n_embedding_dims,
            )
            # Pre-compute item embedding matrix aligned with _item_ids for fast score()
            emb = self._embeddings
            n_dims = emb["item_embeddings"].shape[1]
            self._item_emb_matrix = np.array([
                emb["item_embeddings"][emb["item_id_to_idx"][iid]]
                if iid in emb["item_id_to_idx"]
                else np.zeros(n_dims)
                for iid in self._item_ids
            ])

        pairs = build_training_pairs(
            train_data, self.user_features, self.item_features,
        )

        # Sort by user_id so groups are contiguous
        pairs = pairs.sort("user_id")

        # Expand features with embeddings if enabled
        self._feature_cols = list(FEATURE_COLS)
        if self.use_embeddings and self._embeddings is not None:
            emb = self._embeddings
            user_emb_cols = [f"user_emb_{i}" for i in range(self.n_embedding_dims)]
            item_emb_cols = [f"item_emb_{i}" for i in range(self.n_embedding_dims)]

            # Vectorized embedding lookup via numpy indexing (not per-row dict lookups)
            pair_uids = pairs["user_id"].to_numpy()
            pair_iids = pairs["movie_id"].to_numpy()

            # Map IDs to indices, defaulting unknown to a zero-vector row
            n_dims = emb["user_embeddings"].shape[1]
            user_emb_padded = np.vstack([emb["user_embeddings"], np.zeros(n_dims)])
            item_emb_padded = np.vstack([emb["item_embeddings"], np.zeros(n_dims)])
            zero_user_idx = len(emb["user_ids"])
            zero_item_idx = len(emb["item_ids"])

            u_indices = np.array([emb["user_id_to_idx"].get(uid, zero_user_idx) for uid in pair_uids])
            i_indices = np.array([emb["item_id_to_idx"].get(iid, zero_item_idx) for iid in pair_iids])

            user_emb_matrix = user_emb_padded[u_indices]  # (n_pairs, n_dims)
            item_emb_matrix = item_emb_padded[i_indices]

            emb_columns = []
            for col_idx, col_name in enumerate(user_emb_cols):
                emb_columns.append(pl.Series(name=col_name, values=user_emb_matrix[:, col_idx]))
            for col_idx, col_name in enumerate(item_emb_cols):
                emb_columns.append(pl.Series(name=col_name, values=item_emb_matrix[:, col_idx]))
            pairs = pairs.with_columns(emb_columns)
            self._feature_cols.extend(user_emb_cols + item_emb_cols)

        X = pairs.select(self._feature_cols).to_numpy()
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

    def score(self, items: list[int], context: dict | None = None) -> list[tuple[int, float]]:
        """Score candidate items via batch prediction.

        Args:
            items: Candidate item IDs.
            context: Must contain {"user_id": int}. Thread-safe.
        """
        if self.model is None:
            raise RuntimeError("Policy not fitted. Call fit() first.")

        user_id = (context or {}).get("user_id")
        user_row = self.user_features.filter(pl.col("user_id") == user_id)

        if len(user_row) == 0:
            # Unknown user — zero user features (cold-start fallback)
            user_vals = np.zeros(len(_USER_FEATURE_COLS))
        else:
            user_vals = user_row.select(_USER_FEATURE_COLS).to_numpy()[0]

        # Build item feature matrix for requested items
        request_set = set(items)
        item_indices = [i for i, iid in enumerate(self._item_ids) if iid in request_set]
        known_ids = [self._item_ids[i] for i in item_indices]
        known_features = self._item_feature_matrix[item_indices]

        # Unknown items (not in training data) get score -inf
        known_id_set = set(known_ids)
        unknown_ids = [iid for iid in items if iid not in known_id_set]

        # Append embedding features if enabled
        if self.use_embeddings and self._embeddings is not None:
            emb = self._embeddings
            if user_id in emb["user_id_to_idx"]:
                user_emb = emb["user_embeddings"][emb["user_id_to_idx"][user_id]]
            else:
                user_emb = np.zeros(self.n_embedding_dims)
            user_vals = np.concatenate([user_vals, user_emb])

            # Use precomputed item embedding matrix (indexed same as _item_ids)
            known_features = np.hstack([known_features, self._item_emb_matrix[item_indices]])

        # Broadcast user features across all items
        n_known = len(known_ids)
        if n_known > 0:
            user_block = np.tile(user_vals, (n_known, 1))
            X = np.hstack([user_block, known_features])
            preds = self.model.predict(X)
            results = list(zip(known_ids, [float(p) for p in preds]))
        else:
            results = []

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
