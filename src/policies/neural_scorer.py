"""Neural two-tower ranker — PyTorch user/item towers with dot-product scoring.

Two-tower architecture: user features -> user embedding, item features -> item embedding.
Score = dot product. Trained with BPR (Bayesian Personalized Ranking) loss.
Optional SVD embedding input via use_embeddings flag.
"""

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from src.policies.base import BasePolicy
from src.policies.features import build_features
from src.policies.embeddings import compute_embeddings
from src.evaluation.naive import ndcg_at_k, mrr, hit_rate_at_k


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch tensor via Python list (numpy 2.x compat)."""
    return torch.tensor(arr.tolist(), dtype=torch.float32)


class Tower(nn.Module):
    """Shallow MLP tower: input -> hidden -> embedding."""

    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralScorerPolicy(BasePolicy):
    """Two-tower neural ranker with BPR loss."""

    def __init__(
        self,
        embed_dim: int = 16,
        hidden_dim: int = 32,
        epochs: int = 10,
        lr: float = 1e-3,
        n_negatives: int = 4,
        use_embeddings: bool = False,
        n_embedding_dims: int = 16,
        seed: int = 42,
    ):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.n_negatives = n_negatives
        self.use_embeddings = use_embeddings
        self.n_embedding_dims = n_embedding_dims
        self.seed = seed

        self._user_tower: Tower | None = None
        self._item_tower: Tower | None = None
        self._user_features: dict[int, np.ndarray] = {}
        self._item_features: dict[int, np.ndarray] = {}
        self._item_ids: list[int] = []
        self._item_matrix: np.ndarray | None = None

    def fit(self, train_data: pl.DataFrame) -> "NeuralScorerPolicy":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        features = build_features(train_data)
        user_df = features["user_features"]
        item_df = features["item_features"]

        user_feat_cols = ["user_avg_rating", "user_rating_count", "user_rating_std"]
        item_feat_cols = ["item_avg_rating", "item_popularity", "item_rating_std"]

        # Build feature dicts
        for row in user_df.iter_rows(named=True):
            self._user_features[row["user_id"]] = np.array(
                [row[c] for c in user_feat_cols], dtype=np.float32,
            )
        for row in item_df.iter_rows(named=True):
            self._item_features[row["movie_id"]] = np.array(
                [row[c] for c in item_feat_cols], dtype=np.float32,
            )

        user_input_dim = len(user_feat_cols)
        item_input_dim = len(item_feat_cols)

        # Add SVD embeddings if enabled
        if self.use_embeddings:
            embeddings = compute_embeddings(
                train_data, n_components=self.n_embedding_dims,
            )
            for uid in list(self._user_features):
                if uid in embeddings["user_id_to_idx"]:
                    emb = embeddings["user_embeddings"][embeddings["user_id_to_idx"][uid]]
                    self._user_features[uid] = np.concatenate([
                        self._user_features[uid], emb.astype(np.float32),
                    ])
            for iid in list(self._item_features):
                if iid in embeddings["item_id_to_idx"]:
                    emb = embeddings["item_embeddings"][embeddings["item_id_to_idx"][iid]]
                    self._item_features[iid] = np.concatenate([
                        self._item_features[iid], emb.astype(np.float32),
                    ])
            user_input_dim += self.n_embedding_dims
            item_input_dim += self.n_embedding_dims

        self._item_ids = sorted(self._item_features.keys())

        # Build towers
        self._user_tower = Tower(user_input_dim, self.hidden_dim, self.embed_dim)
        self._item_tower = Tower(item_input_dim, self.hidden_dim, self.embed_dim)

        # BPR training
        optimizer = torch.optim.Adam(
            list(self._user_tower.parameters()) + list(self._item_tower.parameters()),
            lr=self.lr,
        )

        # Build positive pairs: (user, item) where rating >= 4
        positives = train_data.filter(pl.col("rating") >= 4)
        pos_pairs = list(zip(
            positives["user_id"].to_list(),
            positives["movie_id"].to_list(),
        ))

        all_items = list(self._item_features.keys())
        rng = np.random.default_rng(self.seed)

        for epoch in range(self.epochs):
            rng.shuffle(pos_pairs)
            total_loss = 0.0
            n_batches = 0

            for user_id, pos_item in pos_pairs:
                if user_id not in self._user_features or pos_item not in self._item_features:
                    continue

                user_vec = _to_tensor(self._user_features[user_id]).unsqueeze(0)
                pos_vec = _to_tensor(self._item_features[pos_item]).unsqueeze(0)

                # Sample negative items
                neg_items = rng.choice(all_items, size=self.n_negatives, replace=True)
                neg_vecs = _to_tensor(np.array([
                    self._item_features.get(ni, np.zeros(item_input_dim, dtype=np.float32))
                    for ni in neg_items
                ]))

                user_emb = self._user_tower(user_vec)  # (1, embed_dim)
                pos_emb = self._item_tower(pos_vec)  # (1, embed_dim)
                neg_embs = self._item_tower(neg_vecs)  # (n_neg, embed_dim)

                pos_score = (user_emb * pos_emb).sum(dim=1)  # (1,)
                neg_scores = (user_emb * neg_embs).sum(dim=1)  # (n_neg,)

                # BPR loss: -log(sigmoid(pos - neg))
                loss = -torch.log(torch.sigmoid(pos_score - neg_scores) + 1e-8).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        # Pre-compute item embedding matrix for batch scoring
        self._user_tower.eval()
        self._item_tower.eval()
        with torch.no_grad():
            item_vecs = _to_tensor(np.array([
                self._item_features[iid] for iid in self._item_ids
            ]))
            self._item_matrix = self._item_tower(item_vecs).detach().tolist()
            # Store as numpy for dot-product scoring
            self._item_matrix = np.array(self._item_matrix, dtype=np.float32)

        return self

    def score(
        self, items: list[int], context: dict | None = None,
    ) -> list[tuple[int, float]]:
        user_id = (context or {}).get("user_id")
        user_feat = self._user_features.get(user_id)
        if user_feat is None:
            user_feat = np.zeros_like(next(iter(self._user_features.values())))

        with torch.no_grad():
            user_emb_tensor = self._user_tower(
                _to_tensor(user_feat).unsqueeze(0),
            )
            user_emb = np.array(user_emb_tensor.detach().tolist()[0], dtype=np.float32)

        item_id_to_idx = {iid: i for i, iid in enumerate(self._item_ids)}
        results = []
        for iid in items:
            idx = item_id_to_idx.get(iid)
            if idx is not None:
                score = float(np.dot(user_emb, self._item_matrix[idx]))
            else:
                score = float("-inf")
            results.append((iid, score))

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
