"""Collaborative filtering embeddings via truncated SVD.

Computes user and item embeddings from the training-split user-item
interaction matrix. MUST NOT see test interactions — that would leak
collaborative signal into the feature set.
"""

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


def compute_embeddings(
    train_data: pl.DataFrame,
    n_components: int = 16,
    seed: int = 42,
) -> dict:
    """Compute user and item embeddings from rating matrix.

    Args:
        train_data: DataFrame with user_id, movie_id, rating columns.
                    Must be training split only (no test data).
        n_components: Embedding dimensions per user/item.
        seed: Random seed for SVD.

    Returns:
        Dict with user_embeddings (n_users, n_components),
        item_embeddings (n_items, n_components),
        user_ids (list), item_ids (list), user_id_to_idx, item_id_to_idx.
    """
    user_ids = sorted(train_data["user_id"].unique().to_list())
    item_ids = sorted(train_data["movie_id"].unique().to_list())

    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    item_id_to_idx = {iid: i for i, iid in enumerate(item_ids)}

    n_users = len(user_ids)
    n_items = len(item_ids)

    # Cap components to avoid SVD failure
    max_components = min(n_users, n_items) - 1
    n_components = min(n_components, max(max_components, 1))

    # Build sparse user-item rating matrix
    rows = [user_id_to_idx[uid] for uid in train_data["user_id"].to_list()]
    cols = [item_id_to_idx[iid] for iid in train_data["movie_id"].to_list()]
    vals = train_data["rating"].to_list()

    matrix = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

    # Truncated SVD: matrix ≈ U @ diag(S) @ V^T
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    user_embeddings = svd.fit_transform(matrix)  # (n_users, n_components)
    # Item embeddings from V^T (components_ has shape (n_components, n_items))
    item_embeddings = svd.components_.T  # (n_items, n_components)

    return {
        "user_embeddings": user_embeddings,
        "item_embeddings": item_embeddings,
        "user_ids": user_ids,
        "item_ids": item_ids,
        "user_id_to_idx": user_id_to_idx,
        "item_id_to_idx": item_id_to_idx,
    }
