"""Constrained ranking policy — post-processing wrapper for diversity and fairness.

Wraps any BasePolicy and applies constraints to the top-K ranking.
Separates ranking objective from constraint — the base policy optimizes
relevance, the wrapper enforces business rules.
"""

import numpy as np
from sklearn.cluster import KMeans


def compute_item_clusters(
    item_ids: list[int],
    item_features: np.ndarray,
    n_clusters: int = 5,
    seed: int = 42,
) -> dict[int, int]:
    """Cluster items by feature similarity via K-means.

    Returns: {item_id: cluster_id}
    """
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(item_features)
    return dict(zip(item_ids, [int(l) for l in labels]))


class ConstrainedPolicy:
    """Post-processing wrapper that enforces diversity and fairness constraints.

    Works with any policy that has a score() method.
    """

    def __init__(
        self,
        base_policy,
        clusters: dict[int, int],
        min_categories: int = 1,
        max_category_share: float = 1.0,
        k: int = 10,
    ):
        self.base_policy = base_policy
        self.clusters = clusters
        self.min_categories = min_categories
        self.max_category_share = max_category_share
        self.k = k

    def score(
        self, items: list[int], context: dict | None = None,
    ) -> list[tuple[int, float]]:
        """Score and re-rank with constraints applied."""
        scored, _ = self.score_with_metadata(items, context)
        return scored

    def score_with_metadata(
        self, items: list[int], context: dict | None = None,
    ) -> tuple[list[tuple[int, float]], dict]:
        """Score with constraint metadata for Pareto analysis."""
        base_scored = self.base_policy.score(items, context)

        if not self.clusters:
            return base_scored, {
                "categories_in_topk": 0,
                "max_category_share": 0.0,
                "items_swapped": 0,
            }

        # Apply constraints to top-K
        top_k = list(base_scored[:self.k])
        rest = list(base_scored[self.k:])
        swapped = 0

        # Diversity: ensure min_categories in top-K
        top_k_clusters = {self.clusters.get(item_id, -1) for item_id, _ in top_k}
        if len(top_k_clusters) < self.min_categories:
            needed_clusters = set()
            for item_id, _ in rest:
                c = self.clusters.get(item_id, -1)
                if c not in top_k_clusters and c >= 0:
                    needed_clusters.add(c)
                if len(top_k_clusters) + len(needed_clusters) >= self.min_categories:
                    break

            for target_cluster in needed_clusters:
                # Find best item from target cluster in rest
                for i, (item_id, score) in enumerate(rest):
                    if self.clusters.get(item_id, -1) == target_cluster:
                        # Swap with worst item in top-K from over-represented cluster
                        worst_idx = len(top_k) - 1
                        rest.append(top_k[worst_idx])
                        top_k[worst_idx] = (item_id, score)
                        rest.pop(i)
                        swapped += 1
                        top_k_clusters.add(target_cluster)
                        break

        # Fairness cap: no cluster exceeds max_share
        max_count = max(1, int(self.k * self.max_category_share))
        cluster_counts: dict[int, int] = {}
        for item_id, _ in top_k:
            c = self.clusters.get(item_id, -1)
            cluster_counts[c] = cluster_counts.get(c, 0) + 1

        for cluster_id, count in list(cluster_counts.items()):
            while count > max_count:
                # Find the lowest-scored item in this cluster in top-K
                worst_in_cluster = None
                worst_idx = -1
                for i, (item_id, score) in enumerate(top_k):
                    if self.clusters.get(item_id, -1) == cluster_id:
                        if worst_in_cluster is None or score < worst_in_cluster[1]:
                            worst_in_cluster = (item_id, score)
                            worst_idx = i

                # Find best replacement from a different cluster in rest
                replaced = False
                for i, (item_id, score) in enumerate(rest):
                    rc = self.clusters.get(item_id, -1)
                    if rc != cluster_id:
                        rest.append(top_k[worst_idx])
                        top_k[worst_idx] = (item_id, score)
                        rest.pop(i)
                        swapped += 1
                        count -= 1
                        replaced = True
                        break

                if not replaced:
                    break

        # Re-sort top-K by score (maintain relevance ordering)
        top_k.sort(key=lambda x: x[1], reverse=True)

        # Compute metadata
        final_clusters = {}
        for item_id, _ in top_k:
            c = self.clusters.get(item_id, -1)
            final_clusters[c] = final_clusters.get(c, 0) + 1

        max_share = max(final_clusters.values()) / self.k if final_clusters else 0.0

        metadata = {
            "categories_in_topk": len(final_clusters),
            "max_category_share": round(max_share, 3),
            "items_swapped": swapped,
        }

        return top_k + rest, metadata
