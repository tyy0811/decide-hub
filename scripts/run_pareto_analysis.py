"""Run Pareto analysis: sweep constraint thresholds, measure NDCG vs constraint satisfaction.

Usage: python scripts/run_pareto_analysis.py
"""

import numpy as np
from src.policies.data import load_ratings, temporal_split
from src.policies.scorer import ScorerPolicy
from src.policies.constrained import ConstrainedPolicy, compute_item_clusters
from src.policies.features import build_features
from src.evaluation.pareto import compute_pareto_frontier


def main():
    ratings = load_ratings()
    train, test = temporal_split(ratings, n_test=5)

    # Fit scorer
    policy = ScorerPolicy(n_estimators=50).fit(train)

    # Compute item clusters from features
    features = build_features(train)
    item_df = features["item_features"].sort("movie_id")
    item_ids = item_df["movie_id"].to_list()
    item_matrix = item_df.select(["item_avg_rating", "item_popularity", "item_rating_std"]).to_numpy()
    clusters = compute_item_clusters(item_ids, item_matrix, n_clusters=5)

    print("=== Pareto Analysis: NDCG vs Diversity ===\n")
    print(f"{'min_categories':<16} {'NDCG@10':<10} {'Avg Categories':<16} {'Swapped':<10}")
    print("-" * 52)

    points = []
    for min_cats in [1, 2, 3, 4, 5]:
        constrained = ConstrainedPolicy(
            policy, clusters=clusters, min_categories=min_cats, k=10,
        )

        # Evaluate on a subset of test users
        users = test["user_id"].unique().to_list()[:100]
        ndcg_sum = 0.0
        cats_sum = 0.0
        swaps_sum = 0

        for uid in users:
            user_test = test.filter(test["user_id"] == uid)
            relevant = set(user_test["movie_id"].to_list())
            scored, meta = constrained.score_with_metadata(
                item_ids, context={"user_id": uid},
            )
            ranked_ids = [i for i, _ in scored[:10]]

            from src.evaluation.naive import ndcg_at_k
            ndcg_sum += ndcg_at_k(ranked_ids, relevant, 10)
            cats_sum += meta["categories_in_topk"]
            swaps_sum += meta["items_swapped"]

        avg_ndcg = ndcg_sum / len(users)
        avg_cats = cats_sum / len(users)
        avg_swaps = swaps_sum / len(users)

        print(f"{min_cats:<16} {avg_ndcg:<10.4f} {avg_cats:<16.1f} {avg_swaps:<10.1f}")
        points.append((avg_ndcg, avg_cats))

    # Compute Pareto frontier
    frontier = compute_pareto_frontier(points)
    print(f"\nPareto frontier: {len(frontier)} points")
    for ndcg, cats in frontier:
        print(f"  NDCG={ndcg:.4f}, Categories={cats:.1f}")


if __name__ == "__main__":
    main()
