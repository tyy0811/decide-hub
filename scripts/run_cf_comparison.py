"""Compare scorer with and without CF embeddings.

Reports: overall NDCG, warm-user NDCG, cold-start NDCG.

Usage: .venv/bin/python scripts/run_cf_comparison.py
"""

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import polars as pl
from src.policies.data import load_ratings, temporal_split
from src.policies.scorer import ScorerPolicy


def main():
    ratings = load_ratings()
    train, test = temporal_split(ratings, n_test=5)

    # Users in training set (warm) vs not (cold)
    train_users = set(train["user_id"].unique().to_list())
    test_users = test["user_id"].unique().to_list()
    warm_users = [u for u in test_users if u in train_users]
    cold_users = [u for u in test_users if u not in train_users]

    print(f"Test users: {len(test_users)} ({len(warm_users)} warm, {len(cold_users)} cold)\n")

    for use_cf, label in [(False, "Scorer (no CF)"), (True, "Scorer + CF (dim=8)")]:
        policy = ScorerPolicy(
            n_estimators=50,
            use_embeddings=use_cf,
            n_embedding_dims=8,
        ).fit(train)

        metrics = policy.evaluate(test, k=10)
        print(f"=== {label} ===")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

        # Warm-only evaluation
        if warm_users:
            warm_test = test.filter(pl.col("user_id").is_in(warm_users))
            warm_metrics = policy.evaluate(warm_test, k=10)
            print(f"  ndcg@10 (warm only): {warm_metrics['ndcg@10']:.4f}")

        # Cold-only evaluation
        if cold_users:
            cold_test = test.filter(pl.col("user_id").is_in(cold_users))
            cold_metrics = policy.evaluate(cold_test, k=10)
            print(f"  ndcg@10 (cold only): {cold_metrics['ndcg@10']:.4f}")
        print()


if __name__ == "__main__":
    main()
