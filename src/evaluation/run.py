"""Run offline evaluation and print results."""

import asyncio
from src.policies.data import load_ratings, temporal_split
from src.policies.popularity import PopularityPolicy
from src.telemetry import db


async def run_popularity_evaluation() -> dict[str, float]:
    """Run popularity baseline on MovieLens and log to Postgres."""
    print("Loading MovieLens 1M...")
    ratings = load_ratings()
    print(f"  {len(ratings)} ratings loaded")

    print("Splitting train/test (last 5 per user)...")
    train, test = temporal_split(ratings, n_test=5)
    print(f"  Train: {len(train)}, Test: {len(test)}")

    print("Fitting PopularityPolicy...")
    policy = PopularityPolicy().fit(train)
    print(f"  {len(policy.item_counts)} items in catalog")

    print("Evaluating NDCG@10, MRR, HitRate@10...")
    metrics = policy.evaluate(test, k=10)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    return metrics


async def main():
    # Try Postgres logging if available, skip if not
    try:
        await db.init_pool("postgresql://decide_hub:decide_hub@localhost:5432/decide_hub")
        metrics = await run_popularity_evaluation()
        for name, value in metrics.items():
            await db.log_outcome(
                user_id=0, action=f"eval_{name}",
                reward=value, policy_id="popularity_v1",
            )
        print("\nResults logged to Postgres.")
        await db.close_pool()
    except Exception as e:
        print(f"Postgres not available ({e}), running without logging...")
        await run_popularity_evaluation()


if __name__ == "__main__":
    asyncio.run(main())
