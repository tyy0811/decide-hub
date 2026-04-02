"""Run offline evaluation and print results."""

import asyncio
import os
from src.policies.data import load_ratings, temporal_split
from src.policies.popularity import PopularityPolicy
from src.telemetry import db

_DEFAULT_DSN = "postgresql://decide_hub:decide_hub@localhost:5432/decide_hub"


async def run_popularity_evaluation() -> dict[str, float]:
    """Run popularity baseline on MovieLens."""
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
    # Try connecting to Postgres — narrowly scoped to connection only
    dsn = os.environ.get("DATABASE_URL", _DEFAULT_DSN)
    db_available = False
    try:
        await db.init_pool(dsn)
        db_available = True
    except Exception as e:
        print(f"Postgres not available ({e}), running without logging...")

    # Evaluation runs regardless — errors propagate normally
    metrics = await run_popularity_evaluation()

    # Log to Postgres if connected
    if db_available:
        for name, value in metrics.items():
            await db.log_outcome(
                user_id=0, action=f"eval_{name}",
                reward=value, policy_id="popularity_v1",
            )
        print("\nResults logged to Postgres.")
        await db.close_pool()


if __name__ == "__main__":
    asyncio.run(main())
