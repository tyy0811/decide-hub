"""Epsilon-greedy bandit policy — exploration/exploitation with online learning.

Maintains per-arm reward estimates in memory. Arm estimates reset on server
restart — persistent bandit state is a V3 concern. V2 demonstrates the
algorithm and evaluation, not production statefulness.
"""

import numpy as np
import polars as pl

from src.policies.base import BasePolicy
from src.evaluation.naive import ndcg_at_k, mrr, hit_rate_at_k


class EpsilonGreedyPolicy(BasePolicy):
    """Epsilon-greedy multi-armed bandit with warm-start from training data.

    With probability epsilon, explores (random item ordering).
    With probability 1-epsilon, exploits (rank by estimated reward).
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        max_epsilon: float = 0.10,
        seed: int = 42,
    ):
        if epsilon > max_epsilon:
            raise ValueError(
                f"epsilon {epsilon} exceeds max_epsilon {max_epsilon}"
            )
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self._rng = np.random.default_rng(seed)
        self.arm_rewards: dict[int, float] = {}
        self.arm_counts: dict[int, int] = {}
        self._all_items: list[int] = []

    def fit(self, train_data: pl.DataFrame) -> "EpsilonGreedyPolicy":
        """Warm-start arm estimates from training data average ratings."""
        stats = train_data.group_by("movie_id").agg([
            pl.col("rating").sum().alias("total_rating"),
            pl.len().alias("count"),
        ])
        for row in stats.iter_rows(named=True):
            item_id = row["movie_id"]
            self.arm_rewards[item_id] = row["total_rating"]
            self.arm_counts[item_id] = row["count"]
        self._all_items = list(self.arm_counts.keys())
        return self

    def score(
        self, items: list[int], context: dict | None = None,
    ) -> list[tuple[int, float]]:
        """Epsilon-greedy scoring: explore or exploit.

        Explore: random ordering with synthetic descending scores.
        Exploit: rank by estimated reward (arm_rewards / arm_counts).
        """
        if self._rng.random() < self.epsilon:
            shuffled = list(items)
            self._rng.shuffle(shuffled)
            return [
                (item, float(len(items) - i))
                for i, item in enumerate(shuffled)
            ]

        scored = []
        for item in items:
            count = self.arm_counts.get(item, 0)
            estimate = self.arm_rewards[item] / count if count > 0 else 0.0
            scored.append((item, estimate))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def update(self, item_id: int, reward: float) -> None:
        """Update arm estimate with observed reward."""
        self.arm_rewards[item_id] = self.arm_rewards.get(item_id, 0.0) + reward
        self.arm_counts[item_id] = self.arm_counts.get(item_id, 0) + 1

    def evaluate(self, test_data: pl.DataFrame, k: int = 10) -> dict[str, float]:
        """Offline evaluation with epsilon=0 (pure exploitation)."""
        all_items = self._all_items
        users = test_data["user_id"].unique().to_list()

        old_epsilon = self.epsilon
        self.epsilon = 0.0

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

        self.epsilon = old_epsilon

        return {
            f"ndcg@{k}": sum(ndcg_scores) / len(ndcg_scores),
            "mrr": sum(mrr_scores) / len(mrr_scores),
            f"hit_rate@{k}": sum(hit_scores) / len(hit_scores),
        }
