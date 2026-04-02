"""Popularity baseline: rank items by global interaction count."""

import polars as pl
from src.policies.base import BasePolicy
from src.evaluation.naive import ndcg_at_k, mrr, hit_rate_at_k


class PopularityPolicy(BasePolicy):
    def __init__(self):
        self.item_counts: dict[int, int] = {}
        self._context: dict = {}

    def fit(self, train_data: pl.DataFrame) -> "PopularityPolicy":
        counts = train_data.group_by("movie_id").agg(pl.len().alias("count"))
        self.item_counts = dict(zip(
            counts["movie_id"].to_list(),
            counts["count"].to_list(),
        ))
        return self

    def observe(self, context: dict) -> None:
        self._context = context

    def score(self, items: list[int]) -> list[tuple[int, float]]:
        scored = [(item, float(self.item_counts.get(item, 0))) for item in items]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def log_outcome(self, user_id: int, item_id: int, reward: float) -> None:
        pass  # Outcomes are in the dataset for offline evaluation

    def evaluate(self, test_data: pl.DataFrame, k: int = 10) -> dict[str, float]:
        all_items = list(self.item_counts.keys())
        users = test_data["user_id"].unique().to_list()

        ndcg_scores = []
        mrr_scores = []
        hit_scores = []

        for user_id in users:
            user_test = test_data.filter(pl.col("user_id") == user_id)
            relevant = set(user_test["movie_id"].to_list())

            self.observe({"user_id": user_id})
            ranked = self.score(all_items)
            ranked_ids = [item_id for item_id, _ in ranked]

            ndcg_scores.append(ndcg_at_k(ranked_ids, relevant, k))
            mrr_scores.append(mrr(ranked_ids, relevant))
            hit_scores.append(hit_rate_at_k(ranked_ids, relevant, k))

        return {
            f"ndcg@{k}": sum(ndcg_scores) / len(ndcg_scores),
            "mrr": sum(mrr_scores) / len(mrr_scores),
            f"hit_rate@{k}": sum(hit_scores) / len(hit_scores),
        }
