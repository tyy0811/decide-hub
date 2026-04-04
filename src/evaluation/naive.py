"""Naive offline ranking metrics: NDCG@K, MRR, HitRate@K."""

import math


def ndcg_at_k(ranked_items: list[int], relevant_items: set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    if not relevant_items:
        return 0.0

    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(ranked_items[:k])
        if item in relevant_items
    )
    ideal_hits = min(len(relevant_items), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def graded_ndcg_at_k(
    ranked_items: list[int], relevance_grades: dict[int, int], k: int,
) -> float:
    """NDCG@K with graded relevance (gain = 2^grade - 1).

    Args:
        ranked_items: Items in ranked order.
        relevance_grades: {item_id: grade} where grade > 0 means relevant.
        k: Cutoff rank.
    """
    if not relevance_grades:
        return 0.0

    dcg = sum(
        (2 ** relevance_grades.get(item, 0) - 1) / math.log2(i + 2)
        for i, item in enumerate(ranked_items[:k])
    )
    # Ideal: sort grades descending, take top-k
    ideal_gains = sorted(relevance_grades.values(), reverse=True)[:k]
    idcg = sum(
        (2 ** g - 1) / math.log2(i + 2)
        for i, g in enumerate(ideal_gains)
    )
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked_items: list[int], relevant_items: set[int]) -> float:
    """Mean Reciprocal Rank — reciprocal of first relevant item's position."""
    for i, item in enumerate(ranked_items):
        if item in relevant_items:
            return 1.0 / (i + 1)
    return 0.0


def hit_rate_at_k(ranked_items: list[int], relevant_items: set[int], k: int) -> float:
    """1.0 if any relevant item appears in top-K, 0.0 otherwise."""
    return 1.0 if any(item in relevant_items for item in ranked_items[:k]) else 0.0
