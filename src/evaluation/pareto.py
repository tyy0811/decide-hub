"""Pareto frontier computation for reward vs constraint tradeoffs."""

import numpy as np


def compute_pareto_frontier(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Compute 2D Pareto frontier (maximize both dimensions).

    Args:
        points: List of (x, y) tuples to evaluate.

    Returns:
        Pareto-optimal points sorted by x.
    """
    sorted_points = sorted(points, key=lambda p: -p[0])
    frontier = []
    max_y = float("-inf")

    for x, y in sorted_points:
        if y > max_y:
            frontier.append((x, y))
            max_y = y

    return sorted(frontier, key=lambda p: p[0])
