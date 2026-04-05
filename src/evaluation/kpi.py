"""Business metric (KPI) transforms applied to raw rewards.

Pure functions: reward list in, transformed metric list out.
No side effects, no database dependency.
"""


def value_proxy(
    rewards: list[float], prices: list[float],
) -> list[float]:
    """Reward weighted by item price — monetization proxy.

    Higher reward on expensive items contributes more to business value.
    """
    if len(rewards) != len(prices):
        raise ValueError(
            f"rewards length {len(rewards)} != prices length {len(prices)}"
        )
    return [r * p for r, p in zip(rewards, prices)]


def retention_proxy(
    rewards: list[float], threshold: float = 0.5,
) -> list[float]:
    """Binary retention: 1.0 if engagement exceeds threshold, 0.0 otherwise.

    Models "did the user come back?" as a function of engagement level.
    """
    return [1.0 if r >= threshold else 0.0 for r in rewards]


def conversion_proxy(
    user_rewards: list[list[float]], threshold: float = 0.5,
) -> list[float]:
    """Binary conversion: 1.0 if cumulative user engagement exceeds threshold.

    Each inner list is one user's reward sequence. Summed and thresholded.
    """
    return [1.0 if sum(rs) >= threshold else 0.0 for rs in user_rewards]
