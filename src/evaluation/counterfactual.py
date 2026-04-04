"""Counterfactual policy evaluation: IPS and clipped IPS.

Used ONLY with synthetic logged-policy data where propensities are known
by construction. NOT used with MovieLens (which has no logging policy).
"""


def ips_estimate(
    rewards: list[float],
    propensities: list[float],
    target_probs: list[float],
) -> float:
    """Inverse Propensity Scoring estimator.

    IPS = (1/n) * sum(target_prob(a|x) / logging_prob(a|x) * reward)

    Args:
        rewards: Observed rewards under logging policy.
        propensities: Logging policy probabilities P_0(a|x) for chosen actions.
            Must be strictly positive.
        target_probs: Target policy probabilities P_t(a|x) for chosen actions.

    Raises:
        ValueError: If any propensity is <= 0.
    """
    n = len(rewards)
    if n == 0:
        return 0.0

    total = 0.0
    for r, p0, pt in zip(rewards, propensities, target_probs):
        if p0 <= 0:
            raise ValueError(f"Propensity must be > 0, got {p0}")
        weight = pt / p0
        total += weight * r

    return total / n


def clipped_ips_estimate(
    rewards: list[float],
    propensities: list[float],
    target_probs: list[float],
    clip: float = 10.0,
) -> float:
    """Clipped IPS — cap importance weights to reduce variance.

    Same as IPS but weight = min(target/logging, clip).

    Raises:
        ValueError: If any propensity is <= 0.
    """
    n = len(rewards)
    if n == 0:
        return 0.0

    total = 0.0
    for r, p0, pt in zip(rewards, propensities, target_probs):
        if p0 <= 0:
            raise ValueError(f"Propensity must be > 0, got {p0}")
        weight = min(pt / p0, clip)
        total += weight * r

    return total / n
