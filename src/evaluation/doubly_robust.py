"""Doubly Robust off-policy estimator.

Combines IPS with a direct reward model for lower-variance estimates.
DR = (1/n) * sum[ model(x,a) + (P_target/P_logging) * (reward - model(x,a)) ]

Correct if EITHER the reward model or the propensities are correct
(the "doubly robust" property).
"""


def dr_estimate(
    rewards: list[float],
    propensities: list[float],
    target_probs: list[float],
    reward_model_predictions: list[float],
) -> float:
    """Doubly Robust policy evaluation estimator.

    Args:
        rewards: Observed rewards under logging policy.
        propensities: Logging policy P(a|x). Must be > 0.
        target_probs: Target policy P(a|x).
        reward_model_predictions: Predicted E[reward|x,a] from reward model.

    Returns:
        Estimated policy value.
    """
    n = len(rewards)
    if n == 0:
        return 0.0

    total = 0.0
    for r, p0, pt, model_pred in zip(
        rewards, propensities, target_probs, reward_model_predictions,
    ):
        if p0 <= 0:
            raise ValueError(f"Propensity must be > 0, got {p0}")
        weight = pt / p0
        # DR: direct estimate + IPS correction for model error
        total += model_pred + weight * (r - model_pred)

    return total / n
