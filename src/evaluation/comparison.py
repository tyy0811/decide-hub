"""Action distribution comparison — TVD and per-action deltas.

Used by replay (offline change control) and shadow mode (online comparison).
Pure functions, no database dependency.
"""


def compute_action_deltas(
    baseline: dict[str, int], candidate: dict[str, int],
) -> dict[str, float]:
    """Per-action frequency deltas between two distributions.

    Returns dict of signed floats: positive = candidate does this action more.
    Only includes actions present in at least one distribution.
    """
    all_actions = set(baseline) | set(candidate)
    if not all_actions:
        return {}

    baseline_total = sum(baseline.values()) or 1
    candidate_total = sum(candidate.values()) or 1

    return {
        action: (candidate.get(action, 0) / candidate_total)
        - (baseline.get(action, 0) / baseline_total)
        for action in sorted(all_actions)
    }


def total_variation_distance(
    baseline: dict[str, int], candidate: dict[str, int],
) -> float:
    """Total Variation Distance between two action distributions.

    Returns float in [0, 1]. Zero means identical distributions.
    """
    deltas = compute_action_deltas(baseline, candidate)
    if not deltas:
        return 0.0
    return sum(abs(d) for d in deltas.values()) / 2
