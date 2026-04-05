"""Anomaly detection on automation outcomes.

Detects: action distribution drift, permission-block rate spikes,
error rate spikes. Uses z-score at 3 SD threshold to reduce false
positives given small sample sizes.
"""

from dataclasses import dataclass, field

import numpy as np


THRESHOLD_SD = 3.0  # 3 standard deviations to reduce false positives


@dataclass
class AnomalyResult:
    status: str  # "ok" or "alert"
    anomalies: list[dict] = field(default_factory=list)


def detect_distribution_drift(
    baseline_distributions: list[dict[str, int]],
    recent_distributions: list[dict[str, int]],
) -> AnomalyResult:
    """Detect action distribution drift between baseline and recent runs.

    Args:
        baseline_distributions: List of action-count dicts from older runs.
        recent_distributions: List of action-count dicts from recent runs.

    Returns:
        AnomalyResult with status and list of anomalous actions.
    """
    if not baseline_distributions or not recent_distributions:
        return AnomalyResult(status="ok")

    # Convert to proportions per run
    def to_proportions(dist: dict[str, int]) -> dict[str, float]:
        total = sum(dist.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in dist.items()}

    baseline_props = [to_proportions(d) for d in baseline_distributions]
    recent_props = [to_proportions(d) for d in recent_distributions]

    # All action names across all runs
    all_actions = set()
    for d in baseline_props + recent_props:
        all_actions.update(d.keys())

    anomalies = []
    for action in sorted(all_actions):
        baseline_values = np.array([d.get(action, 0.0) for d in baseline_props])
        recent_values = np.array([d.get(action, 0.0) for d in recent_props])

        baseline_mean = float(baseline_values.mean())
        baseline_std = float(baseline_values.std(ddof=1)) if len(baseline_values) > 1 else 0.0

        if baseline_std < 1e-10:
            # No variation in baseline — any deviation is anomalous
            # but only if the recent mean is meaningfully different
            recent_mean = float(recent_values.mean())
            if abs(recent_mean - baseline_mean) > 0.05:
                anomalies.append({
                    "metric": f"action_proportion:{action}",
                    "observed": round(recent_mean, 4),
                    "expected_range": f"{baseline_mean:.4f} (no variation)",
                    "severity": "high",
                })
            continue

        recent_mean = float(recent_values.mean())
        z_score = abs(recent_mean - baseline_mean) / baseline_std

        if z_score > THRESHOLD_SD:
            anomalies.append({
                "metric": f"action_proportion:{action}",
                "observed": round(recent_mean, 4),
                "expected_range": f"{baseline_mean:.4f} ± {THRESHOLD_SD * baseline_std:.4f}",
                "z_score": round(z_score, 2),
                "severity": "high" if z_score > 4.0 else "medium",
            })

    status = "alert" if anomalies else "ok"
    return AnomalyResult(status=status, anomalies=anomalies)


def detect_rate_spike(
    baseline_rates: list[float],
    recent_rates: list[float],
    metric_name: str = "rate",
) -> AnomalyResult:
    """Detect spike in a rate metric (error rate, block rate, etc.)."""
    if not baseline_rates or not recent_rates:
        return AnomalyResult(status="ok")

    baseline = np.array(baseline_rates)
    recent = np.array(recent_rates)

    baseline_mean = float(baseline.mean())
    baseline_std = float(baseline.std(ddof=1)) if len(baseline) > 1 else 0.0

    if baseline_std < 1e-10:
        recent_mean = float(recent.mean())
        if recent_mean > baseline_mean + 0.05:
            return AnomalyResult(
                status="alert",
                anomalies=[{
                    "metric": metric_name,
                    "observed": round(recent_mean, 4),
                    "expected_range": f"{baseline_mean:.4f} (no variation)",
                    "severity": "high",
                }],
            )
        return AnomalyResult(status="ok")

    recent_mean = float(recent.mean())
    z_score = (recent_mean - baseline_mean) / baseline_std

    if z_score > THRESHOLD_SD:
        return AnomalyResult(
            status="alert",
            anomalies=[{
                "metric": metric_name,
                "observed": round(recent_mean, 4),
                "expected_range": f"{baseline_mean:.4f} ± {THRESHOLD_SD * baseline_std:.4f}",
                "z_score": round(z_score, 2),
                "severity": "high" if z_score > 4.0 else "medium",
            }],
        )

    return AnomalyResult(status="ok")
