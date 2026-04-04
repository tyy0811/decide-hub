"""A/B experiment engine — bootstrap CIs, treatment effects, segment breakdown.

No p-values. CIs and effect sizes only. More informative, less prone to
misinterpretation. See DECISIONS.md for rationale.
"""

import numpy as np


def bootstrap_ci(
    data: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean.

    Args:
        data: 1D array of observations.
        confidence: Confidence level (default 0.95).
        n_resamples: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        (lower, upper) bounds of the CI.
    """
    if len(data) == 0:
        raise ValueError("Cannot compute CI on empty data")
    rng = np.random.default_rng(seed)
    n = len(data)
    means = np.array([
        rng.choice(data, size=n, replace=True).mean()
        for _ in range(n_resamples)
    ])
    alpha = 1 - confidence
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lower, upper


def run_experiment(
    control: np.ndarray,
    treatment: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    segments_control: list[str] | None = None,
    segments_treatment: list[str] | None = None,
    seed: int = 42,
) -> dict:
    """Run A/B experiment: compute treatment effect with bootstrap CI.

    Args:
        control: Reward observations for control group.
        treatment: Reward observations for treatment group.
        confidence: CI confidence level.
        n_resamples: Bootstrap resamples.
        segments_control: Optional segment labels for control (same length as control).
        segments_treatment: Optional segment labels for treatment.
        seed: Random seed.

    Returns:
        Dict with baseline_mean, treatment_mean, lift, ci_lower, ci_upper,
        sample sizes, and optional segments breakdown.
    """
    control = np.asarray(control)
    treatment = np.asarray(treatment)

    if len(control) == 0:
        raise ValueError("Control group is empty")
    if len(treatment) == 0:
        raise ValueError("Treatment group is empty")

    # Validate segment lengths if provided
    if segments_control is not None and len(segments_control) != len(control):
        raise ValueError(
            f"segments_control length {len(segments_control)} != "
            f"control length {len(control)}"
        )
    if segments_treatment is not None and len(segments_treatment) != len(treatment):
        raise ValueError(
            f"segments_treatment length {len(segments_treatment)} != "
            f"treatment length {len(treatment)}"
        )

    # Effect = treatment_mean - control_mean
    rng = np.random.default_rng(seed)
    effects = []
    for _ in range(n_resamples):
        c_sample = rng.choice(control, size=len(control), replace=True)
        t_sample = rng.choice(treatment, size=len(treatment), replace=True)
        effects.append(t_sample.mean() - c_sample.mean())

    effects = np.array(effects)
    alpha = 1 - confidence
    ci_lower = float(np.percentile(effects, 100 * alpha / 2))
    ci_upper = float(np.percentile(effects, 100 * (1 - alpha / 2)))

    result = {
        "baseline_mean": float(control.mean()),
        "treatment_mean": float(treatment.mean()),
        "lift": float(treatment.mean() - control.mean()),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence": confidence,
        "sample_size_control": len(control),
        "sample_size_treatment": len(treatment),
    }

    # Segment-wise breakdown
    if segments_control is not None and segments_treatment is not None:
        segment_names = sorted(set(segments_control) | set(segments_treatment))
        segments = {}
        for seg in segment_names:
            c_mask = np.array([s == seg for s in segments_control])
            t_mask = np.array([s == seg for s in segments_treatment])
            c_seg = control[c_mask]
            t_seg = treatment[t_mask]
            if len(c_seg) > 0 and len(t_seg) > 0:
                seg_result = run_experiment(
                    c_seg, t_seg,
                    confidence=confidence,
                    n_resamples=min(n_resamples, 5000),
                    seed=seed,
                )
                segments[seg] = {
                    "lift": seg_result["lift"],
                    "ci_lower": seg_result["ci_lower"],
                    "ci_upper": seg_result["ci_upper"],
                    "n_control": len(c_seg),
                    "n_treatment": len(t_seg),
                }
        result["segments"] = segments

    return result


def minimum_detectable_effect(
    n: int,
    baseline_std: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Minimum detectable effect for a two-sample t-test.

    MDE = (z_alpha/2 + z_power) * baseline_std * sqrt(2/n)

    Args:
        n: Sample size per group.
        baseline_std: Standard deviation of the baseline metric.
        alpha: Significance level (for z-score calculation).
        power: Statistical power.

    Returns:
        Minimum detectable absolute effect.
    """
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    return float((z_alpha + z_power) * baseline_std * np.sqrt(2.0 / n))
