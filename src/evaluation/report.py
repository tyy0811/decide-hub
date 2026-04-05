"""Structured experiment report — JSON dict + markdown renderer."""


def render_markdown(result: dict, confidence: float | None = None) -> str:
    """Render experiment result dict as a markdown report.

    Reads confidence from result["confidence"] if present (set by
    run_experiment). Falls back to the explicit parameter, then 0.95.
    """
    conf = confidence or result.get("confidence", 0.95)
    ci_pct = int(conf * 100)
    lines = [
        "## Experiment Report",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Baseline mean | {result['baseline_mean']:.4f} |",
        f"| Treatment mean | {result['treatment_mean']:.4f} |",
        f"| Lift (treatment - baseline) | {result['lift']:.4f} |",
        f"| {ci_pct}% CI | [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}] |",
        f"| Sample size (control) | {result['sample_size_control']} |",
        f"| Sample size (treatment) | {result['sample_size_treatment']} |",
    ]

    if "mde" in result:
        lines.append(f"| Minimum Detectable Effect (MDE) | {result['mde']:.4f} |")

    # Significance interpretation
    if result["ci_lower"] > 0:
        lines.extend(["", "**Result:** Treatment is significantly better than baseline."])
    elif result["ci_upper"] < 0:
        lines.extend(["", "**Result:** Treatment is significantly worse than baseline."])
    else:
        lines.extend(["", "**Result:** No significant difference detected."])

    # Segment breakdown
    if "segments" in result:
        lines.extend([
            "",
            "### Segment Breakdown",
            "",
            f"| Segment | Lift | {ci_pct}% CI | N (control) | N (treatment) |",
            "|---------|------|--------|-------------|---------------|",
        ])
        for seg_name, seg in sorted(result["segments"].items()):
            lines.append(
                f"| {seg_name} | {seg['lift']:.4f} | "
                f"[{seg['ci_lower']:.4f}, {seg['ci_upper']:.4f}] | "
                f"{seg['n_control']} | {seg['n_treatment']} |"
            )

    return "\n".join(lines) + "\n"
