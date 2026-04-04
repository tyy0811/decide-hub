"""Tests for experiment report renderer."""

from src.evaluation.report import render_markdown


def test_render_markdown_basic():
    """Markdown report includes key fields."""
    result = {
        "baseline_mean": 0.500,
        "treatment_mean": 0.550,
        "lift": 0.050,
        "ci_lower": 0.020,
        "ci_upper": 0.080,
        "sample_size_control": 1000,
        "sample_size_treatment": 1000,
    }
    md = render_markdown(result)
    assert "Baseline" in md
    assert "Treatment" in md
    assert "0.050" in md
    assert "1000" in md


def test_render_markdown_with_segments():
    """Markdown includes segment breakdown."""
    result = {
        "baseline_mean": 0.5,
        "treatment_mean": 0.55,
        "lift": 0.05,
        "ci_lower": 0.02,
        "ci_upper": 0.08,
        "sample_size_control": 200,
        "sample_size_treatment": 200,
        "segments": {
            "high_activity": {"lift": 0.08, "ci_lower": 0.03, "ci_upper": 0.13, "n_control": 100, "n_treatment": 100},
            "low_activity": {"lift": 0.02, "ci_lower": -0.03, "ci_upper": 0.07, "n_control": 100, "n_treatment": 100},
        },
    }
    md = render_markdown(result)
    assert "high_activity" in md
    assert "low_activity" in md


def test_render_markdown_with_mde():
    """Markdown includes MDE when provided."""
    result = {
        "baseline_mean": 0.5,
        "treatment_mean": 0.55,
        "lift": 0.05,
        "ci_lower": 0.02,
        "ci_upper": 0.08,
        "sample_size_control": 1000,
        "sample_size_treatment": 1000,
        "mde": 0.03,
    }
    md = render_markdown(result)
    assert "MDE" in md or "Minimum" in md


def test_render_markdown_confidence_label():
    """CI label reflects the actual confidence level, not hardcoded 95%."""
    result = {
        "baseline_mean": 0.5,
        "treatment_mean": 0.55,
        "lift": 0.05,
        "ci_lower": 0.01,
        "ci_upper": 0.09,
        "sample_size_control": 500,
        "sample_size_treatment": 500,
    }
    md_80 = render_markdown(result, confidence=0.80)
    assert "80% CI" in md_80
    assert "95% CI" not in md_80

    md_99 = render_markdown(result, confidence=0.99)
    assert "99% CI" in md_99
