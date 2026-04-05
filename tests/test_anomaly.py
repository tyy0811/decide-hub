"""Tests for anomaly detection on automation outcomes."""

import numpy as np
import pytest

from src.telemetry.anomaly import (
    detect_distribution_drift,
    detect_rate_spike,
    AnomalyResult,
)


def test_no_drift_returns_ok():
    """Identical distributions produce no anomaly."""
    baseline = [{"a": 10, "b": 20}, {"a": 11, "b": 19}, {"a": 10, "b": 20}]
    recent = [{"a": 10, "b": 20}, {"a": 11, "b": 19}]
    result = detect_distribution_drift(baseline, recent)
    assert result.status == "ok"
    assert len(result.anomalies) == 0


def test_detects_clear_drift():
    """Major distribution shift triggers anomaly."""
    # Baseline: ~50/50 split
    baseline = [{"a": 50, "b": 50}] * 20
    # Recent: 90/10 split — clear drift
    recent = [{"a": 90, "b": 10}] * 5
    result = detect_distribution_drift(baseline, recent)
    assert result.status == "alert"
    assert len(result.anomalies) > 0
    assert any("a" in a["metric"] or "b" in a["metric"] for a in result.anomalies)


def test_small_variation_no_alert():
    """Normal variation within 3 SD does not trigger."""
    baseline = [
        {"a": 48, "b": 52}, {"a": 52, "b": 48},
        {"a": 50, "b": 50}, {"a": 49, "b": 51},
    ] * 5  # 20 baseline runs
    recent = [{"a": 53, "b": 47}] * 5
    result = detect_distribution_drift(baseline, recent)
    assert result.status == "ok"


def test_rate_spike_detected():
    """Error rate spike triggers anomaly."""
    baseline_rates = [0.05, 0.04, 0.06, 0.05, 0.04, 0.05] * 3  # ~5% baseline
    recent_rates = [0.30, 0.35, 0.28, 0.32, 0.31]  # ~30% — clear spike
    result = detect_rate_spike(baseline_rates, recent_rates, metric_name="error_rate")
    assert result.status == "alert"


def test_rate_no_spike():
    """Normal rate variation does not trigger."""
    baseline_rates = [0.05, 0.04, 0.06, 0.05, 0.07] * 4
    recent_rates = [0.06, 0.05, 0.07, 0.04, 0.06]
    result = detect_rate_spike(baseline_rates, recent_rates, metric_name="error_rate")
    assert result.status == "ok"


def test_empty_baseline_returns_ok():
    """Empty baseline (no history) returns ok — can't detect anomaly."""
    result = detect_distribution_drift([], [{"a": 50}])
    assert result.status == "ok"


def test_anomaly_result_structure():
    """AnomalyResult has expected fields."""
    result = AnomalyResult(status="ok", anomalies=[])
    assert hasattr(result, "status")
    assert hasattr(result, "anomalies")
