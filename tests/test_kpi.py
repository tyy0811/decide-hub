"""Tests for KPI metric transforms."""

import numpy as np
import pytest
from src.evaluation.kpi import value_proxy, retention_proxy, conversion_proxy


def test_value_proxy_weights_by_price():
    """value_proxy multiplies reward by item price."""
    rewards = [1.0, 0.0, 1.0]
    prices = [10.0, 20.0, 5.0]
    result = value_proxy(rewards, prices)
    assert result == [10.0, 0.0, 5.0]


def test_value_proxy_empty():
    assert value_proxy([], []) == []


def test_retention_proxy_binary_threshold():
    """retention_proxy converts engagement to binary (did user return?)."""
    rewards = [0.0, 0.3, 0.5, 0.7, 1.0]
    result = retention_proxy(rewards, threshold=0.5)
    assert result == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_retention_proxy_default_threshold():
    """Default threshold is 0.5."""
    assert retention_proxy([0.4, 0.6]) == [0.0, 1.0]


def test_conversion_proxy_cumulative():
    """conversion_proxy: 1 if cumulative reward exceeds threshold."""
    # Each user's total engagement → binary conversion
    user_rewards = [[0.1, 0.2, 0.3], [0.8, 0.9]]
    result = conversion_proxy(user_rewards, threshold=0.5)
    assert result == [1.0, 1.0]  # sum > 0.5 for both


def test_conversion_proxy_below_threshold():
    user_rewards = [[0.1, 0.1]]
    result = conversion_proxy(user_rewards, threshold=0.5)
    assert result == [0.0]


def test_value_proxy_length_mismatch_raises():
    """value_proxy rejects mismatched list lengths."""
    with pytest.raises(ValueError, match="rewards length"):
        value_proxy([1.0, 0.0, 1.0], [10.0])
