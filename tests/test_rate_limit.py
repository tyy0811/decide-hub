"""Tests for rate limiting and backpressure."""

import time
from src.serving.rate_limit import SlidingWindowRateLimiter, check_entity_cap


def test_rate_limiter_allows_within_limit():
    limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
    for _ in range(5):
        assert limiter.allow() is True


def test_rate_limiter_blocks_over_limit():
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
    assert limiter.allow() is True
    assert limiter.allow() is True
    assert limiter.allow() is False


def test_rate_limiter_resets_after_window():
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=0.1)
    assert limiter.allow() is True
    assert limiter.allow() is False
    time.sleep(0.15)
    assert limiter.allow() is True


def test_entity_cap_within_limit():
    assert check_entity_cap(50, max_entities=100) is True


def test_entity_cap_over_limit():
    assert check_entity_cap(150, max_entities=100) is False
