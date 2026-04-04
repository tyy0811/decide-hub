"""Rate limiting and backpressure for automation endpoints.

- SlidingWindowRateLimiter: in-memory deque-based per-endpoint rate limit
- check_entity_cap: per-run entity count validation
- check_backpressure: DB-based write rate check
"""

import time
from collections import deque


class SlidingWindowRateLimiter:
    """Sliding window rate limiter using a deque of timestamps."""

    def __init__(self, max_requests: int = 5, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()

    def allow(self) -> bool:
        now = time.monotonic()
        # Remove expired timestamps
        while self._timestamps and now - self._timestamps[0] > self.window_seconds:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_requests:
            return False
        self._timestamps.append(now)
        return True

    def retry_after(self) -> float:
        """Seconds until the next request would be allowed."""
        if not self._timestamps:
            return 0.0
        oldest = self._timestamps[0]
        return max(0.0, self.window_seconds - (time.monotonic() - oldest))


def check_entity_cap(entity_count: int, max_entities: int = 100) -> bool:
    """Check if entity count is within the per-run cap."""
    return entity_count <= max_entities


async def check_backpressure(threshold: int = 500) -> bool:
    """Check if automation_outcomes write rate exceeds threshold.

    Returns True if backpressure is detected (should reject request).
    """
    from src.telemetry.db import get_pool
    pool = get_pool()
    count = await pool.fetchval(
        "SELECT count(*) FROM automation_outcomes "
        "WHERE created_at > now() - interval '1 minute'"
    )
    return count > threshold
