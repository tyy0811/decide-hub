"""Optional PostHog event tracking — no-op without POSTHOG_API_KEY.

Set POSTHOG_API_KEY and optionally POSTHOG_HOST to enable.
The app runs identically without PostHog configured.
"""

import os
import sys

_client = None
_enabled = False


def _init():
    global _client, _enabled
    api_key = os.environ.get("POSTHOG_API_KEY")
    if not api_key:
        return
    try:
        import posthog
        host = os.environ.get("POSTHOG_HOST", "https://app.posthog.com")
        _client = posthog
        _client.project_api_key = api_key
        _client.host = host
        _enabled = True
    except ImportError:
        print("PostHog package not installed, tracking disabled", file=sys.stderr)


def _get_client():
    return _client if _enabled else None


def capture_event(event: str, properties: dict, distinct_id: str = "system") -> None:
    """Capture a PostHog event. Silent no-op if PostHog is not configured."""
    if not _enabled or _client is None:
        return
    try:
        _client.capture(distinct_id=distinct_id, event=event, properties=properties)
    except Exception as e:
        print(f"PostHog capture failed: {e}", file=sys.stderr)


# Initialize on import
_init()
