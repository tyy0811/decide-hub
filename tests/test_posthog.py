"""Tests for PostHog event tracking."""

import os
from unittest.mock import patch, MagicMock

from src.telemetry.posthog import capture_event, _get_client


def test_capture_noop_without_api_key():
    """No-op when POSTHOG_API_KEY is not set."""
    with patch.dict(os.environ, {}, clear=True):
        # Should not raise
        capture_event("test_event", {"key": "value"})


def test_capture_calls_client_with_key():
    """Event is captured when API key is configured."""
    mock_client = MagicMock()
    with patch("src.telemetry.posthog._client", mock_client):
        with patch("src.telemetry.posthog._enabled", True):
            capture_event("rank_request", {"policy": "popularity", "user_id": 42})
            mock_client.capture.assert_called_once()
            call_kwargs = mock_client.capture.call_args
            assert call_kwargs[1]["event"] == "rank_request"


def test_capture_swallows_exceptions():
    """Capture never raises — failures are logged, not propagated."""
    mock_client = MagicMock()
    mock_client.capture.side_effect = RuntimeError("network error")
    with patch("src.telemetry.posthog._client", mock_client):
        with patch("src.telemetry.posthog._enabled", True):
            # Should not raise
            capture_event("test_event", {})
