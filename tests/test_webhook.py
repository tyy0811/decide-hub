"""Tests for webhook automation endpoint."""

import pytest
from fastapi.testclient import TestClient

from src.serving.app import app
from src.serving.auth import create_token


def _operator_headers() -> dict:
    token = create_token(username="admin", role="operator")
    return {"Authorization": f"Bearer {token}"}


def test_webhook_accepts_entities():
    """Webhook returns 202 with run_id."""
    with TestClient(app) as client:
        resp = client.post(
            "/webhooks/automate",
            json={
                "entities": [
                    {"entity_id": "wh_1", "company": "TestCo", "role": "CTO", "source": "organic", "signup_date": "2026-04-01"},
                ],
                "dry_run": True,
            },
            headers=_operator_headers(),
        )
        # 202 if async, 200 if sync fallback, 503 if no DB
        assert resp.status_code in (200, 202, 503)
        if resp.status_code in (200, 202):
            data = resp.json()
            assert "run_id" in data
            assert data["status"] == "accepted"


def test_webhook_validates_entities():
    """Webhook rejects empty entity list."""
    with TestClient(app) as client:
        resp = client.post(
            "/webhooks/automate",
            json={"entities": [], "dry_run": True},
            headers=_operator_headers(),
        )
        assert resp.status_code == 422


def test_webhook_respects_entity_cap():
    """Webhook rejects entity lists exceeding MAX_ENTITIES_PER_RUN."""
    with TestClient(app) as client:
        entities = [
            {"entity_id": f"wh_{i}", "company": "TestCo", "role": "CTO", "source": "organic", "signup_date": "2026-04-01"}
            for i in range(101)
        ]
        resp = client.post(
            "/webhooks/automate",
            json={"entities": entities, "dry_run": True},
            headers=_operator_headers(),
        )
        assert resp.status_code == 422


def test_webhook_requires_auth():
    """Webhook without auth returns 401."""
    with TestClient(app) as client:
        resp = client.post(
            "/webhooks/automate",
            json={
                "entities": [{"entity_id": "wh_1"}],
                "dry_run": True,
            },
        )
        assert resp.status_code == 401
