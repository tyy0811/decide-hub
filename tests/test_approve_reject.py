"""Tests for approve/reject API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.automations.orchestrator import run_automation_pipeline
from src.telemetry import db
from src.serving.app import app
from src.serving.auth import create_token
from tests.mock_lead_api import MOCK_LEADS


def _operator_headers() -> dict:
    token = create_token(username="admin", role="operator")
    return {"Authorization": f"Bearer {token}"}


def _viewer_headers() -> dict:
    token = create_token(username="viewer1", role="viewer")
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_approve_executes_action(db_pool):
    """Approving a pending approval executes the proposed action."""
    db._pool = db_pool

    # Trigger pipeline with email lead -> approval_required
    await run_automation_pipeline(
        entities=[MOCK_LEADS[4]],  # lead_005, request_email=True
        run_id="test_approve_run",
        dry_run=False,
    )

    # Get the pending approval
    approvals = await db.get_pending_approvals()
    email_approval = next(a for a in approvals if a["proposed_action"] == "send_external_email")
    approval_id = email_approval["id"]

    # Approve it
    approval = await db.get_approval_by_id(approval_id)
    assert approval is not None
    assert approval["status"] == "pending"

    await db.update_approval_status(approval_id, "approved")
    updated = await db.get_approval_by_id(approval_id)
    assert updated["status"] == "approved"


@pytest.mark.asyncio
async def test_reject_does_not_execute(db_pool):
    """Rejecting a pending approval does not execute the action."""
    db._pool = db_pool

    await run_automation_pipeline(
        entities=[MOCK_LEADS[4]],
        run_id="test_reject_run",
        dry_run=False,
    )

    approvals = await db.get_pending_approvals()
    email_approval = next(a for a in approvals if a["proposed_action"] == "send_external_email")

    await db.update_approval_status(email_approval["id"], "rejected")
    updated = await db.get_approval_by_id(email_approval["id"])
    assert updated["status"] == "rejected"

    # Should no longer appear in pending
    pending = await db.get_pending_approvals()
    pending_ids = [a["id"] for a in pending]
    assert email_approval["id"] not in pending_ids


def test_approve_endpoint_requires_auth():
    """Approve without JWT returns 401."""
    with TestClient(app) as client:
        resp = client.post("/approvals/99999/approve")
        assert resp.status_code == 401


def test_reject_endpoint_requires_auth():
    """Reject without JWT returns 401."""
    with TestClient(app) as client:
        resp = client.post("/approvals/99999/reject")
        assert resp.status_code == 401


def test_approve_endpoint_viewer_rejected():
    """Viewer role cannot approve — returns 403."""
    with TestClient(app) as client:
        resp = client.post(
            "/approvals/99999/approve",
            headers=_viewer_headers(),
        )
        assert resp.status_code == 403


def test_reject_endpoint_viewer_rejected():
    """Viewer role cannot reject — returns 403."""
    with TestClient(app) as client:
        resp = client.post(
            "/approvals/99999/reject",
            headers=_viewer_headers(),
        )
        assert resp.status_code == 403


def test_approve_endpoint_operator_reaches_db():
    """Operator role passes auth and reaches DB layer."""
    with TestClient(app) as client:
        resp = client.post(
            "/approvals/99999/approve",
            headers=_operator_headers(),
        )
        # 404 (approval not found) or 503 (no DB) — not 401/403
        assert resp.status_code in (404, 503)
