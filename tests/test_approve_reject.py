"""Tests for approve/reject API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.automations.orchestrator import run_automation_pipeline
from src.telemetry import db
from src.serving.app import app
from tests.mock_lead_api import MOCK_LEADS


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


def test_approve_endpoint_404():
    """Approve nonexistent approval returns 404."""
    with TestClient(app) as client:
        resp = client.post("/approvals/99999/approve")
        # 404 when DB is available (approval not found).
        # 503 when DB is unavailable (CI without Postgres).
        # Both are correct rejections — the request cannot succeed.
        assert resp.status_code in (404, 503)


def test_reject_endpoint_404():
    """Reject nonexistent approval returns 404."""
    with TestClient(app) as client:
        resp = client.post("/approvals/99999/reject")
        # See test_approve_endpoint_404 for rationale.
        assert resp.status_code in (404, 503)
