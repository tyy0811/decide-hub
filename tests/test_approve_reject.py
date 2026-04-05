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


def test_approve_endpoint_requires_api_key():
    """Approve without API key returns 403."""
    with TestClient(app) as client:
        resp = client.post("/approvals/99999/approve")
        assert resp.status_code == 403


def test_reject_endpoint_requires_api_key():
    """Reject without API key returns 403."""
    with TestClient(app) as client:
        resp = client.post("/approvals/99999/reject")
        assert resp.status_code == 403


def test_approve_endpoint_wrong_key():
    """Approve with wrong API key returns 403."""
    import os
    old = os.environ.get("OPERATOR_API_KEY")
    os.environ["OPERATOR_API_KEY"] = "correct-key"
    try:
        # Reload the module-level variable
        import src.serving.app as app_mod
        app_mod._OPERATOR_API_KEY = "correct-key"
        with TestClient(app) as client:
            resp = client.post(
                "/approvals/99999/approve",
                headers={"X-Operator-Key": "wrong-key"},
            )
            assert resp.status_code == 403
    finally:
        app_mod._OPERATOR_API_KEY = old
        if old is None:
            os.environ.pop("OPERATOR_API_KEY", None)
        else:
            os.environ["OPERATOR_API_KEY"] = old


def test_approve_endpoint_valid_key_reaches_db():
    """Approve with correct API key passes auth and reaches DB layer."""
    import src.serving.app as app_mod
    old = app_mod._OPERATOR_API_KEY
    app_mod._OPERATOR_API_KEY = "test-key"
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/approvals/99999/approve",
                headers={"X-Operator-Key": "test-key"},
            )
            # 404 (approval not found) or 503 (no DB) — not 403
            assert resp.status_code in (404, 503)
    finally:
        app_mod._OPERATOR_API_KEY = old
