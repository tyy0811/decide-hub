"""Tests for audit trail logging."""

import pytest
from src.telemetry.audit import log_audit_event


@pytest.mark.asyncio
async def test_audit_event_logged(db_pool):
    """Audit events are written to action_audit_log."""
    from src.telemetry import db as db_module
    db_module._pool = db_pool

    await log_audit_event(
        entity_id="test_entity",
        run_id="test_run",
        actor="system",
        action_type="execute",
        action="priority_outreach",
        rule_matched="high_value_recent",
        permission_result="allowed",
        reason=None,
    )

    pool = db_module.get_pool()
    rows = await pool.fetch("SELECT * FROM action_audit_log WHERE entity_id = 'test_entity'")
    assert len(rows) == 1
    assert rows[0]["actor"] == "system"
    assert rows[0]["action_type"] == "execute"
    assert rows[0]["action"] == "priority_outreach"


@pytest.mark.asyncio
async def test_audit_event_with_reason(db_pool):
    """Audit events can include a reason string."""
    from src.telemetry import db as db_module
    db_module._pool = db_pool

    await log_audit_event(
        entity_id="test_entity_2",
        run_id="test_run",
        actor="operator",
        action_type="approve",
        action="send_external_email",
        rule_matched="email_request",
        permission_result="approval_required",
        reason="Verified recipient domain",
    )

    pool = db_module.get_pool()
    rows = await pool.fetch("SELECT * FROM action_audit_log WHERE entity_id = 'test_entity_2'")
    assert rows[0]["reason"] == "Verified recipient domain"


@pytest.mark.asyncio
async def test_orchestrator_creates_audit_entries(db_pool):
    """Orchestrator logs audit events for each entity processed."""
    from src.telemetry import db as db_module
    from src.automations.orchestrator import run_automation_pipeline
    from tests.mock_lead_api import MOCK_LEADS
    db_module._pool = db_pool

    await run_automation_pipeline(
        entities=MOCK_LEADS[:3],
        run_id="test_audit_run",
        dry_run=False,
    )

    pool = db_module.get_pool()
    rows = await pool.fetch(
        "SELECT * FROM action_audit_log WHERE run_id = 'test_audit_run'"
    )
    assert len(rows) == 3  # One audit entry per entity
    actors = {r["actor"] for r in rows}
    assert actors == {"system"}
