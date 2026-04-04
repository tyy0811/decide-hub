"""Tests for shadow mode — candidate rules run alongside production."""

import yaml
import pytest

from src.automations.orchestrator import run_automation_pipeline
from tests.mock_lead_api import MOCK_LEADS


def _write_shadow_config(tmp_path, rules: list[dict]) -> str:
    """Write a temporary rules config and return its path."""
    path = tmp_path / "shadow_rules.yml"
    path.write_text(yaml.dump({"rules": rules}))
    return str(path)


@pytest.mark.asyncio
async def test_shadow_mode_no_divergence(db_pool):
    """Same rules config produces zero shadow TVD."""
    from src.telemetry import db as db_module
    db_module._pool = db_pool

    result = await run_automation_pipeline(
        entities=MOCK_LEADS[:2],
        run_id="test_shadow_same",
        dry_run=False,
        shadow_rules_config="src/automations/rules_config.yml",
    )

    assert "shadow_tvd" in result
    assert result["shadow_tvd"] == 0.0


@pytest.mark.asyncio
async def test_shadow_mode_detects_divergence(db_pool, tmp_path):
    """Different rules config produces nonzero shadow TVD."""
    from src.telemetry import db as db_module
    db_module._pool = db_pool

    # Route everything to priority_outreach — diverges from the varied
    # production actions (standard_sequence, flag_for_review, delete_lead, etc.)
    shadow_path = _write_shadow_config(tmp_path, [
        {"name": "catch_all", "condition": "true", "action": "priority_outreach"},
    ])

    result = await run_automation_pipeline(
        entities=MOCK_LEADS[:4],
        run_id="test_shadow_diff",
        dry_run=False,
        shadow_rules_config=shadow_path,
    )

    assert result["shadow_tvd"] > 0.0
    assert len(result["shadow_action_deltas"]) > 0


@pytest.mark.asyncio
async def test_shadow_mode_writes_to_db(db_pool, tmp_path):
    """Shadow outcomes are written to shadow_outcomes table."""
    from src.telemetry import db as db_module
    db_module._pool = db_pool

    # Route everything to priority_outreach — leads 3-4 get different
    # production actions so will show as diverged
    shadow_path = _write_shadow_config(tmp_path, [
        {"name": "catch_all", "condition": "true", "action": "priority_outreach"},
    ])

    result = await run_automation_pipeline(
        entities=MOCK_LEADS[:4],
        run_id="test_shadow_db",
        dry_run=False,
        shadow_rules_config=shadow_path,
    )

    pool = db_module.get_pool()
    rows = await pool.fetch(
        "SELECT * FROM shadow_outcomes WHERE run_id = 'test_shadow_db'"
    )
    assert len(rows) == 4
    # Leads with non-priority_outreach production actions should diverge
    diverged = [r for r in rows if r["diverged"]]
    assert len(diverged) > 0


@pytest.mark.asyncio
async def test_shadow_mode_not_active_without_config(db_pool):
    """No shadow fields in result when shadow_rules_config is None."""
    from src.telemetry import db as db_module
    db_module._pool = db_pool

    result = await run_automation_pipeline(
        entities=MOCK_LEADS[:2],
        run_id="test_shadow_none",
        dry_run=False,
    )

    assert "shadow_tvd" not in result


@pytest.mark.asyncio
async def test_shadow_does_not_apply_permissions(db_pool):
    """Shadow side logs raw rule output, not post-permission output."""
    from src.telemetry import db as db_module
    db_module._pool = db_pool

    # Use production rules as shadow — lead_005 triggers send_external_email
    # which is approval_required in permissions, but shadow should still log
    # send_external_email as the shadow_action
    result = await run_automation_pipeline(
        entities=[MOCK_LEADS[4]],  # lead_005 with request_email=True
        run_id="test_shadow_noperm",
        dry_run=False,
        shadow_rules_config="src/automations/rules_config.yml",
    )

    pool = db_module.get_pool()
    rows = await pool.fetch(
        "SELECT * FROM shadow_outcomes WHERE run_id = 'test_shadow_noperm'"
    )
    assert len(rows) == 1
    assert rows[0]["shadow_action"] == "send_external_email"
