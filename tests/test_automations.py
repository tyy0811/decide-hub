import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from tests.mock_lead_api import MOCK_LEADS
from src.automations.crawler import fetch_entities


@pytest.mark.asyncio
async def test_fetch_entities():
    """Crawler fetches entities from a URL."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"leads": MOCK_LEADS}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        entities = await fetch_entities("http://localhost:9999/leads")

    assert len(entities) == 5
    assert entities[0]["entity_id"] == "lead_001"


# --- Enrichment ---

from datetime import date
from src.automations.enrichment import enrich_entity, EnrichedEntity


def test_enrich_entity_basic():
    """Enrichment computes expected fields."""
    raw = {
        "entity_id": "lead_001",
        "company": "TechCorp",
        "role": "CTO",
        "source": "organic",
        "signup_date": "2026-03-20",
    }
    enriched = enrich_entity(raw, today=date(2026, 4, 3))
    assert isinstance(enriched, EnrichedEntity)
    assert enriched.entity_id == "lead_001"
    assert enriched.lead_score > 0
    assert enriched.days_since_signup == 14
    assert enriched.source_quality_tier in ("high", "medium", "low")


def test_enrich_entity_missing_company():
    """Missing company field is handled gracefully."""
    raw = {
        "entity_id": "lead_003",
        "company": "",
        "role": "PM",
        "source": "paid_ad",
        "signup_date": "2025-06-15",
    }
    enriched = enrich_entity(raw, today=date(2026, 4, 3))
    assert enriched.company_size_bucket == "unknown"
    assert enriched.has_missing_fields is True


def test_enrich_entity_high_value_lead():
    """CTO from organic source with recent signup gets high lead score."""
    raw = {
        "entity_id": "lead_001",
        "company": "TechCorp",
        "role": "CTO",
        "source": "organic",
        "signup_date": "2026-04-01",
    }
    enriched = enrich_entity(raw, today=date(2026, 4, 3))
    assert enriched.lead_score >= 70  # High-value signals


# --- Rules engine ---

from src.automations.rules import apply_rules, load_rules_config


def _make_entity(**overrides) -> EnrichedEntity:
    defaults = dict(
        entity_id="test_001", company="TestCo", role="Engineer",
        source="organic", signup_date="2026-04-01",
        company_size_bucket="mid", lead_score=50,
        days_since_signup=5, source_quality_tier="high",
        has_missing_fields=False, request_email=False,
    )
    defaults.update(overrides)
    return EnrichedEntity(**defaults)


def test_rules_priority_outreach():
    """High score + recent signup -> priority_outreach."""
    entity = _make_entity(lead_score=80, days_since_signup=3)
    action, rule = apply_rules(entity)
    assert action == "priority_outreach"


def test_rules_standard_sequence():
    """Medium score -> standard_sequence."""
    entity = _make_entity(lead_score=50, days_since_signup=20)
    action, rule = apply_rules(entity)
    assert action == "standard_sequence"


def test_rules_flag_for_review():
    """Missing critical field -> flag_for_review."""
    entity = _make_entity(has_missing_fields=True, lead_score=50)
    action, rule = apply_rules(entity)
    assert action == "flag_for_review"


def test_rules_deprioritize():
    """Low score + old signup -> deprioritize."""
    entity = _make_entity(lead_score=20, days_since_signup=200)
    action, rule = apply_rules(entity)
    assert action == "deprioritize"


def test_rules_send_external_email():
    """Entity requesting email -> send_external_email."""
    entity = _make_entity(lead_score=80, days_since_signup=3, request_email=True)
    action, rule = apply_rules(entity)
    assert action == "send_external_email"


def test_load_rules_config():
    """Rules config loads from YAML."""
    rules = load_rules_config()
    assert len(rules) > 0
    assert all("action" in r for r in rules)


def test_rules_missing_fields_blocks_email():
    """Missing fields + email request -> flag_for_review, NOT send_external_email."""
    entity = _make_entity(has_missing_fields=True, request_email=True, lead_score=80)
    action, rule = apply_rules(entity)
    assert action == "flag_for_review", (
        f"Expected flag_for_review for incomplete entity, got {action}"
    )


def test_rules_malformed_condition_raises():
    """Unrecognized condition syntax raises ValueError, not silent pass."""
    from src.automations.rules import _evaluate_condition
    entity = _make_entity()
    with pytest.raises(ValueError, match="Unrecognized condition"):
        _evaluate_condition("unknown_garbage", entity)


def test_rules_unknown_field_raises():
    """Comparison on a nonexistent field raises ValueError."""
    from src.automations.rules import _evaluate_condition
    entity = _make_entity()
    with pytest.raises(ValueError, match="Unknown field"):
        _evaluate_condition("nonexistent_field >= 5", entity)


def test_enrich_unknown_company_defaults_to_unknown():
    """Unseen company name defaults to 'unknown', not 'mid'."""
    raw = {
        "entity_id": "lead_new",
        "company": "NeverSeenBeforeCo",
        "role": "Engineer",
        "source": "organic",
        "signup_date": "2026-04-01",
    }
    enriched = enrich_entity(raw, today=date(2026, 4, 3))
    assert enriched.company_size_bucket == "unknown"


# --- Permissions ---

from src.automations.permissions import check_permission, load_permissions_config


def test_permission_allowed():
    result = check_permission("priority_outreach")
    assert result == "allowed"


def test_permission_blocked():
    result = check_permission("delete_lead")
    assert result == "blocked"


def test_permission_approval_required():
    result = check_permission("send_external_email")
    assert result == "approval_required"


def test_permission_unknown_action_blocked():
    """Unknown actions default to blocked."""
    result = check_permission("totally_unknown_action")
    assert result == "blocked"


def test_all_rule_actions_have_permissions():
    """Cross-config validation: every rule action has a permissions entry."""
    rule_actions = {r["action"] for r in load_rules_config()}
    perm_actions = set(load_permissions_config().keys())
    missing = rule_actions - perm_actions
    assert not missing, f"Actions without permissions: {missing}"


# --- Orchestrator ---

from src.automations.orchestrator import run_automation_pipeline


@pytest.mark.asyncio
async def test_orchestrator_processes_entities(db_pool):
    """Orchestrator processes entities through full pipeline."""
    from src.telemetry import db as db_module
    db_module._pool = db_pool

    result = await run_automation_pipeline(
        entities=MOCK_LEADS[:2],  # Two normal leads
        run_id="test_run_001",
        dry_run=False,
    )

    assert result["status"] == "completed"
    assert result["entities_processed"] == 2
    assert result["entities_failed"] == 0


@pytest.mark.asyncio
async def test_orchestrator_approval_required(db_pool):
    """Approval-required action appears in pending_approvals table."""
    from src.telemetry import db as db_module
    db_module._pool = db_pool

    # Lead 005 has request_email=True -> send_external_email -> approval_required
    email_lead = MOCK_LEADS[4]

    result = await run_automation_pipeline(
        entities=[email_lead],
        run_id="test_run_approval",
        dry_run=False,
    )

    approvals = await db_module.get_pending_approvals()
    email_approvals = [a for a in approvals if a["entity_id"] == "lead_005"]
    assert len(email_approvals) == 1
    assert email_approvals[0]["proposed_action"] == "send_external_email"


@pytest.mark.asyncio
async def test_orchestrator_idempotent_rerun(db_pool):
    """Rerunning with same entities on same day does not duplicate rows."""
    from src.telemetry import db as db_module
    db_module._pool = db_pool

    entities = [MOCK_LEADS[0]]

    await run_automation_pipeline(entities=entities, run_id="test_idem_001", dry_run=False)
    await run_automation_pipeline(entities=entities, run_id="test_idem_002", dry_run=False)

    pool = await db_module.get_pool()
    outcomes = await pool.fetch(
        "SELECT * FROM automation_outcomes WHERE entity_id = 'lead_001'"
    )
    # Should have only 1 outcome (second run skipped due to idempotency)
    assert len(outcomes) == 1


@pytest.mark.asyncio
async def test_orchestrator_failed_entity_logged(db_pool):
    """Failed entity is logged with correct error_type."""
    from src.telemetry import db as db_module
    db_module._pool = db_pool

    bad_entity = {"entity_id": "lead_bad", "signup_date": "not-a-date", "role": None}  # Will fail in enrichment

    result = await run_automation_pipeline(
        entities=[bad_entity],
        run_id="test_run_fail",
        dry_run=False,
    )

    assert result["entities_failed"] == 1
    failed = await db_module.get_failed_entities(run_id="test_run_fail")
    assert len(failed) == 1


@pytest.mark.asyncio
async def test_orchestrator_dry_run(db_pool):
    """Dry run returns what would happen without executing."""
    from src.telemetry import db as db_module
    db_module._pool = db_pool

    result = await run_automation_pipeline(
        entities=MOCK_LEADS[:2],
        run_id="test_dry_run",
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["entities_processed"] == 2

    # Dry run should NOT write to automation_outcomes
    pool = await db_module.get_pool()
    outcomes = await pool.fetch(
        "SELECT * FROM automation_outcomes WHERE run_id = 'test_dry_run'"
    )
    assert len(outcomes) == 0
