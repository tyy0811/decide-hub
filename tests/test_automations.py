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
