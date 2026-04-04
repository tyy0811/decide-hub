"""Tests for policy replay runner."""

import json
import tempfile
from pathlib import Path

import yaml

from src.automations.enrichment import EnrichedEntity
from src.evaluation.replay import replay_contexts, ReplayResult


def _make_contexts():
    """Build minimal frozen contexts for testing."""
    enriched_base = dict(
        company="TestCo", role="Engineer", source="organic",
        signup_date="2026-04-01", company_size_bucket="mid",
        lead_score=80, days_since_signup=3,
        source_quality_tier="high", has_missing_fields=False,
        request_email=False,
    )
    return [
        {
            "entity": {"entity_id": "e1", "company": "TestCo", "role": "Engineer", "source": "organic", "signup_date": "2026-04-01"},
            "enriched": {**enriched_base, "entity_id": "e1"},
            "action": "priority_outreach",
            "rule_matched": "high_value_recent",
        },
        {
            "entity": {"entity_id": "e2", "company": "TestCo", "role": "Engineer", "source": "organic", "signup_date": "2026-04-01"},
            "enriched": {**enriched_base, "entity_id": "e2"},
            "action": "priority_outreach",
            "rule_matched": "high_value_recent",
        },
    ]


def test_replay_no_drift():
    """Same rules config produces zero TVD."""
    contexts = _make_contexts()
    # Use the production rules — should match frozen actions
    result = replay_contexts(contexts, "src/automations/rules_config.yml")
    assert isinstance(result, ReplayResult)
    assert result.tvd == 0.0
    assert len(result.per_entity_changes) == 0


def test_replay_detects_drift():
    """Modified rules config produces nonzero TVD and per-entity changes."""
    contexts = _make_contexts()

    # Create a rules config that routes everything to standard_sequence
    modified_rules = {
        "rules": [
            {"name": "catch_all", "condition": "true", "action": "standard_sequence"},
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        yaml.dump(modified_rules, f)
        tmp_path = f.name

    result = replay_contexts(contexts, tmp_path)
    assert result.tvd > 0.0
    assert len(result.per_entity_changes) == 2
    for change in result.per_entity_changes:
        assert change["baseline_action"] == "priority_outreach"
        assert change["candidate_action"] == "standard_sequence"
        assert "candidate_rule_matched" in change
        assert "entity_id" in change


def test_replay_result_action_deltas():
    """Action deltas show directional shift."""
    contexts = _make_contexts()

    modified_rules = {
        "rules": [
            {"name": "catch_all", "condition": "true", "action": "standard_sequence"},
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        yaml.dump(modified_rules, f)
        tmp_path = f.name

    result = replay_contexts(contexts, tmp_path)
    assert result.action_deltas["standard_sequence"] > 0
    assert result.action_deltas["priority_outreach"] < 0
