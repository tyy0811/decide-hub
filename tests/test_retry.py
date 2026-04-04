"""Tests for retry logic and dead-letter queue."""

import pytest

from src.telemetry import db
from src.automations.orchestrator import run_automation_pipeline


@pytest.mark.asyncio
async def test_failed_entity_stores_raw_data(db_pool):
    """Failed entity stores raw pre-enrichment payload for retry."""
    db._pool = db_pool

    bad_entity = {"entity_id": "retry_test_1", "signup_date": "not-a-date", "role": None}

    await run_automation_pipeline(
        entities=[bad_entity],
        run_id="test_retry_store",
        dry_run=False,
    )

    failed = await db.get_failed_entities(run_id="test_retry_store")
    assert len(failed) == 1
    assert failed[0]["entity_data"] is not None
    assert failed[0]["entity_data"]["entity_id"] == "retry_test_1"


@pytest.mark.asyncio
async def test_retry_config_sets_max_retries(db_pool):
    """Failed entities get max_retries from retry config."""
    db._pool = db_pool

    bad_entity = {"entity_id": "retry_test_2", "signup_date": "not-a-date", "role": None}

    await run_automation_pipeline(
        entities=[bad_entity],
        run_id="test_retry_config",
        dry_run=False,
    )

    failed = await db.get_failed_entities(run_id="test_retry_config")
    assert len(failed) == 1
    # role=None triggers AttributeError during enrichment.
    # Not in retry_config.yml — falls through to default.max_retries = 0.
    assert failed[0]["max_retries"] == 0
    assert failed[0]["status"] == "dead_letter"


@pytest.mark.asyncio
async def test_get_retryable_entities(db_pool):
    """get_retryable_entities returns only entities eligible for retry."""
    db._pool = db_pool
    pool = db.get_pool()

    # Insert a retryable entity (retry_count < max_retries, status = 'failed')
    await pool.execute(
        "INSERT INTO automation_runs (run_id, status) VALUES ('retry_run', 'completed')"
    )
    await pool.execute(
        "INSERT INTO failed_entities (entity_id, run_id, error_type, error_message, "
        "retry_count, max_retries, status, entity_data) "
        "VALUES ('retryable_1', 'retry_run', 'fetch_timeout', 'timeout', 0, 2, 'failed', $1)",
        '{"entity_id": "retryable_1", "company": "TestCo", "role": "CTO", "source": "organic", "signup_date": "2026-04-01"}',
    )
    # Insert a dead-lettered entity
    await pool.execute(
        "INSERT INTO failed_entities (entity_id, run_id, error_type, error_message, "
        "retry_count, max_retries, status, entity_data) "
        "VALUES ('dead_1', 'retry_run', 'validation_error', 'bad data', 0, 0, 'dead_letter', $1)",
        '{"entity_id": "dead_1"}',
    )

    retryable = await db.get_retryable_entities()
    entity_ids = [r["entity_id"] for r in retryable]
    assert "retryable_1" in entity_ids
    assert "dead_1" not in entity_ids
