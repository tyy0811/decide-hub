import pytest
from src.telemetry import db


@pytest.mark.asyncio
async def test_log_outcome_and_retrieve(db_pool):
    db._pool = db_pool

    await db.log_outcome(user_id=1, action="movie_42", reward=5.0, policy_id="popularity_v1")
    await db.log_outcome(user_id=1, action="movie_99", reward=3.0, policy_id="popularity_v1")

    pool = db.get_pool()
    rows = await pool.fetch("SELECT * FROM outcomes WHERE user_id = 1")
    assert len(rows) == 2
    assert rows[0]["policy_id"] == "popularity_v1"


@pytest.mark.asyncio
async def test_create_and_complete_run(db_pool):
    db._pool = db_pool

    await db.create_run("run_001")
    await db.complete_run("run_001", entities_processed=10, entities_failed=2, action_distribution={"priority_outreach": 5, "deprioritize": 3})

    runs = await db.get_runs(limit=1)
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run_001"
    assert runs[0]["status"] == "completed"
    assert runs[0]["entities_processed"] == 10
    # JSONB round-trip: codec guarantees dict, never str
    ad = runs[0]["action_distribution"]
    assert isinstance(ad, dict), f"Expected dict, got {type(ad)}"
    assert ad == {"priority_outreach": 5, "deprioritize": 3}


@pytest.mark.asyncio
async def test_pending_approvals(db_pool):
    db._pool = db_pool

    await db.create_approval("entity_42", "send_external_email", reason="High-value lead")
    approvals = await db.get_pending_approvals()
    assert len(approvals) == 1
    assert approvals[0]["entity_id"] == "entity_42"
    assert approvals[0]["status"] == "pending"


@pytest.mark.asyncio
async def test_failed_entities(db_pool):
    db._pool = db_pool

    await db.create_run("run_fail_001")
    await db.log_failed_entity("entity_99", "run_fail_001", "validation_error", "Missing company field")

    failed = await db.get_failed_entities(run_id="run_fail_001")
    assert len(failed) == 1
    assert failed[0]["error_type"] == "validation_error"


@pytest.mark.asyncio
async def test_idempotent_outcome_insert(db_pool):
    """Second insert for same entity on same day is silently skipped."""
    db._pool = db_pool

    await db.create_run("run_idem_001")

    inserted = await db.insert_outcome_idempotent(
        run_id="run_idem_001", entity_id="ent_X",
        enriched_fields={"score": 42}, action_taken="priority_outreach",
        rule_matched="high_value", permission_result="allowed",
    )
    assert inserted is True

    # Same entity_id on the same calendar day — should be a no-op
    inserted2 = await db.insert_outcome_idempotent(
        run_id="run_idem_001", entity_id="ent_X",
        enriched_fields={"score": 99}, action_taken="deprioritize",
        rule_matched="low_value", permission_result="allowed",
    )
    assert inserted2 is False

    pool = db.get_pool()
    rows = await pool.fetch(
        "SELECT * FROM automation_outcomes WHERE entity_id = 'ent_X'"
    )
    assert len(rows) == 1
    assert rows[0]["action_taken"] == "priority_outreach"
