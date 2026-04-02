import pytest
from src.telemetry import db


@pytest.mark.asyncio
async def test_log_outcome_and_retrieve(db_pool):
    db._pool = db_pool

    await db.log_outcome(user_id=1, action="movie_42", reward=5.0, policy_id="popularity_v1")
    await db.log_outcome(user_id=1, action="movie_99", reward=3.0, policy_id="popularity_v1")

    pool = await db.get_pool()
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
