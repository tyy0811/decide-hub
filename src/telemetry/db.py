"""asyncpg database layer — pool + parameterized query helpers."""

import json
import asyncpg
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_pool: asyncpg.Pool | None = None


async def init_pool(dsn: str, run_schema_on_connect: bool = True) -> asyncpg.Pool:
    global _pool
    _pool = await asyncpg.create_pool(dsn, init=_init_connection)
    if run_schema_on_connect:
        await run_schema()
    return _pool


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Per-connection setup: register JSONB codec so JSONB columns return dict."""
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_pool() first.")
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def run_schema(schema_path: str | None = None) -> None:
    pool = get_pool()
    path = Path(schema_path) if schema_path else _PROJECT_ROOT / "schema.sql"
    sql = path.read_text()
    async with pool.acquire() as conn:
        await conn.execute(sql)


# --- Outcomes (ranking) ---

async def log_outcome(user_id: int, action: str, reward: float, policy_id: str) -> None:
    pool = get_pool()
    await pool.execute(
        "INSERT INTO outcomes (user_id, action, reward, policy_id) "
        "VALUES ($1, $2, $3, $4)",
        user_id, action, reward, policy_id,
    )


# --- Automation runs ---

async def create_run(run_id: str) -> None:
    pool = get_pool()
    await pool.execute(
        "INSERT INTO automation_runs (run_id, status) VALUES ($1, 'running')",
        run_id,
    )


async def complete_run(
    run_id: str,
    entities_processed: int,
    entities_failed: int,
    action_distribution: dict,
) -> None:
    pool = get_pool()
    await pool.execute(
        "UPDATE automation_runs SET status = 'completed', "
        "entities_processed = $2, entities_failed = $3, "
        "action_distribution = $4, completed_at = NOW() "
        "WHERE run_id = $1",
        run_id, entities_processed, entities_failed,
        action_distribution,
    )


async def get_runs(limit: int = 20) -> list[dict]:
    pool = get_pool()
    rows = await pool.fetch(
        "SELECT * FROM automation_runs ORDER BY started_at DESC LIMIT $1",
        limit,
    )
    return [dict(r) for r in rows]


# --- Automation outcomes ---

async def log_automation_outcome(
    run_id: str,
    entity_id: str,
    enriched_fields: dict,
    action_taken: str,
    rule_matched: str | None,
    permission_result: str,
) -> None:
    pool = get_pool()
    await pool.execute(
        "INSERT INTO automation_outcomes "
        "(run_id, entity_id, enriched_fields, action_taken, rule_matched, permission_result) "
        "VALUES ($1, $2, $3, $4, $5, $6)",
        run_id, entity_id, enriched_fields,
        action_taken, rule_matched, permission_result,
    )


# --- Pending approvals ---

async def create_approval(
    entity_id: str, proposed_action: str, reason: str | None = None,
) -> None:
    pool = get_pool()
    await pool.execute(
        "INSERT INTO pending_approvals (entity_id, proposed_action, reason) "
        "VALUES ($1, $2, $3)",
        entity_id, proposed_action, reason,
    )


async def get_pending_approvals() -> list[dict]:
    pool = get_pool()
    rows = await pool.fetch(
        "SELECT * FROM pending_approvals WHERE status = 'pending' "
        "ORDER BY created_at DESC",
    )
    return [dict(r) for r in rows]


async def get_approval_by_id(approval_id: int) -> dict | None:
    pool = get_pool()
    row = await pool.fetchrow(
        "SELECT * FROM pending_approvals WHERE id = $1", approval_id,
    )
    return dict(row) if row else None


async def update_approval_status(approval_id: int, status: str) -> None:
    pool = get_pool()
    await pool.execute(
        "UPDATE pending_approvals SET status = $1 WHERE id = $2",
        status, approval_id,
    )


# --- Failed entities ---

async def log_failed_entity(
    entity_id: str, run_id: str, error_type: str, error_message: str,
) -> None:
    pool = get_pool()
    await pool.execute(
        "INSERT INTO failed_entities (entity_id, run_id, error_type, error_message) "
        "VALUES ($1, $2, $3, $4)",
        entity_id, run_id, error_type, error_message,
    )


async def get_failed_entities(run_id: str | None = None) -> list[dict]:
    pool = get_pool()
    if run_id:
        rows = await pool.fetch(
            "SELECT * FROM failed_entities WHERE run_id = $1 "
            "ORDER BY created_at DESC",
            run_id,
        )
    else:
        rows = await pool.fetch(
            "SELECT * FROM failed_entities ORDER BY created_at DESC LIMIT 100",
        )
    return [dict(r) for r in rows]


# --- Idempotency ---

async def insert_outcome_idempotent(
    run_id: str,
    entity_id: str,
    enriched_fields: dict,
    action_taken: str,
    rule_matched: str | None,
    permission_result: str,
) -> bool:
    """Insert automation outcome, skipping if entity already processed today.

    Uses INSERT ... ON CONFLICT DO NOTHING on the unique constraint
    (entity_id, processed_date). Returns True if the row was inserted,
    False if it was a duplicate.
    """
    pool = get_pool()
    result = await pool.execute(
        "INSERT INTO automation_outcomes "
        "(run_id, entity_id, enriched_fields, action_taken, rule_matched, permission_result) "
        "VALUES ($1, $2, $3, $4, $5, $6) "
        "ON CONFLICT (entity_id, processed_date) DO NOTHING",
        run_id, entity_id, enriched_fields,
        action_taken, rule_matched, permission_result,
    )
    # asyncpg returns "INSERT 0 1" on success, "INSERT 0 0" on conflict
    return result == "INSERT 0 1"


# --- Shadow outcomes ---

async def insert_shadow_outcome(
    run_id: str,
    entity_id: str,
    production_action: str,
    shadow_action: str,
    production_rule: str,
    shadow_rule: str,
) -> None:
    pool = get_pool()
    await pool.execute(
        "INSERT INTO shadow_outcomes "
        "(run_id, entity_id, production_action, shadow_action, "
        "production_rule, shadow_rule, diverged) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7)",
        run_id, entity_id, production_action, shadow_action,
        production_rule, shadow_rule,
        production_action != shadow_action,
    )


async def get_shadow_outcomes(run_id: str) -> list[dict]:
    pool = get_pool()
    rows = await pool.fetch(
        "SELECT * FROM shadow_outcomes WHERE run_id = $1 ORDER BY id",
        run_id,
    )
    return [dict(r) for r in rows]
