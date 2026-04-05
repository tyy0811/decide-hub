import asyncio
import json
import os
import asyncpg
import pytest
from pathlib import Path

# Allow default JWT secret in tests
os.environ.setdefault("ALLOW_INSECURE_AUTH", "true")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Mirror the JSONB codec from src/telemetry/db.py."""
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db_pool():
    """Create a test database pool and run schema."""
    dsn = "postgresql://decide_hub:decide_hub@localhost:5432/decide_hub"
    pool = await asyncpg.create_pool(dsn, init=_init_connection)
    # Run schema (idempotent due to IF NOT EXISTS)
    schema_sql = (_PROJECT_ROOT / "schema.sql").read_text()
    async with pool.acquire() as conn:
        await conn.execute(schema_sql)
    yield pool
    # Truncate all tables and reset caches after test
    async with pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE outcomes, automation_outcomes, pending_approvals, "
            "failed_entities, shadow_outcomes, action_audit_log CASCADE"
        )
        await conn.execute("DELETE FROM automation_runs")
    from src.telemetry.db import _reset_retry_config
    _reset_retry_config()
    await pool.close()
