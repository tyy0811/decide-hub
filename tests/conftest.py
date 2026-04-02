import asyncio
import asyncpg
import pytest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db_pool():
    """Create a test database pool and run schema."""
    dsn = "postgresql://decide_hub:decide_hub@localhost:5432/decide_hub"
    pool = await asyncpg.create_pool(dsn)
    # Run schema (idempotent due to IF NOT EXISTS)
    schema_sql = (_PROJECT_ROOT / "schema.sql").read_text()
    async with pool.acquire() as conn:
        await conn.execute(schema_sql)
    yield pool
    # Truncate all tables after test
    async with pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE outcomes, automation_outcomes, pending_approvals, "
            "failed_entities CASCADE"
        )
        await conn.execute("DELETE FROM automation_runs")
    await pool.close()
