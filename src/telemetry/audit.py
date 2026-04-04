"""Audit trail — log every action decision with actor, type, and reason.

Fire-and-forget: failures logged to stderr, never block the pipeline.
"""

import sys
from src.telemetry.db import get_pool


async def log_audit_event(
    *,
    entity_id: str,
    run_id: str | None,
    actor: str,
    action_type: str,
    action: str,
    rule_matched: str | None,
    permission_result: str | None,
    reason: str | None,
) -> None:
    """Insert an audit log entry. Never raises — prints to stderr on failure."""
    try:
        pool = get_pool()
        await pool.execute(
            "INSERT INTO action_audit_log "
            "(entity_id, run_id, actor, action_type, action, "
            "rule_matched, permission_result, reason) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
            entity_id, run_id, actor, action_type, action,
            rule_matched, permission_result, reason,
        )
    except Exception as e:
        print(f"Audit log failed: {e}", file=sys.stderr)
