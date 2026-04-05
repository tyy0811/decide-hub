"""Automation orchestrator: collect -> enrich -> rules -> permissions -> execute -> log.

Per-entity error handling: one failure doesn't kill the run.
Idempotency: DB unique constraint on (entity_id, processed_date) prevents duplicates.
"""

import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

from src.automations.enrichment import enrich_entity
from src.automations.rules import apply_rules, load_rules_config
from src.automations.permissions import check_permission, load_permissions_config
from src.evaluation.comparison import compute_action_deltas, total_variation_distance
from src.telemetry import db
from src.telemetry.audit import log_audit_event
from src.serving.ws import ws_manager
from src.telemetry.metrics import (
    automation_runs, rule_hits, permission_results,
    failed_entities_counter, enrichment_duration,
)

_PERMISSION_TO_AUDIT = {
    "allowed": "execute",
    "blocked": "block",
    "approval_required": "queue_approval",
}


async def _record_shadow_if_needed(
    shadow_action: str | None,
    shadow_rule: str | None,
    shadow_action_counts: Counter,
    run_id: str,
    entity_id: str,
    production_action: str,
    production_rule: str,
) -> None:
    """Record shadow outcome only after a successful idempotent insert.

    This ensures shadow accounting matches production dedupe semantics:
    duplicate/replayed entities that production discards are also excluded
    from shadow TVD and delta calculations.
    """
    if shadow_action is None:
        return
    shadow_action_counts[shadow_action] += 1
    try:
        await db.insert_shadow_outcome(
            run_id=run_id,
            entity_id=entity_id,
            production_action=production_action,
            shadow_action=shadow_action,
            production_rule=production_rule,
            shadow_rule=shadow_rule,
        )
    except Exception as e:
        print(f"Shadow outcome logging failed: {e}", file=sys.stderr)


async def _emit_post_insert(
    entity_id: str, run_id: str, action: str, rule_name: str, permission: str,
) -> None:
    """Emit audit event and WS broadcast after a successful idempotent insert.

    Called only when insert_outcome_idempotent returns True, so audit/telemetry
    reflects committed state transitions only — no false events for duplicates.
    """
    await log_audit_event(
        entity_id=entity_id,
        run_id=run_id,
        actor="system",
        action_type=_PERMISSION_TO_AUDIT.get(permission, "unknown"),
        action=action,
        rule_matched=rule_name,
        permission_result=permission,
        reason=None,
    )
    await ws_manager.broadcast({
        "event": "entity_processed",
        "run_id": run_id,
        "entity_id": entity_id,
        "action": action,
        "permission": permission,
    })


async def run_automation_pipeline(
    entities: list[dict],
    run_id: str,
    dry_run: bool = False,
    shadow_rules_config: str | None = None,
    suppress_failure_logging: bool = False,
) -> dict:
    """Run the full automation pipeline on a list of entities.

    Returns run summary dict.
    """
    today = date.today()
    rules = load_rules_config()
    permissions = load_permissions_config()
    shadow_rules = None
    if shadow_rules_config:
        shadow_path = Path(shadow_rules_config).resolve()
        allowed_dir = Path(__file__).resolve().parent
        if not str(shadow_path).startswith(str(allowed_dir)):
            raise ValueError(
                f"shadow_rules_config must be under {allowed_dir}, "
                f"got {shadow_path}"
            )
        shadow_rules = load_rules_config(path=shadow_path)
    shadow_action_counts: Counter = Counter()

    if not dry_run:
        await db.create_run(run_id)
        await ws_manager.broadcast({
            "event": "run_started",
            "run_id": run_id,
            "entity_count": len(entities),
        })

    processed = 0
    failed = 0
    action_counts: Counter = Counter()
    results = []

    for raw_entity in entities:
        entity_id = raw_entity.get("entity_id", "unknown")
        _error_category = "unknown_error"

        try:
            # Enrich
            _error_category = "enrichment_error"
            enrich_start = time.monotonic()
            enriched = enrich_entity(raw_entity, today=today)
            enrichment_duration.observe(time.monotonic() - enrich_start)

            # Apply rules (configs loaded once, not per entity)
            _error_category = "validation_error"
            action, rule_name = apply_rules(enriched, rules=rules)
            rule_hits.labels(action=action).inc()

            # Shadow mode: compute candidate action (logged after idempotent insert)
            shadow_action = None
            shadow_rule = None
            if shadow_rules is not None:
                shadow_action, shadow_rule = apply_rules(enriched, rules=shadow_rules)

            # Check permissions
            _error_category = "validation_error"
            permission = check_permission(action, permissions=permissions)
            permission_results.labels(result=permission).inc()

            if dry_run:
                action_counts[action] += 1
                # Dry runs have no idempotency, so shadow counts all entities
                if shadow_action is not None:
                    shadow_action_counts[shadow_action] += 1
                processed += 1
                results.append({
                    "entity_id": entity_id,
                    "action": action,
                    "permission": permission,
                    "rule": rule_name,
                })
                continue

            if permission == "blocked":
                inserted = await db.insert_outcome_idempotent(
                    run_id=run_id,
                    entity_id=entity_id,
                    enriched_fields=enriched.model_dump(),
                    action_taken=action,
                    rule_matched=rule_name,
                    permission_result="blocked",
                )
                if inserted:
                    await _emit_post_insert(entity_id, run_id, action, rule_name, permission)
                    await _record_shadow_if_needed(
                        shadow_action, shadow_rule, shadow_action_counts,
                        run_id, entity_id, action, rule_name,
                    )
                    action_counts[f"{action}:blocked"] += 1
                    processed += 1
                continue

            if permission == "approval_required":
                inserted = await db.insert_outcome_idempotent(
                    run_id=run_id,
                    entity_id=entity_id,
                    enriched_fields=enriched.model_dump(),
                    action_taken=action,
                    rule_matched=rule_name,
                    permission_result="approval_required",
                )
                if inserted:
                    await db.create_approval(
                        entity_id=entity_id,
                        proposed_action=action,
                        reason=f"Rule '{rule_name}' matched, requires approval",
                    )
                    await _emit_post_insert(entity_id, run_id, action, rule_name, permission)
                    await _record_shadow_if_needed(
                        shadow_action, shadow_rule, shadow_action_counts,
                        run_id, entity_id, action, rule_name,
                    )
                    action_counts[f"{action}:approval_required"] += 1
                    processed += 1
                continue

            if permission != "allowed":
                raise ValueError(
                    f"Unexpected permission level '{permission}' for action '{action}'. "
                    f"Expected 'allowed', 'blocked', or 'approval_required'."
                )

            # Permission is "allowed" — execute
            inserted = await db.insert_outcome_idempotent(
                run_id=run_id,
                entity_id=entity_id,
                enriched_fields=enriched.model_dump(),
                action_taken=action,
                rule_matched=rule_name,
                permission_result="allowed",
            )
            if inserted:
                await _emit_post_insert(entity_id, run_id, action, rule_name, permission)
                await _record_shadow_if_needed(
                    shadow_action, shadow_rule, shadow_action_counts,
                    run_id, entity_id, action, rule_name,
                )
                action_counts[action] += 1
                processed += 1

            results.append({
                "entity_id": entity_id,
                "action": action,
                "permission": permission,
                "rule": rule_name,
            })

        except Exception as e:
            failed += 1
            failed_entities_counter.labels(error_type=_error_category).inc()
            if not dry_run and not suppress_failure_logging:
                try:
                    await db.log_failed_entity(
                        entity_id=entity_id,
                        run_id=run_id,
                        error_type=_error_category,
                        error_message=f"{type(e).__name__}: {e}",
                        entity_data=raw_entity,
                    )
                except Exception:
                    pass  # Don't let failure logging kill the run

    # Compute shadow metrics before completing run (so they get persisted)
    shadow_tvd = None
    shadow_action_deltas_dict = None
    if shadow_rules is not None:
        production_counts: Counter = Counter()
        for ctx_action in action_counts:
            base = ctx_action.split(":")[0]
            production_counts[base] += action_counts[ctx_action]
        shadow_tvd = total_variation_distance(
            dict(production_counts), dict(shadow_action_counts),
        )
        shadow_action_deltas_dict = compute_action_deltas(
            dict(production_counts), dict(shadow_action_counts),
        )

    # Complete run (includes shadow summary for dashboard)
    automation_runs.labels(status="dry_run" if dry_run else "completed").inc()
    if not dry_run:
        await db.complete_run(
            run_id=run_id,
            entities_processed=processed,
            entities_failed=failed,
            action_distribution=dict(action_counts),
            shadow_tvd=shadow_tvd,
            shadow_action_deltas=shadow_action_deltas_dict,
        )
        await ws_manager.broadcast({
            "event": "run_completed",
            "run_id": run_id,
            "entities_processed": processed,
            "entities_failed": failed,
            "action_distribution": dict(action_counts),
        })

    result = {
        "run_id": run_id,
        "status": "completed",
        "entities_processed": processed,
        "entities_failed": failed,
        "action_distribution": dict(action_counts),
        "dry_run": dry_run,
        "results": results,
    }
    if shadow_tvd is not None:
        result["shadow_tvd"] = shadow_tvd
        result["shadow_action_deltas"] = shadow_action_deltas_dict

    return result
