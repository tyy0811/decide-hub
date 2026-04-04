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
from src.telemetry.metrics import (
    automation_runs, rule_hits, permission_results,
    failed_entities_counter, enrichment_duration,
)

_PERMISSION_TO_AUDIT = {
    "allowed": "execute",
    "blocked": "block",
    "approval_required": "queue_approval",
}


async def run_automation_pipeline(
    entities: list[dict],
    run_id: str,
    dry_run: bool = False,
    shadow_rules_config: str | None = None,
) -> dict:
    """Run the full automation pipeline on a list of entities.

    Returns run summary dict.
    """
    today = date.today()
    rules = load_rules_config()
    permissions = load_permissions_config()
    shadow_rules = (
        load_rules_config(path=Path(shadow_rules_config))
        if shadow_rules_config
        else None
    )
    shadow_action_counts: Counter = Counter()

    if not dry_run:
        await db.create_run(run_id)

    processed = 0
    failed = 0
    action_counts: Counter = Counter()
    results = []

    for raw_entity in entities:
        entity_id = raw_entity.get("entity_id", "unknown")

        try:
            # Enrich
            enrich_start = time.monotonic()
            enriched = enrich_entity(raw_entity, today=today)
            enrichment_duration.observe(time.monotonic() - enrich_start)

            # Apply rules (configs loaded once, not per entity)
            action, rule_name = apply_rules(enriched, rules=rules)
            rule_hits.labels(action=action).inc()

            # Shadow mode: run candidate rules (no permissions applied)
            if shadow_rules is not None:
                shadow_action, shadow_rule = apply_rules(enriched, rules=shadow_rules)
                shadow_action_counts[shadow_action] += 1
                if not dry_run:
                    try:
                        await db.insert_shadow_outcome(
                            run_id=run_id,
                            entity_id=entity_id,
                            production_action=action,
                            shadow_action=shadow_action,
                            production_rule=rule_name,
                            shadow_rule=shadow_rule,
                        )
                    except Exception as e:
                        print(f"Shadow outcome logging failed: {e}", file=sys.stderr)

            # Check permissions
            permission = check_permission(action, permissions=permissions)
            permission_results.labels(result=permission).inc()

            if not dry_run:
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

            if dry_run:
                action_counts[action] += 1
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
            failed_entities_counter.labels(error_type=type(e).__name__).inc()
            if not dry_run:
                try:
                    await db.log_failed_entity(
                        entity_id=entity_id,
                        run_id=run_id,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        entity_data=raw_entity,
                    )
                except Exception:
                    pass  # Don't let failure logging kill the run

    # Complete run
    automation_runs.labels(status="dry_run" if dry_run else "completed").inc()
    if not dry_run:
        await db.complete_run(
            run_id=run_id,
            entities_processed=processed,
            entities_failed=failed,
            action_distribution=dict(action_counts),
        )

    result = {
        "run_id": run_id,
        "status": "completed",
        "entities_processed": processed,
        "entities_failed": failed,
        "action_distribution": dict(action_counts),
        "dry_run": dry_run,
        "results": results,
    }

    if shadow_rules is not None:
        production_counts: Counter = Counter()
        for ctx_action in action_counts:
            # Strip permission suffixes for comparison
            base = ctx_action.split(":")[0]
            production_counts[base] += action_counts[ctx_action]
        result["shadow_tvd"] = total_variation_distance(
            dict(production_counts), dict(shadow_action_counts),
        )
        result["shadow_action_deltas"] = compute_action_deltas(
            dict(production_counts), dict(shadow_action_counts),
        )

    return result
