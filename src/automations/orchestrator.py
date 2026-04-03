"""Automation orchestrator: collect -> enrich -> rules -> permissions -> execute -> log.

Per-entity error handling: one failure doesn't kill the run.
Idempotency: DB unique constraint on (entity_id, processed_date) prevents duplicates.
"""

from collections import Counter
from datetime import date

from src.automations.enrichment import enrich_entity
from src.automations.rules import apply_rules
from src.automations.permissions import check_permission
from src.telemetry import db


async def run_automation_pipeline(
    entities: list[dict],
    run_id: str,
    dry_run: bool = False,
) -> dict:
    """Run the full automation pipeline on a list of entities.

    Returns run summary dict.
    """
    today = date.today()

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
            enriched = enrich_entity(raw_entity, today=today)

            # Apply rules
            action, rule_name = apply_rules(enriched)

            # Check permissions
            permission = check_permission(action)

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
            if not dry_run:
                try:
                    await db.log_failed_entity(
                        entity_id=entity_id,
                        run_id=run_id,
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )
                except Exception:
                    pass  # Don't let failure logging kill the run

    # Complete run
    if not dry_run:
        await db.complete_run(
            run_id=run_id,
            entities_processed=processed,
            entities_failed=failed,
            action_distribution=dict(action_counts),
        )

    return {
        "run_id": run_id,
        "status": "completed",
        "entities_processed": processed,
        "entities_failed": failed,
        "action_distribution": dict(action_counts),
        "dry_run": dry_run,
        "results": results,
    }
