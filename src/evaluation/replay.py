"""Policy replay — run frozen contexts through candidate rules for change control.

Compares candidate action distribution against a known-good baseline.
Used by CI regression gate and manual change review.
"""

from dataclasses import dataclass, field
from collections import Counter

from src.automations.enrichment import EnrichedEntity
from src.automations.rules import apply_rules, load_rules_config
from src.evaluation.comparison import compute_action_deltas, total_variation_distance


@dataclass
class ReplayResult:
    tvd: float
    action_deltas: dict[str, float]
    per_entity_changes: list[dict] = field(default_factory=list)


def replay_contexts(
    contexts: list[dict], rules_config_path: str,
) -> ReplayResult:
    """Replay frozen contexts through a candidate rules config.

    Args:
        contexts: List of frozen context dicts, each with keys:
            "enriched" (dict of EnrichedEntity fields),
            "action" (baseline action string),
            "rule_matched" (baseline rule name).
        rules_config_path: Path to candidate rules YAML.

    Returns:
        ReplayResult with TVD, per-action deltas, and per-entity changes.
    """
    candidate_rules = load_rules_config(path=rules_config_path)

    baseline_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    changes: list[dict] = []

    for ctx in contexts:
        enriched = EnrichedEntity(**ctx["enriched"])
        baseline_action = ctx["action"]
        baseline_counts[baseline_action] += 1

        candidate_action, candidate_rule = apply_rules(enriched, rules=candidate_rules)
        candidate_counts[candidate_action] += 1

        if candidate_action != baseline_action:
            changes.append({
                "entity_id": enriched.entity_id,
                "baseline_action": baseline_action,
                "candidate_action": candidate_action,
                "candidate_rule_matched": candidate_rule,
            })

    action_deltas = compute_action_deltas(dict(baseline_counts), dict(candidate_counts))
    tvd = total_variation_distance(dict(baseline_counts), dict(candidate_counts))

    return ReplayResult(tvd=tvd, action_deltas=action_deltas, per_entity_changes=changes)
