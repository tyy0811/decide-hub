"""Rule-driven action selector — config-loaded, operator-editable."""

from pathlib import Path
import yaml
from src.automations.enrichment import EnrichedEntity

_RULES_PATH = Path(__file__).parent / "rules_config.yml"


def load_rules_config(path: Path | None = None) -> list[dict]:
    """Load rules from YAML config."""
    config_path = path or _RULES_PATH
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["rules"]


def _evaluate_condition(condition: str, entity: EnrichedEntity) -> bool:
    """Evaluate a rule condition against an enriched entity.

    Supports: field checks, comparisons, 'and' conjunctions, 'true'.
    """
    if condition.strip() == "true":
        return True

    # Split on 'and' for conjunctions
    parts = [p.strip() for p in condition.split(" and ")]

    for part in parts:
        # Boolean field check (e.g., "has_missing_fields", "request_email")
        if part in ("has_missing_fields", "request_email"):
            if not getattr(entity, part, False):
                return False
            continue

        # Comparison: "field >= value" or "field < value"
        for op in (">=", "<=", ">", "<", "=="):
            if op in part:
                field, val = part.split(op)
                field = field.strip()
                val = val.strip()
                entity_val = getattr(entity, field, None)
                if entity_val is None:
                    return False
                val = type(entity_val)(val)
                if op == ">=" and not (entity_val >= val):
                    return False
                if op == "<=" and not (entity_val <= val):
                    return False
                if op == ">" and not (entity_val > val):
                    return False
                if op == "<" and not (entity_val < val):
                    return False
                if op == "==" and not (entity_val == val):
                    return False
                break

    return True


def apply_rules(
    entity: EnrichedEntity, rules: list[dict] | None = None,
) -> tuple[str, str]:
    """Apply rules to an enriched entity. First match wins.

    Returns: (action, rule_name)
    """
    if rules is None:
        rules = load_rules_config()

    for rule in rules:
        if _evaluate_condition(rule["condition"], entity):
            return rule["action"], rule["name"]

    return "standard_sequence", "default_fallback"
