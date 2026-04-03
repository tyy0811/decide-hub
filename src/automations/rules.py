"""Rule-driven action selector — config-loaded, operator-editable."""

from pathlib import Path
import yaml
from src.automations.enrichment import EnrichedEntity

_RULES_PATH = Path(__file__).parent / "rules_config.yml"

# Fields on EnrichedEntity that can be used as boolean checks
_BOOLEAN_FIELDS = frozenset(
    name for name, info in EnrichedEntity.model_fields.items()
    if info.annotation is bool
)

# All valid field names for comparison conditions
_ALL_FIELDS = frozenset(EnrichedEntity.model_fields.keys())

_COMPARISON_OPS = (">=", "<=", ">", "<", "==")


def load_rules_config(path: Path | None = None) -> list[dict]:
    """Load rules from YAML config."""
    config_path = path or _RULES_PATH
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["rules"]


def _evaluate_condition(condition: str, entity: EnrichedEntity) -> bool:
    """Evaluate a rule condition against an enriched entity.

    Supports: boolean field checks, comparisons (>=, <=, >, <, ==),
    'and' conjunctions, and literal 'true'.

    Raises ValueError on unrecognized condition syntax.
    """
    if condition.strip() == "true":
        return True

    parts = [p.strip() for p in condition.split(" and ")]

    for part in parts:
        # Boolean field check (e.g., "has_missing_fields", "request_email")
        if part in _BOOLEAN_FIELDS:
            if not getattr(entity, part):
                return False
            continue

        # Comparison: "field >= value" or "field < value"
        matched_op = False
        for op in _COMPARISON_OPS:
            if op in part:
                field, val = part.split(op, 1)
                field = field.strip()
                val = val.strip()

                if field not in _ALL_FIELDS:
                    raise ValueError(
                        f"Unknown field '{field}' in rule condition: {condition!r}. "
                        f"Valid fields: {sorted(_ALL_FIELDS)}"
                    )

                entity_val = getattr(entity, field)
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

                matched_op = True
                break

        if not matched_op:
            raise ValueError(
                f"Unrecognized condition part: {part!r} in rule: {condition!r}. "
                f"Expected a boolean field, a comparison (field >= value), or 'true'."
            )

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
