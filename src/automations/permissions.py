"""Action guardrails — allow/block/approval_required between rules and execution."""

from pathlib import Path
import yaml

_PERMISSIONS_PATH = Path(__file__).parent / "permissions_config.yml"

_VALID_LEVELS = frozenset({"allowed", "blocked", "approval_required"})


def load_permissions_config(path: Path | None = None) -> dict[str, str]:
    """Load permissions from YAML config.

    Raises ValueError if any permission value is not a recognized level.
    """
    config_path = path or _PERMISSIONS_PATH
    with open(config_path) as f:
        config = yaml.safe_load(f)
    permissions = config["permissions"]
    for action, level in permissions.items():
        if level not in _VALID_LEVELS:
            raise ValueError(
                f"Invalid permission level '{level}' for action '{action}'. "
                f"Must be one of: {sorted(_VALID_LEVELS)}"
            )
    return permissions


def check_permission(action: str, permissions: dict[str, str] | None = None) -> str:
    """Check permission for an action.

    Returns: "allowed", "blocked", or "approval_required".
    Unknown actions default to "blocked" (fail-safe).
    """
    if permissions is None:
        permissions = load_permissions_config()
    return permissions.get(action, "blocked")
