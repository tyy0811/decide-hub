"""Action guardrails — allow/block/approval_required between rules and execution."""

from pathlib import Path
import yaml

_PERMISSIONS_PATH = Path(__file__).parent / "permissions_config.yml"


def load_permissions_config(path: Path | None = None) -> dict[str, str]:
    """Load permissions from YAML config."""
    config_path = path or _PERMISSIONS_PATH
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["permissions"]


def check_permission(action: str, permissions: dict[str, str] | None = None) -> str:
    """Check permission for an action.

    Returns: "allowed", "blocked", or "approval_required".
    Unknown actions default to "blocked" (fail-safe).
    """
    if permissions is None:
        permissions = load_permissions_config()
    return permissions.get(action, "blocked")
