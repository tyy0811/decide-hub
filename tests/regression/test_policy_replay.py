"""CI regression gate: fail if action distribution drifts beyond threshold.

Run as part of `pytest tests/` — no special invocation needed.
Set ALLOW_DRIFT=true to override for intentional rule changes.
"""

import json
import os
from pathlib import Path

from src.evaluation.replay import replay_contexts

TVD_THRESHOLD = 0.15
_FIXTURE_PATH = Path("tests/fixtures/frozen_contexts.json")


def _load_frozen_contexts() -> list[dict]:
    return json.loads(_FIXTURE_PATH.read_text())


def test_policy_replay_drift():
    """Fail if current rules_config.yml causes action distribution drift."""
    if os.environ.get("ALLOW_DRIFT", "").lower() == "true":
        return  # Intentional drift — human approved

    contexts = _load_frozen_contexts()
    result = replay_contexts(contexts, "src/automations/rules_config.yml")

    assert result.tvd <= TVD_THRESHOLD, (
        f"Action distribution drift TVD={result.tvd:.3f} exceeds threshold {TVD_THRESHOLD}.\n"
        f"Per-action deltas: {result.action_deltas}\n"
        f"Entities changed: {len(result.per_entity_changes)}\n"
        f"Set ALLOW_DRIFT=true to override after human review."
    )
