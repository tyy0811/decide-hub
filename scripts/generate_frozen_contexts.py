"""Generate frozen contexts fixture from mock leads.

Runs the enrichment + rules pipeline on mock leads and writes the
context-action pairs to tests/fixtures/frozen_contexts.json.
Repeat the 10 base entities with variations to get 100 entries.
"""

import json
from datetime import date
from pathlib import Path

from src.automations.enrichment import enrich_entity
from src.automations.rules import apply_rules, load_rules_config

FIXTURE_PATH = Path("tests/fixtures/frozen_contexts.json")

# Base entities — varied to cover all rule branches
BASE_ENTITIES = [
    {"entity_id": "frozen_001", "company": "TechCorp", "role": "CTO", "source": "organic", "signup_date": "2026-03-20"},
    {"entity_id": "frozen_002", "company": "StartupInc", "role": "Engineer", "source": "referral", "signup_date": "2026-04-01"},
    {"entity_id": "frozen_003", "company": "", "role": "PM", "source": "paid_ad", "signup_date": "2025-06-15"},
    {"entity_id": "frozen_004", "company": "OldCo", "role": "Intern", "source": "cold_outbound", "signup_date": "2024-01-01"},
    {"entity_id": "frozen_005", "company": "BigEnterprise", "role": "VP Sales", "source": "organic", "signup_date": "2026-03-28", "request_email": True},
    {"entity_id": "frozen_006", "company": "TechCorp", "role": "Director", "source": "organic", "signup_date": "2026-03-25"},
    {"entity_id": "frozen_007", "company": "StartupInc", "role": "Manager", "source": "paid_ad", "signup_date": "2026-03-15"},
    {"entity_id": "frozen_008", "company": "BigEnterprise", "role": "CEO", "source": "referral", "signup_date": "2026-04-02"},
    {"entity_id": "frozen_009", "company": "OldCo", "role": "Engineer", "source": "cold_outbound", "signup_date": "2025-12-01"},
    {"entity_id": "frozen_010", "company": "TechCorp", "role": "Founder", "source": "organic", "signup_date": "2026-01-15"},
]


def main():
    today = date(2026, 4, 3)  # Fixed date for reproducibility
    rules = load_rules_config()
    contexts = []

    # Repeat base entities with unique IDs to reach 100
    for batch in range(10):
        for i, base in enumerate(BASE_ENTITIES):
            entity = {**base, "entity_id": f"frozen_{batch * 10 + i + 1:03d}"}
            enriched = enrich_entity(entity, today=today)
            action, rule_name = apply_rules(enriched, rules=rules)
            contexts.append({
                "entity": entity,
                "enriched": enriched.model_dump(),
                "action": action,
                "rule_matched": rule_name,
            })

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(contexts, indent=2) + "\n")
    print(f"Wrote {len(contexts)} frozen contexts to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
