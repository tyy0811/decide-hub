"""FastAPI fixture serving realistic-looking lead/entity data for tests."""

from fastapi import FastAPI

mock_app = FastAPI()

MOCK_LEADS = [
    {
        "entity_id": "lead_001",
        "company": "TechCorp",
        "role": "CTO",
        "source": "organic",
        "signup_date": "2026-03-20",
    },
    {
        "entity_id": "lead_002",
        "company": "StartupInc",
        "role": "Engineer",
        "source": "referral",
        "signup_date": "2026-04-01",
    },
    {
        "entity_id": "lead_003",
        "company": "",  # Missing company — triggers flag_for_review
        "role": "PM",
        "source": "paid_ad",
        "signup_date": "2025-06-15",
    },
    {
        "entity_id": "lead_004",
        "company": "OldCo",
        "role": "Intern",
        "source": "cold_outbound",
        "signup_date": "2024-01-01",
    },
    {
        "entity_id": "lead_005",
        "company": "BigEnterprise",
        "role": "VP Sales",
        "source": "organic",
        "signup_date": "2026-03-28",
        "request_email": True,  # Triggers send_external_email -> approval_required
    },
]


@mock_app.get("/leads")
async def get_leads():
    return {"leads": MOCK_LEADS}
