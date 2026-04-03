"""Entity enrichment pipeline — raw entity -> computed fields."""

from datetime import date, datetime
from pydantic import BaseModel


class EnrichedEntity(BaseModel):
    entity_id: str
    company: str
    role: str
    source: str
    signup_date: str
    company_size_bucket: str  # "small", "mid", "large", "enterprise", "unknown"
    lead_score: int  # 0-100
    days_since_signup: int
    source_quality_tier: str  # "high", "medium", "low"
    has_missing_fields: bool
    request_email: bool = False


# Simple heuristics for demonstration
ROLE_SCORES = {
    "cto": 30, "vp sales": 30, "ceo": 30, "founder": 25,
    "director": 20, "manager": 15, "engineer": 10,
    "pm": 15, "intern": 5,
}

SOURCE_TIERS = {
    "organic": "high",
    "referral": "high",
    "paid_ad": "medium",
    "cold_outbound": "low",
}

# Company name -> size bucket (simple rule-based lookup)
COMPANY_SIZES = {
    "techcorp": "mid",
    "startupinc": "small",
    "bigenterprise": "enterprise",
    "oldco": "small",
}


def enrich_entity(raw: dict, today: date | None = None) -> EnrichedEntity:
    """Enrich a raw entity with computed fields."""
    if today is None:
        today = date.today()

    entity_id = raw.get("entity_id", "unknown")
    company = raw.get("company", "")
    role = raw.get("role", "")
    source = raw.get("source", "")
    signup_date_str = raw.get("signup_date", "")

    # Missing fields check
    has_missing = not company or not role or not signup_date_str

    # Company size bucket
    company_key = company.lower().replace(" ", "")
    company_size = COMPANY_SIZES.get(company_key, "unknown" if not company else "mid")

    # Days since signup
    try:
        signup_date = datetime.strptime(signup_date_str, "%Y-%m-%d").date()
        days_since = (today - signup_date).days
    except (ValueError, TypeError):
        days_since = 999

    # Source quality tier
    source_tier = SOURCE_TIERS.get(source, "low")

    # Lead score (0-100 heuristic)
    score = 0
    score += ROLE_SCORES.get(role.lower(), 5)
    score += {"high": 20, "medium": 10, "low": 5}.get(source_tier, 0)
    if days_since <= 7:
        score += 30
    elif days_since <= 30:
        score += 20
    elif days_since <= 90:
        score += 10
    if company_size in ("enterprise", "large"):
        score += 15
    elif company_size == "mid":
        score += 10

    score = min(score, 100)

    return EnrichedEntity(
        entity_id=entity_id,
        company=company,
        role=role,
        source=source,
        signup_date=signup_date_str,
        company_size_bucket=company_size,
        lead_score=score,
        days_since_signup=days_since,
        source_quality_tier=source_tier,
        has_missing_fields=has_missing,
        request_email=raw.get("request_email", False),
    )
