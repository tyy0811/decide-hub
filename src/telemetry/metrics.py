"""Prometheus counters and histograms for observability."""

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# --- Ranking ---
rank_requests = Counter(
    "decidehub_rank_requests_total",
    "Total rank API requests",
    ["policy"],
)

# --- Automation ---
automation_runs = Counter(
    "decidehub_automation_runs_total",
    "Total automation runs",
    ["status"],
)

rule_hits = Counter(
    "decidehub_rule_hits_total",
    "Rule match counts by action",
    ["action"],
)

permission_results = Counter(
    "decidehub_permission_results_total",
    "Permission check results",
    ["result"],  # allowed, blocked, approval_required
)

failed_entities_counter = Counter(
    "decidehub_failed_entities_total",
    "Failed entity count by error type",
    ["error_type"],
)

# --- Latency ---
api_latency = Histogram(
    "decidehub_api_latency_seconds",
    "API endpoint latency",
    ["endpoint"],
)

enrichment_duration = Histogram(
    "decidehub_enrichment_duration_seconds",
    "Entity enrichment duration",
)


def get_metrics() -> bytes:
    """Return Prometheus metrics in exposition format."""
    return generate_latest()


def get_content_type() -> str:
    return CONTENT_TYPE_LATEST
