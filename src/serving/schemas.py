"""Pydantic request/response models for the API."""

from pydantic import BaseModel, Field


# --- Ranking ---

class RankRequest(BaseModel):
    user_id: int
    candidate_items: list[int] = Field(default_factory=list, description="Item IDs to rank. If empty, ranks all items.")
    k: int = Field(default=10, ge=1, le=100)
    policy: str = Field(default="popularity", pattern="^(popularity|scorer|bandit|retrieval)$")
    query: str | None = Field(default=None, description="Query string for retrieval policies")


class ScoredItem(BaseModel):
    item_id: int
    score: float


class RankResponse(BaseModel):
    user_id: int
    policy: str
    items: list[ScoredItem]


# --- Evaluation ---

class EvaluateRequest(BaseModel):
    policy: str = Field(default="popularity", pattern="^(popularity|scorer|bandit|retrieval)$")
    k: int = Field(default=10, ge=1, le=100)


class EvaluateResponse(BaseModel):
    policy: str
    k: int
    metrics: dict[str, float]


# --- Automation ---

class AutomateRequest(BaseModel):
    source_url: str = Field(description="URL of entity data source (mock API in tests)")
    dry_run: bool = False
    shadow_rules_config: str | None = Field(default=None, description="Path to shadow rules YAML for candidate comparison")


class EntityResult(BaseModel):
    entity_id: str
    action: str
    permission: str
    rule: str


class AutomateResponse(BaseModel):
    run_id: str
    status: str
    entities_processed: int
    entities_failed: int
    action_distribution: dict[str, int]
    dry_run: bool
    results: list[EntityResult] = Field(default_factory=list)
    shadow_tvd: float | None = None
    shadow_action_deltas: dict[str, float] | None = None


# --- Approvals ---

class ApprovalItem(BaseModel):
    id: int
    entity_id: str
    proposed_action: str
    reason: str | None
    status: str
    created_at: str


class ApprovalsResponse(BaseModel):
    approvals: list[ApprovalItem]


class ApprovalActionResponse(BaseModel):
    id: int
    entity_id: str
    proposed_action: str
    status: str


# --- Runs ---

class RunItem(BaseModel):
    run_id: str
    status: str
    entities_processed: int
    entities_failed: int
    action_distribution: dict
    shadow_tvd: float | None = None
    shadow_action_deltas: dict[str, float] | None = None
    started_at: str
    completed_at: str | None


class RunsResponse(BaseModel):
    runs: list[RunItem]


# --- Errors ---

class FailedEntityItem(BaseModel):
    entity_id: str
    run_id: str
    error_type: str
    error_message: str


class FailedEntitiesResponse(BaseModel):
    failed_entities: list[FailedEntityItem]
    total: int


class RetryResponse(BaseModel):
    retried: int
    succeeded: int
    dead_lettered: int
    still_failing: int


# --- Anomalies ---

class AnomalyItem(BaseModel):
    metric: str
    observed: float
    expected_range: str
    severity: str
    z_score: float | None = None


class AnomalyResponse(BaseModel):
    status: str  # "ok" or "alert"
    anomalies: list[AnomalyItem]
    baseline_window: int
    recent_window: int


# --- Run Detail ---

class RunOutcomeItem(BaseModel):
    entity_id: str
    action_taken: str
    rule_matched: str | None
    permission_result: str


class AuditEventItem(BaseModel):
    entity_id: str
    actor: str
    action_type: str
    reason: str | None


class RunDetailResponse(BaseModel):
    run_id: str
    status: str
    entities_processed: int
    entities_failed: int
    action_distribution: dict
    started_at: str
    completed_at: str | None
    outcomes: list[RunOutcomeItem]
    audit_events: list[AuditEventItem]


# --- Evaluation Results Cache ---

class EvalResultItem(BaseModel):
    policy: str
    k: int
    metrics: dict[str, float]


class EvalResultsResponse(BaseModel):
    results: list[EvalResultItem]
