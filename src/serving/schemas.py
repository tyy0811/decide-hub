"""Pydantic request/response models for the API."""

from pydantic import BaseModel, Field


# --- Ranking ---

class RankRequest(BaseModel):
    user_id: int
    candidate_items: list[int] = Field(default_factory=list, description="Item IDs to rank. If empty, ranks all items.")
    k: int = Field(default=10, ge=1, le=100)
    policy: str = Field(default="popularity", pattern="^(popularity|scorer)$")


class ScoredItem(BaseModel):
    item_id: int
    score: float


class RankResponse(BaseModel):
    user_id: int
    policy: str
    items: list[ScoredItem]


# --- Evaluation ---

class EvaluateRequest(BaseModel):
    policy: str = Field(default="popularity", pattern="^(popularity|scorer)$")
    k: int = Field(default=10, ge=1, le=100)


class EvaluateResponse(BaseModel):
    policy: str
    k: int
    metrics: dict[str, float]


# --- Automation ---

class AutomateRequest(BaseModel):
    source_url: str = Field(description="URL of entity data source (mock API in tests)")
    dry_run: bool = False


class AutomateResponse(BaseModel):
    run_id: str
    status: str
    entities_processed: int
    entities_failed: int
    action_distribution: dict[str, int]
    dry_run: bool


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


# --- Runs ---

class RunItem(BaseModel):
    run_id: str
    status: str
    entities_processed: int
    entities_failed: int
    action_distribution: dict
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
