"""FastAPI application — /rank, /evaluate, /approvals, /runs endpoints."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.policies.base import BasePolicy
from src.policies.data import load_ratings, temporal_split
from src.policies.popularity import PopularityPolicy
from src.policies.scorer import ScorerPolicy
from src.serving.schemas import (
    RankRequest, RankResponse, ScoredItem,
    EvaluateRequest, EvaluateResponse,
    ApprovalsResponse, ApprovalItem,
    RunsResponse, RunItem,
    FailedEntitiesResponse, FailedEntityItem,
)
from src.telemetry import db

_DEFAULT_DSN = "postgresql://decide_hub:decide_hub@localhost:5432/decide_hub"

# Global state: fitted policies and data
_policies: dict[str, BasePolicy] = {}
_train_data = None
_test_data = None
_db_available = False


def get_policies() -> dict[str, BasePolicy]:
    return _policies


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _policies, _train_data, _test_data, _db_available

    # Reset state before (re-)initializing
    _policies.clear()
    _train_data = None
    _test_data = None
    _db_available = False

    # Load data and fit policies
    ratings = load_ratings()
    _train_data, _test_data = temporal_split(ratings, n_test=5)

    pop = PopularityPolicy().fit(_train_data)
    _policies["popularity"] = pop

    try:
        scorer = ScorerPolicy(n_estimators=50).fit(_train_data)
        _policies["scorer"] = scorer
    except Exception as e:
        print(f"Warning: ScorerPolicy failed to fit: {e}")

    # Try connecting to Postgres (schema sync happens in init_pool)
    dsn = os.environ.get("DATABASE_URL", _DEFAULT_DSN)
    try:
        await db.init_pool(dsn)
        _db_available = True
    except Exception:
        print("Warning: Postgres not available, logging disabled")

    yield

    if _db_available:
        await db.close_pool()
        _db_available = False


app = FastAPI(title="decide-hub", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "policies": list(_policies.keys())}


@app.post("/rank", response_model=RankResponse)
async def rank(req: RankRequest):
    policy = _policies.get(req.policy)
    if not policy:
        raise HTTPException(404, f"Policy '{req.policy}' not loaded")

    # Determine candidate items
    if req.candidate_items:
        candidates = req.candidate_items
    elif hasattr(policy, "item_counts"):
        candidates = list(policy.item_counts.keys())
    elif hasattr(policy, "_item_ids"):
        candidates = policy._item_ids
    else:
        raise HTTPException(500, "No candidate items available")

    policy.observe({"user_id": req.user_id})
    scored = policy.score(candidates)
    top_k = scored[:req.k]

    # Note: we do NOT log predictions as "outcomes" here.
    # Outcomes (observed rewards) should come from a separate feedback
    # endpoint when the user actually engages with a recommended item.

    return RankResponse(
        user_id=req.user_id,
        policy=req.policy,
        items=[ScoredItem(item_id=iid, score=s) for iid, s in top_k],
    )


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest):
    policy = _policies.get(req.policy)
    if not policy:
        raise HTTPException(404, f"Policy '{req.policy}' not loaded")

    metrics = policy.evaluate(_test_data, k=req.k)

    # Log evaluation results to Postgres for audit trail
    if _db_available:
        for name, value in metrics.items():
            await db.log_outcome(
                user_id=0, action=f"eval_{name}",
                reward=value, policy_id=req.policy,
            )

    return EvaluateResponse(
        policy=req.policy,
        k=req.k,
        metrics=metrics,
    )


@app.get("/approvals", response_model=ApprovalsResponse)
async def get_approvals():
    if not _db_available:
        raise HTTPException(503, "Database not available")

    rows = await db.get_pending_approvals()
    return ApprovalsResponse(
        approvals=[
            ApprovalItem(
                id=r["id"],
                entity_id=r["entity_id"],
                proposed_action=r["proposed_action"],
                reason=r.get("reason"),
                status=r["status"],
                created_at=str(r["created_at"]),
            )
            for r in rows
        ]
    )


@app.get("/runs", response_model=RunsResponse)
async def get_runs():
    if not _db_available:
        raise HTTPException(503, "Database not available")

    rows = await db.get_runs()
    return RunsResponse(
        runs=[
            RunItem(
                run_id=r["run_id"],
                status=r["status"],
                entities_processed=r["entities_processed"],
                entities_failed=r["entities_failed"],
                action_distribution=r.get("action_distribution") or {},
                started_at=str(r["started_at"]),
                completed_at=str(r["completed_at"]) if r.get("completed_at") else None,
            )
            for r in rows
        ]
    )


@app.get("/failed-entities", response_model=FailedEntitiesResponse)
async def get_failed_entities(run_id: str | None = None):
    if not _db_available:
        raise HTTPException(503, "Database not available")

    rows = await db.get_failed_entities(run_id)
    return FailedEntitiesResponse(
        failed_entities=[
            FailedEntityItem(
                entity_id=r["entity_id"],
                run_id=r["run_id"],
                error_type=r["error_type"],
                error_message=r["error_message"],
            )
            for r in rows
        ],
        total=len(rows),
    )
