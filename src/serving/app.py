"""FastAPI application — /rank, /evaluate, /automate, /approvals, /runs, /metrics endpoints."""

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Path, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from src.policies.base import BasePolicy
from src.policies.data import load_ratings, temporal_split
from src.policies.popularity import PopularityPolicy
from src.policies.scorer import ScorerPolicy
from src.policies.bandit import EpsilonGreedyPolicy
from src.policies.retrieval import RetrievalPolicy
import polars as pl
from src.automations.crawler import fetch_entities
from src.automations.orchestrator import run_automation_pipeline
from src.serving.schemas import (
    RankRequest, RankResponse, ScoredItem,
    EvaluateRequest, EvaluateResponse,
    AutomateRequest, AutomateResponse, EntityResult,
    ApprovalsResponse, ApprovalItem, ApprovalActionResponse,
    RunsResponse, RunItem,
    FailedEntitiesResponse, FailedEntityItem,
    RetryResponse,
    AnomalyResponse, AnomalyItem,
    RunDetailResponse, RunOutcomeItem, AuditEventItem,
    EvalResultItem, EvalResultsResponse,
    LoginRequest, LoginResponse,
    WebhookRequest, WebhookResponse,
)
from src.telemetry.anomaly import detect_distribution_drift, detect_rate_spike
from src.serving.auth import authenticate_user, create_token, get_current_user, require_role
from src.serving.ws import ws_manager
from src.serving.rate_limit import SlidingWindowRateLimiter, check_backpressure
from src.telemetry import db
from src.telemetry.audit import log_audit_event
from src.telemetry.metrics import get_metrics, get_content_type, rank_requests, api_latency, rate_limited_total

_DEFAULT_DSN = "postgresql://decide_hub:decide_hub@localhost:5432/decide_hub"

# Global state: fitted policies and data
_policies: dict[str, BasePolicy] = {}
_train_data = None
_test_data = None
_db_available = False
# Per-process rate limiter — with multiple Uvicorn workers, the effective
# limit is multiplied by the number of workers. Use a shared backend
# (Redis) if multi-worker rate limiting is needed.
_automate_limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60.0)

MAX_ENTITIES_PER_RUN = 100


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

    try:
        bandit = EpsilonGreedyPolicy(epsilon=0.1).fit(_train_data)
        _policies["bandit"] = bandit
    except Exception as e:
        print(f"Warning: EpsilonGreedyPolicy failed to fit: {e}")

    try:
        import json
        from pathlib import Path
        corpus_path = Path("data/retrieval_corpus.json")
        if corpus_path.exists():
            corpus_data = json.loads(corpus_path.read_text())
            doc_rows = [
                {"doc_id": d["id"], "title": d["title"], "text": d["text"]}
                for d in corpus_data["documents"]
            ]
            retrieval = RetrievalPolicy(corpus_path=corpus_path).fit(
                pl.DataFrame(doc_rows)
            )
            _policies["retrieval"] = retrieval
    except Exception as e:
        print(f"Warning: RetrievalPolicy failed to fit: {e}")

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


@app.get("/metrics")
async def metrics():
    return Response(content=get_metrics(), media_type=get_content_type())


@app.post("/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(401, "Invalid credentials")
    token = create_token(username=user["username"], role=user["role"])
    return LoginResponse(token=token, username=user["username"], role=user["role"])


@app.post("/rank", response_model=RankResponse)
async def rank(req: RankRequest):
    start = time.time()
    rank_requests.labels(policy=req.policy).inc()

    policy = _policies.get(req.policy)
    if not policy:
        raise HTTPException(404, f"Policy '{req.policy}' not loaded")

    if req.policy == "retrieval" and not req.query:
        raise HTTPException(422, "Retrieval policy requires a 'query' field")

    # Determine candidate items
    if req.candidate_items:
        candidates = req.candidate_items
    elif hasattr(policy, "item_counts"):
        candidates = list(policy.item_counts.keys())
    elif hasattr(policy, "_item_ids"):
        candidates = policy._item_ids
    elif hasattr(policy, "_all_items"):
        candidates = policy._all_items
    elif hasattr(policy, "_doc_ids"):
        candidates = policy._doc_ids
    else:
        raise HTTPException(500, "No candidate items available")

    ctx = {"user_id": req.user_id}
    if req.query:
        ctx["query"] = req.query
    scored = policy.score(candidates, context=ctx)
    top_k = scored[:req.k]

    api_latency.labels(endpoint="/rank").observe(time.time() - start)

    return RankResponse(
        user_id=req.user_id,
        policy=req.policy,
        items=[ScoredItem(item_id=iid, score=s) for iid, s in top_k],
    )


# Cache of last evaluation results (populated by POST /evaluate)
# Keyed on (policy, k) to deduplicate — re-evaluation overwrites prior result.
_eval_cache: dict[tuple[str, int], dict] = {}


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest):
    policy = _policies.get(req.policy)
    if not policy:
        raise HTTPException(404, f"Policy '{req.policy}' not loaded")

    metrics = await asyncio.to_thread(policy.evaluate, _test_data, req.k)

    # Log evaluation results to Postgres for audit trail
    if _db_available:
        for name, value in metrics.items():
            await db.log_outcome(
                user_id=0, action=f"eval_{name}",
                reward=value, policy_id=req.policy,
            )

    _eval_cache[(req.policy, req.k)] = {"policy": req.policy, "k": req.k, "metrics": metrics}

    return EvaluateResponse(
        policy=req.policy,
        k=req.k,
        metrics=metrics,
    )


@app.get("/runs/{run_id}", response_model=RunDetailResponse)
async def get_run_detail_endpoint(
    run_id: str = Path(pattern=r"^(run|webhook|retry)_[a-f0-9_]+$"),
    user: dict = Depends(get_current_user),
):
    if not _db_available:
        raise HTTPException(503, "Database not available")
    detail = await db.get_run_detail(run_id)
    if not detail:
        raise HTTPException(404, f"Run {run_id} not found")
    return RunDetailResponse(
        run_id=detail["run_id"],
        status=detail["status"],
        entities_processed=detail["entities_processed"],
        entities_failed=detail["entities_failed"],
        action_distribution=detail.get("action_distribution") or {},
        started_at=str(detail["started_at"]),
        completed_at=str(detail["completed_at"]) if detail.get("completed_at") else None,
        outcomes=detail["outcomes"],
        audit_events=detail["audit_events"],
    )


@app.get("/evaluate/results", response_model=EvalResultsResponse)
async def get_eval_results(user: dict = Depends(get_current_user)):
    """Return cached evaluation results. Populated by POST /evaluate or make eval."""
    return EvalResultsResponse(
        results=[EvalResultItem(**r) for r in _eval_cache.values()]
    )


@app.post("/automate", response_model=AutomateResponse)
async def automate(req: AutomateRequest, user: dict = Depends(require_role("operator"))):
    # Rate limit check
    if not _automate_limiter.allow():
        rate_limited_total.labels(endpoint="/automate", reason="run_frequency").inc()
        raise HTTPException(
            429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(int(_automate_limiter.retry_after()) + 1)},
        )

    if not _db_available and not req.dry_run:
        raise HTTPException(503, "Database not available for non-dry-run")

    # Backpressure check
    if _db_available and not req.dry_run:
        if await check_backpressure():
            rate_limited_total.labels(endpoint="/automate", reason="backpressure").inc()
            raise HTTPException(
                429,
                detail="Backpressure: write rate too high",
                headers={"Retry-After": "30"},
            )

    try:
        entities = await fetch_entities(req.source_url)
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch entities: {e}")

    # Entity cap validation
    if len(entities) > MAX_ENTITIES_PER_RUN:
        raise HTTPException(
            422,
            detail=f"Entity count {len(entities)} exceeds maximum {MAX_ENTITIES_PER_RUN}",
        )

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    result = await run_automation_pipeline(
        entities=entities,
        run_id=run_id,
        dry_run=req.dry_run,
        shadow_rules_config=req.shadow_rules_config,
    )

    return AutomateResponse(
        run_id=result["run_id"],
        status=result["status"],
        entities_processed=result["entities_processed"],
        entities_failed=result["entities_failed"],
        action_distribution=result["action_distribution"],
        dry_run=result["dry_run"],
        results=result.get("results", []),
        shadow_tvd=result.get("shadow_tvd"),
        shadow_action_deltas=result.get("shadow_action_deltas"),
    )


@app.get("/approvals", response_model=ApprovalsResponse)
async def get_approvals(user: dict = Depends(get_current_user)):
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


@app.post("/approvals/{approval_id}/approve", response_model=ApprovalActionResponse)
async def approve_action(
    approval_id: int, user: dict = Depends(require_role("operator")),
):
    if not _db_available:
        raise HTTPException(503, "Database not available")

    # Atomic transition: avoids TOCTOU race from separate read+check+update
    approval = await db.claim_approval(approval_id, "approved")
    if not approval:
        existing = await db.get_approval_by_id(approval_id)
        if not existing:
            raise HTTPException(404, f"Approval {approval_id} not found")
        raise HTTPException(409, f"Approval {approval_id} is already {existing['status']}")

    # Approval recorded — action is NOT executed yet.
    # Execution requires an action executor (not yet built).
    # The approval stays in "approved" state until an executor
    # picks it up. No fake "completed run" records are created.
    await log_audit_event(
        entity_id=approval["entity_id"],
        run_id=None,
        actor=f"operator:{user['username']}",
        action_type="approve",
        action=approval["proposed_action"],
        rule_matched=None,
        permission_result="approved_pending_execution",
        reason=None,
    )

    return ApprovalActionResponse(
        id=approval_id,
        entity_id=approval["entity_id"],
        proposed_action=approval["proposed_action"],
        status="approved_pending_execution",
    )


@app.post("/approvals/{approval_id}/reject", response_model=ApprovalActionResponse)
async def reject_action(
    approval_id: int, user: dict = Depends(require_role("operator")),
):
    if not _db_available:
        raise HTTPException(503, "Database not available")

    # Atomic transition: avoids TOCTOU race from separate read+check+update
    approval = await db.claim_approval(approval_id, "rejected")
    if not approval:
        existing = await db.get_approval_by_id(approval_id)
        if not existing:
            raise HTTPException(404, f"Approval {approval_id} not found")
        raise HTTPException(409, f"Approval {approval_id} is already {existing['status']}")

    await log_audit_event(
        entity_id=approval["entity_id"],
        run_id=None,
        actor=f"operator:{user['username']}",
        action_type="reject",
        action=approval["proposed_action"],
        rule_matched=None,
        permission_result="rejected",
        reason=None,
    )

    return ApprovalActionResponse(
        id=approval_id,
        entity_id=approval["entity_id"],
        proposed_action=approval["proposed_action"],
        status="rejected",
    )


@app.get("/runs", response_model=RunsResponse)
async def get_runs(user: dict = Depends(get_current_user)):
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
                shadow_tvd=r.get("shadow_tvd"),
                shadow_action_deltas=r.get("shadow_action_deltas"),
                started_at=str(r["started_at"]),
                completed_at=str(r["completed_at"]) if r.get("completed_at") else None,
            )
            for r in rows
        ]
    )


@app.get("/failed-entities", response_model=FailedEntitiesResponse)
async def get_failed_entities(run_id: str | None = None, user: dict = Depends(get_current_user)):
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


@app.post("/automate/retry", response_model=RetryResponse)
async def retry_failed(user: dict = Depends(require_role("operator"))):
    if not _automate_limiter.allow():
        rate_limited_total.labels(endpoint="/automate/retry", reason="run_frequency").inc()
        raise HTTPException(
            429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(int(_automate_limiter.retry_after()) + 1)},
        )

    if not _db_available:
        raise HTTPException(503, "Database not available")

    retryable = await db.get_retryable_entities()
    succeeded = 0
    dead_lettered = 0
    still_failing = 0

    for entity_row in retryable:
        raw = entity_row.get("entity_data")
        if not raw:
            new_status = await db.increment_retry_count(entity_row["id"])
            if new_status == "dead_letter":
                dead_lettered += 1
            else:
                still_failing += 1
            continue

        try:
            result = await run_automation_pipeline(
                entities=[raw],
                run_id=f"retry_{entity_row['id']}",
                dry_run=False,
                suppress_failure_logging=True,
            )
            if result["entities_failed"] == 0:
                await db.delete_failed_entity(entity_row["id"])
                succeeded += 1
            else:
                new_status = await db.increment_retry_count(entity_row["id"])
                if new_status == "dead_letter":
                    dead_lettered += 1
                else:
                    still_failing += 1
        except Exception:
            new_status = await db.increment_retry_count(entity_row["id"])
            if new_status == "dead_letter":
                dead_lettered += 1
            else:
                still_failing += 1

    return RetryResponse(
        retried=len(retryable),
        succeeded=succeeded,
        dead_lettered=dead_lettered,
        still_failing=still_failing,
    )


@app.get("/anomalies", response_model=AnomalyResponse)
async def get_anomalies(
    baseline_runs: int = Query(default=20, ge=1),
    recent_runs: int = Query(default=5, ge=1),
    user: dict = Depends(get_current_user),
):
    """Check automation outcomes for anomalies.

    Compares action distribution and error rates of recent runs
    against a trailing baseline window. Read-only endpoint — no auth
    required (anomaly status is operational visibility, not a mutation).
    """
    if not _db_available:
        raise HTTPException(503, "Database not available")

    pool = db.get_pool()

    # Single query for both distribution and error-rate analysis —
    # ensures both checks operate on the same set of runs.
    rows = await pool.fetch(
        "SELECT action_distribution, entities_failed, entities_processed "
        "FROM automation_runs "
        "WHERE status = 'completed' AND action_distribution IS NOT NULL "
        "ORDER BY completed_at DESC LIMIT $1",
        baseline_runs + recent_runs,
    )

    if len(rows) < recent_runs + 1:
        return AnomalyResponse(
            status="ok", anomalies=[],
            baseline_window=0, recent_window=len(rows),
        )

    # Split into recent and baseline (same rows for both checks)
    recent_rows = rows[:recent_runs]
    baseline_rows = rows[recent_runs:]

    # Detect distribution drift
    recent_dists = [dict(r["action_distribution"]) for r in recent_rows]
    baseline_dists = [dict(r["action_distribution"]) for r in baseline_rows]
    drift_result = detect_distribution_drift(baseline_dists, recent_dists)

    # Detect error rate spike (same runs)
    recent_error_rates = [
        r["entities_failed"] / max(r["entities_processed"], 1)
        for r in recent_rows
    ]
    baseline_error_rates = [
        r["entities_failed"] / max(r["entities_processed"], 1)
        for r in baseline_rows
    ]
    error_result = detect_rate_spike(
        baseline_error_rates, recent_error_rates, metric_name="error_rate",
    )

    # Combine results
    all_anomalies = drift_result.anomalies + error_result.anomalies
    status = "alert" if all_anomalies else "ok"

    return AnomalyResponse(
        status=status,
        anomalies=[AnomalyItem(**a) for a in all_anomalies],
        baseline_window=len(baseline_rows),
        recent_window=len(recent_rows),
    )


@app.post("/webhooks/automate", response_model=WebhookResponse, status_code=202)
async def webhook_automate(
    req: WebhookRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_role("operator")),
):
    """Webhook: accept entities directly, process asynchronously.

    Returns 202 Accepted with run_id. Poll GET /runs/{run_id} for completion.
    """
    if not _db_available and not req.dry_run:
        raise HTTPException(503, "Database not available for non-dry-run")

    if not _automate_limiter.allow():
        rate_limited_total.labels(endpoint="/webhooks/automate", reason="run_frequency").inc()
        raise HTTPException(429, detail="Rate limit exceeded",
                            headers={"Retry-After": str(int(_automate_limiter.retry_after()) + 1)})

    if len(req.entities) > MAX_ENTITIES_PER_RUN:
        raise HTTPException(422, f"Entity count {len(req.entities)} exceeds maximum {MAX_ENTITIES_PER_RUN}")

    run_id = f"webhook_{uuid.uuid4().hex[:12]}"

    if req.dry_run:
        # Dry run: execute synchronously (no DB writes)
        await run_automation_pipeline(
            entities=req.entities, run_id=run_id, dry_run=True,
            shadow_rules_config=req.shadow_rules_config,
        )
        return WebhookResponse(run_id=run_id, status="accepted", entity_count=len(req.entities))

    # Persist run record before returning 202 — ensures the run_id is
    # durable and pollable even if the background task never executes.
    await db.create_run(run_id)

    # Async execution via BackgroundTasks
    background_tasks.add_task(
        run_automation_pipeline,
        entities=req.entities,
        run_id=run_id,
        dry_run=False,
        shadow_rules_config=req.shadow_rules_config,
    )

    return WebhookResponse(run_id=run_id, status="accepted", entity_count=len(req.entities))


@app.websocket("/ws/runs")
async def websocket_runs(websocket: WebSocket, token: str = Query(default="")):
    """WebSocket endpoint for live run updates. Requires JWT token as query param."""
    from src.serving.auth import decode_token
    try:
        decode_token(token)
    except (ValueError, Exception):
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        ws_manager.disconnect(websocket)
