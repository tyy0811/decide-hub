# Phase 2 + 3: Scope-Broadening Extensions & Full-Stack Depth — Design

**Date:** 2026-04-04
**Status:** Validated via brainstorming session
**Prerequisite:** Phase 1 shipped (replay, shadow, audit, approve/reject, retry, rate limiting)

---

## Overview

Phase 2 adds ML depth — new policy types, evaluation methods, and stronger
features. Phase 3 adds full-stack depth — multi-page dashboard, real-time
updates, authentication, and new intake methods. Together they make
decide-hub the lead portfolio repo for applied ML, experimentation, and
full-stack AI engineering roles.

**Phase 2 scope:** ~9-14 days, 5 items
**Phase 3 scope:** ~4-7 days, 5 items

---

## Build Order

Phase 2 items are all independent — they plug into existing interfaces
(`BasePolicy`, `src/evaluation/`, Postgres telemetry) without depending
on each other.

Phase 3 has one soft dependency: 3.1 (multi-page) is the natural first
item since 3.3 (auth) benefits from its routing structure.

```
Phase 2: All independent.
         Recommended: 2.1 first (bandit before experimentation = better demo).
         2.2, 2.3, 2.4, 2.5 in any order based on which roles you're targeting.

Phase 3: 3.1 first.
         Then 3.2, 3.3 in any order.
         3.4, 3.5 independent.
```

Cross-phase connection worth knowing: 2.1 (bandit) and 2.3 (experimentation)
have a natural synergy — the experimentation layer can compare bandit vs
static policy as treatment/control, which is a more compelling demo than
popularity vs scorer. Not a dependency, but sequencing 2.1 before 2.3
gives the experimentation layer a better demo for free.

---

## Phase 2: Scope-Broadening Extensions

### 2.1 Contextual Bandit (3-5 days)

**What it adds:** A new policy type that balances exploitation with
exploration, evaluated using the existing IPS/counterfactual infrastructure.

**Files:**
- `src/policies/bandit.py` — `EpsilonGreedyPolicy(BasePolicy)`
- `src/evaluation/bandit_comparison.py` — static vs bandit head-to-head
- `tests/test_bandit.py`, `tests/test_bandit_comparison.py`

**Design decisions:**

- **Epsilon-greedy first.** Simpler to implement and explain than Thompson
  sampling. "With probability epsilon, pick a random item; otherwise use
  the learned model." Thompson sampling is a stretch goal.

- **Exploration guardrail.** Epsilon exposed as config with hard cap
  (`max_epsilon=0.10`). Connects to permissions layer conceptually — the
  system limits how much the bandit can explore.

- **In-memory arm state.** `self.arm_rewards` and `self.arm_counts` updated
  on each `log_outcome()` call. No Redis, no database state. Server restart
  resets estimates. DECISIONS.md: "Persistent bandit state is a V3 concern;
  V2 demonstrates the algorithm and evaluation, not production statefulness."

- **Comparison output.** Simulate 10K interaction rounds, track cumulative
  reward at each timestep. Two curves: static scorer (flat — doesn't learn)
  vs bandit (starts lower due to exploration, should converge or overtake).
  If the bandit doesn't overtake, that's an honest finding worth reporting.

- **Interface preserved.** Extends `BasePolicy` — `fit()` initializes arm
  estimates, `score()` returns epsilon-greedy rankings, `evaluate()` runs
  offline metrics. `/rank` works unchanged with `policy=bandit`.

---

### 2.2 Retrieval/Search Mode (1-2 days)

**What it adds:** A retrieval policy ranking documents instead of items,
demonstrating that the evaluation framework generalizes across decision
domains.

**Files:**
- `src/policies/retrieval.py` — `RetrievalPolicy(BasePolicy)` wrapping TF-IDF
- `tests/test_retrieval.py`
- `tests/fixtures/retrieval_corpus.json` — ~50-100 docs + 10-20 queries
- Update benchmark table in README

**Design decisions:**

- **TF-IDF + cosine similarity** via scikit-learn's `TfidfVectorizer`. No
  FAISS — adding a vector index to rank 100 documents is engineering theater.
  scikit-learn is already installed.

- **Corpus with graded relevance.** Queries must produce interesting NDCG/MRR
  scores (0.6-0.9 range), not trivial 1.0s. Include queries where multiple
  documents are partially relevant and the most relevant document shares
  vocabulary with distractors. Binary-relevance-with-unique-keywords would
  make the benchmark table meaningless.

- **Same interface.** `fit()` builds TF-IDF matrix. `score(candidates,
  context={"query": "..."})` returns ranked documents. `evaluate()` runs
  NDCG/MRR/HitRate. `/rank` works with `policy=retrieval`.

- **Portfolio story:** "Same harness, different domain." The retrieval
  implementation is the least important part — the point is that the same
  `BasePolicy` interface, evaluation metrics, and CI regression gates apply
  identically to document retrieval.

---

### 2.3 KPI / Experimentation Layer (2-3 days)

**What it adds:** A/B test simulation and statistical analysis —
difference-in-means, bootstrap CIs, segment-wise breakdown, and minimum
detectable effect calculation.

**Files:**
- `src/evaluation/experiment.py` — `run_experiment()` engine
- `src/evaluation/kpi.py` — business metric definitions
- `src/evaluation/report.py` — structured experiment report (JSON + markdown)
- `tests/test_experiment.py` — statistical property tests
- `scripts/run_experiment.py` — CLI to run and print report

**Design decisions:**

- **No p-values.** CIs and effect sizes only. More informative, less prone
  to misinterpretation. DECISIONS.md entry explaining this choice — it's a
  strong interview talking point showing statistical maturity.

- **Bootstrap CIs.** 10K resamples, configurable. Treatment effect computed
  as difference-in-means.

- **Coverage test.** `test_experiment.py` generates data with a known true
  effect (+0.05), runs 200 experiments with different seeds, checks the 95%
  CI contains the true effect ~95% of the time (±3-4pp tolerance). Proves
  the CI implementation is calibrated, not just that it returns two numbers.

- **Segment-wise breakdown.** Split by a categorical feature (e.g., user
  activity bucket), compute per-segment treatment effects.

- **MDE calculator.** Given sample size and baseline variance, compute the
  smallest detectable effect. Useful for experiment planning.

- **KPI transforms.** `kpi.py` defines 2-3 metric transforms on raw rewards:
  `value_proxy` (reward weighted by item price), `retention_proxy` (binary
  return), `conversion_proxy` (threshold on engagement). Pure functions.

- **Report structure.** Dict with `baseline_mean`, `treatment_mean`, `lift`,
  `ci_lower`, `ci_upper`, `segments`, `sample_size`, `mde`. A
  `render_markdown()` function formats for terminal or README.

- **Data source.** Uses existing simulator for control (logging policy) vs
  treatment (scorer). If bandit (2.1) is built first, compare bandit vs
  scorer for a more compelling demo.

---

### 2.4 Collaborative Filtering Features for Scorer (1-2 days)

**What it adds:** User and item embedding features from matrix factorization,
fed into the existing LightGBM scorer. Expected to close the NDCG gap
documented in DECISIONS.md #3.

**Files:**
- `src/policies/embeddings.py` — `compute_embeddings()` via truncated SVD
- Modify `src/policies/scorer.py` — add embedding features
- `tests/test_embeddings.py`
- Update benchmark table in README (before/after + warm/cold split)
- DECISIONS.md entry

**Design decisions:**

- **Truncated SVD** via scikit-learn's `TruncatedSVD` on sparse user-item
  interaction matrix. No new dependencies (`implicit`, `LightFM`).

- **Training split only.** SVD must only see training interactions. Including
  test interactions leaks collaborative signal into the feature set —
  the CF equivalent of entity memorization. DECISIONS.md: "Embeddings
  computed on training split only."

- **16-32 dimensions** per user and per item. Low enough not to dominate
  existing 6 aggregate features, high enough for meaningful patterns.

- **Integration.** `build_features()` in `scorer.py` concatenates user
  embedding + item embedding to existing feature vector when available.
  `fit()` calls `compute_embeddings()` and caches. `score()` looks up by
  user/item ID.

- **Cold-start fallback.** Unknown users/items get a zero vector. Report
  NDCG separately for warm users vs cold-start users in the benchmark
  table — the improvement likely only comes from warm users, and reporting
  that honestly shows you understand the limitation.

- **Either outcome is valuable.** If scorer beats popularity with CF
  features, DECISIONS.md #3 gets a documented improvement arc. If it
  doesn't, that's an honest finding worth reporting.

---

### 2.5 Anomaly Detection on Automation Outcomes (1-2 days)

**What it adds:** Lightweight monitoring detecting when automation behavior
shifts. Extends telemetry from "what happened" to "is something wrong."

**Files:**
- `src/telemetry/anomaly.py` — detection functions
- Modify `src/serving/app.py` — `GET /anomalies` endpoint
- `tests/test_anomaly.py` — tests with synthetic distributions
- Dashboard anomaly indicator on overview page

**Design decisions:**

- **Z-score at 3 standard deviations.** Compare action category proportions
  in the last 5 runs against the preceding 20 runs. 3 SD instead of 2 to
  reduce false positives given the small sample sizes (5 runs of 20-50
  entities). DECISIONS.md sentence explaining the false-positive tradeoff.

- **Three monitored signals:**
  - Action distribution drift (proportion per action vs historical baseline)
  - Permission-block rate vs trailing average
  - Error rate vs trailing average

- **API endpoint.** `GET /anomalies` returns `status` ("ok" or "alert"),
  list of triggered anomalies (metric, observed, expected range, severity),
  and window timestamps.

- **Dashboard indicator.** Green/amber/red on overview page. No separate
  anomaly page.

- **No Prometheus AlertManager.** The API endpoint + dashboard indicator
  demonstrates the same concept without infrastructure overhead.

- **~80 lines of pure Python** operating on data already in Postgres.

---

## Phase 3: Full-Stack + Integration Depth

### 3.1 Multi-page Dashboard with Routing (1-2 days)

**What it adds:** Multi-page app with dedicated views for run details,
approvals, and policy comparison.

**Pages:**
- `/` — overview (current dashboard, compacted)
- `/runs/[id]` — run detail (entity list, actions, errors)
- `/approvals` — dedicated approval queue
- `/policies` — policy comparison view

**Files:**
- `operator_ui/src/app/runs/[id]/page.tsx`
- `operator_ui/src/app/approvals/page.tsx`
- `operator_ui/src/app/policies/page.tsx`
- `operator_ui/src/components/NavBar.tsx`
- Modify `operator_ui/src/app/page.tsx`
- New API: `GET /runs/{run_id}` (entity-level outcomes)
- New API: `GET /evaluate/results` (cached evaluation results)

**Design decisions:**

- **Next.js App Router** with file-based routing. Already set up.

- **Run detail page.** Entity-level breakdown for a single run. New
  `GET /runs/{run_id}` endpoint querying `automation_outcomes` joined with
  `action_audit_log`. `RunsTable` links each run_id to `/runs/{id}`.

- **Policies page.** Displays benchmark table and IPS results from cached
  data via `GET /evaluate/results`. Does NOT trigger live evaluation —
  page must load in under a second. Results populated by `make eval` or
  manual `POST /evaluate` calls.

- **NavBar.** Top nav: Overview, Runs, Approvals, Policies. Highlights
  current page. Added to root layout.

---

### 3.2 WebSocket Live Updates (1-2 days)

**What it adds:** Real-time push updates to the dashboard during automation
runs.

**Files:**
- `src/serving/ws.py` — WebSocket manager
- Modify `src/serving/app.py` — `GET /ws/runs` endpoint
- Modify `src/automations/orchestrator.py` — emit events
- Modify `operator_ui/src/components/RunsTable.tsx` — WebSocket subscription
- `tests/test_ws.py`

**Design decisions:**

- **Three event types (JSON):**
  - `run_started` — `{run_id, entity_count, timestamp}`
  - `entity_processed` — `{run_id, entity_id, action, permission, timestamp}`
  - `run_completed` — `{run_id, summary}`

- **In-process broadcast.** Simple connection set, `broadcast(event)` to all
  clients. No Redis pub/sub. DECISIONS.md: "In-process WebSocket broadcast.
  Multi-process would need Redis pub/sub — a deployment concern, not a
  design concern."

- **Fire-and-forget from orchestrator.** Calls `broadcast()` at run start,
  after each entity, and at completion. No pipeline blocking. No-op if no
  clients connected.

- **Frontend batching.** `useRef` buffer flushing every 500ms to prevent
  React re-rendering 100 times during a 100-entity run. ~5 lines of code,
  prevents visible jank.

- **Graceful fallback.** Falls back to existing fetch-once pattern if
  WebSocket disconnects. No polling.

- **Independent of 3.1.** Works on the existing single-page dashboard's
  RunsTable. More useful on a run detail page, but not dependent on it.

---

### 3.3 Authentication + Role-based Access (1-2 days)

**What it adds:** JWT auth with two roles protecting write endpoints.

**Files:**
- `src/serving/auth.py` — JWT creation/validation, FastAPI dependency
- Modify `src/serving/app.py` — auth on protected endpoints
- `operator_ui/src/app/login/page.tsx`
- `operator_ui/src/lib/auth.ts` — token storage, fetch wrapper
- Modify `operator_ui/src/components/ApprovalsList.tsx` — conditional buttons
- `tests/test_auth.py`

**Design decisions:**

- **JWT HS256** with secret from `JWT_SECRET` env var. `POST /auth/login`
  returns token with `role` claim.

- **Hardcoded user store.** 2-3 users in-memory dict. No database user
  table. DECISIONS.md: "Hardcoded users. Production would use an identity
  provider."

- **Role enforcement:**
  - `viewer`: all GET endpoints
  - `operator`: also POST endpoints (`/automate`, `/approvals/*/approve`,
    `/approvals/*/reject`, `/automate/retry`, `/webhooks/automate`,
    `/automate/upload`)
  - Unauthenticated: `/health` and `/metrics` only

- **Per-endpoint dependencies.** `get_current_user` extracts JWT,
  `require_role("operator")` chains on top. Not global middleware —
  keeps `/health` and `/metrics` open.

- **Audit integration.** Approve/reject endpoints thread JWT username
  into `log_audit_event()` as `actor=f"operator:{username}"` instead of
  hardcoded `"operator"`. This is the payoff of building audit (1.4)
  before auth.

- **Frontend.** Login page stores token in localStorage. Fetch wrapper
  attaches `Authorization: Bearer` header. 401 redirects to login.
  Approve/reject buttons conditionally rendered based on decoded role.

---

### 3.4 n8n / Webhook Orchestration (0.5-1 day)

**What it adds:** Webhook endpoint for external orchestration, plus an
example n8n workflow.

**Files:**
- Modify `src/serving/app.py` — `POST /webhooks/automate` endpoint
- `docs/n8n-workflow.json` — importable n8n workflow
- `tests/test_webhook.py`

**Design decisions:**

- **Separate endpoint.** `/automate` fetches entities via crawler.
  `/webhooks/automate` accepts `{"entities": [...], "dry_run": false}`
  directly — the caller already has the data.

- **Async execution.** FastAPI `BackgroundTasks` — endpoint creates run_id,
  enqueues orchestrator call, returns 202 Accepted immediately. Caller
  polls `GET /runs/{run_id}` for completion. DECISIONS.md: "202 Accepted
  (not 200 OK) — standard pattern for long-running operations."

- **n8n workflow.** JSON file with HTTP Request → Wait → Poll pattern.
  Demonstrates integration without requiring n8n to be running.

- **Rate limiting.** Shares existing `_automate_limiter`.

- **Auth.** Requires `operator` role if 3.3 is built.

---

### 3.5 CSV/File Upload Intake (0.5-1 day)

**What it adds:** CSV upload endpoint for operator-driven entity intake.

**Files:**
- Modify `src/serving/app.py` — `POST /automate/upload` accepting `UploadFile`
- Modify `src/serving/schemas.py` — `EntityRow` validation model
- `tests/test_upload.py`
- `tests/fixtures/sample_entities.csv`

**Design decisions:**

- **`csv.DictReader`** on uploaded bytes. No pandas dependency.

- **Pydantic validation.** Each row validated against `EntityRow` model
  matching existing entity shape (`entity_id`, `company`, `role`, `source`,
  `signup_date`).

- **Validate all before processing.** Return 422 with row-level errors
  (row number, field, what's wrong) if any row fails. No partial processing
  of bad files.

- **Entity cap.** Same `MAX_ENTITIES_PER_RUN = 100`.

- **Synchronous execution.** Like `/automate` — operator-initiated, they'll
  wait for the result.

- **Rate limiting.** Shares existing `_automate_limiter`.

- **No dashboard upload UI.** Endpoint for API consumers. Drag-and-drop
  component is scope creep for 0.5-1 day.

- **Sample CSV in README.** Reference `tests/fixtures/sample_entities.csv`
  in the development section so operators can see expected format.

---

## Key Design Choices Summary

| Item | Key choices |
|------|------------|
| 2.1 Bandit | Epsilon-greedy, in-memory arm state, hard epsilon cap, cumulative reward plot |
| 2.2 Retrieval | TF-IDF (no FAISS), graded relevance corpus with vocabulary overlap |
| 2.3 Experimentation | Bootstrap CIs (no p-values), coverage test, segment-wise breakdown |
| 2.4 CF Features | Truncated SVD on training split only, warm/cold NDCG split in benchmarks |
| 2.5 Anomaly | Z-score at 3 SD, 5-vs-20 run windowing, API + dashboard indicator |
| 3.1 Multi-page | App Router routing, cached eval results via GET endpoint |
| 3.2 WebSocket | In-process broadcast, 3 event types, useRef buffer flush at 500ms |
| 3.3 Auth | JWT HS256, hardcoded users, per-endpoint role deps, username in audit |
| 3.4 Webhook | BackgroundTasks async, 202 Accepted, poll for completion |
| 3.5 CSV Upload | csv.DictReader + Pydantic, validate-all-before-processing |
