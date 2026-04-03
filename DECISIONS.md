# Design Decisions

## 1. MovieLens for ranking metrics, synthetic data for counterfactual evaluation

MovieLens 1M is a standard recommendation benchmark with user/item features and
implicit feedback (ratings as engagement proxy). It is well-suited for NDCG, MRR,
and HitRate evaluation.

However, MovieLens has no logging policy controlling item exposure — users
self-selected what to rate. Pretending popularity fractions are propensities
would be methodologically dishonest. IPS requires known propensities.

We use a synthetic logged-policy simulator (~70 LOC) where the logging policy
is a softmax over item features with known temperature. True propensities are
recorded at generation time, making IPS exact rather than estimated.

## 2. IPS over more complex causal methods

Inverse Propensity Scoring is the simplest unbiased off-policy estimator.
For V1, it demonstrates the methodology without the implementation complexity
of doubly-robust estimators or direct method baselines. Clipped IPS reduces
variance with a single hyperparameter (clip bound).

## 3. LightGBM LambdaRank over neural ranker

LightGBM with the `lambdarank` objective is CPU-friendly, trains in seconds
on MovieLens, and optimizes a ranking loss directly (not squared error on
ratings). A neural ranker (two-tower, MLP) would require GPU setup, longer
training, and hyperparameter tuning — all of which are scope creep for V1.

The scorer currently underperforms the popularity baseline on MovieLens
(NDCG@10 0.002 vs 0.018) because it uses only 6 aggregate features (user/item
means, counts, stds) without collaborative filtering signals. This is expected:
a pointwise feature set without user-item interaction history cannot beat a
global popularity count on a dataset where popular items dominate test sets.
The value is in the ranking infrastructure (LambdaRank training, batch
prediction, policy interface), not the V1 feature set.

## 4. Rule-driven automation actions, not ML-driven

V1 demonstrates the orchestration pattern: collect -> enrich -> rules ->
permissions -> execute -> log. The rules are YAML-configurable, showing the
system is operator-editable. ML-driven routing (e.g., a bandit selecting
actions) is a V2 extension that builds on this pipeline.

## 5. Postgres over DuckDB

Postgres handles concurrent write-heavy logging (API requests + automation
outcomes + approval queue) better than DuckDB's single-writer model.
It also aligns with production infrastructure expectations.

## 6. IPS evaluation results on synthetic data

The IPS estimate for the greedy target policy (temperature=0.5) against the
exploratory logging policy (temperature=1.0) was 0.8879, compared to naive
average reward of 0.8120. The greedy policy concentrates probability on
higher-reward items, and IPS correctly accounts for the distribution shift,
producing a higher estimated value. Clipped IPS (M=10) produced the same
estimate (0.8879), indicating importance weights stayed within bounds for
this temperature gap.

## 7. Permission layer between rules and execution

The pipeline is: rules decide what to do, permissions decide whether it's
allowed. This separation means the system knows what it can do (allowed),
what it must not do (blocked), and when a human must intervene
(approval_required).

The YAML config makes this operator-editable without code changes. In a real
org, the ML team adjusts routing rules while compliance sets permission levels.

Permission values are validated at load time — typos like "allowd" raise
immediately rather than silently disabling guardrails. Unknown actions default
to blocked (fail-safe). The orchestrator explicitly checks for "allowed"
rather than treating anything-not-blocked as executable.

## 8. Per-entity error handling instead of fail-fast

One bad entity should not kill a 500-entity run. Partial failures are normal
in production automation — network timeouts, validation errors, missing fields.
Each failure is logged to the `failed_entities` table with error_type and
error_message, enabling targeted reruns.

Idempotency is enforced at the database level via a unique constraint on
(entity_id, processed_date), using INSERT ... ON CONFLICT DO NOTHING. This
prevents duplicate processing on retry without race conditions.

## 9. Next.js for the operator dashboard

Next.js provides React component architecture, TypeScript, Tailwind CSS, and
file-based routing out of the box. The four dashboard components (RunsTable,
ApprovalsList, ActionChart, ErrorSummary) are read-only in V1.

All components guard against API errors (503 when DB unavailable) by checking
response.ok before parsing payloads.

## 10. Postgres schema created on Day 1 (infrastructure-first)

Database tables are infrastructure, not a feature. Schema creation on Day 1
means every subsequent day writes to a live database. The app runs the
idempotent schema on every startup via init_pool(), so schema changes apply
regardless of whether the Docker entrypoint already ran.

## 11. asyncpg + raw SQL instead of SQLAlchemy

Five tables with straightforward queries (INSERT, SELECT with WHERE,
COUNT/GROUP BY). SQLAlchemy would add async session management ceremony,
Alembic migration setup, and context manager boilerplate — none of which
adds value at this scale.

A per-connection JSONB codec ensures JSONB columns always return `dict`
(not `str`), eliminating type ambiguity for all callers.

## 12. Separate interfaces for ranking and automation modules

The ranking policy interface (fit -> observe -> score -> evaluate) and
the automation pipeline (collect -> enrich -> rules -> permissions ->
execute -> log) have genuinely different shapes. They share the real common
ground: `telemetry/db.py`, `telemetry/metrics.py`, Postgres schema, and
CI gates. The shared infrastructure is the connection, not a base class
that papers over different decision shapes.
