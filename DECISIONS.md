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
file-based routing out of the box. The dashboard has seven components:
three interactive (HealthStatus, RankingDemo, EvalMetrics) and four
read-only (RunsTable, ApprovalsList, ActionChart, ErrorSummary).

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

The ranking policy interface (fit -> score(items, context) -> evaluate) and
the automation pipeline (collect -> enrich -> rules -> permissions ->
execute -> log) have genuinely different shapes. They share the real common
ground: `telemetry/db.py`, `telemetry/metrics.py`, Postgres schema, and
CI gates. The shared infrastructure is the connection, not a base class
that papers over different decision shapes.

## 13. Policy replay for change control

A policy change that improves average reward by 2% but changes 40% of
individual decisions needs human review — the aggregate metric hides
distributional shift.

The replay runner (`src/evaluation/replay.py`) loads 100 frozen
context-action pairs and re-runs them through the candidate rules config.
It measures Total Variation Distance (TVD) between the baseline and
candidate action distributions. CI fails if TVD exceeds 0.15 (15% shift).

TVD provides the single-number CI gate. Per-action deltas provide the
debugging output (which specific actions shifted and by how much).
Per-entity changes list exactly which entities would receive different
treatment, with the candidate rule that fired.

Set `ALLOW_DRIFT=true` to override for intentional rule changes after
human review. This escape hatch exists because not all drift is bad —
but all drift should be acknowledged.

## 14. Shadow mode for safe policy deployment

Shadow mode lets you evaluate a new policy on live traffic without risk —
the candidate policy observes but never executes.

The shadow fork happens after enrichment: same enriched entities go through
both production and candidate rules. Permissions are NOT applied to the
shadow side — shadow logs raw rule output so you can see what the candidate
rules would route, even for actions that permissions would block. The
comparison surface is pre-permission rule output, which isolates the
variable shadow mode exists to test: rule changes.

Shadow data is stored in `shadow_outcomes` with a per-entity row and a
`diverged` boolean for cheap filtering. Distribution comparison uses the
same TVD + per-action delta functions as the offline replay runner.

## 15. Epsilon-greedy bandit with in-memory arm state

The bandit maintains per-arm reward estimates (`arm_rewards`, `arm_counts`)
in memory on the policy instance, updated via `update(item_id, reward)`.
Server restart resets all estimates to the warm-start values from training
data. Persistent bandit state (Redis, database) is a V3 concern — V2
demonstrates the algorithm and evaluation, not production statefulness.

Epsilon is hard-capped at `max_epsilon=0.10`. This limits exploration to
10% of interactions, analogous to how the permissions layer limits what
actions the automation pipeline can take. The cap is configurable but
exists to prevent unbounded random behavior.

Warm-start normalizes MovieLens ratings from [1, 5] to [0, 1] via
`(rating - 1) / 4`. The `update()` method validates that rewards are in
[0, 1], enforcing scale consistency at the mutation boundary so mixing
warm-start and online feedback never corrupts estimates.

Evaluation uses `_exploit_scores()` — a pure method that ranks by arm
estimates without touching `self.epsilon` or the RNG. This avoids mutating
the live singleton in `_policies` during `/evaluate` requests, which would
otherwise cause concurrent `/rank` requests to observe epsilon=0 and stop
exploring. The non-mutating path also eliminates the need for
try/finally restoration.

The bandit comparison module simulates online interaction rounds against
a static best-arm baseline. The environment uses per-arm bias (linspace
-3 to +3) so arms have genuinely different expected rewards — without
bias, symmetry makes every arm average ~0.5, leaving nothing to learn.
The static policy picks a single arm with the highest estimated marginal
reward (from a warmup phase) and never adapts. The comparison produces
cumulative reward curves: the bandit's ability to learn gives it a
substantial advantage (+4669 reward over 10K rounds in the default config).
