# Phase 4: Model Depth + Advanced ML — Design

**Date:** 2026-04-05
**Status:** Validated via brainstorming session
**Prerequisite:** Phases 1-3 shipped
**Scope:** 10 items, ~10-18 days. Build selectively based on target roles.

---

## Overview

Phase 4 adds ML depth — new policy types, advanced evaluation methods,
explainability, and business-domain framing. Unlike Phases 2-3 which
unlock new job categories, Phase 4 items strengthen signal for specific
roles. Each item is self-contained but grouped for efficient integration.

---

## Groupings & Build Order

```
Group A (Observability):    4.1 Grafana, 4.2 PostHog — independent
Group B (New Policies):     4.5 Pointwise LTR → 4.3 Neural Ranker
Group C (Evaluation Depth): 4.8 SHAP → 4.7 Doubly Robust → 4.10 Online Sim
Group D (Business Framing): 4.4 pLTV, 4.6 Multi-dataset, 4.9 Constrained — independent
```

Cross-group dependencies:
- 4.3 (neural ranker) benefits from 2.4 (CF embeddings) for optional
  SVD embedding input — flips architecture from "fighting the design"
  to "playing to its strengths"
- 4.7 (DR) extends 2.3 (experimentation) infrastructure
- 4.10 (online sim) extends 2.1 (bandit comparison) into a general framework

---

## Group A: Observability & Infrastructure

### 4.1 Grafana Dashboards (0.5-1 day)

**What it adds:** Pre-provisioned Grafana dashboards visualizing existing
Prometheus metrics. Zero-config setup via Docker volumes.

**Files:**
- `docker/grafana/provisioning/datasources/prometheus.yml`
- `docker/grafana/provisioning/dashboards/dashboard.yml`
- `docker/grafana/dashboards/decide-hub.json`
- Modify `docker-compose.yml` — add Grafana (port 3001) + Prometheus services

**Design decisions:**

- **Prometheus as intermediary.** App exposes `/metrics`, Prometheus
  scrapes it, Grafana reads from Prometheus. Standard stack.

- **6 dashboard panels:**
  1. Ranking requests over time (by policy)
  2. Automation run throughput
  3. Rule hit distribution (pie chart)
  4. Permission results breakdown
  5. API latency histograms (p50/p95/p99)
  6. Rate-limited requests

- **Port 3001.** Next.js occupies 3000. Explicit in compose `ports:` mapping.

- **File-based provisioning.** Auto-loads on startup. No manual Grafana
  UI configuration. `docker compose up` gives a working dashboard.

- **No alerting.** Would duplicate `/anomalies` endpoint from 2.5.
  Dashboards are visualization only.

---

### 4.2 PostHog Integration (0.5-1 day)

**What it adds:** Optional event tracking to PostHog alongside Postgres logging.

**Files:**
- `src/telemetry/posthog.py` — thin wrapper, `capture_event()` helper
- Modify `src/serving/app.py` — call in `/rank` and `/automate`
- `tests/test_posthog.py` — mocked client tests

**Design decisions:**

- **Optional client.** If `POSTHOG_API_KEY` env var is not set, all
  capture calls are no-ops. App runs identically without PostHog.

- **2 events captured:**
  1. `rank_request` — policy, user_id, k, latency_ms
  2. `automation_triggered` — run_id, entity_count, dry_run, source

- **No PostHog in Docker compose.** PostHog is SaaS or 10+ container
  self-hosted. Integration sends to configurable `POSTHOG_HOST`.

---

## Group B: Model Depth — New Policies

### 4.5 Pointwise LTR Baseline (1 day)

Build before 4.3 — establishes the pattern for adding new policies.

**What it adds:** A pointwise regression scorer (`LGBMRegressor`) as a
baseline for the existing pairwise LambdaRank scorer. Same features,
different objective — clean A/B on the loss function.

**Files:**
- `src/policies/ltr_scorer.py` — `PointwiseScorerPolicy(BasePolicy)`
- `tests/test_ltr_scorer.py`
- Update benchmark table, schema pattern, app.py registration

**Design decisions:**

- **Why pointwise, not another pairwise.** The existing `ScorerPolicy`
  already uses `LGBMRanker` with `lambdarank` objective and user-level
  grouping — it IS the pairwise scorer. Adding another pairwise variant
  would be redundant. The valuable comparison is pointwise vs pairwise:
  "pointwise predicts ratings, pairwise optimizes ranking directly."

- **LGBMRegressor.** Predicts individual ratings, ranks by predicted
  value. Same 6 features. The pairwise scorer should outperform because
  it optimizes item ordering within each user's candidate set, not
  individual rating prediction.

- **Update DECISIONS.md #3.** Current entry says "LightGBM LambdaRank"
  without calling out that it's pairwise with user-level grouping.
  Update to: "Pairwise LambdaRank (existing) vs pointwise regression
  baseline: pairwise outperforms by Xpp NDCG@10 because it optimizes
  item ordering within each user's candidate set."

---

### 4.3 Neural Ranker — Two-Tower (2-3 days)

**What it adds:** A PyTorch two-tower model as a third ranking approach.

**Files:**
- `src/policies/neural_scorer.py` — `NeuralScorerPolicy(BasePolicy)`
- `tests/test_neural_scorer.py`
- Update benchmark table, schema pattern, app.py registration

**Design decisions:**

- **Two-tower architecture.** User tower (input → 32 → 16) and item
  tower (input → 32 → 16). Score = dot product of embeddings. Standard
  industry architecture (YouTube, Google, Meta) — recognizable in interviews.

- **Feature input.** Same 6 aggregate features as LightGBM by default.
  Optional `use_embeddings=True` flag to use SVD embeddings from 2.4
  instead. Two-tower is designed for embedding inputs — feeding it
  aggregate statistics fights the architecture. With SVD embeddings,
  the result may flip (neural beats LightGBM on embeddings, loses on
  aggregates). Document whichever outcome.

- **BPR loss.** Bayesian Personalized Ranking on positive/negative item
  pairs per user. Positive = rating >= 4. Negative = random unrated items.

- **No GPU required.** ~2K parameters. CPU training takes seconds.

- **Raw PyTorch.** `torch.nn.Module` + `torch.optim.Adam`. No Lightning.
  The point is PyTorch fluency, not framework choice.

- **Expected result.** May underperform LightGBM on this small dataset.
  "Two-tower is designed for large-scale retrieval with rich features;
  on a small dataset with aggregate features, gradient boosting wins."
  With SVD embeddings, the result may differ — that's the interesting
  finding worth documenting.

- **Benchmark table grows to 4+ rows:** LightGBM (aggregate), LightGBM
  (+ CF), Neural (aggregate), Neural (+ CF). Whichever outcome, it's
  a documented comparison.

---

## Group C: Evaluation Depth

### 4.8 Feature Importance + SHAP Explainability (1 day)

Build first in Group C — quickest item.

**What it adds:** SHAP summary plot for LightGBM scorer.

**Files:**
- `scripts/generate_shap_plot.py` — fits scorer, computes SHAP, saves plot
- `docs/shap_summary.png` — output image
- Update README

**Design decisions:**

- **LightGBM native SHAP.** `model.predict(X, pred_contrib=True)` —
  fast, exact, no approximation. Neural ranker would need KernelSHAP
  (slow, approximate). Keep to LightGBM.

- **Script, not module.** One-shot analysis artifact, not runtime
  functionality. No SHAP dependency in the main app.

- **Beeswarm summary plot.** Shows feature importance + directionality.
  "High item_popularity pushes score up" — more informative than a
  bar chart.

- **With/without CF embeddings.** If 2.4 is built, generate two plots.
  Shows whether CF features are contributing or just noise.

---

### 4.7 Doubly Robust Estimator (1-2 days)

**What it adds:** DR estimator alongside existing IPS for lower-variance
off-policy evaluation.

**Files:**
- `src/evaluation/doubly_robust.py` — `dr_estimate()` function
- `tests/test_doubly_robust.py`
- `scripts/compare_estimators.py` — IPS vs DR head-to-head
- DECISIONS.md entry

**Design decisions:**

- **Formula.** DR = (1/n) * sum[ reward_model(x,a) + (target/logging) *
  (reward - reward_model(x,a)) ]. Correct if *either* the reward model
  or propensities are correct (doubly robust property).

- **Reward model.** Logistic regression on (context, action) → reward
  probability. scikit-learn `LogisticRegression`. Doesn't need to be
  good — even a mediocre model reduces variance vs pure IPS.

- **Same interface as IPS.** Takes rewards, propensities, target_probs,
  plus `reward_model_predictions`. Parallel to existing functions.

- **Comparison output.** 100 random seeds. Report mean estimate, std,
  MSE for IPS and DR. DR should show lower variance (tighter std).

- **Key test cases:**
  1. DR converges to IPS when reward model is uninformative (constant)
  2. DR has lower variance than IPS with a reasonable reward model
  3. Doubly robust property: correct when propensities wrong but model
     right, and correct when model wrong but propensities right

---

### 4.10 Online Simulation Environment (2-3 days)

**What it adds:** General-purpose online simulation where policies
interact with synthetic users, tracking regret curves.

**Files:**
- `src/evaluation/online_sim.py` — `OnlineEnvironment` + `run_simulation()`
- `tests/test_online_sim.py`
- `scripts/run_regret_comparison.py`

**Design decisions:**

- **Generalizes bandit_comparison.** The 2.1 module is a specialized
  two-policy comparison. Online sim handles any number of policies with
  configurable environments and regret tracking.

- **OnlineEnvironment class.** Wraps synthetic data generation. Exposes
  `get_context() -> np.ndarray` and `step(action) -> float`. Clean
  interface — policies don't know the reward model internals.

- **Fixed seed for fair comparison.** `seed` parameter on `__init__`
  determines the context sequence. All policies see the same contexts
  in the same order per comparison run. Otherwise regret curves aren't
  comparable.

- **Regret definition.** `regret(t) = sum(optimal[1..t]) - sum(policy[1..t])`.
  Optimal policy knows true reward probabilities, always picks best arm
  per context. Static policies have linear regret. Bandit should have
  sublinear regret (learning).

- **Policies compared.** All available: popularity, scorer, bandit,
  neural (if built). Output: per-policy regret curves + summary stats.

- **Not replacing bandit_comparison.** Existing module stays — it's
  simpler and self-contained. Online sim is the generalized version.

---

## Group D: Business Framing

### 4.4 pLTV Proxy Objective (1-2 days)

**What it adds:** Scorer trained on cumulative future user value instead
of individual ratings.

**Files:**
- `src/policies/pltv_scorer.py` — `PLTVScorerPolicy(BasePolicy)`
- `src/policies/labels.py` — label construction functions
- `tests/test_pltv.py`
- Update benchmark table

**Design decisions:**

- **Label construction.** For each (user, item, timestamp), compute
  `sum(ratings within N_days after this interaction)`. Items rated by
  high-future-value users get higher labels. Default N=30 days.

- **Temporal leakage prevention.** Discard training samples where the
  N-day window crosses the train/test split boundary. Same principle
  as CF embedding leakage (2.4) — future information must not leak
  into training labels. Document in DECISIONS.md alongside the 2.4 note.

- **Same LightGBM, different target.** Same features, same model class.
  Only the label changes. Clean isolation of the objective function's
  impact on ranking quality.

- **Fintech framing.** "Future engagement" → "expected customer lifetime
  value." Algorithm identical, framing domain-relevant. Cover-letter
  adjustment, not a code change.

- **Custom metric.** "pLTV lift" — average future value of users who
  received top-K from pLTV scorer vs rating scorer. Measures whether
  the pLTV objective actually surfaces retention-correlated items.

---

### 4.6 Multiple Datasets via DatasetAdapter (1-2 days)

**What it adds:** Adapter pattern generalizing data loading, with Amazon
Reviews (Books) as a second dataset.

**Files:**
- `src/policies/data_adapter.py` — `DatasetAdapter` Protocol + implementations
- `tests/test_data_adapter.py`
- Update README with cross-dataset benchmark table

**Design decisions:**

- **Protocol, not ABC.** Lighter weight, no inheritance ceremony. Two
  methods: `load() -> pl.DataFrame` and `name -> str`. Returns DataFrame
  with `user_id`, `item_id`, `rating`, `timestamp`.

- **Contract tests.** Both adapters tested for required columns with
  correct types. This is the Protocol equivalent of ABC enforcement.

- **Amazon Reviews (Books subset).** ~50K ratings. Real data, different
  domain, public, CPU-friendly. Download script like MovieLens.

- **Minimal refactoring.** `load_ratings()` stays as-is (backward
  compatible). Adapter is an optional path for cross-dataset comparison.

- **Cross-dataset benchmark table.** One row per (dataset, policy).
  The interesting finding: whether relative policy rankings change
  across datasets. Same insight pattern as demandops-lite's cross-city
  comparison.

---

### 4.9 Cost-sensitive / Constrained Optimization (2-3 days)

**What it adds:** Constraints on ranking — diversity and fairness caps
with Pareto frontier analysis.

**Files:**
- `src/policies/constrained.py` — `ConstrainedPolicy` wrapper
- `src/evaluation/pareto.py` — Pareto frontier computation
- `tests/test_constrained.py`
- `scripts/run_pareto_analysis.py`
- DECISIONS.md entry

**Design decisions:**

- **Wrapper, not a new policy.** `ConstrainedPolicy` wraps any
  `BasePolicy` and applies post-processing constraints. Works with
  any existing policy. Separates ranking objective from constraint.
  This is the production pattern (re-rank with business rules after
  ML scoring).

- **Two constraint types:**
  1. Diversity: top-K must contain items from >= M distinct categories
  2. Exposure fairness: no category > P% of top-K slots

- **Item categories via K-means.** Cluster items by their 3 rating
  features (avg_rating, popularity, rating_std) into 5 groups. Self-
  contained, no new data files. "I derived categories from existing
  features via K-means" is a pragmatic engineering decision.

- **score_with_metadata().** Standard `score()` returns the ranked list
  (BasePolicy contract). Separate `score_with_metadata()` returns both
  ranking and constraint metadata: `{"categories_in_topk": 4,
  "max_category_share": 0.35, "items_swapped": 2}`. Pareto script
  calls metadata version; `/rank` calls standard.

- **Pareto frontier.** Sweep constraint threshold (M from 1 to 5, P
  from 20% to 100%), plot NDCG vs constraint satisfaction at each point.
  Shows the tradeoff — tighter constraints cost NDCG.

---

## Key Design Choices Summary

| Item | Key choices |
|------|------------|
| 4.1 Grafana | 6 panels, file provisioning, port 3001, no alerting |
| 4.2 PostHog | Optional no-op client, 2 events, SaaS only |
| 4.5 Pointwise LTR | LGBMRegressor baseline vs existing pairwise, update DECISIONS.md #3 |
| 4.3 Neural Ranker | Two-tower, BPR loss, optional SVD embeddings, raw PyTorch |
| 4.8 SHAP | Script not module, native LightGBM SHAP, with/without CF |
| 4.7 Doubly Robust | IPS + reward model, test doubly-robust property directly |
| 4.10 Online Sim | OnlineEnvironment class, fixed seed, regret curves, multi-policy |
| 4.4 pLTV | Label construction, discard cross-boundary samples, fintech framing |
| 4.6 Multi-dataset | Protocol adapter, Amazon Books, contract tests, cross-dataset table |
| 4.9 Constrained | Wrapper pattern, K-means categories, score_with_metadata(), Pareto sweep |
