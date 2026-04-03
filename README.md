# decide-hub

Decision-policy engine for ranking, counterfactual evaluation, and safe operational automation.

## Architecture

```
+-------------------------------------------------------------+
|                        FastAPI                               |
|  /rank  /evaluate  /automate  /approvals  /runs  /metrics   |
+----------------------+--------------------------------------+
|   Ranking Policies   |      Automation Pipeline             |
|                      |                                      |
|  PopularityPolicy    |  Crawler (httpx)                     |
|  ScorerPolicy (LGBM) |  -> Enrichment                      |
|                      |  -> Rules (YAML config)              |
|  fit -> observe      |  -> Permissions (YAML config)        |
|  -> score -> evaluate|  -> Execute or Queue                 |
+----------------------+--------------------------------------+
|                 Shared Telemetry Layer                       |
|         asyncpg (Postgres)  +  Prometheus metrics           |
+-------------------------------------------------------------+
|              Operator Dashboard (Next.js)                    |
|  RunsTable  ApprovalsList  ActionChart  ErrorSummary        |
+-------------------------------------------------------------+
```

## Quick Start

```bash
# Start Postgres
docker compose up -d postgres

# Install and run
make install
make eval    # Run offline evaluation (downloads MovieLens 1M on first run)
make serve   # Start API on :8000
make test    # Run test suite
```

## Ranking Benchmarks (MovieLens 1M)

| Policy | NDCG@10 | MRR | HitRate@10 |
|--------|---------|-----|------------|
| Popularity | 0.0177 | 0.0473 | 0.0954 |
| LightGBM Scorer | 0.0019 | 0.0120 | 0.0100 |

The scorer uses only 6 aggregate features without collaborative filtering signals.
See [DECISIONS.md](DECISIONS.md) #3 for why this is expected and what the value is.

## Counterfactual Evaluation (Synthetic Data)

| Estimator | Value |
|-----------|-------|
| Naive average | 0.8120 |
| IPS (target temp=0.5) | 0.8879 |
| Clipped IPS (M=10) | 0.8879 |

Evaluated on synthetic logged-policy data where propensities are known
by construction. See [DECISIONS.md](DECISIONS.md) #1 for methodology.

## Automation Pipeline

```
Source API -> Crawler -> Enrichment -> Rules -> Permissions -> Execute/Queue -> Log
```

- **Rules:** YAML-configured routing (operator-editable, validated at load)
- **Permissions:** Separate safety policy (allow/block/approval_required)
- **Dry run:** `POST /automate {"source_url": "...", "dry_run": true}` previews per-entity results
- **Failure handling:** Per-entity error isolation, `failed_entities` table
- **Idempotency:** DB unique constraint prevents duplicate processing on retry

## Operator Dashboard

Next.js + React + Tailwind dashboard at `:3000`:
- Recent automation runs with status
- Pending approvals (actions requiring human review)
- Action distribution chart
- Failed entities grouped by error type

## Development

```bash
make install   # Create venv and install deps
make test      # Python tests (excludes E2E)
make e2e       # Playwright E2E (requires Postgres + API + Next.js)
make eval      # Run offline ranking evaluation
make serve     # Start FastAPI dev server
make db-reset  # Reset Postgres (destroys data)
```

## Docker

```bash
docker compose up --build -d   # Full stack: Postgres + API + Dashboard
docker compose down             # Stop all
```

## Roadmap

- Collaborative filtering features for scorer (user-item interaction matrix)
- Contextual bandits (exploration/exploitation with safety bounds)
- Policy replay + change control (regression testing for policy changes)
- KPI/experimentation layer (A/B test simulation, confidence intervals)
