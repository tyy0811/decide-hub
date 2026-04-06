# Development Guide

## Analysis Scripts

```bash
python scripts/run_experiment.py          # A/B experiment with bootstrap CIs
python scripts/run_bandit_comparison.py   # Bandit vs static cumulative reward
python scripts/run_regret_comparison.py   # Multi-policy regret curves
python scripts/run_cf_comparison.py       # Scorer with/without CF embeddings
python scripts/run_pareto_analysis.py     # Constrained optimization tradeoffs
python scripts/compare_estimators.py      # IPS vs Doubly Robust variance
python scripts/generate_shap_plot.py      # SHAP feature importance plot
```

## API Examples

Get an auth token:

```bash
TOKEN=$(curl -s http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")
```

CSV upload:

```bash
curl -X POST http://localhost:8000/automate/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@tests/fixtures/sample_entities.csv" \
  -F "dry_run=true"
```

Webhook (async — returns 202, poll /runs/{run_id}):

```bash
curl -X POST http://localhost:8000/webhooks/automate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"entities": [{"entity_id": "test", "company": "Co", "role": "CTO", "source": "organic", "signup_date": "2026-04-01"}]}'
```

See `tests/fixtures/sample_entities.csv` for the expected CSV format.

## Authentication

JWT HS256 with two roles:
- **`operator`**: read + write (automate, approve/reject, retry, webhook, upload)
- **`viewer`**: read-only (runs, approvals, metrics)
- `/health` and `/metrics` are open (no auth required)

Demo accounts (local dev only, requires `ALLOW_INSECURE_AUTH=true`):
- `admin/admin` (operator)
- `operator1/operator1` (operator)
- `viewer1/viewer1` (viewer)

See [DECISIONS.md](../DECISIONS.md) #21 for the design rationale.
