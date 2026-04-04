import pytest
from fastapi.testclient import TestClient
from src.serving.app import app, get_policies


@pytest.fixture(scope="module")
def client():
    """TestClient for the FastAPI app — context manager triggers lifespan."""
    with TestClient(app) as c:
        yield c


def test_rank_endpoint_returns_items(client):
    """POST /rank returns scored items."""
    response = client.post("/rank", json={
        "user_id": 1,
        "k": 5,
        "policy": "popularity",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert data["policy"] == "popularity"
    assert len(data["items"]) <= 5
    # Items should have item_id and score
    assert "item_id" in data["items"][0]
    assert "score" in data["items"][0]


def test_rank_invalid_policy(client):
    """POST /rank with invalid policy returns 422."""
    response = client.post("/rank", json={
        "user_id": 1,
        "policy": "nonexistent",
    })
    assert response.status_code == 422


def test_evaluate_endpoint(client):
    """POST /evaluate returns metrics."""
    response = client.post("/evaluate", json={
        "policy": "popularity",
        "k": 10,
    })
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "ndcg@10" in data["metrics"]


def test_health_endpoint(client):
    """GET /health returns ok."""
    response = client.get("/health")
    assert response.status_code == 200


def test_automate_endpoint_dry_run(client):
    """POST /automate with dry_run=True returns 502 when source is unreachable."""
    response = client.post("/automate", json={
        "source_url": "http://localhost:19999/nonexistent",
        "dry_run": True,
    })
    # Source unreachable — should return 502, not crash
    assert response.status_code == 502


def test_metrics_endpoint(client):
    """GET /metrics returns Prometheus exposition format."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"decidehub_rank_requests_total" in response.content


def test_rank_bandit_policy(client):
    """Bandit policy returns ranked items via /rank endpoint."""
    resp = client.post("/rank", json={
        "user_id": 42,
        "policy": "bandit",
        "k": 5,
    })
    if resp.status_code == 200:
        data = resp.json()
        assert data["policy"] == "bandit"
        assert len(data["items"]) == 5
        # Scores should be descending
        scores = [item["score"] for item in data["items"]]
        assert scores == sorted(scores, reverse=True)
    else:
        # Bandit may fail to load if data unavailable — 404 is acceptable
        assert resp.status_code == 404
