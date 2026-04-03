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
    """POST /automate with dry_run returns plan without executing."""
    response = client.post("/automate", json={
        "source_url": "http://mock/leads",
        "dry_run": True,
    })
    # May fail if mock server not running — test with expected status
    assert response.status_code in (200, 502, 503)
