"""Tests for JWT authentication and role-based access."""

import pytest

from src.serving.auth import (
    create_token,
    decode_token,
    USERS,
    authenticate_user,
)


def test_create_and_decode_token():
    """Token roundtrip: create -> decode returns same claims."""
    token = create_token(username="admin", role="operator")
    claims = decode_token(token)
    assert claims["username"] == "admin"
    assert claims["role"] == "operator"


def test_decode_invalid_token():
    """Invalid token raises ValueError."""
    with pytest.raises(ValueError):
        decode_token("not.a.valid.jwt")


def test_decode_expired_token():
    """Expired token raises ValueError."""
    token = create_token(username="admin", role="operator", expires_seconds=-1)
    with pytest.raises(ValueError, match="expired"):
        decode_token(token)


def test_authenticate_valid_user():
    """Valid credentials return user dict."""
    user = authenticate_user("admin", "admin")
    assert user is not None
    assert user["role"] == "operator"


def test_authenticate_invalid_password():
    """Wrong password returns None."""
    user = authenticate_user("admin", "wrong")
    assert user is None


def test_authenticate_unknown_user():
    """Unknown username returns None."""
    user = authenticate_user("nonexistent", "password")
    assert user is None


def test_viewer_role_exists():
    """At least one viewer user exists."""
    viewers = [u for u in USERS.values() if u["role"] == "viewer"]
    assert len(viewers) >= 1


# --- API integration tests ---

from fastapi.testclient import TestClient
from src.serving.app import app


def test_login_endpoint():
    """Valid login returns token."""
    with TestClient(app) as client:
        resp = client.post("/auth/login", json={"username": "admin", "password": "admin"})
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert data["role"] == "operator"


def test_login_invalid():
    """Invalid credentials return 401."""
    with TestClient(app) as client:
        resp = client.post("/auth/login", json={"username": "admin", "password": "wrong"})
        assert resp.status_code == 401


def test_health_no_auth_required():
    """Health endpoint works without auth."""
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
