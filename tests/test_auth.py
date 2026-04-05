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
