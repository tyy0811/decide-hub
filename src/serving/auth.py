"""JWT authentication with role-based access control.

Hardcoded user store — production would use an identity provider.
See DECISIONS.md.
"""

import os
import time

import jwt
from fastapi import Depends, HTTPException, Request


_DEFAULT_SECRET = "decide-hub-dev-secret-change-in-prod"
JWT_SECRET = os.environ.get("JWT_SECRET", _DEFAULT_SECRET)
JWT_ALGORITHM = "HS256"

if JWT_SECRET == _DEFAULT_SECRET:
    import sys
    print(
        "WARNING: using default JWT_SECRET — set JWT_SECRET env var before "
        "exposing this service. Default credentials are for local development only.",
        file=sys.stderr,
    )

# Hardcoded users — local development / demo only.
# Production would use an identity provider (OAuth2, SAML).
USERS: dict[str, dict] = {
    "admin": {"password": "admin", "role": "operator"},
    "operator1": {"password": "operator1", "role": "operator"},
    "viewer1": {"password": "viewer1", "role": "viewer"},
}


def authenticate_user(username: str, password: str) -> dict | None:
    """Validate credentials. Returns user dict or None."""
    user = USERS.get(username)
    if not user or user["password"] != password:
        return None
    return {"username": username, **user}


def create_token(
    username: str,
    role: str,
    expires_seconds: int = 86400,
) -> str:
    """Create a signed JWT with username and role claims."""
    payload = {
        "username": username,
        "role": role,
        "exp": int(time.time()) + expires_seconds,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT. Raises ValueError on failure."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise ValueError("Token expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")


async def get_current_user(request: Request) -> dict:
    """FastAPI dependency: extract and validate JWT from Authorization header.

    Returns dict with username and role.
    Raises 401 if token is missing or invalid.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")
    token = auth_header[7:]
    try:
        return decode_token(token)
    except ValueError as e:
        raise HTTPException(401, str(e))


def require_role(role: str):
    """FastAPI dependency factory: require a specific role.

    Usage: Depends(require_role("operator"))
    """
    async def _check(user: dict = Depends(get_current_user)):
        if user.get("role") != role:
            raise HTTPException(403, f"Requires {role} role")
        return user
    return _check
