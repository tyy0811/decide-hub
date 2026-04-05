"""Tests for CSV upload endpoint."""

import io
from fastapi.testclient import TestClient

from src.serving.app import app
from src.serving.auth import create_token


def _operator_headers() -> dict:
    token = create_token(username="admin", role="operator")
    return {"Authorization": f"Bearer {token}"}


def test_upload_valid_csv():
    """Valid CSV processes successfully."""
    with TestClient(app) as client:
        csv_content = (
            "entity_id,company,role,source,signup_date\n"
            "up_1,TestCo,CTO,organic,2026-04-01\n"
            "up_2,StartupInc,Engineer,referral,2026-03-28\n"
        )
        resp = client.post(
            "/automate/upload",
            files={"file": ("entities.csv", io.BytesIO(csv_content.encode()), "text/csv")},
            data={"dry_run": "true"},
            headers=_operator_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["entities_uploaded"] == 2


def test_upload_invalid_csv_missing_entity_id():
    """CSV missing required entity_id column returns 422."""
    with TestClient(app) as client:
        csv_content = (
            "company,role,source,signup_date\n"
            "TestCo,CTO,organic,2026-04-01\n"
        )
        resp = client.post(
            "/automate/upload",
            files={"file": ("entities.csv", io.BytesIO(csv_content.encode()), "text/csv")},
            data={"dry_run": "true"},
            headers=_operator_headers(),
        )
        assert resp.status_code == 422


def test_upload_empty_csv():
    """CSV with only headers returns 422."""
    with TestClient(app) as client:
        csv_content = "entity_id,company,role,source,signup_date\n"
        resp = client.post(
            "/automate/upload",
            files={"file": ("entities.csv", io.BytesIO(csv_content.encode()), "text/csv")},
            data={"dry_run": "true"},
            headers=_operator_headers(),
        )
        assert resp.status_code == 422


def test_upload_exceeds_entity_cap():
    """CSV with >100 entities returns 422."""
    with TestClient(app) as client:
        rows = ["entity_id,company,role,source,signup_date"]
        for i in range(101):
            rows.append(f"up_{i},TestCo,CTO,organic,2026-04-01")
        csv_content = "\n".join(rows) + "\n"
        resp = client.post(
            "/automate/upload",
            files={"file": ("entities.csv", io.BytesIO(csv_content.encode()), "text/csv")},
            data={"dry_run": "true"},
            headers=_operator_headers(),
        )
        assert resp.status_code == 422


def test_upload_requires_auth():
    """Upload without auth returns 401."""
    with TestClient(app) as client:
        csv_content = "entity_id,company,role,source,signup_date\nup_1,TestCo,CTO,organic,2026-04-01\n"
        resp = client.post(
            "/automate/upload",
            files={"file": ("entities.csv", io.BytesIO(csv_content.encode()), "text/csv")},
            data={"dry_run": "true"},
        )
        assert resp.status_code == 401
