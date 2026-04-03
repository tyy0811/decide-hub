"""Tiny mock lead server for E2E tests. Runs on port 9999.

Each request returns unique entity IDs (UUID suffix) so idempotency
checks don't skip them on re-runs within the same day.
"""
import json
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler


def _make_leads():
    suffix = uuid.uuid4().hex[:8]
    return [
        {"entity_id": f"e2e_cto_{suffix}", "company": "TechCorp", "role": "CTO", "source": "organic", "signup_date": "2026-03-20"},
        {"entity_id": f"e2e_pm_{suffix}", "company": "", "role": "PM", "source": "paid_ad", "signup_date": "2025-06-15"},
        {"entity_id": f"e2e_vp_{suffix}", "company": "BigEnterprise", "role": "VP Sales", "source": "organic", "signup_date": "2026-03-28", "request_email": True},
    ]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"leads": _make_leads()}).encode())

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    HTTPServer(("", 9999), Handler).serve_forever()
