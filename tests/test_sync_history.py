"""Sync history: every import recorded (success + failure), newest first, owner-only."""

import types
import uuid

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.ingestion import ingestion_service as svc


def _adapter(source="csv"):
    return types.SimpleNamespace(get_source_name=lambda: source)


def _merchant():
    c = TestClient(app)
    c.post("/auth/register", json={"email": f"sy_{uuid.uuid4().hex[:8]}@example.com", "password": "Strong-pass-123"})
    sid = c.post("/stores", json={"display_name": "S", "industry": "fashion", "country": "EG"}).json()["store_id"]
    return c, sid


def test_success_and_failure_recorded(monkeypatch):
    c, sid = _merchant()
    result = types.SimpleNamespace(created=3, updated=1, failed=1, total_rows=5, unmapped_columns=["Notes"])
    monkeypatch.setattr(svc, "_ingest_products_impl", lambda adapter, store_id, *a, **kw: result)
    assert svc.ingest_products(_adapter("shopify"), sid, "fashion") is result

    def boom(adapter, store_id, *a, **kw):
        raise RuntimeError("WooCommerce returned 401")
    monkeypatch.setattr(svc, "_ingest_products_impl", boom)
    with pytest.raises(RuntimeError):
        svc.ingest_products(_adapter("woocommerce"), sid, "fashion")

    runs = c.get(f"/stores/{sid}/syncs").json()["runs"]
    assert [(r["source"], r["status"]) for r in runs] == [("woocommerce", "failed"), ("shopify", "succeeded")]
    assert "401" in runs[0]["error"]
    assert (runs[1]["created"], runs[1]["updated"], runs[1]["failed"], runs[1]["unmapped_columns"]) == (3, 1, 1, ["Notes"])


def test_history_is_private():
    c, sid = _merchant()
    other, _ = _merchant()
    assert other.get(f"/stores/{sid}/syncs").status_code == 404
    assert c.get(f"/stores/{sid}/syncs").json() == {"runs": []}