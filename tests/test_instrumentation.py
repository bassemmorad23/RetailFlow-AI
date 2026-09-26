"""Analytics instrumentation: case resolved_at, fulfilment snapshots + refresh, AI reply metadata."""

import uuid

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.commerce import order_status as st
from app.commerce import repository as repo
from app.commerce.models import CustomerDetails, OrderItem
from app.inbox.service import _ai_meta
from app.schemas.models import AgentReply, EmotionLabel, EmotionResult, IntentLabel, IntentResult, ProductRecommendation

CUSTOMER = CustomerDetails(name="Sara", phone="+201012345678", country="EG", region_code="EG-GZ",
                           region_name="Giza", city="Dokki", address_line="12 Tahrir St")


def _order(sid, pushed=True, status="approved"):
    o = repo.create_order(sid, channel="web", customer_external_id="v", customer=CUSTOMER,
                          items=[OrderItem(product_id="tee", name="T", quantity=1, unit_price=100)],
                          currency="EGP", shipping_fee=10, idempotency_key=uuid.uuid4().hex)
    if status != "pending_approval":
        repo.change_order_status(sid, o["id"], new_status="approved", by="m")
    if pushed:
        repo.set_platform_ref(sid, o["id"], platform="shopify", platform_order_id=f"gid://Order/{uuid.uuid4().hex[:6]}")
    return repo.get_order(sid, o["id"])


# ---------------------------------------------------------------- cases

def test_case_resolved_at_set_and_cleared():
    sid = f"store_in_{uuid.uuid4().hex[:8]}"
    case = repo.create_case(sid, case_type="return", description="x")
    repo.set_case_status(sid, case["id"], "resolved")
    assert repo.get_case(sid, case["id"])["resolved_at"] is not None
    repo.set_case_status(sid, case["id"], "open")
    assert repo.get_case(sid, case["id"])["resolved_at"] is None


# ---------------------------------------------------------------- fulfilment snapshots

def test_live_status_saves_snapshot(monkeypatch):
    sid = f"store_in_{uuid.uuid4().hex[:8]}"
    o = _order(sid)
    monkeypatch.setattr(st, "get_credentials", lambda s, src: {"shop": "t.myshopify.com", "access_token": "x"})
    node = {"displayFulfillmentStatus": "FULFILLED", "cancelledAt": None,
            "fulfillments": [{"displayStatus": "IN_TRANSIT", "trackingInfo": [{"company": "Aramex", "number": "9"}]}]}
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": {"node": node}})))
    assert st.live_status(sid, o, client) == ("shipped", ["Aramex 9"])
    snap = repo.get_order(sid, o["id"])["fulfilment"]
    assert (snap["state"], snap["tracking"]) == ("shipped", ["Aramex 9"]) and snap["checked_at"]


def test_refresh_only_open_pushed_orders(monkeypatch):
    sid = f"store_in_{uuid.uuid4().hex[:8]}"
    open_order = _order(sid)
    done = _order(sid)
    repo.set_fulfilment(sid, done["id"], "delivered", [])
    _order(sid, pushed=False)                      # CSV-style: nothing to refresh
    _order(sid, status="pending_approval")         # not approved yet
    calls = []
    monkeypatch.setattr(st, "_shopify_state", lambda s, oid, http: calls.append(oid) or ("preparing", []))

    assert st.refresh_platform_orders(sid) == {"checked": 1, "changed": 1, "failed": 0}
    assert calls == [open_order["platform"]["order_id"]]
    assert st.refresh_platform_orders(sid) == {"checked": 1, "changed": 0, "failed": 0}  # same state again

    monkeypatch.setattr(st, "_shopify_state", lambda s, oid, http: None)
    assert st.refresh_platform_orders(sid)["failed"] == 1


def test_refresh_endpoint_owner_only(monkeypatch):
    monkeypatch.setattr(st, "_shopify_state", lambda s, oid, http: ("shipped", []))
    c = TestClient(app)
    c.post("/auth/register", json={"email": f"in_{uuid.uuid4().hex[:8]}@example.com", "password": "Strong-pass-123"})
    sid = c.post("/stores", json={"display_name": "S", "industry": "fashion", "country": "EG"}).json()["store_id"]
    _order(sid)
    assert c.post(f"/stores/{sid}/orders/refresh-platform").json() == {"checked": 1, "changed": 1, "failed": 0}
    other = TestClient(app)
    other.post("/auth/register", json={"email": f"in_{uuid.uuid4().hex[:8]}@example.com", "password": "Strong-pass-123"})
    assert other.post(f"/stores/{sid}/orders/refresh-platform").status_code == 404


# ---------------------------------------------------------------- AI reply metadata

def test_ai_meta_includes_order_and_aftersales_outcomes():
    reply = AgentReply(
        reply_text="hi", conversation_id="c",
        emotion=EmotionResult(label=EmotionLabel.NEUTRAL, confidence=0.9),
        intent=IntentResult(label=IntentLabel.READY_TO_BUY, confidence=0.8),
        recommendations=[ProductRecommendation(product_id="tee", name="T", price=1, reason="r")],
        order_event="order_created", aftersales_outcome=None,
    )
    meta = _ai_meta(reply)
    assert (meta["intent"], meta["product_ids"], meta["order_event"], meta["aftersales"]) == \
           ("ready_to_buy", ["tee"], "order_created", None)