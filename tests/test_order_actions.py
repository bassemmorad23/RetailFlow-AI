"""Merchant order actions: approve / reject / push / ship, customer notifications, isolation."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.channels.base import SendResult
from app.commerce import order_actions as actions
from app.commerce import order_notify
from app.commerce import repository as orders
from app.commerce.models import CustomerDetails, OrderItem
from app.commerce.platform_orders import PushResult
from app.commerce.stock import StockCheck
from app.inbox import repository as inbox_repo

CUSTOMER = CustomerDetails(name="Sara", phone="+201012345678", country="EG", region_code="EG-GZ",
                           region_name="Giza", city="Dokki", address_line="12 Tahrir St")
ITEM = OrderItem(product_id="tee", variant_sku="tee-L", name="Cotton T-Shirt",
                 variant_attrs={"size": "L"}, quantity=2, unit_price=320.0)


@pytest.fixture
def env(monkeypatch):
    st = {"stock": "in_stock", "platform": None, "push": PushResult(True, "gid://Order/1", "#1001"),
          "sent": [], "pushes": 0}

    def fake_push(store_id, platform, order):
        st["pushes"] += 1
        if isinstance(st["push"], Exception):
            raise st["push"]
        return st["push"]

    def fake_send(channel, store_id, recipient, text):
        st["sent"].append(text)
        return SendResult(True, [f"m_{uuid.uuid4().hex[:6]}"])

    monkeypatch.setattr(actions, "check_stock", lambda sid, reqs: [StockCheck(st["stock"], "live") for _ in reqs])
    monkeypatch.setattr(actions, "store_platform", lambda sid: st["platform"])
    monkeypatch.setattr(actions, "push_order", fake_push)
    monkeypatch.setattr(order_notify, "send", fake_send)
    return st


def _merchant():
    c = TestClient(app)
    c.post("/auth/register", json={"email": f"oa_{uuid.uuid4().hex[:8]}@example.com", "password": "Strong-pass-123"})
    sid = c.post("/stores", json={"display_name": "S", "industry": "fashion", "country": "EG"}).json()["store_id"]
    return c, sid


def _order(sid, channel="web", shipping_fee=60.0):
    cid = inbox_repo.get_or_create_conversation(sid, channel, "2010" + uuid.uuid4().hex[:6])["id"]
    inbox_repo.add_message(sid, cid, "customer", "yes", delivery_status="received")
    o = orders.create_order(sid, channel=channel, customer_external_id="201012345678", customer=CUSTOMER,
                            items=[ITEM], currency="EGP", shipping_fee=shipping_fee,
                            idempotency_key=uuid.uuid4().hex, conversation_id=cid)
    return o, cid


def _post(c, sid, oid, action, body=None):
    return c.post(f"/stores/{sid}/orders/{oid}/{action}", json=body if body is not None else {})


# ---------------------------------------------------------------- approve

def test_csv_store_approve_confirms_customer(env):
    c, sid = _merchant()
    o, _ = _order(sid)
    r = _post(c, sid, o["id"], "approve")
    assert r.status_code == 200
    body = r.json()
    assert (body["status"], body["push"]["status"]) == ("approved", "not_needed")
    assert len(env["sent"]) == 1 and "SF-1001" in env["sent"][0] and "تم تأكيد" in env["sent"][0]
    assert "700.00 EGP" in env["sent"][0]


def test_manual_shipping_requires_fee(env):
    c, sid = _merchant()
    o, _ = _order(sid, shipping_fee=None)
    r = _post(c, sid, o["id"], "approve")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "shipping_fee_required"
    body = _post(c, sid, o["id"], "approve", {"shipping_fee": 45}).json()
    assert (body["shipping_fee"], body["total"], body["shipping_status"]) == (45.0, 685.0, "quoted")


def test_out_of_stock_blocks_approval(env):
    env["stock"] = "out_of_stock"
    c, sid = _merchant()
    o, _ = _order(sid)
    r = _post(c, sid, o["id"], "approve")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "out_of_stock"
    assert orders.get_order(sid, o["id"])["status"] == "pending_approval" and env["sent"] == []


def test_cannot_approve_twice(env):
    c, sid = _merchant()
    o, _ = _order(sid)
    _post(c, sid, o["id"], "approve")
    assert _post(c, sid, o["id"], "approve").json()["detail"]["code"] == "invalid_status"


# ---------------------------------------------------------------- platform push

def test_platform_push_success_then_notify(env):
    env["platform"] = "shopify"
    c, sid = _merchant()
    o, _ = _order(sid)
    body = _post(c, sid, o["id"], "approve").json()
    assert body["push"]["status"] == "succeeded" and body["platform"]["order_id"] == "gid://Order/1"
    assert len(env["sent"]) == 1


def test_push_failure_no_notify_then_retry(env):
    env["platform"] = "woocommerce"
    env["push"] = PushResult(False, error="http_500")
    c, sid = _merchant()
    o, _ = _order(sid)
    body = _post(c, sid, o["id"], "approve").json()
    assert (body["status"], body["push"]["status"], body["push"]["error"]) == ("approved", "failed", "http_500")
    assert env["sent"] == []  # customer not told "confirmed" yet

    env["push"] = PushResult(True, "55", "55")
    body = _post(c, sid, o["id"], "retry-push").json()
    assert body["push"]["status"] == "succeeded" and body["push"]["attempts"] == 2
    assert len(env["sent"]) == 1
    assert _post(c, sid, o["id"], "retry-push").json()["detail"]["code"] == "nothing_to_retry"


def test_push_crash_never_leaves_pending(env):
    env["platform"] = "shopify"
    env["push"] = RuntimeError("boom")
    c, sid = _merchant()
    o, _ = _order(sid)
    assert _post(c, sid, o["id"], "approve").json()["push"]["status"] == "failed"


# ---------------------------------------------------------------- reject / ship / deliver

def test_reject_hides_internal_reason(env):
    c, sid = _merchant()
    o, _ = _order(sid)
    body = _post(c, sid, o["id"], "reject", {"reason": "suspected_fake", "message": "Please call us."}).json()
    assert body["status"] == "rejected" and body["rejection"]["reason"] == "suspected_fake"
    assert "suspected" not in env["sent"][0].lower() and "Please call us." in env["sent"][0]


def test_ship_and_deliver_csv_only(env):
    c, sid = _merchant()
    o, _ = _order(sid)
    _post(c, sid, o["id"], "approve")
    assert _post(c, sid, o["id"], "ship").json()["status"] == "shipped"
    assert "shipped" in env["sent"][-1]
    assert _post(c, sid, o["id"], "deliver").json()["status"] == "delivered"

    env["platform"] = "shopify"
    o2, _ = _order(sid)
    _post(c, sid, o2["id"], "approve")
    assert _post(c, sid, o2["id"], "ship").json()["detail"]["code"] == "managed_by_platform"


def test_closed_reply_window_leaves_internal_note(env):
    c, sid = _merchant()
    o, cid = _order(sid, channel="whatsapp")
    inbox_repo._db()["inbox_conversations"].update_one(
        {"store_id": sid, "id": cid},
        {"$set": {"last_customer_message_at": datetime.now(timezone.utc) - timedelta(hours=30)}})
    _post(c, sid, o["id"], "approve")
    assert env["sent"] == []
    msgs, _ = inbox_repo.list_messages(sid, cid)
    assert any("not notified" in m["text"] for m in msgs if m["delivery_status"] == "internal")


# ---------------------------------------------------------------- listing + isolation

def test_list_and_isolation(env):
    c, sid = _merchant()
    other, other_sid = _merchant()
    o, _ = _order(sid)
    _order(sid)
    _post(c, sid, o["id"], "approve")

    assert len(c.get(f"/stores/{sid}/orders").json()["orders"]) == 2
    assert [x["id"] for x in c.get(f"/stores/{sid}/orders", params={"status": "approved"}).json()["orders"]] == [o["id"]]

    assert other.get(f"/stores/{sid}/orders").status_code == 404
    assert other.get(f"/stores/{other_sid}/orders/{o['id']}").status_code == 404
    assert _post(other, other_sid, o["id"], "reject", {"reason": "other"}).status_code == 404
    assert TestClient(app).get(f"/stores/{sid}/orders").status_code == 401