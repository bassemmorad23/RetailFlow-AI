"""Commerce Intelligence: exact numbers on a seeded store, definitions, isolation, period validation."""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.commerce import repository as orders
from app.commerce.models import CustomerDetails, OrderItem
from app.inbox import repository as inbox
from app.products.product_store import _get_collection as products_col
from app.response.response_generator import FALLBACK_REPLY
from app.schemas.models import Product, Variant

CUSTOMER = CustomerDetails(name="Sara", phone="+201012345678", country="EG", region_code="EG-GZ",
                           region_name="Giza", city="Dokki", address_line="12 Tahrir St")


def _tee(qty):
    return [OrderItem(product_id="tee", name="Cotton T-Shirt", quantity=qty, unit_price=300)]


def _merchant():
    c = TestClient(app)
    c.post("/auth/register", json={"email": f"an_{uuid.uuid4().hex[:8]}@example.com", "password": "Strong-pass-123"})
    sid = c.post("/stores", json={"display_name": "S", "industry": "fashion", "country": "EG"}).json()["store_id"]
    return c, sid


def _order(sid, channel, cid, items, fee, status=None):
    o = orders.create_order(sid, channel=channel, customer_external_id="x", customer=CUSTOMER, items=items,
                            currency="EGP", shipping_fee=fee, idempotency_key=uuid.uuid4().hex, conversation_id=cid)
    if status:
        orders.change_order_status(sid, o["id"], new_status=status, by="m")
    return o


def _seed(sid):
    a = inbox.get_or_create_conversation(sid, "web", "va")["id"]
    inbox.add_message(sid, a, "customer", "I want the shirt", delivery_status="received")
    inbox.add_message(sid, a, "ai", "Great", delivery_status="sent",
                      ai_meta={"intent": "ready_to_buy", "product_ids": ["tee"]})
    _order(sid, "web", a, _tee(2), 60, status="approved")                       # 660 confirmed

    b = inbox.get_or_create_conversation(sid, "instagram", "ib")["id"]
    inbox.add_message(sid, b, "customer", "It arrived torn", delivery_status="received")
    inbox.add_message(sid, b, "ai", "Sorry", delivery_status="sent", ai_meta={"intent": "complaint", "aftersales": "escalated"})
    inbox.add_message(sid, b, "human", "We'll replace it", delivery_status="sent")
    orders.create_case(sid, case_type="return", description="torn", conversation_id=b)

    c = inbox.get_or_create_conversation(sid, "instagram", "ic")["id"]
    inbox.add_message(sid, c, "customer", "??", delivery_status="received")
    inbox.add_message(sid, c, "system", FALLBACK_REPLY, delivery_status="sent")
    _order(sid, "instagram", c, _tee(1), None)                                  # 300 placed, pending, fee missing
    _order(sid, "instagram", c, _tee(5), 60, status="rejected")                 # excluded

    products_col().insert_one(Product(
        product_id="tee", store_id=sid, name="Cotton T-Shirt", price=300, stock_status="in_stock",
        variants=[Variant(sku="tee-M", price=300, attributes={"size": "M"}, stock_status="in_stock", stock_quantity=2),
                  Variant(sku="tee-L", price=300, attributes={"size": "L"}, stock_status="out_of_stock", stock_quantity=0)],
    ).model_dump())
    return a, b, c


@pytest.fixture
def seeded():
    c, sid = _merchant()
    ids = _seed(sid)
    other_c, other_sid = _merchant()
    _seed(other_sid)  # a second store with the same data must never leak in
    return c, sid, ids, other_c


def _get(c, sid, section, **params):
    r = c.get(f"/stores/{sid}/analytics/{section}", params={"period": "today", **params})
    assert r.status_code == 200, r.text
    return r.json()


def test_overview_kpis_and_attention(seeded):
    c, sid, _, _ = seeded
    body = _get(c, sid, "overview")
    k = body["kpis"]
    assert body["currency"] == "EGP" and body["period"]["timezone"] == "Africa/Cairo"
    assert (k["orders"]["value"], k["revenue_placed"]["value"], k["revenue_confirmed"]["value"]) == (2, 960.0, 660.0)
    assert k["average_order_value"]["value"] == 480.0
    assert (k["conversations"]["value"], k["conversion_rate"]["value"]) == (3, 66.7)
    assert (k["handled_by_ai_rate"]["value"], k["human_takeover_rate"]["value"], k["escalation_rate"]["value"]) == (33.3, 33.3, 33.3)
    assert k["revenue_placed"]["previous"] == 0 and k["revenue_placed"]["change_pct"] is None
    assert (k["pending_orders"]["value"], k["orders_needing_attention"]["value"]) == (1, 1)
    types = {i["type"]: i["count"] for i in body["needs_attention"]}
    assert types["orders_awaiting_approval"] == 1 and types["shipping_fee_missing"] == 1 and types["open_support_cases"] == 1
    assert body["needs_attention"][0]["severity"] == "high"
    assert body["usage"]["limit"] > 0 and body["usage"]["remaining"] <= body["usage"]["limit"]


def test_sales(seeded):
    c, sid, _, _ = seeded
    body = _get(c, sid, "sales")
    assert body["totals"]["conversion_rate"]["value"] == 66.7
    assert body["by_status"] == {"approved": 1, "pending_approval": 1, "rejected": 1}
    day = body["daily"][-1]
    assert (day["orders"], day["revenue_placed"], day["conversations"]) == (2, 960.0, 3)


def test_ai_performance(seeded):
    c, sid, _, _ = seeded
    body = _get(c, sid, "ai")
    assert (body["ai_replies"]["value"], body["merchant_replies"]["value"]) == (2, 1)
    assert (body["handled_by_ai"]["value"], body["escalated"]["value"]) == (1, 1)
    assert body["orders_from_ai_conversations"]["value"] == 2
    assert body["first_response_seconds_median"]["value"] is not None
    assert body["resolution_time"]["available"] is False


def test_channels(seeded):
    c, sid, _, _ = seeded
    by = {x["channel"]: x for x in _get(c, sid, "channels")["channels"]}
    assert (by["web"]["conversations"], by["web"]["orders"], by["web"]["revenue_confirmed"], by["web"]["conversion_rate"]) == (1, 1, 660.0, 100.0)
    assert (by["instagram"]["conversations"], by["instagram"]["orders"], by["instagram"]["revenue_placed"]) == (2, 1, 300.0)
    assert by["instagram"]["escalation_rate"] == 50.0 and by["web"]["handled_by_ai_rate"] == 100.0


def test_products(seeded):
    c, sid, _, _ = seeded
    body = _get(c, sid, "products")
    assert body["most_ordered"] == [{"product_id": "tee", "name": "Cotton T-Shirt", "quantity": 3, "orders": 2, "revenue": 900.0}]
    rec = body["most_recommended"][0]
    assert (rec["product_id"], rec["name"], rec["conversations"], rec["conversion_rate"]) == ("tee", "Cotton T-Shirt", 1, 100.0)
    assert body["low_stock"] == [{"product_id": "tee", "name": "Cotton T-Shirt", "variant": {"size": "M"}, "level": "low_stock"}]
    assert body["out_of_stock"] == [{"product_id": "tee", "name": "Cotton T-Shirt", "variant": {"size": "L"}}]
    assert "external" not in str(body)


def test_customers(seeded):
    c, sid, _, _ = seeded
    body = _get(c, sid, "customers")
    assert (body["new_conversations"]["value"], body["returning_conversations"]["value"]) == (3, 0)
    assert {i["intent"]: i["count"] for i in body["top_intents"]} == {"ready_to_buy": 1, "complaint": 1}
    assert body["top_support_issues"] == [{"type": "return", "count": 1}]
    assert body["unresolved_by_ai"]["fallback_replies"]["value"] == 1
    assert body["top_product_questions"]["available"] is False


def test_support_and_resolution(seeded):
    c, sid, ids, _ = seeded
    case = orders.find_open_case(sid, ids[1])
    orders.set_case_status(sid, case["id"], "resolved")
    body = _get(c, sid, "support")
    assert (body["opened"]["value"], body["resolved"]["value"], body["pending_now"]["value"]) == (1, 1, 0)
    assert body["ai_resolved_rate"]["value"] == 0.0            # 0 resolved by AI / 1 escalated
    assert body["resolution_time_hours_median"]["value"] is not None
    assert body["daily"][-1] == {"date": body["period"]["to"], "opened": 1, "resolved": 1}


def test_order_operations(seeded):
    c, sid, _, _ = seeded
    body = _get(c, sid, "orders")
    assert body["by_status_now"] == {"approved": 1, "pending_approval": 1, "rejected": 1}
    actions = {a["reason"]: a["count"] for a in body["requiring_action"]}
    assert actions["awaiting_approval"] == 1 and actions["shipping_fee_missing"] == 1
    assert body["average_approval_minutes"]["value"] is not None


def test_isolation_and_validation(seeded):
    c, sid, _, other = seeded
    assert other.get(f"/stores/{sid}/analytics/overview").status_code == 404
    assert TestClient(app).get(f"/stores/{sid}/analytics/overview").status_code == 401
    assert c.get(f"/stores/{sid}/analytics/sales", params={"period": "1y"}).status_code == 422
    r = c.get(f"/stores/{sid}/analytics/sales", params={"period": "custom", "from": "2026-01-01", "to": "2026-01-02"})
    assert r.status_code in (200, 422)  # 422 only if 'to' is in the future on the test machine
    assert c.get(f"/stores/{sid}/analytics/sales", params={"period": "custom"}).status_code == 422