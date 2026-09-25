"""Order status: identity rules, neutral answers, lockout, live platform status, no PII leaks."""

import json
import uuid

import httpx
import pytest

from app.commerce import order_status as st
from app.commerce import repository as orders
from app.commerce.models import CustomerDetails, OrderItem
from app.schemas.models import IntentLabel, IntentResult

CUSTOMER = CustomerDetails(name="Sara", phone="+201012345678", country="EG", region_code="EG-GZ",
                           region_name="Giza", city="Dokki", address_line="12 Tahrir St")
ITEM = OrderItem(product_id="tee", variant_sku="tee-L", name="Cotton T-Shirt", variant_attrs={"size": "L"},
                 quantity=2, unit_price=320.0)


@pytest.fixture(autouse=True)
def no_live(monkeypatch):
    monkeypatch.setattr(st, "live_status", lambda store_id, order: None)


def _store():
    return f"store_st_{uuid.uuid4().hex[:8]}"


def _order(sid, customer_id="wa_owner", channel="whatsapp"):
    return orders.create_order(sid, channel=channel, customer_external_id=customer_id, customer=CUSTOMER,
                               items=[ITEM], currency="EGP", shipping_fee=60.0, idempotency_key=uuid.uuid4().hex)


def _ask(sid, text, customer_id="wa_owner", channel="whatsapp", cid="conv_1"):
    return st.order_status_text(store_id=sid, conversation_id=cid, channel=channel,
                                customer_external_id=customer_id, text=text, store_country="EG")


# ---------------------------------------------------------------- trigger

@pytest.mark.parametrize("text,intent,expected", [
    ("Where is my order?", IntentLabel.OTHER, True),
    ("طلبي فين؟", IntentLabel.OTHER, True),
    ("any update on sf-1004", IntentLabel.OTHER, True),
    ("hmm", IntentLabel.ORDER_STATUS, True),
    ("Do you have size M?", IntentLabel.ASKING_AVAILABILITY, False),
])
def test_wants_status(text, intent, expected):
    assert st.wants_status(IntentResult(label=intent, confidence=0.4), text) is expected


# ---------------------------------------------------------------- identity rules

def test_own_orders_shown_without_details():
    sid = _store()
    o = _order(sid)
    text = _ask(sid, "where is my order?")
    assert f"Order {o['number']}: received — waiting for the store to confirm it" in text
    assert "2 × Cotton T-Shirt" in text and "700.00 EGP" in text


def test_number_plus_matching_phone_from_another_identity():
    sid = _store()
    o = _order(sid)
    text = _ask(sid, f"status of {o['number']} my phone 010 1234 5678", customer_id="web_visitor", channel="web")
    assert f"Order {o['number']}" in text


@pytest.mark.parametrize("message", [
    "status of {n}",                          # number only, other identity
    "status of {n} phone 01099999999",        # wrong phone
    "status of SF-9999 phone 01012345678",    # order doesn't exist
])
def test_unverified_gets_the_same_neutral_answer(message):
    sid = _store()
    o = _order(sid)
    assert _ask(sid, message.format(n=o["number"]), customer_id="stranger", channel="web") == st.NEUTRAL


def test_other_stores_orders_never_visible():
    a, b = _store(), _store()
    o = _order(a)
    assert _ask(b, f"{o['number']} 01012345678", customer_id="wa_owner") == st.NEUTRAL


def test_no_orders_and_no_number_is_neutral():
    assert _ask(_store(), "where is my order?") == st.NEUTRAL


def test_lockout_after_failed_attempts_even_with_correct_details():
    sid = _store()
    o = _order(sid)
    for _ in range(st.FAILED_LIMIT):
        _ask(sid, f"{o['number']} 01099999999", customer_id="stranger", channel="web", cid="conv_x")
    locked = _ask(sid, f"{o['number']} 01012345678", customer_id="stranger", channel="web", cid="conv_x")
    assert locked == st.LOCKED
    assert _ask(sid, f"{o['number']} 01012345678", customer_id="stranger", channel="web", cid="conv_other") != st.LOCKED


def test_number_without_phone_does_not_count_as_failure():
    sid = _store()
    o = _order(sid)
    for _ in range(st.FAILED_LIMIT + 2):
        _ask(sid, f"status {o['number']}", customer_id="stranger", channel="web", cid="conv_y")
    assert _ask(sid, f"{o['number']} 01012345678", customer_id="stranger", channel="web", cid="conv_y") != st.LOCKED


def test_never_reveals_address_or_phone():
    sid = _store()
    _order(sid)
    text = _ask(sid, "where is my order")
    assert "Tahrir" not in text and "1012345678" not in text


# ---------------------------------------------------------------- live status

def test_platform_status_used_when_available(monkeypatch):
    sid = _store()
    o = _order(sid)
    orders.set_platform_ref(sid, o["id"], platform="shopify", platform_order_id="gid://Order/1")
    monkeypatch.setattr(st, "live_status", lambda store_id, order: ("shipped", ["Aramex 123 https://t.example/123"]))
    text = _ask(sid, "where is my order")
    assert ": shipped." in text and "Tracking: Aramex 123 https://t.example/123" in text


def test_live_failure_falls_back_with_note():
    sid = _store()
    o = _order(sid)
    orders.set_platform_ref(sid, o["id"], platform="woocommerce", platform_order_id="27")
    assert "couldn't check the latest update right now" in _ask(sid, "where is my order")


def _shopify(node, monkeypatch):
    monkeypatch.setattr(st, "get_credentials", lambda sid, src: {"shop": "t.myshopify.com", "access_token": "x"})
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": {"node": node}})))
    return st._shopify_status("s", "gid://Order/1", client)


def test_shopify_status_mapping(monkeypatch):
    fulfilled = {"displayFulfillmentStatus": "FULFILLED", "cancelledAt": None, "fulfillments": [
        {"displayStatus": "IN_TRANSIT", "trackingInfo": [{"company": "Aramex", "number": "123", "url": "https://t/123"}]}]}
    assert _shopify(fulfilled, monkeypatch) == ("shipped", ["Aramex 123 https://t/123"])
    delivered = {**fulfilled, "fulfillments": [{"displayStatus": "DELIVERED", "trackingInfo": []}]}
    assert _shopify(delivered, monkeypatch)[0] == "delivered"
    assert _shopify({**fulfilled, "cancelledAt": "2026-09-25T00:00:00Z"}, monkeypatch) == ("cancelled", [])
    assert _shopify({"displayFulfillmentStatus": "UNFULFILLED", "fulfillments": []}, monkeypatch)[0] == \
           "confirmed — being prepared"


def test_woocommerce_status_mapping(monkeypatch):
    monkeypatch.setattr(st, "get_credentials", lambda sid, src: {"site_url": "https://s.example", "username": "u", "password": "p"})
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"status": "processing"})))
    assert st._wc_status("s", "27", client) == ("confirmed — being prepared", [])