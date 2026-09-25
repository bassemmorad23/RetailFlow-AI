"""After-sales: resolve first, escalate when needed; one case per conversation; merchant APIs."""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.commerce import aftersales as asl
from app.commerce import order_status as st
from app.commerce import repository as repo
from app.commerce.models import CustomerDetails, OrderItem
from app.commerce.policies import StorePolicies
from app.inbox import repository as inbox_repo
from app.schemas.models import IntentLabel, IntentResult

COMPLAINT = IntentResult(label=IntentLabel.COMPLAINT, confidence=0.7)
OTHER = IntentResult(label=IntentLabel.OTHER, confidence=0.9)
POLICIES = StorePolicies(returns="Returns within 14 days, unused, with tags.", delivery="2-4 working days in Egypt.")
CUSTOMER = CustomerDetails(name="Sara", phone="+201012345678", country="EG", region_code="EG-GZ",
                           region_name="Giza", city="Dokki", address_line="12 Tahrir St")


@pytest.fixture
def env(monkeypatch):
    s = {"problem": {"problem_type": "return", "policy_topic": "other", "summary": "Wants to return the shirt"},
         "status": None}
    monkeypatch.setattr(asl, "read_problem", lambda text: s["problem"])
    monkeypatch.setattr(st, "live_status", lambda store_id, order: s["status"])
    return s


def _setup(with_order=True, order_status=None):
    sid = f"store_as_{uuid.uuid4().hex[:8]}"
    cid = inbox_repo.get_or_create_conversation(sid, "whatsapp", "wa_cust")["id"]
    order = None
    if with_order:
        order = repo.create_order(sid, channel="whatsapp", customer_external_id="wa_cust", customer=CUSTOMER,
                                  items=[OrderItem(product_id="tee", name="Cotton T-Shirt", quantity=1, unit_price=300)],
                                  currency="EGP", shipping_fee=60, idempotency_key=uuid.uuid4().hex)
        if order_status:
            repo.change_order_status(sid, order["id"], new_status="approved", by="m")
            if order_status == "shipped":
                repo.change_order_status(sid, order["id"], new_status="shipped", by="m")
    return sid, cid, order


def _run(sid, cid, text="I want to return it", intent=COMPLAINT, policies=POLICIES):
    return asl.process_aftersales(store_id=sid, conversation_id=cid, channel="whatsapp", customer_external_id="wa_cust",
                                  text=text, intent=intent, policies=policies, store_country="EG")


def _conv(sid, cid):
    return inbox_repo.get_conversation(sid, cid)


# ---------------------------------------------------------------- trigger

@pytest.mark.parametrize("text,expected", [
    ("I want to return the shirt", True), ("the item arrived damaged", True), ("عايز استرجع التيشيرت", True),
    ("الطلب موصلش", True), ("what's your return policy?", True), ("Do you have size M?", False),
])
def test_trigger(text, expected):
    assert asl.wants_aftersales(OTHER, text) is expected


def test_not_after_sales_does_nothing(env):
    sid, cid, _ = _setup()
    assert _run(sid, cid, text="hello", intent=OTHER) is None


# ---------------------------------------------------------------- resolved by the AI

def test_policy_question_answered_no_case(env):
    env["problem"] = {"problem_type": "policy_question", "policy_topic": "returns", "summary": "asks return policy"}
    sid, cid, _ = _setup(with_order=False)
    t = _run(sid, cid, text="what's your return policy?")
    assert t.outcome == "resolved" and "Returns within 14 days" in t.state_text
    assert repo.find_open_case(sid, cid) is None and not _conv(sid, cid).get("needs_attention")


def test_late_order_not_shipped_is_explained(env):
    env["problem"] = {"problem_type": "late", "policy_topic": "other", "summary": "order is late"}
    sid, cid, _ = _setup(order_status="approved")
    t = _run(sid, cid, text="my order is late")
    assert t.outcome == "resolved" and "hasn't shipped yet" in t.state_text and "2-4 working days" in t.state_text


def test_late_with_tracking_shares_tracking(env):
    env["problem"] = {"problem_type": "late", "policy_topic": "other", "summary": "order is late"}
    env["status"] = ("shipped", ["Aramex 123"])
    sid, cid, _ = _setup(order_status="shipped")
    t = _run(sid, cid, text="it's late")
    assert t.outcome == "resolved" and "Tracking: Aramex 123" in t.state_text


# ---------------------------------------------------------------- escalated

def test_return_escalates_with_policy_and_order(env):
    sid, cid, order = _setup()
    t = _run(sid, cid)
    assert t.outcome == "escalated"
    assert (t.case["type"], t.case["order_id"], t.case["status"]) == ("return", order["id"], "open")
    assert "Returns within 14 days" in t.state_text and t.case["number"] in t.state_text
    assert "Never approve" in t.state_text
    assert _conv(sid, cid)["needs_attention"] is True
    msgs, _ = inbox_repo.list_messages(sid, cid)
    note = [m["text"] for m in msgs if m["delivery_status"] == "internal"][-1]
    assert "Wants to return the shirt" in note and "Explained returns policy" in note and "Checked order" in note


def test_not_received_after_shipping_escalates(env):
    env["problem"] = {"problem_type": "not_received", "policy_topic": "other", "summary": "says not received"}
    sid, cid, _ = _setup(order_status="shipped")
    t = _run(sid, cid, text="I never received it")
    assert t.outcome == "escalated" and t.case["type"] == "delivery_issue"


def test_missing_policy_escalates_without_inventing(env):
    env["problem"] = {"problem_type": "policy_question", "policy_topic": "warranty", "summary": "asks warranty"}
    sid, cid, _ = _setup(with_order=False)
    t = _run(sid, cid, text="what's the warranty?")
    assert t.outcome == "escalated" and "NOT SET" in t.state_text


def test_human_request_escalates(env):
    env["problem"] = {"problem_type": "human", "policy_topic": "other", "summary": "wants a person"}
    sid, cid, _ = _setup(with_order=False)
    assert _run(sid, cid, text="let me speak to someone").outcome == "escalated"


# ---------------------------------------------------------------- order not identified: ask once

def test_asks_for_order_once_then_escalates(env):
    sid, cid, _ = _setup(with_order=False)
    first = _run(sid, cid)
    assert first.outcome == "ask_order" and "order number" in first.state_text
    assert repo.find_open_case(sid, cid) is None
    env["problem"] = {"problem_type": "none", "policy_topic": "other", "summary": ""}
    second = _run(sid, cid, text="I don't have it", intent=OTHER)  # continues the same problem
    assert second.outcome == "escalated" and second.case["order_id"] is None
    msgs, _ = inbox_repo.list_messages(sid, cid)
    assert any("Order not identified" in m["text"] for m in msgs if m["delivery_status"] == "internal")


# ---------------------------------------------------------------- one case per conversation

def test_second_problem_updates_same_case(env):
    sid, cid, _ = _setup()
    first = _run(sid, cid).case
    env["problem"] = {"problem_type": "exchange", "policy_topic": "other", "summary": "Or exchange for size L"}
    second = _run(sid, cid, text="or exchange it").case
    assert first["id"] == second["id"] and "Or exchange for size L" in second["description"]
    assert len(repo.list_cases(sid)) == 1


# ---------------------------------------------------------------- merchant APIs

def _merchant():
    c = TestClient(app)
    c.post("/auth/register", json={"email": f"as_{uuid.uuid4().hex[:8]}@example.com", "password": "Strong-pass-123"})
    sid = c.post("/stores", json={"display_name": "S", "industry": "fashion", "country": "EG"}).json()["store_id"]
    return c, sid


def test_resolving_case_clears_attention(env):
    c, sid = _merchant()
    cid = inbox_repo.get_or_create_conversation(sid, "web", "v1")["id"]
    case = asl._escalate(sid, cid, "return", "wants return", None, [], [], []).case
    assert c.get(f"/stores/{sid}/conversations", params={"needs_attention": True}).json()["conversations"]
    r = c.post(f"/stores/{sid}/cases/{case['id']}/status", json={"status": "resolved"})
    assert r.status_code == 200 and r.json()["status"] == "resolved"
    assert c.get(f"/stores/{sid}/conversations", params={"needs_attention": True}).json()["conversations"] == []


def test_manual_clear_and_case_isolation(env):
    c, sid = _merchant()
    other, _ = _merchant()
    cid = inbox_repo.get_or_create_conversation(sid, "web", "v2")["id"]
    case = asl._escalate(sid, cid, "other", "question", None, [], [], []).case
    assert c.post(f"/stores/{sid}/conversations/{cid}/attention/clear").json()["needs_attention"] is False
    assert other.get(f"/stores/{sid}/cases").status_code == 404
    assert other.post(f"/stores/{sid}/cases/{case['id']}/status", json={"status": "closed"}).status_code == 404


def test_policies_api(env):
    c, sid = _merchant()
    other, _ = _merchant()
    assert c.put(f"/stores/{sid}/policies", json={"returns": "14 days"}).status_code == 200
    assert c.get(f"/stores/{sid}/policies").json()["returns"] == "14 days"
    assert c.put(f"/stores/{sid}/policies", json={"returns": "x" * 2001}).status_code == 422
    assert c.put(f"/stores/{sid}/policies", json={"loyalty": "x"}).status_code == 422
    assert other.get(f"/stores/{sid}/policies").status_code == 404