"""Commerce foundation: orders, numbering, transitions, cases, draft state. Isolated test DB."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.commerce import repository as repo
from app.commerce import workflow
from app.commerce.models import MAX_ORDER_LINES, CustomerDetails, OrderDraft, OrderItem
from app.inbox import repository as inbox_repo

CUSTOMER = CustomerDetails(name="Sara", phone="01000000000", governorate="Cairo",
                           city="Nasr City", address_line="12 Abbas St")


def _store() -> str:
    return f"store_com_{uuid.uuid4().hex[:8]}"


def _item(price=450.0, qty=1, pid="shirt") -> OrderItem:
    return OrderItem(product_id=pid, variant_sku=f"{pid}-M", name="Cotton Shirt",
                     variant_attrs={"size": "M"}, quantity=qty, unit_price=price)


def _order(store_id, key=None, items=None, shipping=60.0):
    return repo.create_order(store_id, channel="whatsapp", customer_external_id="2010",
                             customer=CUSTOMER, items=items or [_item()], currency="EGP",
                             shipping_fee=shipping, idempotency_key=key or uuid.uuid4().hex)


# ---------------------------------------------------------------- orders

def test_numbers_sequential_per_store():
    a, b = _store(), _store()
    assert [_order(a)["number"], _order(a)["number"]] == ["SF-1001", "SF-1002"]
    assert _order(b)["number"] == "SF-1001"


def test_totals_computed_server_side():
    o = _order(_store(), items=[_item(450.0, 2), _item(199.99, 1, pid="cap")], shipping=60)
    assert (o["subtotal"], o["shipping_fee"], o["total"]) == (1099.99, 60.0, 1159.99)
    assert o["payment_method"] == "cod" and o["status"] == "pending_approval"


def test_idempotent_creation():
    sid, key = _store(), uuid.uuid4().hex
    first, second = _order(sid, key=key), _order(sid, key=key)
    assert first["id"] == second["id"] and first["number"] == second["number"]


def test_empty_order_rejected():
    with pytest.raises(ValueError):
        repo.create_order(_store(), channel="whatsapp", customer_external_id="2010",
                          customer=CUSTOMER, items=[], currency="EGP",
                          shipping_fee=60, idempotency_key=uuid.uuid4().hex)


def test_order_isolated_per_store():
    a, b = _store(), _store()
    o = _order(a)
    assert repo.get_order(b, o["id"]) is None
    assert repo.get_order_by_number(b, o["number"]) is None
    assert repo.get_order_by_number(a, o["number"].lower())["id"] == o["id"]


def test_customer_order_lookup():
    sid = _store()
    o = _order(sid)
    assert [x["id"] for x in repo.list_customer_orders(sid, "whatsapp", "2010")] == [o["id"]]
    assert repo.list_customer_orders(sid, "whatsapp", "someone_else") == []


# ---------------------------------------------------------------- transitions

def test_allowed_transitions_recorded():
    sid = _store()
    o = _order(sid)
    repo.change_order_status(sid, o["id"], new_status="approved", by="user_1")
    done = repo.change_order_status(sid, o["id"], new_status="shipped", by="user_1")
    assert [h["status"] for h in done["status_history"]] == ["pending_approval", "approved", "shipped"]


@pytest.mark.parametrize("path", [["delivered"], ["rejected", "approved"], ["cancelled", "shipped"]])
def test_invalid_transitions_rejected(path):
    sid = _store()
    o = _order(sid)
    with pytest.raises(repo.InvalidTransition):
        for status in path:
            repo.change_order_status(sid, o["id"], new_status=status, by="user_1")


def test_cross_store_status_change_impossible():
    a, b = _store(), _store()
    o = _order(a)
    with pytest.raises(LookupError):
        repo.change_order_status(b, o["id"], new_status="approved", by="attacker")
    assert repo.get_order(a, o["id"])["status"] == "pending_approval"


# ---------------------------------------------------------------- validation

def test_order_line_limits():
    with pytest.raises(ValidationError):
        OrderDraft(items=[_item()] * (MAX_ORDER_LINES + 1))
    with pytest.raises(ValidationError):
        _item(qty=0)
    with pytest.raises(ValidationError):
        _item(price=-1)


# ---------------------------------------------------------------- support cases

def test_support_case_numbering_and_isolation():
    a, b = _store(), _store()
    c = repo.create_case(a, case_type="return", description="  Wrong size  ", order_id="ord_x")
    assert c["number"] == "CASE-1001" and c["status"] == "open" and c["description"] == "Wrong size"
    assert repo.get_case(b, c["id"]) is None
    assert repo.set_case_status(a, c["id"], "resolved") is True
    assert repo.set_case_status(b, c["id"], "closed") is False


# ---------------------------------------------------------------- draft workflow

def _conversation(store_id: str) -> str:
    return inbox_repo.get_or_create_conversation(store_id, "whatsapp", uuid.uuid4().hex[:8])["id"]


def test_draft_save_load_and_version_check():
    sid = _store()
    cid = _conversation(sid)
    state = workflow.load_draft(sid, cid)
    assert state.draft is None and state.version == 0

    draft = OrderDraft(items=[_item()])
    assert workflow.save_draft(sid, cid, draft, expected_version=0) is True
    assert workflow.save_draft(sid, cid, draft, expected_version=0) is False  # stale

    loaded = workflow.load_draft(sid, cid)
    assert loaded.version == 1 and loaded.draft.items[0].product_id == "shirt"


def test_draft_expires():
    sid = _store()
    cid = _conversation(sid)
    workflow.save_draft(sid, cid, OrderDraft(), expected_version=0)
    inbox_repo._db()["inbox_conversations"].update_one(
        {"store_id": sid, "id": cid},
        {"$set": {"workflow.expires_at": datetime.now(timezone.utc) - timedelta(minutes=1)}})
    state = workflow.load_draft(sid, cid)
    assert state.draft is None and state.version == 1
    assert workflow.save_draft(sid, cid, OrderDraft(), expected_version=state.version) is True


def test_draft_clear_and_isolation():
    a, b = _store(), _store()
    cid = _conversation(a)
    workflow.save_draft(a, cid, OrderDraft(), expected_version=0)
    with pytest.raises(LookupError):
        workflow.load_draft(b, cid)
    assert workflow.save_draft(b, cid, OrderDraft(), expected_version=1) is False
    assert workflow.clear_draft(a, cid, expected_version=1) is True
    assert workflow.load_draft(a, cid).draft is None