"""Confirmation -> order: pure 'yes' only, re-check prices/stock, idempotent creation, merchant notified."""

import types
import uuid

import pytest

from app.commerce import order_flow as flow
from app.commerce import repository as orders
from app.commerce.item_resolver import ResolvedItem
from app.commerce.models import OrderItem
from app.commerce.order_extractor import OrderExtraction
from app.commerce.shipping import ShippingSettings
from app.commerce.stock import StockCheck
from app.inbox import repository as inbox_repo
from app.schemas.models import IntentLabel, IntentResult, Product, Variant

READY = IntentResult(label=IntentLabel.READY_TO_BUY, confidence=0.8)
OTHER = IntentResult(label=IntentLabel.OTHER, confidence=0.9)
DETAILS = {"name": "Sara", "phone": "01012345678", "region": "Giza", "city": "Dokki", "address_line": "12 Tahrir St"}


def _item(qty=2):
    return OrderItem(product_id="tee", variant_sku="tee-L", name="Cotton T-Shirt",
                     variant_attrs={"size": "L"}, quantity=qty, unit_price=320.0)


@pytest.fixture
def env(monkeypatch):
    st = {"price": 320.0, "stock": "in_stock", "extraction": OrderExtraction(),
          "shipping": ShippingSettings(method="fixed", fixed={"fee": 60})}
    monkeypatch.setattr(flow, "extract_order_details", lambda text, pending_region_suggestions=None: st["extraction"])
    monkeypatch.setattr(flow, "resolve_item", lambda sid, req, recs: ResolvedItem(
        "ok", item=_item(req.get("quantity") or 2), product_name="Cotton T-Shirt", stock="in_stock"))
    monkeypatch.setattr(flow, "get_shipping", lambda sid: st["shipping"])
    monkeypatch.setattr(flow, "get_settings", lambda sid: types.SimpleNamespace(country="EG", currency="EGP"))
    monkeypatch.setattr(flow, "fetch_products", lambda sid, ids: [Product.model_construct(
        product_id="tee", name="Cotton T-Shirt", price=300.0, attributes={}, specifications={},
        variants=[Variant.model_construct(sku="tee-L", price=st["price"], attributes={"size": "L"}, specifications={})])])
    monkeypatch.setattr(flow, "check_stock", lambda sid, reqs: [StockCheck(st["stock"], "live") for _ in reqs])
    return st


def _conv():
    sid = f"store_conf_{uuid.uuid4().hex[:8]}"
    cid = inbox_repo.get_or_create_conversation(sid, "web", "v_" + uuid.uuid4().hex[:6])["id"]
    return sid, cid


def _turn(sid, cid, text="msg", intent=READY):
    return flow.process_order_turn(store_id=sid, conversation_id=cid, channel="web", customer_external_id="v1",
                                   text=text, intent=intent, recommendations=[])


def _ready_summary(env, sid, cid):
    env["extraction"] = OrderExtraction(items=[{"product": "it", "quantity": 2}], customer=DETAILS)
    t = _turn(sid, cid)
    assert t.event == "awaiting_confirmation"
    env["extraction"] = OrderExtraction()
    return t


def _internal_notes(sid, cid):
    msgs, _ = inbox_repo.list_messages(sid, cid)
    return [m["text"] for m in msgs if m["delivery_status"] == "internal"]


# ---------------------------------------------------------------- happy path

@pytest.mark.parametrize("text", ["yes", "Confirm!", "تمام", "aywa"])
def test_plain_confirmation_creates_order(env, text):
    sid, cid = _conv()
    _ready_summary(env, sid, cid)
    t = _turn(sid, cid, text=text, intent=OTHER)

    assert t.event == "order_created"
    o = t.order
    assert (o["number"], o["status"], o["subtotal"], o["shipping_fee"], o["total"]) == \
           ("SF-1001", "pending_approval", 640.0, 60.0, 700.0)
    assert o["customer"]["phone"] == "+201012345678" and o["customer"]["region_code"] == "EG-GZ"
    assert "ORDER PLACED: order number SF-1001" in t.state_text
    assert flow.workflow.load_draft(sid, cid).draft is None
    assert any("New order SF-1001" in n for n in _internal_notes(sid, cid))
    assert "order.created" in [e["type"] for e in inbox_repo.events_after(sid, 0)]


def test_extractor_confirm_flag_also_works(env):
    sid, cid = _conv()
    _ready_summary(env, sid, cid)
    env["extraction"] = OrderExtraction(confirm=True)
    assert _turn(sid, cid, text="yes please go ahead with it", intent=OTHER).event == "order_created"


def test_manual_shipping_order_has_no_total(env):
    env["shipping"] = ShippingSettings(method="manual")
    sid, cid = _conv()
    _ready_summary(env, sid, cid)
    t = _turn(sid, cid, text="yes", intent=OTHER)
    o = t.order
    assert (o["shipping_fee"], o["total"], o["shipping_status"]) == (None, None, "pending_merchant")
    assert "store will confirm the shipping fee" in t.state_text


# ---------------------------------------------------------------- not a confirmation

def test_yes_with_a_change_is_not_confirmation(env):
    sid, cid = _conv()
    _ready_summary(env, sid, cid)
    env["extraction"] = OrderExtraction(items=[{"product": "it", "quantity": 3}])
    t = _turn(sid, cid, text="yes but make it 3", intent=OTHER)
    assert t.event == "awaiting_confirmation" and t.order is None
    assert orders.get_order_by_number(sid, "SF-1001") is None


def test_yes_before_summary_does_nothing(env):
    sid, cid = _conv()
    env["extraction"] = OrderExtraction(items=[{"product": "it"}])
    _turn(sid, cid)  # collecting_details
    env["extraction"] = OrderExtraction()
    assert _turn(sid, cid, text="yes", intent=OTHER).event == "collecting_details"


# ---------------------------------------------------------------- re-check at confirmation

def test_price_change_requires_reconfirmation(env):
    sid, cid = _conv()
    old_hash = _ready_summary(env, sid, cid).draft.confirmation_hash
    env["price"] = 350.0
    t = _turn(sid, cid, text="yes", intent=OTHER)
    assert t.event == "awaiting_confirmation" and t.order is None
    assert "price of Cotton T-Shirt changed to 350.00 EGP" in t.state_text
    assert t.draft.confirmation_hash != old_hash
    assert _turn(sid, cid, text="yes", intent=OTHER).order["subtotal"] == 700.0  # confirms the new summary


def test_out_of_stock_at_confirmation_removes_item(env):
    sid, cid = _conv()
    _ready_summary(env, sid, cid)
    env["stock"] = "out_of_stock"
    t = _turn(sid, cid, text="yes", intent=OTHER)
    assert t.event == "collecting_items" and t.order is None
    assert "just went out of stock" in t.state_text