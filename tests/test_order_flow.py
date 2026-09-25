"""Order draft flow: start rules, items, details, summary + hash, cancel. Real draft storage (test DB)."""

import types
import uuid

import pytest

from app.commerce import order_flow as flow
from app.commerce.item_resolver import ResolvedItem
from app.commerce.models import OrderItem
from app.commerce.order_extractor import OrderExtraction
from app.commerce.shipping import ShippingSettings
from app.inbox import repository as inbox_repo
from app.schemas.models import IntentLabel, IntentResult

READY = IntentResult(label=IntentLabel.READY_TO_BUY, confidence=0.8)
OTHER = IntentResult(label=IntentLabel.OTHER, confidence=0.9)
DETAILS = {"name": "Sara", "phone": "01012345678", "region": "Giza", "city": "Dokki", "address_line": "12 Tahrir St"}


def _tee(size="L", qty=1):
    return OrderItem(product_id="tee", variant_sku=f"tee-{size}", name="Cotton T-Shirt",
                     variant_attrs={"size": size}, quantity=qty, unit_price=300.0)


@pytest.fixture
def env(monkeypatch):
    st = {"extraction": OrderExtraction(), "resolved": ResolvedItem("ok", item=_tee(), product_name="Cotton T-Shirt",
                                                                    stock="in_stock"),
          "shipping": ShippingSettings(method="fixed", fixed={"fee": 60}), "extract_calls": 0}

    def fake_extract(text, pending_region_suggestions=None):
        st["extract_calls"] += 1
        return st["extraction"]

    monkeypatch.setattr(flow, "extract_order_details", fake_extract)
    monkeypatch.setattr(flow, "resolve_item", lambda sid, req, recs: st["resolved"])
    monkeypatch.setattr(flow, "get_shipping", lambda sid: st["shipping"])
    monkeypatch.setattr(flow, "get_settings", lambda sid: types.SimpleNamespace(country="EG", currency="EGP"))
    return st


def _conv(channel="web"):
    sid = f"store_flow_{uuid.uuid4().hex[:8]}"
    cid = inbox_repo.get_or_create_conversation(sid, channel, "2010" + uuid.uuid4().hex[:6])["id"]
    return sid, cid


def _turn(sid, cid, intent=READY, channel="web", sender="v1"):
    return flow.process_order_turn(store_id=sid, conversation_id=cid, channel=channel,
                                   customer_external_id=sender, text="msg", intent=intent, recommendations=[])


# ---------------------------------------------------------------- start rules

def test_no_draft_and_not_buying_does_nothing(env):
    sid, cid = _conv()
    assert _turn(sid, cid, intent=OTHER) is None
    assert env["extract_calls"] == 0  # no LLM cost outside ordering


def test_ordering_unavailable_without_shipping(env):
    env["shipping"] = None
    sid, cid = _conv()
    t = _turn(sid, cid)
    assert t.event == "unavailable" and "not set up" in t.state_text
    assert flow.workflow.load_draft(sid, cid).draft is None


# ---------------------------------------------------------------- collecting

def test_start_adds_item_and_asks_for_details(env):
    env["extraction"] = OrderExtraction(items=[{"product": "it", "attributes": {"size": "L"}, "quantity": None}])
    sid, cid = _conv()
    t = _turn(sid, cid)
    assert t.event == "collecting_details"
    assert "1 × Cotton T-Shirt (size: L) — 300.00 EGP each" in t.state_text
    assert "Missing: full name, phone number, region / governorate, city or area, delivery address" in t.state_text


def test_active_draft_continues_whatever_the_intent(env):
    env["extraction"] = OrderExtraction(items=[{"product": "it"}])
    sid, cid = _conv()
    _turn(sid, cid)
    env["extraction"] = OrderExtraction(customer={"name": "Sara"})
    t = _turn(sid, cid, intent=OTHER)
    assert t is not None and "full name: Sara" in t.state_text


def test_problems_become_notes(env):
    env["extraction"] = OrderExtraction(items=[{"product": "shirt"}], customer={"region": "Gizza", "phone": "12"})
    env["resolved"] = ResolvedItem("needs_variant", product_name="Cotton T-Shirt", options={"size": ["M", "L"]})
    sid, cid = _conv()
    t = _turn(sid, cid)
    assert t.event == "collecting_items"
    assert "Ask which option of Cotton T-Shirt they want (size: M, L)" in t.state_text
    assert "did you mean Giza?" in t.state_text and "phone number is not valid" in t.state_text


def test_out_of_stock_not_added(env):
    env["extraction"] = OrderExtraction(items=[{"product": "cap"}])
    env["resolved"] = ResolvedItem("out_of_stock", product_name="Cap")
    sid, cid = _conv()
    t = _turn(sid, cid)
    assert t.draft.items == [] and "OUT OF STOCK" in t.state_text


def test_whatsapp_number_confirmation_prompt(env):
    env["extraction"] = OrderExtraction(items=[{"product": "it"}])
    sid, cid = _conv("whatsapp")
    t = _turn(sid, cid, channel="whatsapp", sender="201012345678")
    assert "+201012345678 comes from their WhatsApp" in t.state_text


# ---------------------------------------------------------------- summary

def _complete(env, sid, cid, qty=2):
    env["resolved"] = ResolvedItem("ok", item=_tee(qty=qty), product_name="Cotton T-Shirt", stock="in_stock")
    env["extraction"] = OrderExtraction(items=[{"product": "it", "quantity": qty}], customer=DETAILS)
    return _turn(sid, cid)


def test_complete_draft_shows_summary_with_totals(env):
    sid, cid = _conv()
    t = _complete(env, sid, cid)
    assert t.event == "awaiting_confirmation" and t.draft.confirmation_hash
    for expected in ("Subtotal: 600.00 EGP", "Shipping: 60.00 EGP", "Total: 660.00 EGP",
                     "Sara, +201012345678, 12 Tahrir St, Dokki, Giza, EG", "NOT placed"):
        assert expected in t.state_text


def test_manual_shipping_never_states_total(env):
    env["shipping"] = ShippingSettings(method="manual")
    sid, cid = _conv()
    t = _complete(env, sid, cid)
    assert "Total:" not in t.state_text and "store will confirm the shipping fee" in t.state_text


def test_any_change_after_summary_changes_hash(env):
    sid, cid = _conv()
    first = _complete(env, sid, cid, qty=2).draft.confirmation_hash
    second = _complete(env, sid, cid, qty=3).draft.confirmation_hash
    assert first != second
    assert flow.workflow.load_draft(sid, cid).draft.items[0].quantity == 3  # restated line replaced, not duplicated


# ---------------------------------------------------------------- cancel

def test_cancel_clears_draft(env):
    env["extraction"] = OrderExtraction(items=[{"product": "it"}])
    sid, cid = _conv()
    _turn(sid, cid)
    env["extraction"] = OrderExtraction(cancel=True)
    t = _turn(sid, cid, intent=OTHER)
    assert t.event == "cancelled"
    assert flow.workflow.load_draft(sid, cid).draft is None