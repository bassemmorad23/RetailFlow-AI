"""Shopify calculated shipping: rate parsing, customer choice in the draft, re-check at confirmation."""

import json
import types
import uuid

import httpx
import pytest

from app.commerce import order_flow as flow
from app.commerce import platform_shipping as ps
from app.commerce.item_resolver import ResolvedItem
from app.commerce.models import DraftCustomer, OrderItem, ShippingOption
from app.commerce.order_extractor import OrderExtraction
from app.commerce.shipping import ShippingSettings
from app.commerce.stock import StockCheck
from app.inbox import repository as inbox_repo
from app.schemas.models import IntentLabel, IntentResult, Product, Variant

READY = IntentResult(label=IntentLabel.READY_TO_BUY, confidence=0.8)
OTHER = IntentResult(label=IntentLabel.OTHER, confidence=0.9)
DETAILS = {"name": "Sara", "phone": "01012345678", "region": "Giza", "city": "Dokki", "address_line": "12 Tahrir St"}
STANDARD = ShippingOption(handle="std", title="Standard", price=50.0)
EXPRESS = ShippingOption(handle="exp", title="Express", price=90.0)


def _item():
    return OrderItem(product_id="tee", variant_sku="tee-L", name="Cotton T-Shirt",
                     variant_attrs={"size": "L"}, quantity=2, unit_price=320.0)


PRODUCT = Product.model_construct(
    product_id="tee", name="Cotton T-Shirt", price=300.0, attributes={}, specifications={},
    external_id=None, variants=[Variant.model_construct(sku="tee-L", price=320.0, attributes={"size": "L"},
                                                        specifications={}, external_id="gid://V/2")])


# ---------------------------------------------------------------- Shopify rate parsing

@pytest.fixture
def shop(monkeypatch):
    monkeypatch.setattr(ps, "get_credentials", lambda sid, src: {"shop": "t.myshopify.com", "access_token": "x"})
    monkeypatch.setattr(ps, "fetch_products", lambda sid, ids: [PRODUCT])


def _rates(handler):
    customer = DraftCustomer(country="EG", region_code="EG-GZ", city="Dokki", address_line="12 Tahrir St")
    return ps.shopify_shipping_rates("s", [_item()], customer, "EGP", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_rates_parsed_and_request_uses_platform_ids(shop):
    seen = {}

    def handler(request):
        seen["vars"] = json.loads(request.content)["variables"]["input"]
        return httpx.Response(200, json={"data": {"draftOrderCalculate": {"userErrors": [], "calculatedDraftOrder": {
            "availableShippingRates": [
                {"handle": "std", "title": "Standard", "price": {"amount": "50.0", "currencyCode": "EGP"}},
                {"handle": "usd", "title": "Intl", "price": {"amount": "5.0", "currencyCode": "USD"}},  # other currency
            ]}}}})

    assert _rates(handler) == [STANDARD]
    assert seen["vars"]["lineItems"] == [{"variantId": "gid://V/2", "quantity": 2}]
    assert seen["vars"]["shippingAddress"]["provinceCode"] == "GZ"


@pytest.mark.parametrize("response", [
    httpx.Response(500),
    httpx.Response(200, json={"errors": [{"message": "denied"}]}),
    httpx.Response(200, json={"data": {"draftOrderCalculate": {"userErrors": [{"message": "bad"}]}}}),
    httpx.Response(200, json={"data": {"draftOrderCalculate": {"userErrors": [],
                                                               "calculatedDraftOrder": {"availableShippingRates": []}}}}),
])
def test_no_usable_rates_returns_none(shop, response):
    assert _rates(lambda r: response) is None


# ---------------------------------------------------------------- choice in the draft

@pytest.fixture
def env(monkeypatch):
    st = {"options": [STANDARD, EXPRESS], "extraction": OrderExtraction()}
    monkeypatch.setattr(flow, "extract_order_details", lambda text, **kw: st["extraction"])
    monkeypatch.setattr(flow, "resolve_item", lambda sid, req, recs: ResolvedItem("ok", item=_item(), stock="in_stock"))
    monkeypatch.setattr(flow, "get_shipping", lambda sid: ShippingSettings(method="platform", fallback="fixed", fixed={"fee": 60}))
    monkeypatch.setattr(flow, "get_settings", lambda sid: types.SimpleNamespace(country="EG", currency="EGP"))
    monkeypatch.setattr(flow, "store_platform", lambda sid: "shopify")
    monkeypatch.setattr(flow, "shopify_shipping_rates", lambda sid, items, customer, currency: st["options"])
    monkeypatch.setattr(flow, "fetch_products", lambda sid, ids: [PRODUCT])
    monkeypatch.setattr(flow, "check_stock", lambda sid, reqs: [StockCheck("in_stock", "live") for _ in reqs])
    return st


def _conv():
    sid = f"store_ship_{uuid.uuid4().hex[:8]}"
    return sid, inbox_repo.get_or_create_conversation(sid, "web", "v_" + uuid.uuid4().hex[:6])["id"]


def _turn(sid, cid, intent=OTHER, text="msg"):
    return flow.process_order_turn(store_id=sid, conversation_id=cid, channel="web", customer_external_id="v1",
                                   text=text, intent=intent, recommendations=[])


def _complete(env, sid, cid):
    env["extraction"] = OrderExtraction(items=[{"product": "it", "quantity": 2}], customer=DETAILS)
    return _turn(sid, cid, intent=READY)


def test_several_rates_customer_chooses(env):
    sid, cid = _conv()
    t = _complete(env, sid, cid)
    assert t.event == "choosing_shipping" and t.draft.confirmation_hash is None
    assert "1. Standard — 50.00 EGP" in t.state_text and "2. Express — 90.00 EGP" in t.state_text

    env["extraction"] = OrderExtraction(shipping_choice="Express")
    t = _turn(sid, cid)
    assert t.event == "awaiting_confirmation"
    assert "Shipping (Express): 90.00 EGP" in t.state_text and "Total: 730.00 EGP" in t.state_text


def test_choice_by_number(env):
    sid, cid = _conv()
    _complete(env, sid, cid)
    env["extraction"] = OrderExtraction(shipping_choice="1")
    assert "Shipping (Standard): 50.00 EGP" in _turn(sid, cid).state_text


def test_unclear_choice_asks_again(env):
    sid, cid = _conv()
    _complete(env, sid, cid)
    env["extraction"] = OrderExtraction(shipping_choice="teleport")
    t = _turn(sid, cid)
    assert t.event == "choosing_shipping" and "wasn't clear which shipping option" in t.state_text


def test_single_rate_used_automatically(env):
    env["options"] = [STANDARD]
    sid, cid = _conv()
    t = _complete(env, sid, cid)
    assert t.event == "awaiting_confirmation" and "Shipping (Standard): 50.00 EGP" in t.state_text


def test_no_rates_uses_configured_fallback(env):
    env["options"] = None
    sid, cid = _conv()
    t = _complete(env, sid, cid)
    assert t.event == "awaiting_confirmation" and "Shipping: 60.00 EGP" in t.state_text


def test_rate_change_at_confirmation_then_order_keeps_title(env):
    sid, cid = _conv()
    _complete(env, sid, cid)
    env["extraction"] = OrderExtraction(shipping_choice="Express")
    _turn(sid, cid)
    env["extraction"] = OrderExtraction()
    env["options"] = [STANDARD, ShippingOption(handle="exp", title="Express", price=95.0)]
    t = _turn(sid, cid, text="yes")
    assert t.event == "awaiting_confirmation" and t.order is None and "95.00 EGP" in t.state_text
    t = _turn(sid, cid, text="yes")
    assert t.event == "order_created"
    assert (t.order["shipping_fee"], t.order["shipping_title"]) == (95.0, "Express")