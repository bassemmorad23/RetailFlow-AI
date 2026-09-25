"""Shopify / WooCommerce order creation (faked HTTP): payloads, duplicate safety, errors."""

import json

import httpx
import pytest

from app.commerce import platform_orders as po
from app.schemas.models import Product, Variant

CREDS = {"shopify": {"shop": "t.myshopify.com", "access_token": "tok"},
         "woocommerce": {"site_url": "https://shop.example", "username": "u", "password": "p"}}


def _order(shipping_fee=60.0, sku="tee-L", pid="tee"):
    return {"id": "ord_abc", "number": "SF-1001", "currency": "EGP", "shipping_fee": shipping_fee,
            "items": [{"product_id": pid, "variant_sku": sku, "name": "Cotton T-Shirt", "variant_attrs": {},
                       "quantity": 2, "unit_price": 320.0}],
            "customer": {"name": "Sara Ahmed", "phone": "+201012345678", "country": "EG", "region_code": "EG-GZ",
                         "region_name": "Giza", "city": "Dokki", "address_line": "12 Tahrir St", "notes": ""}}


@pytest.fixture
def catalog(monkeypatch):
    st = {"products": [
        Product.model_construct(product_id="tee", name="T", price=300.0, attributes={}, specifications={},
                                external_id=None, external_parent_id="PARENT",
                                variants=[Variant.model_construct(sku="tee-L", price=320.0, attributes={},
                                                                  specifications={}, external_id="VAR")]),
        Product.model_construct(product_id="cap", name="Cap", price=100.0, attributes={}, specifications={},
                                external_id="CAPID", external_parent_id=None, variants=[]),
    ]}
    monkeypatch.setattr(po, "get_credentials", lambda sid, src: CREDS.get(src))
    monkeypatch.setattr(po, "fetch_products", lambda sid, ids: [p for p in st["products"] if p.product_id in ids])
    return st


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------- Shopify

def _shopify_handler(calls, *, existing=None, currency="EGP", create_response=None):
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if "lookup" in body["query"]:
            edges = [{"node": existing}] if existing else []
            return httpx.Response(200, json={"data": {"shop": {"currencyCode": currency}, "orders": {"edges": edges}}})
        return httpx.Response(200, json=create_response or {"data": {"orderCreate": {
            "order": {"id": "gid://shopify/Order/9", "name": "#1009"}, "userErrors": []}}})
    return handler


def test_shopify_creates_order_with_confirmed_prices(catalog):
    calls = []
    r = po.push_order("s", "shopify", _order(), client=_client(_shopify_handler(calls)))
    assert (r.ok, r.platform_order_id, r.platform_order_number) == (True, "gid://shopify/Order/9", "#1009")

    order = calls[1]["variables"]["order"]
    assert order["lineItems"] == [{"variantId": "VAR", "quantity": 2,
                                   "priceSet": {"shopMoney": {"amount": "320.00", "currencyCode": "EGP"}}}]
    assert order["shippingLines"][0]["priceSet"]["shopMoney"]["amount"] == "60.00"
    assert order["financialStatus"] == "PENDING" and "sf-ord_abc" in order["tags"] and "COD" in order["tags"]
    addr = order["shippingAddress"]
    assert (addr["firstName"], addr["lastName"], addr["provinceCode"], addr["countryCode"]) == ("Sara", "Ahmed", "GZ", "EG")
    assert calls[1]["variables"]["options"]["inventoryBehaviour"] == "DECREMENT_OBEYING_POLICY"


def test_shopify_retry_reuses_existing_order(catalog):
    calls = []
    r = po.push_order("s", "shopify", _order(),
                      client=_client(_shopify_handler(calls, existing={"id": "gid://shopify/Order/5", "name": "#1005"})))
    assert r.ok and r.platform_order_id == "gid://shopify/Order/5"
    assert len(calls) == 1  # never created a second order


def test_shopify_currency_mismatch_blocks(catalog):
    calls = []
    r = po.push_order("s", "shopify", _order(), client=_client(_shopify_handler(calls, currency="USD")))
    assert not r.ok and "currency_mismatch" in r.error and len(calls) == 1


def test_shopify_user_errors_reported(catalog):
    calls = []
    bad = {"data": {"orderCreate": {"order": None, "userErrors": [{"field": ["x"], "message": "Province invalid"}]}}}
    r = po.push_order("s", "shopify", _order(), client=_client(_shopify_handler(calls, create_response=bad)))
    assert not r.ok and r.error == "rejected: Province invalid"


def test_shopify_manual_shipping_has_no_shipping_line(catalog):
    calls = []
    po.push_order("s", "shopify", _order(shipping_fee=None), client=_client(_shopify_handler(calls)))
    assert "shippingLines" not in calls[1]["variables"]["order"]


def test_missing_platform_ids_blocks(catalog):
    catalog["products"][0].variants[0].external_id = None
    calls = []
    r = po.push_order("s", "shopify", _order(), client=_client(_shopify_handler(calls)))
    assert r.error == "missing_platform_ids_resync_products" and len(calls) == 1


def test_timeout_is_flagged_for_manual_check(catalog):
    catalog["products"][0].external_parent_id = "20"
    catalog["products"][0].variants[0].external_id = "21"

    def handler(request):
        raise httpx.ReadTimeout("slow")
    r = po.push_order("s", "woocommerce", _order(), client=_client(handler))
    assert r.error == "timeout_check_platform_before_retry"


# ---------------------------------------------------------------- WooCommerce

def test_woocommerce_creates_cod_processing_order(catalog):
    catalog["products"][0].external_parent_id = "20"
    catalog["products"][0].variants[0].external_id = "21"
    sent = {}

    def handler(request):
        sent["path"] = request.url.path
        sent["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": 77, "number": "77"})

    r = po.push_order("s", "woocommerce", _order(), client=_client(handler))
    assert (r.ok, r.platform_order_id, r.platform_order_number) == (True, "77", "77")
    b = sent["body"]
    assert sent["path"] == "/wp-json/wc/v3/orders"
    assert (b["payment_method"], b["status"], b["set_paid"]) == ("cod", "processing", False)
    assert b["line_items"] == [{"quantity": 2, "subtotal": "640.00", "total": "640.00", "product_id": 20, "variation_id": 21}]
    assert b["shipping_lines"][0]["total"] == "60.00"
    assert b["billing"]["state"] == "GZ" and b["billing"]["phone"] == "+201012345678"
    assert {"key": "storeflow_order_number", "value": "SF-1001"} in b["meta_data"]


def test_woocommerce_simple_product(catalog):
    catalog["products"][1].external_id = "55"
    sent = {}

    def handler(request):
        sent["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": 78})

    po.push_order("s", "woocommerce", _order(sku=None, pid="cap"), client=_client(handler))
    assert sent["body"]["line_items"] == [{"quantity": 2, "subtotal": "640.00", "total": "640.00", "product_id": 55}]


def test_woocommerce_non_numeric_ids_rejected_not_crashing(catalog):
    r = po.push_order("s", "woocommerce", _order(sku=None, pid="cap"),
                      client=_client(lambda req: pytest.fail("must not call WooCommerce")))
    assert r.error == "invalid_platform_ids_resync_products"


def test_woocommerce_error_message_surfaced(catalog):
    catalog["products"][0].external_parent_id = "20"
    catalog["products"][0].variants[0].external_id = "21"
    r = po.push_order("s", "woocommerce", _order(),
                      client=_client(lambda req: httpx.Response(400, json={"message": "Invalid product"})))
    assert not r.ok and r.error == "http_400: Invalid product"


def test_not_connected(catalog, monkeypatch):
    monkeypatch.setattr(po, "get_credentials", lambda sid, src: None)
    assert po.push_order("s", "shopify", _order()).error == "not_connected"
    assert po.push_order("s", "woocommerce", _order()).error == "not_connected"