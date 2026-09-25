"""Live stock lookups (faked HTTP) + platform ids carried through ingestion."""

import json

import httpx
import pytest

from app.commerce import stock_live
from app.commerce.stock_live import LiveStockError, shopify_live_stock, woocommerce_live_stock
from app.ingestion.canonical_converter import to_canonical_grouped
from app.ingestion.deterministic_mapper import map_columns

CREDS = {
    "shopify": {"shop": "test.myshopify.com", "access_token": "tok"},
    "woocommerce": {"site_url": "https://shop.example", "username": "u", "password": "p"},
}


@pytest.fixture(autouse=True)
def creds(monkeypatch):
    monkeypatch.setattr(stock_live, "get_credentials", lambda sid, src: CREDS.get(src))


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------- Shopify

def test_shopify_batched_single_call_and_parsing():
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"data": {"nodes": [
            {"id": "gid://V/1", "inventoryQuantity": 4, "inventoryPolicy": "DENY", "inventoryItem": {"tracked": True}},
            {"id": "gid://V/2", "inventoryQuantity": 0, "inventoryPolicy": "DENY", "inventoryItem": {"tracked": True}},
            {"id": "gid://V/3", "inventoryQuantity": 0, "inventoryPolicy": "DENY", "inventoryItem": {"tracked": False}},
            None,  # deleted variant
        ]}})

    out = shopify_live_stock("s", ["gid://V/1", "gid://V/2", "gid://V/3", "gid://V/9"], client=_client(handler))
    assert len(calls) == 1
    assert out == {"gid://V/1": ("in_stock", 4), "gid://V/2": ("out_of_stock", 0), "gid://V/3": ("in_stock", None)}


@pytest.mark.parametrize("response", [
    httpx.Response(500),
    httpx.Response(429),
    httpx.Response(200, json={"errors": [{"message": "Access denied"}]}),
])
def test_shopify_failures_raise(response):
    with pytest.raises(LiveStockError):
        shopify_live_stock("s", ["gid://V/1"], client=_client(lambda r: response))


def test_shopify_network_error_raises():
    def handler(request):
        raise httpx.ConnectTimeout("slow")
    with pytest.raises(LiveStockError):
        shopify_live_stock("s", ["gid://V/1"], client=_client(handler))


def test_not_connected_raises(monkeypatch):
    monkeypatch.setattr(stock_live, "get_credentials", lambda sid, src: None)
    with pytest.raises(LiveStockError):
        shopify_live_stock("s", ["gid://V/1"])


# ---------------------------------------------------------------- WooCommerce

def test_woocommerce_simple_and_variations_batched():
    paths = []

    def handler(request):
        paths.append((request.url.path, request.url.params.get("include")))
        if request.url.path.endswith("/products"):
            return httpx.Response(200, json=[{"id": 10, "manage_stock": True, "stock_quantity": 2}])
        return httpx.Response(200, json=[
            {"id": 21, "manage_stock": False, "stock_status": "outofstock"},
            {"id": 22, "manage_stock": False, "stock_status": "instock"},
        ])

    out = woocommerce_live_stock("s", [(None, "10"), ("20", "21"), ("20", "22")], client=_client(handler))
    assert sorted(paths) == [("/wp-json/wc/v3/products", "10"), ("/wp-json/wc/v3/products/20/variations", "21,22")]
    assert out == {"10": ("in_stock", 2), "21": ("out_of_stock", None), "22": ("in_stock", None)}


def test_woocommerce_failure_raises():
    with pytest.raises(LiveStockError):
        woocommerce_live_stock("s", [(None, "10")], client=_client(lambda r: httpx.Response(401)))


def test_empty_request_makes_no_call():
    assert shopify_live_stock("s", []) == {} and woocommerce_live_stock("s", []) == {}


# ---------------------------------------------------------------- ids through ingestion

def test_platform_ids_stored_on_products_and_variants():
    rows = [
        {"Product ID": "tee", "Product Name": "Tee", "Price": "300", "Size": "M",
         "external_id": "gid://V/1", "external_parent_id": "gid://P/1"},
        {"Product ID": "tee", "Product Name": "Tee", "Price": "300", "Size": "L",
         "external_id": "gid://V/2", "external_parent_id": "gid://P/1"},
        {"Product ID": "cap", "Product Name": "Cap", "Price": "100",
         "external_id": "gid://V/3", "external_parent_id": "gid://P/2"},
    ]
    cols = sorted({k for r in rows for k in r if not k.startswith("external_")})
    products = {p.product_id: p for p in to_canonical_grouped(rows, map_columns(cols, "fashion"), "fashion", "s")}

    tee, cap = products["tee"], products["cap"]
    assert {v.attributes["size"]: v.external_id for v in tee.variants} == {"M": "gid://V/1", "L": "gid://V/2"}
    assert tee.external_parent_id == "gid://P/1" and tee.external_id is None
    assert (cap.external_id, cap.external_parent_id) == ("gid://V/3", "gid://P/2")