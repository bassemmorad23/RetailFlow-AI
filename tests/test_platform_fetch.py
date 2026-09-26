"""Live single-product fetch for webhooks (faked HTTP): same rows as the full sync, active/published only."""

import json

import httpx
import pytest

from app.ingestion import shopify_adapter as sa
from app.ingestion import woocommerce_adapter as wa


@pytest.fixture(autouse=True)
def creds(monkeypatch):
    monkeypatch.setattr(sa, "get_credentials", lambda s, src: {"shop": "t.myshopify.com", "access_token": "x"})
    monkeypatch.setattr(wa, "get_credentials", lambda s, src: {"site_url": "https://w.example", "username": "u", "password": "p"})


_REAL_CLIENT = httpx.Client


def _client(handler):
    return _REAL_CLIENT(transport=httpx.MockTransport(handler))


def _node(status="ACTIVE"):
    return {"id": "gid://shopify/Product/1", "handle": "tee", "title": "Tee", "status": status,
            "variants": {"edges": [{"node": {"id": "gid://shopify/ProductVariant/11", "sku": "tee-m", "price": "300",
                                             "inventoryQuantity": 4, "inventoryPolicy": "DENY",
                                             "inventoryItem": {"tracked": True},
                                             "selectedOptions": [{"name": "Size", "value": "M"}]}}]}}


# ---------------------------------------------------------------- Shopify

def test_shopify_single_product_rows_match_full_sync_format():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"data": {"product": _node()}})

    rows = sa.ShopifyAdapter().fetch_product_rows("s", "gid://shopify/Product/1", client=_client(handler))
    assert rows == sa._flatten_product_with_variants(_node())
    assert rows[0]["external_parent_id"] == "gid://shopify/Product/1" and rows[0]["stock"] == 4 and rows[0]["Size"] == "M"
    assert "product(id: $id)" in seen["query"] and seen["variables"] == {"id": "gid://shopify/Product/1"}


@pytest.mark.parametrize("product", [None, _node(status="DRAFT"), _node(status="ARCHIVED")])
def test_shopify_missing_or_inactive_is_none(product):
    client = _client(lambda r: httpx.Response(200, json={"data": {"product": product}}))
    assert sa.ShopifyAdapter().fetch_product_rows("s", "gid://shopify/Product/1", client=client) is None


def test_shopify_inventory_item_to_product():
    client = _client(lambda r: httpx.Response(200, json={"data": {"inventoryItem": {
        "variant": {"product": {"id": "gid://shopify/Product/1"}}}}}))
    assert sa.ShopifyAdapter().product_id_for_inventory_item("s", "gid://shopify/InventoryItem/5", client=client) \
        == "gid://shopify/Product/1"


def test_shopify_full_sync_skips_drafts(monkeypatch):
    page = {"data": {"products": {"edges": [{"cursor": "a", "node": _node()},
                                            {"cursor": "b", "node": {**_node("DRAFT"), "handle": "draft"}}],
                                  "pageInfo": {"hasNextPage": False, "endCursor": None}}}}
    monkeypatch.setattr(sa.httpx, "Client", lambda **kw: _client(lambda r: httpx.Response(200, json=page)))
    rows = sa.ShopifyAdapter().fetch_raw_products("s")
    assert {r["product_id"] for r in rows} == {"tee"}


# ---------------------------------------------------------------- WooCommerce

def test_wc_simple_and_variable_products():
    def handler(request):
        path = request.url.path
        if path.endswith("/products/5"):
            return httpx.Response(200, json={"id": 5, "name": "Mug", "status": "publish", "type": "simple",
                                             "regular_price": "50", "manage_stock": True, "stock_quantity": 2})
        if path.endswith("/products/20"):
            return httpx.Response(200, json={"id": 20, "name": "Shirt", "status": "publish", "type": "variable"})
        if path.endswith("/products/20/variations"):
            return httpx.Response(200, json=[{"id": 21, "regular_price": "300", "stock_status": "instock",
                                              "attributes": [{"name": "Size", "option": "M"}]}])
        return httpx.Response(404)

    a = wa.WooCommerceAdapter()
    mug = a.fetch_product_rows("s", "5", client=_client(handler))
    assert (mug[0]["product_id"], mug[0]["stock"], mug[0]["external_id"]) == ("5", 2, "5")
    shirt = a.fetch_product_rows("s", "20", client=_client(handler))
    assert (shirt[0]["external_id"], shirt[0]["external_parent_id"], shirt[0]["Size"]) == ("21", "20", "M")


@pytest.mark.parametrize("response", [httpx.Response(404), httpx.Response(200, json={"id": 5, "status": "draft"})])
def test_wc_missing_or_unpublished_is_none(response):
    assert wa.WooCommerceAdapter().fetch_product_rows("s", "5", client=_client(lambda r: response)) is None


def test_wc_full_sync_requests_published_only(monkeypatch):
    seen = []

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx.Response(200, json=[])

    monkeypatch.setattr(wa.httpx, "Client", lambda **kw: _client(handler))
    assert wa.WooCommerceAdapter().fetch_raw_products("s") == []
    assert seen[0]["status"] == "publish"