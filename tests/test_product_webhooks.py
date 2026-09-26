"""Product webhooks: signatures, dedupe, topic routing, background processing, WooCommerce registration."""

import base64
import hashlib
import hmac
import json
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.config import settings
from app.ingestion import webhooks as wh
from app.settings.store_credentials import _get_collection as creds_col

SECRET = "shpss_test_secret"


def _sig(secret: str, body: bytes) -> str:
    return base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()


@pytest.fixture
def env(monkeypatch):
    st = {"refreshed": [], "removed": [], "fail": False}

    def refresh(store_id, platform, pid):
        if st["fail"]:
            raise RuntimeError("platform down")
        st["refreshed"].append((store_id, platform, pid))
        return "upserted"

    monkeypatch.setattr(settings, "SHOPIFY_CLIENT_SECRET", SECRET)
    monkeypatch.setattr(wh, "refresh_product", refresh)
    monkeypatch.setattr(wh, "remove_platform_product", lambda sid, pid: st["removed"].append((sid, pid)) or [pid])
    return st


# ---------------------------------------------------------------- Shopify

def _shop_store():
    sid, shop = f"store_wh_{uuid.uuid4().hex[:8]}", f"s{uuid.uuid4().hex[:6]}.myshopify.com"
    creds_col().insert_one({"store_id": sid, "source": "shopify", "credentials": {"shop": shop, "access_token": "x"}})
    return sid, shop


def _shopify(shop, topic, payload, delivery=None, secret=SECRET):
    body = json.dumps(payload).encode()
    return TestClient(app).post("/webhooks/shopify", content=body, headers={
        "X-Shopify-Hmac-Sha256": _sig(secret, body), "X-Shopify-Topic": topic, "X-Shopify-Shop-Domain": shop,
        "X-Shopify-Webhook-Id": delivery or uuid.uuid4().hex, "Content-Type": "application/json"})


def test_shopify_update_refetches_product(env):
    sid, shop = _shop_store()
    r = _shopify(shop, "products/update", {"id": 1, "admin_graphql_api_id": "gid://shopify/Product/1"})
    assert r.json() == {"status": "accepted"}
    assert env["refreshed"] == [(sid, "shopify", "gid://shopify/Product/1")]


def test_shopify_delete_and_inventory(env, monkeypatch):
    sid, shop = _shop_store()
    _shopify(shop, "products/delete", {"id": 7})
    assert env["removed"] == [(sid, "gid://shopify/Product/7")]
    monkeypatch.setattr(wh.ShopifyAdapter, "product_id_for_inventory_item",
                        lambda self, s, item: "gid://shopify/Product/9" if item.endswith("/55") else None)
    _shopify(shop, "inventory_levels/update", {"inventory_item_id": 55, "available": 0})
    assert env["refreshed"] == [(sid, "shopify", "gid://shopify/Product/9")]


def test_shopify_signature_duplicates_and_unknown(env):
    sid, shop = _shop_store()
    assert _shopify(shop, "products/update", {"id": 1}, secret="wrong").status_code == 401
    assert _shopify("unknown.myshopify.com", "products/update", {"id": 1}).json() == {"status": "ignored"}
    assert _shopify(shop, "orders/create", {"id": 1}).json() == {"status": "ignored"}
    assert _shopify(shop, "products/update", {"id": 1}, delivery="d1").json() == {"status": "accepted"}
    assert _shopify(shop, "products/update", {"id": 1}, delivery="d1").json() == {"status": "duplicate"}
    assert len(env["refreshed"]) == 1


def test_processing_failure_is_recorded(env):
    env["fail"] = True
    sid, shop = _shop_store()
    _shopify(shop, "products/update", {"id": 1}, delivery="dx")
    doc = wh._db()["platform_webhook_deliveries"].find_one({"_id": f"shopify:{sid}:dx"})
    assert doc["status"] == "failed" and "platform down" in doc["error"]


# ---------------------------------------------------------------- WooCommerce

def _wc_store(secret="wc_secret"):
    sid = f"store_wh_{uuid.uuid4().hex[:8]}"
    wh._db()["platform_webhook_secrets"].insert_one({"store_id": sid, "platform": "woocommerce", "secret": secret})
    return sid


def _wc(sid, topic, payload, delivery=None, secret="wc_secret"):
    body = json.dumps(payload).encode()
    return TestClient(app).post(f"/webhooks/woocommerce/{sid}", content=body, headers={
        "X-WC-Webhook-Signature": _sig(secret, body), "X-WC-Webhook-Topic": topic,
        "X-WC-Webhook-Delivery-ID": delivery or uuid.uuid4().hex, "Content-Type": "application/json"})


def test_wc_routing(env):
    sid = _wc_store()
    _wc(sid, "product.updated", {"id": 20, "type": "variable"})
    _wc(sid, "product.updated", {"id": 21, "parent_id": 20, "type": "variation"})          # variation -> parent
    _wc(sid, "action.woocommerce_product_set_stock", {"action": "woocommerce_product_set_stock", "arg": 5})
    _wc(sid, "product.deleted", {"id": 30})
    assert [p for _, _, p in env["refreshed"]] == ["20", "20", "5"]
    assert env["removed"] == [(sid, "30")]


def test_wc_ping_signature_and_duplicates(env):
    sid = _wc_store()
    client = TestClient(app)
    assert client.post(f"/webhooks/woocommerce/{sid}", content=b"webhook_id=12").json() == {"status": "pong"}
    assert _wc(sid, "product.updated", {"id": 1}, secret="wrong").status_code == 401
    no_secret = f"store_wh_{uuid.uuid4().hex[:8]}"
    assert _wc(no_secret, "product.updated", {"id": 1}).status_code == 401
    assert _wc(sid, "product.updated", {"id": 1}, delivery="w1").json() == {"status": "accepted"}
    assert _wc(sid, "product.updated", {"id": 1}, delivery="w1").json() == {"status": "duplicate"}
    assert len(env["refreshed"]) == 1


def test_ensure_woocommerce_webhooks_is_idempotent(monkeypatch):
    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "https://api.example")
    sid = f"store_wh_{uuid.uuid4().hex[:8]}"
    creds_col().insert_one({"store_id": sid, "source": "woocommerce", "credentials": {
        "site_url": "https://w.example", "auth_method": "basic", "username": "u", "password": "p"}})
    hooks, created_bodies = [], []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=hooks)
        if request.method == "POST":
            body = json.loads(request.content)
            created_bodies.append(body)
            hooks.append({"id": len(hooks) + 1, "topic": body["topic"], "delivery_url": body["delivery_url"],
                          "status": "active"})
            return httpx.Response(201, json=hooks[-1])
        return httpx.Response(200, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    first = wh.ensure_woocommerce_webhooks(sid, client)
    assert first["ok"] and sorted(first["created"]) == sorted(wh.WC_TOPICS)
    assert {b["delivery_url"] for b in created_bodies} == {f"https://api.example/webhooks/woocommerce/{sid}"}
    secret = wh._wc_secret(sid)
    assert secret and all(b["secret"] == secret for b in created_bodies)
    assert wh.ensure_woocommerce_webhooks(sid, client) == {"ok": True, "created": []}   # nothing duplicated
    assert wh.ensure_woocommerce_webhooks("store_none") == {"ok": False, "error": "not_connected"}


def test_ensure_shopify_webhooks_is_idempotent(monkeypatch):
    from app.ingestion import shopify_adapter as sa
    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "https://api.example")
    monkeypatch.setattr(sa, "get_credentials", lambda s, src: {"shop": "t.myshopify.com", "access_token": "x"})
    monkeypatch.setattr(wh, "get_credentials", lambda s, src: {"shop": "t.myshopify.com", "access_token": "x"})
    subs, created = [], []

    def handler(request):
        body = json.loads(request.content)
        if "webhookSubscriptions" in body["query"]:
            return httpx.Response(200, json={"data": {"webhookSubscriptions": {"edges": [{"node": n} for n in subs]}}})
        created.append(body["variables"])
        subs.append({"id": str(len(subs)), "topic": body["variables"]["topic"], "uri": body["variables"]["uri"]})
        return httpx.Response(200, json={"data": {"webhookSubscriptionCreate": {"webhookSubscription": {"id": "1"},
                                                                                  "userErrors": []}}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    first = wh.ensure_shopify_webhooks("s", client)
    assert first["ok"] and sorted(first["created"]) == sorted(wh._SHOPIFY_TOPIC_ENUMS.values())
    assert {c["uri"] for c in created} == {"https://api.example/webhooks/shopify"}
    assert wh.ensure_shopify_webhooks("s", client) == {"ok": True, "created": []}


def test_ensure_shopify_webhooks_reports_errors(monkeypatch):
    from app.ingestion import shopify_adapter as sa
    monkeypatch.setattr(sa, "get_credentials", lambda s, src: {"shop": "t.myshopify.com", "access_token": "x"})
    monkeypatch.setattr(wh, "get_credentials", lambda s, src: {"shop": "t.myshopify.com", "access_token": "x"})

    def handler(request):
        if "webhookSubscriptions" in json.loads(request.content)["query"]:
            return httpx.Response(200, json={"data": {"webhookSubscriptions": {"edges": []}}})
        return httpx.Response(200, json={"data": {"webhookSubscriptionCreate": {
            "webhookSubscription": None, "userErrors": [{"field": ["uri"], "message": "Address is invalid"}]}}})

    result = wh.ensure_shopify_webhooks("s", httpx.Client(transport=httpx.MockTransport(handler)))
    assert result["ok"] is False and "Address is invalid" in result["errors"][0]
    monkeypatch.setattr(wh, "get_credentials", lambda s, src: None)
    assert wh.ensure_shopify_webhooks("s") == {"ok": False, "error": "not_connected"}