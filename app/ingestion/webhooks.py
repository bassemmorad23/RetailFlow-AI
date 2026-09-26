"""
Real-time product updates from Shopify and WooCommerce (webhooks = primary path;
the 6-hourly sync is reconciliation for anything missed).

Flow for every delivery:
  verify signature -> dedupe by delivery id -> 200 immediately ->
  background: RE-FETCH the product from the platform (latest state, same rows as
  the full sync) -> upsert, or remove if it's gone / unpublished.
Re-fetching makes out-of-order and duplicate deliveries harmless.

Shopify: HMAC-SHA256 (base64) of the raw body with the app's client secret.
         Topics products/create|update|delete, inventory_levels/update — registered
         per store through the API by ensure_shopify_webhooks() (after OAuth).
WooCommerce: HMAC-SHA256 (base64) with a per-store secret; webhooks registered
         per store by ensure_woocommerce_webhooks().
"""

import base64
import hashlib
import hmac
import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

from app.config import settings
from app.ingestion.platform_updates import remove_platform_product, upsert_platform_products
from app.ingestion.shopify_adapter import ShopifyAdapter, _graphql
from app.ingestion.woocommerce_adapter import WooCommerceAdapter
from app.settings.store_credentials import _get_collection as creds_col
from app.settings.store_credentials import get_credentials
from app.log_context import set_request_context

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhooks", tags=["platform webhooks"])

MAX_BODY = 1_000_000
DELIVERY_TTL = timedelta(days=7)
SHOPIFY_TOPICS = ("products/create", "products/update", "products/delete", "inventory_levels/update")
WC_TOPICS = ("product.created", "product.updated", "product.deleted", "product.restored",
             "action.woocommerce_product_set_stock", "action.woocommerce_variation_set_stock")


@lru_cache(maxsize=1)
def _db():
    db = MongoClient(settings.MONGO_URI, tz_aware=True)[settings.MONGO_DB]
    db["platform_webhook_deliveries"].create_index("expires_at", expireAfterSeconds=0)
    db["platform_webhook_secrets"].create_index([("store_id", 1), ("platform", 1)], unique=True)
    return db


def _valid_b64_hmac(secret: str, body: bytes, header: str | None) -> bool:
    if not secret or not header:
        return False
    expected = base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()
    return hmac.compare_digest(expected, header.strip())


def _first_delivery(platform: str, store_id: str, delivery_id: str, topic: str) -> bool:
    """True the first time we see this delivery; False for duplicates."""
    now = datetime.now(timezone.utc)
    try:
        _db()["platform_webhook_deliveries"].insert_one({
            "_id": f"{platform}:{store_id}:{delivery_id}", "store_id": store_id, "topic": topic,
            "status": "received", "received_at": now, "expires_at": now + DELIVERY_TTL})
        return True
    except DuplicateKeyError:
        return False


def _finish(platform: str, store_id: str, delivery_id: str, status: str, error: str | None = None) -> None:
    _db()["platform_webhook_deliveries"].update_one(
        {"_id": f"{platform}:{store_id}:{delivery_id}"}, {"$set": {"status": status, "error": error}})


async def _body(request: Request) -> bytes:
    body = await request.body()
    if len(body) > MAX_BODY:
        raise HTTPException(status_code=413, detail="Payload too large")
    return body


# ---------------------------------------------------------------- product refresh (shared)

def refresh_product(store_id: str, platform: str, platform_product_id: str) -> str:
    """Re-fetch one product and apply it. Returns 'upserted' or 'removed'."""
    adapter = ShopifyAdapter() if platform == "shopify" else WooCommerceAdapter()
    rows = adapter.fetch_product_rows(store_id, platform_product_id)
    if rows is None:
        remove_platform_product(store_id, platform_product_id)
        return "removed"
    upsert_platform_products(store_id, rows)
    return "upserted"


def _run(platform: str, store_id: str, delivery_id: str, work) -> None:
    set_request_context(store_id=store_id)
    try:
        outcome = work()
        _finish(platform, store_id, delivery_id, "processed")
        logger.info("Product webhook processed", extra={"webhook_platform": platform, "webhook_outcome": outcome})
    except Exception as exc:  # the 6-hourly reconciliation repairs anything missed here
        _finish(platform, store_id, delivery_id, "failed", f"{type(exc).__name__}: {str(exc)[:200]}")
        logger.exception("Product webhook processing failed", extra={"webhook_platform": platform})


# ---------------------------------------------------------------- Shopify

def _store_for_shop(shop: str) -> str | None:
    doc = creds_col().find_one({"source": "shopify", "credentials.shop": shop}, {"_id": 0, "store_id": 1})
    return doc["store_id"] if doc else None


def shopify_work(store_id: str, topic: str, payload: dict):
    if topic == "products/delete":
        pid = f"gid://shopify/Product/{payload.get('id')}"
        return lambda: (remove_platform_product(store_id, pid), "removed")[1]
    if topic in ("products/create", "products/update"):
        pid = payload.get("admin_graphql_api_id") or f"gid://shopify/Product/{payload.get('id')}"
        return lambda: refresh_product(store_id, "shopify", pid)
    if topic == "inventory_levels/update":
        item = f"gid://shopify/InventoryItem/{payload.get('inventory_item_id')}"

        def work():
            product = ShopifyAdapter().product_id_for_inventory_item(store_id, item)
            return refresh_product(store_id, "shopify", product) if product else "unknown_item"
        return work
    return None


@router.post("/shopify")
async def shopify_webhook(request: Request, background: BackgroundTasks) -> dict:
    body = await _body(request)
    if not _valid_b64_hmac(settings.SHOPIFY_CLIENT_SECRET, body, request.headers.get("X-Shopify-Hmac-Sha256")):
        raise HTTPException(status_code=401, detail="Invalid signature")
    topic = request.headers.get("X-Shopify-Topic", "")
    delivery = request.headers.get("X-Shopify-Webhook-Id") or request.headers.get("X-Shopify-Event-Id") or ""
    store_id = _store_for_shop(request.headers.get("X-Shopify-Shop-Domain", ""))
    if store_id is None:
        return {"status": "ignored"}           # shop not connected (anymore): ack so Shopify stops retrying
    try:
        payload = json.loads(body or b"{}")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    work = shopify_work(store_id, topic, payload)
    if work is None:
        return {"status": "ignored"}
    if delivery and not _first_delivery("shopify", store_id, delivery, topic):
        return {"status": "duplicate"}
    background.add_task(_run, "shopify", store_id, delivery, work)
    return {"status": "accepted"}


# ---------------------------------------------------------------- WooCommerce

def _wc_secret(store_id: str) -> str | None:
    doc = _db()["platform_webhook_secrets"].find_one({"store_id": store_id, "platform": "woocommerce"})
    return doc["secret"] if doc else None


def _wc_product_id(payload: dict) -> str | None:
    """Product id from a product payload or a stock action payload ({"action", "arg"})."""
    if "arg" in payload:
        arg = payload["arg"]
        arg = arg.get("id") if isinstance(arg, dict) else arg
        return str(arg) if arg not in (None, "", 0) else None
    parent = payload.get("parent_id")
    if parent:                                      # a variation -> refresh its parent product
        return str(parent)
    return str(payload["id"]) if payload.get("id") else None


def wc_work(store_id: str, topic: str, payload: dict):
    pid = _wc_product_id(payload)
    if pid is None:
        return None
    if topic == "product.deleted":
        return lambda: (remove_platform_product(store_id, pid), "removed")[1]
    if topic in WC_TOPICS:
        return lambda: refresh_product(store_id, "woocommerce", pid)
    return None


@router.post("/woocommerce/{store_id}")
async def woocommerce_webhook(store_id: str, request: Request, background: BackgroundTasks) -> dict:
    body = await _body(request)
    if body.startswith(b"webhook_id="):             # WooCommerce's ping when a webhook is created
        return {"status": "pong"}
    if not _valid_b64_hmac(_wc_secret(store_id) or "", body, request.headers.get("X-WC-Webhook-Signature")):
        raise HTTPException(status_code=401, detail="Invalid signature")
    topic = request.headers.get("X-WC-Webhook-Topic", "")
    delivery = request.headers.get("X-WC-Webhook-Delivery-ID", "")
    try:
        payload = json.loads(body or b"{}")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    work = wc_work(store_id, topic, payload)
    if work is None:
        return {"status": "ignored"}
    if delivery and not _first_delivery("woocommerce", store_id, delivery, topic):
        return {"status": "duplicate"}
    background.add_task(_run, "woocommerce", store_id, delivery, work)
    return {"status": "accepted"}


# ---------------------------------------------------------------- WooCommerce registration

def wc_delivery_url(store_id: str) -> str:
    return f"{settings.PUBLIC_BASE_URL.rstrip('/')}/webhooks/woocommerce/{store_id}"


def ensure_woocommerce_webhooks(store_id: str, client: httpx.Client | None = None) -> dict:
    """
    Make sure this store's WooCommerce sends us every product topic. Idempotent:
    creates only what's missing. A new secret replaces our old webhooks.
    """
    creds = get_credentials(store_id, "woocommerce")
    if creds is None:
        return {"ok": False, "error": "not_connected"}
    base = creds["site_url"].rstrip("/") + "/wp-json/wc/v3/webhooks"
    auth = (creds["username"], creds["password"])
    url = wc_delivery_url(store_id)

    secret = _wc_secret(store_id)
    new_secret = secret is None
    if new_secret:  # saved BEFORE creating webhooks, so a half-finished run can't leave a mismatch
        secret = secrets.token_urlsafe(32)
        _db()["platform_webhook_secrets"].update_one(
            {"store_id": store_id, "platform": "woocommerce"}, {"$set": {"secret": secret}}, upsert=True)

    http = client or httpx.Client(timeout=30.0)
    try:
        resp = http.get(base, params={"per_page": 100}, auth=auth)
        resp.raise_for_status()
        ours = [h for h in resp.json() if h.get("delivery_url") == url]
        if new_secret:
            for h in ours:
                http.delete(f"{base}/{h['id']}", params={"force": "true"}, auth=auth)
            ours = []
        active = {h["topic"] for h in ours if h.get("status") == "active"}
        created = []
        for topic in WC_TOPICS:
            if topic in active:
                continue
            r = http.post(base, auth=auth, json={"name": f"StoreFlow {topic}", "topic": topic,
                                                 "delivery_url": url, "secret": secret, "status": "active"})
            r.raise_for_status()
            created.append(topic)
    except httpx.HTTPError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}"}
    finally:
        if client is None:
            http.close()

    return {"ok": True, "created": created}


# ---------------------------------------------------------------- Shopify registration

_SHOPIFY_TOPIC_ENUMS = {"products/create": "PRODUCTS_CREATE", "products/update": "PRODUCTS_UPDATE",
                        "products/delete": "PRODUCTS_DELETE", "inventory_levels/update": "INVENTORY_LEVELS_UPDATE"}
_LIST_SUBSCRIPTIONS = "query { webhookSubscriptions(first: 100) { edges { node { id topic uri } } } }"
_CREATE_SUBSCRIPTION = """
mutation create($topic: WebhookSubscriptionTopic!, $uri: String!) {
  webhookSubscriptionCreate(topic: $topic, webhookSubscription: {uri: $uri, format: JSON}) {
    webhookSubscription { id }
    userErrors { field message }
  }
}
"""


def shopify_delivery_url() -> str:
    return f"{settings.PUBLIC_BASE_URL.rstrip('/')}/webhooks/shopify"


def ensure_shopify_webhooks(store_id: str, client: httpx.Client | None = None) -> dict:
    """Make sure this shop sends us every product topic. Idempotent: creates only what's missing."""
    if get_credentials(store_id, "shopify") is None:
        return {"ok": False, "error": "not_connected"}
    url = shopify_delivery_url()
    http = client or httpx.Client(timeout=30.0)
    created, errors = [], []
    try:
        edges = (_graphql(store_id, http, _LIST_SUBSCRIPTIONS, {}).get("webhookSubscriptions") or {}).get("edges", [])
        have = {e["node"]["topic"] for e in edges if e["node"].get("uri") == url}
        for enum in _SHOPIFY_TOPIC_ENUMS.values():
            if enum in have:
                continue
            result = _graphql(store_id, http, _CREATE_SUBSCRIPTION, {"topic": enum, "uri": url}) \
                .get("webhookSubscriptionCreate") or {}
            if result.get("userErrors"):
                errors.append(f"{enum}: {result['userErrors'][0].get('message')}")
            else:
                created.append(enum)
    except (httpx.HTTPError, RuntimeError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}", "created": created}
    finally:
        if client is None:
            http.close()
    return {"ok": not errors, "created": created, **({"errors": errors} if errors else {})}