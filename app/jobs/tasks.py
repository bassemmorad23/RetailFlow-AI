"""
Scheduled jobs (business logic only — no scheduler here).

sync_products            every 6h   reconciliation for missed/failed webhooks: full platform
                                    sync (adds/updates, removes platform-deleted products with
                                    guards) + re-registers any missing product webhooks
refresh_fulfilment       every 2h   fulfilment snapshots for open pushed orders
refresh_instagram_tokens daily      refresh long-lived tokens expiring within 15 days (or of
                                    unknown expiry); a dead token -> merchant must reconnect
"""

from datetime import datetime, timedelta, timezone

import httpx

from app.commerce.order_status import refresh_platform_orders
from app.commerce.repository import TERMINAL_FULFILMENT
from app.commerce.repository import _db as orders_db
from app.ingestion.ingestion_service import ingest_products
from app.ingestion.shopify_adapter import ShopifyAdapter
from app.ingestion.webhooks import ensure_shopify_webhooks, ensure_woocommerce_webhooks
from app.ingestion.woocommerce_adapter import WooCommerceAdapter
from app.jobs.runner import JobSpec
from app.settings.store_credentials import _get_collection as creds_col
from app.settings.store_credentials import get_credentials, set_credentials
from app.settings.store_settings import get_industry

TOKEN_REFRESH_WINDOW = timedelta(days=15)
TOKEN_URGENT = timedelta(days=7)
_IG_REFRESH_URL = "https://graph.instagram.com/refresh_access_token"


def _stores_with(*sources: str) -> list[str]:
    return sorted(set(creds_col().distinct("store_id", {"source": {"$in": list(sources)}})))


# ---------------------------------------------------------------- sync_products (reconciliation)

def sync_store_products(store_id: str) -> str:
    outcomes = []
    for source, adapter, ensure in (("shopify", ShopifyAdapter, ensure_shopify_webhooks),
                                    ("woocommerce", WooCommerceAdapter, ensure_woocommerce_webhooks)):
        if get_credentials(store_id, source) is None:
            continue
        result = ingest_products(adapter(), store_id, get_industry(store_id))  # recorded in sync history
        hooks = ensure(store_id)
        outcomes.append(f"{source}: {result.created} new, {result.updated} updated, "
                        f"webhooks {'ok' if hooks.get('ok') else 'FAILED'}")
        if not hooks.get("ok"):
            raise RuntimeError(f"{source} webhooks not registered: {hooks.get('error') or hooks.get('errors')}")
    return "; ".join(outcomes) or "no platform connected"


# ---------------------------------------------------------------- refresh_fulfilment

def stores_with_open_platform_orders() -> list[str]:
    return sorted(orders_db()["orders"].distinct("store_id", {
        "status": "approved", "platform.order_id": {"$exists": True, "$ne": None},
        "fulfilment.state": {"$nin": list(TERMINAL_FULFILMENT)}}))


def refresh_store_fulfilment(store_id: str) -> str:
    r = refresh_platform_orders(store_id)
    if r["checked"] and r["failed"] == r["checked"]:
        raise RuntimeError(f"all {r['checked']} fulfilment checks failed")
    return f"{r['checked']} checked, {r['changed']} changed, {r['failed']} failed"


# ---------------------------------------------------------------- refresh_instagram_tokens

def refresh_instagram_token(store_id: str, client: httpx.Client | None = None, now: datetime | None = None) -> str:
    creds = get_credentials(store_id, "instagram")
    if creds is None:
        return "not connected"
    now = now or datetime.now(timezone.utc)
    expires = creds.get("token_expires_at")
    expires = datetime.fromisoformat(expires) if isinstance(expires, str) else expires
    if expires and expires - now > TOKEN_REFRESH_WINDOW:
        return "not due"

    http = client or httpx.Client(timeout=15.0)
    try:
        resp = http.get(_IG_REFRESH_URL, params={"grant_type": "ig_refresh_token"},
                        headers={"Authorization": f"Bearer {creds['access_token']}"})
        data = resp.json() if resp.status_code < 500 else {}
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(f"Instagram unreachable: {type(exc).__name__}") from exc
    finally:
        if client is None:
            http.close()

    if resp.status_code == 200 and data.get("access_token"):
        new_expiry = now + timedelta(seconds=int(data.get("expires_in") or 0))
        set_credentials(store_id, "instagram", {**creds, "access_token": data["access_token"],
                                                "token_expires_at": new_expiry.isoformat(),
                                                "token_refreshed_at": now.isoformat(), "token_status": "ok"})
        return f"refreshed, valid until {new_expiry.date().isoformat()}"

    # Refresh refused. Only flag the merchant when the token is really at risk.
    if expires is None or expires - now <= TOKEN_URGENT:
        set_credentials(store_id, "instagram", {**creds, "token_status": "needs_reconnect"})
    raise RuntimeError(f"Instagram refused the refresh ({resp.status_code})")


JOBS: dict[str, JobSpec] = {
    "sync_products": JobSpec("sync_products", lambda: _stores_with("shopify", "woocommerce"),
                             sync_store_products, min_interval=timedelta(hours=5)),
    "refresh_fulfilment": JobSpec("refresh_fulfilment", stores_with_open_platform_orders,
                                  refresh_store_fulfilment, min_interval=timedelta(minutes=90)),
    "refresh_instagram_tokens": JobSpec("refresh_instagram_tokens", lambda: _stores_with("instagram"),
                                        refresh_instagram_token, min_interval=timedelta(hours=20)),
}