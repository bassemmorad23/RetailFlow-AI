"""
Shopify OAuth flow.

Standard OAuth 2.0 authorization code grant:
1. install() builds Shopify's authorize URL with our client_id + scopes
2. Merchant approves on Shopify → Shopify redirects to callback with code
3. handle_callback() exchanges code for permanent access_token
4. Token saved in store_credentials for later API calls

WHY STATE TOKENS:
Shopify requires 'state' parameter for CSRF protection. We generate a
random token per install, save it in Mongo, verify it on callback.
Any callback without matching state = rejected.

WHY HMAC VERIFICATION:
Shopify signs callback params with our client_secret. Verifying prevents
attacks where someone constructs a fake callback URL. Reject if bad.
"""

import hashlib
import hmac
import secrets
from functools import lru_cache
from urllib.parse import urlencode

import httpx
from pymongo import MongoClient
from pymongo.collection import Collection

from app.config import settings

_SHOPIFY_SCOPES = "read_products,read_inventory,read_orders,write_orders,write_draft_orders"


@lru_cache(maxsize=1)
def _get_state_collection() -> Collection:
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB]
    return db["oauth_state"]


def build_authorize_url(shop: str, store_id: str) -> str:
    """
    Build the URL where we redirect the merchant to grant permissions.

    'shop' is their myshopify domain (e.g. 'test-store.myshopify.com').
    'store_id' is our internal ID — bundled into state so we know who
    it is when Shopify calls back.
    """
    # Generate random state token and save it with store_id
    state = secrets.token_urlsafe(32)
    col = _get_state_collection()
    col.insert_one({
        "state": state,
        "store_id": store_id,
        "shop": shop,
    })

    params = {
        "client_id": settings.SHOPIFY_CLIENT_ID,
        "scope": _SHOPIFY_SCOPES,
        "redirect_uri": f"{settings.SHOPIFY_REDIRECT_BASE_URL}/oauth/shopify/callback",
        "state": state,
    }
    return f"https://{shop}/admin/oauth/authorize?{urlencode(params)}"


def verify_state(state: str) -> dict | None:
    """
    Look up the state token. Returns {store_id, shop} if valid.
    Deletes it after use (single-use tokens).
    """
    col = _get_state_collection()
    doc = col.find_one_and_delete({"state": state})
    if doc is None:
        return None
    return {"store_id": doc["store_id"], "shop": doc["shop"]}


def verify_hmac(params: dict, hmac_from_shopify: str) -> bool:
    """
    Shopify signs callback query params with client_secret.
    We recompute the signature and compare.
    """
    # Shopify's signing: sort params (excluding hmac itself), URL-encode, sign with client_secret
    sorted_params = "&".join(
        f"{k}={v}" for k, v in sorted(params.items()) if k != "hmac"
    )
    computed = hmac.new(
        settings.SHOPIFY_CLIENT_SECRET.encode("utf-8"),
        sorted_params.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(computed, hmac_from_shopify)


def exchange_code_for_token(shop: str, code: str) -> str:
    """
    Exchange the temporary code for a permanent access token.
    Called during OAuth callback.
    """
    resp = httpx.post(
        f"https://{shop}/admin/oauth/access_token",
        json={
            "client_id": settings.SHOPIFY_CLIENT_ID,
            "client_secret": settings.SHOPIFY_CLIENT_SECRET,
            "code": code,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["access_token"]