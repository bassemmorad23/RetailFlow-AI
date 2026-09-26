"""
Per-store credentials for external sources (WooCommerce, Shopify, etc.).

WHY SEPARATE FROM store_settings:
Credentials are sensitive — API keys, OAuth tokens. Isolating them
in their own collection means different access controls, easier
auditing, and cleaner rotation later.

STRUCTURE:
One document per (store_id, source). Store may have multiple sources
if merchant syncs from both WC and Shopify.
"""

from functools import lru_cache
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection
from app.config import settings


@lru_cache(maxsize=1)
def _get_collection() -> Collection:
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB]
    return db["store_credentials"]

def _validate_woocommerce_creds(creds: dict[str, Any]) -> None:
    """
    Validate WooCommerce credentials structure and enforce HTTPS.
    """
    from urllib.parse import urlparse

    # Required fields
    required = {"site_url", "auth_method", "username", "password"}
    missing = required - set(creds.keys())
    if missing:
        raise ValueError(
            f"WooCommerce credentials missing required fields: {missing}. "
            f"Required: site_url, auth_method, username, password."
        )

    # auth_method must be known value
    valid_methods = {"application_password", "consumer_key"}
    if creds["auth_method"] not in valid_methods:
        raise ValueError(
            f"auth_method must be one of {valid_methods}. "
            f"Got: {creds['auth_method']}"
        )

    # HTTPS enforcement (localhost exception for dev)
    url = creds["site_url"]
    parsed = urlparse(url)
    host = parsed.hostname or ""
    is_local = host in ("localhost", "127.0.0.1") or host.endswith(".local")

    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and is_local:
        return
    raise ValueError(
        f"WooCommerce site_url must use HTTPS in production. "
        f"Got: {url}. HTTP sends credentials in plaintext and is only "
        f"allowed for localhost/dev URLs."
    )



def set_credentials(store_id: str, source: str, creds: dict[str, Any]) -> None:
    """
    Save or update credentials for a store+source pair.
    Overwrites any existing credentials for that pair.
    """
    
    if source == "woocommerce":
        _validate_woocommerce_creds(creds)
    
    col = _get_collection()
    col.update_one(
        {"store_id": store_id, "source": source},
        {"$set": {"store_id": store_id, "source": source, "credentials": creds}},
        upsert=True,
    )


def get_credentials(store_id: str, source: str) -> dict[str, Any] | None:
    """
    Load credentials for a store+source. Returns None if not configured.
    """
    col = _get_collection()
    doc = col.find_one({"store_id": store_id, "source": source})
    if doc is None:
        return None
    return doc.get("credentials")

def find_store_by_ig_account(ig_account_id: str) -> str | None:
    """Match either Instagram id: webhooks may use user_id, /me returns id."""
    col = _get_collection()
    doc = col.find_one({
        "source": "instagram",
        "$or": [
            {"credentials.instagram_business_account_id": ig_account_id},
            {"credentials.instagram_user_id": ig_account_id},
        ],
    })
    return doc["store_id"] if doc else None


def find_store_by_fb_page(page_id: str) -> str | None:
    """Find store that owns a given Facebook Page."""
    col = _get_collection()
    doc = col.find_one({
        "source": "messenger",
        "credentials.page_id": page_id,
    })
    return doc["store_id"] if doc else None



