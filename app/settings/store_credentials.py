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


def set_credentials(store_id: str, source: str, creds: dict[str, Any]) -> None:
    """
    Save or update credentials for a store+source pair.
    Overwrites any existing credentials for that pair.
    """
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