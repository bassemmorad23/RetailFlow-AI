"""
Per-store configuration in MongoDB.

One document per store, keyed by store_id, holding all merchant
configuration: industry, channel toggles (future), sync credentials
(future), branding (future).

WHY THIS EXISTS AS A DEDICATED MODULE:
- Single source of truth for store config reads/writes
- Multi-tenant scoping enforced in one place (store_id required
  on every operation)
- Callers get a validated Pydantic object, not a raw dict
- Migration path clear: adding a new field = schema change here,
  not scattered updates across the codebase

DESIGN CHOICE — LENIENT INDUSTRY LOOKUP:
get_industry() returns None when the store has no industry assigned,
rather than raising. Downstream code decides what to do (fall back
to generic behavior, prompt the merchant to configure, etc.).
This matches lenient V1 architecture — unknown or unset data flows
through the pipeline without breaking it.
"""

from functools import lru_cache
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection

from app.config import settings
from app.industries.registry import list_industries
from app.schemas.models import StoreSettings
from app.billing.plans import PLANS


# ---------------------------------------------------------------------------
# MongoDB collection access
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_collection() -> Collection:
    """
    Return the store_settings collection.

    Cached with lru_cache to reuse one MongoDB connection across all
    calls in a single process. Same pattern as conversation_memory.py.
    """
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB]
    return db["store_settings"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_BILLING_FIELDS = ("plan", "enterprise_monthly_limit", "billing_anchor_day")


def get_settings(store_id: str) -> StoreSettings:
    """
    Load a store's settings. Creates defaults on first access.
    Backfills billing fields on older documents and persists them,
    so billing_anchor_day is fixed once and never drifts.
    """
    col = _get_collection()
    doc = col.find_one({"store_id": store_id})

    if doc is None:
        fresh = StoreSettings(store_id=store_id, industry=None)
        col.insert_one(fresh.model_dump())
        return fresh

    doc.pop("_id", None)
    missing = [f for f in _BILLING_FIELDS if f not in doc]
    store_settings = StoreSettings(**doc)

    if missing:
        col.update_one(
            {"store_id": store_id},
            {"$set": {f: getattr(store_settings, f) for f in missing}},
        )

    return store_settings


def get_industry(store_id: str) -> Optional[str]:
    """
    Quick lookup for just the industry ID.

    Returns None if the store has no industry assigned. Callers
    that need full settings should use get_settings() instead.

    Convenience helper — same result as get_settings(store_id).industry
    but reads slightly less from Mongo.
    """
    col = _get_collection()
    doc = col.find_one(
        {"store_id": store_id},
        projection={"industry": 1, "_id": 0},
    )
    if doc is None:
        return None
    return doc.get("industry")


def set_industry(store_id: str, industry_id: Optional[str]) -> StoreSettings:
    """
    Set (or clear) a store's industry.

    Validates that industry_id matches a registered IndustryConfig
    before writing — prevents typos or references to industries that
    don't exist. Pass None to clear the industry (e.g. for testing
    or when a merchant deletes their assignment).

    Rejects unknown industries with ValueError. The caller (usually
    the merchant dashboard onboarding endpoint) is expected to catch
    this and surface a friendly error to the merchant.
    """
    if industry_id is not None:
        registered = list_industries()
        if industry_id not in registered:
            raise ValueError(
                f"Unknown industry '{industry_id}'. Registered "
                f"industries: {registered}"
            )

    col = _get_collection()
    col.update_one(
        {"store_id": store_id},
        {"$set": {"store_id": store_id, "industry": industry_id}},
        upsert=True,
    )
    return get_settings(store_id)



def set_plan(
    store_id: str,
    plan: str,
    enterprise_monthly_limit: Optional[int] = None,
) -> StoreSettings:
    """
    Change a store's plan. The new limit applies immediately because
    limits are always read from the plan config at enforcement time.

    Enterprise requires an explicit positive limit (no unlimited stores).
    Non-enterprise plans clear any custom limit.
    The billing anchor day is NOT changed on upgrade/downgrade.
    """
    if plan not in PLANS:
        raise ValueError(f"Unknown plan '{plan}'. Available: {list(PLANS)}")

    if plan == "enterprise":
        if enterprise_monthly_limit is None or enterprise_monthly_limit <= 0:
            raise ValueError("Enterprise plan requires a positive enterprise_monthly_limit.")
    else:
        enterprise_monthly_limit = None

    get_settings(store_id)  # ensures document + anchor exist

    col = _get_collection()
    col.update_one(
        {"store_id": store_id},
        {"$set": {"plan": plan, "enterprise_monthly_limit": enterprise_monthly_limit}},
    )
    return get_settings(store_id)