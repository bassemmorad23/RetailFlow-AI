"""
AI message usage tracking and limit checks.

One document per (store_id, kind, period_key) in `usage_counters`:
- kind="monthly": period_key = billing period start date (from anchor day)
- kind="daily":   period_key = today's UTC date

Limits are read from the plan config at check time, so plan
upgrades/downgrades apply immediately with no data migration.

Only successful AI replies are counted (caller decides when to record).
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from functools import lru_cache

from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection

from app.billing.plans import DAILY_MESSAGE_CAP_PER_STORE, get_monthly_limit
from app.config import settings
from app.settings.store_settings import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_collection() -> Collection:
    client = MongoClient(settings.MONGO_URI)
    col = client[settings.MONGO_DB]["usage_counters"]
    col.create_index(
        [("store_id", ASCENDING), ("kind", ASCENDING), ("period_key", ASCENDING)],
        unique=True,
    )
    return col


@dataclass(frozen=True)
class UsageStatus:
    allowed: bool
    reason: str | None  # None | "daily_cap" | "monthly_limit"
    monthly_used: int
    monthly_limit: int
    daily_used: int
    daily_cap: int
    period_start: str


def current_period_start(anchor_day: int, today: date) -> date:
    """Start of the billing period that contains `today`."""
    if today.day >= anchor_day:
        return date(today.year, today.month, anchor_day)
    if today.month == 1:
        return date(today.year - 1, 12, anchor_day)
    return date(today.year, today.month - 1, anchor_day)


def _keys(store_id: str) -> tuple[str, str]:
    today = datetime.now(timezone.utc).date()
    anchor = get_settings(store_id).billing_anchor_day
    return current_period_start(anchor, today).isoformat(), today.isoformat()


def _read_count(store_id: str, kind: str, period_key: str) -> int:
    doc = _get_collection().find_one(
        {"store_id": store_id, "kind": kind, "period_key": period_key},
        projection={"count": 1, "_id": 0},
    )
    return int(doc["count"]) if doc else 0


def check_usage(store_id: str) -> UsageStatus:
    """Return whether this store may generate another AI reply right now."""
    store = get_settings(store_id)
    monthly_limit = get_monthly_limit(store.plan, store.enterprise_monthly_limit)
    month_key, day_key = _keys(store_id)

    monthly_used = _read_count(store_id, "monthly", month_key)
    daily_used = _read_count(store_id, "daily", day_key)

    reason = None
    if daily_used >= DAILY_MESSAGE_CAP_PER_STORE:
        reason = "daily_cap"
    elif monthly_limit <= 0 or monthly_used >= monthly_limit:
        # monthly_limit <= 0 means misconfigured: fail closed, never unlimited
        reason = "monthly_limit"

    return UsageStatus(
        allowed=reason is None,
        reason=reason,
        monthly_used=monthly_used,
        monthly_limit=monthly_limit,
        daily_used=daily_used,
        daily_cap=DAILY_MESSAGE_CAP_PER_STORE,
        period_start=month_key,
    )


def record_ai_message(store_id: str) -> None:
    """Atomically increment monthly + daily counters for one AI reply."""
    month_key, day_key = _keys(store_id)
    col = _get_collection()
    now = datetime.now(timezone.utc)
    for kind, key in (("monthly", month_key), ("daily", day_key)):
        col.update_one(
            {"store_id": store_id, "kind": kind, "period_key": key},
            {"$inc": {"count": 1}, "$set": {"updated_at": now}},
            upsert=True,
        )