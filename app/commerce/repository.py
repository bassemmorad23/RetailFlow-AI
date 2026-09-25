"""
Commerce storage (MongoDB): orders, support cases, per-store numbering.

TENANT ISOLATION: every read/write filters by store_id AND the object id.

SAFETY:
- Totals computed here from item prices; callers can't set them.
- Order creation is idempotent per (store_id, idempotency_key): webhook
  retries or double confirmations return the SAME order.
- Status changes are conditional (only from an allowed current status),
  so two merchants/tabs can't both approve or approve-and-reject.
"""

import uuid
from datetime import datetime, timezone
from functools import lru_cache

from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from app.commerce.models import ORDER_TRANSITIONS, CustomerDetails, OrderItem
from app.config import settings

_PROJ = {"_id": 0}
_FIRST_NUMBER = 1000  # first issued number is 1001


class InvalidTransition(Exception):
    pass


@lru_cache(maxsize=1)
def _db() -> Database:
    db = MongoClient(settings.MONGO_URI, tz_aware=True)[settings.MONGO_DB]

    orders = db["orders"]
    orders.create_index("id", unique=True)
    orders.create_index([("store_id", ASCENDING), ("number", ASCENDING)], unique=True)
    orders.create_index([("store_id", ASCENDING), ("idempotency_key", ASCENDING)], unique=True)
    orders.create_index([("store_id", ASCENDING), ("created_at", DESCENDING)])
    orders.create_index([("store_id", ASCENDING), ("channel", ASCENDING), ("customer_external_id", ASCENDING)])

    cases = db["support_cases"]
    cases.create_index("id", unique=True)
    cases.create_index([("store_id", ASCENDING), ("number", ASCENDING)], unique=True)
    cases.create_index([("store_id", ASCENDING), ("created_at", DESCENDING)])
    return db


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _next_number(store_id: str, kind: str) -> int:
    doc = _db()["commerce_counters"].find_one_and_update(
        {"_id": f"{store_id}:{kind}"},
        {"$inc": {"seq": 1}, "$setOnInsert": {"store_id": store_id}},
        upsert=True, return_document=ReturnDocument.AFTER,
    )
    return _FIRST_NUMBER + doc["seq"]


def _money(value: float) -> float:
    return round(float(value), 2)


# ---------------------------------------------------------------- orders

def create_order(
    store_id: str,
    *,
    channel: str,
    customer_external_id: str,
    customer: CustomerDetails,
    items: list[OrderItem],
    currency: str,
    shipping_fee: float | None,
    idempotency_key: str,
    conversation_id: str | None = None,
) -> dict:
    """Create a pending_approval COD order, or return the existing one for this idempotency key."""
    if not items:
        raise ValueError("Order must contain at least one item")

    col = _db()["orders"]
    existing = col.find_one({"store_id": store_id, "idempotency_key": idempotency_key}, _PROJ)
    if existing:
        return existing

    subtotal = _money(sum(i.unit_price * i.quantity for i in items))
    shipping = _money(shipping_fee) if shipping_fee is not None else None
    now = _now()
    doc = {
        "id": f"ord_{uuid.uuid4().hex}",
        "store_id": store_id,
        "number": f"SF-{_next_number(store_id, 'order')}",
        "conversation_id": conversation_id,
        "channel": channel,
        "customer_external_id": customer_external_id,
        "customer": customer.model_dump(),
        "items": [i.model_dump() for i in items],
        "currency": currency,
        "subtotal": subtotal,
        "shipping_fee": shipping,
        "shipping_status": "quoted" if shipping is not None else "pending_merchant",
        "total": _money(subtotal + shipping) if shipping is not None else None,
        "payment_method": "cod",
        "status": "pending_approval",
        "status_history": [{"status": "pending_approval", "at": now, "by": "customer"}],
        "platform": None,
        "idempotency_key": idempotency_key,
        "created_at": now,
        "updated_at": now,
    }
    try:
        col.insert_one(doc)
    except DuplicateKeyError:  # same key raced in; return the winner
        return col.find_one({"store_id": store_id, "idempotency_key": idempotency_key}, _PROJ)
    doc.pop("_id", None)
    return doc


def get_order(store_id: str, order_id: str) -> dict | None:
    return _db()["orders"].find_one({"store_id": store_id, "id": order_id}, _PROJ)


def get_order_by_number(store_id: str, number: str) -> dict | None:
    return _db()["orders"].find_one({"store_id": store_id, "number": number.strip().upper()}, _PROJ)


def list_customer_orders(store_id: str, channel: str, customer_external_id: str, limit: int = 10) -> list[dict]:
    """Orders placed by this customer on this channel (newest first)."""
    return list(
        _db()["orders"].find(
            {"store_id": store_id, "channel": channel, "customer_external_id": customer_external_id}, _PROJ,
        ).sort("created_at", DESCENDING).limit(limit)
    )


def change_order_status(store_id: str, order_id: str, *, new_status: str, by: str) -> dict:
    """
    Move an order to new_status if allowed from its current status.
    Conditional update: if someone else changed it first, this fails cleanly.
    Raises LookupError (not found) or InvalidTransition.
    """
    col = _db()["orders"]
    current = get_order(store_id, order_id)
    if current is None:
        raise LookupError(order_id)
    if new_status not in ORDER_TRANSITIONS.get(current["status"], set()):
        raise InvalidTransition(f"{current['status']} -> {new_status}")

    now = _now()
    updated = col.find_one_and_update(
        {"store_id": store_id, "id": order_id, "status": current["status"]},
        {"$set": {"status": new_status, "updated_at": now},
         "$push": {"status_history": {"status": new_status, "at": now, "by": by}}},
        projection=_PROJ, return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise InvalidTransition("Order status changed concurrently")
    return updated


def set_platform_ref(store_id: str, order_id: str, *, platform: str, platform_order_id: str,
                     platform_order_number: str | None = None) -> None:
    _db()["orders"].update_one(
        {"store_id": store_id, "id": order_id},
        {"$set": {"platform": {"type": platform, "order_id": platform_order_id,
                               "order_number": platform_order_number},
                  "updated_at": _now()}},
    )


# ---------------------------------------------------------------- support cases

def create_case(
    store_id: str,
    *,
    case_type: str,
    description: str,
    conversation_id: str | None = None,
    order_id: str | None = None,
) -> dict:
    now = _now()
    doc = {
        "id": f"case_{uuid.uuid4().hex}",
        "store_id": store_id,
        "number": f"CASE-{_next_number(store_id, 'case')}",
        "conversation_id": conversation_id,
        "order_id": order_id,
        "type": case_type,
        "description": description.strip()[:2000],
        "status": "open",
        "created_at": now,
        "updated_at": now,
    }
    _db()["support_cases"].insert_one(doc)
    doc.pop("_id", None)
    return doc


def get_case(store_id: str, case_id: str) -> dict | None:
    return _db()["support_cases"].find_one({"store_id": store_id, "id": case_id}, _PROJ)


def set_case_status(store_id: str, case_id: str, status: str) -> bool:
    if status not in ("open", "in_progress", "resolved", "closed"):
        raise ValueError(f"Invalid case status '{status}'")
    result = _db()["support_cases"].update_one(
        {"store_id": store_id, "id": case_id},
        {"$set": {"status": status, "updated_at": _now()}},
    )
    return result.matched_count == 1