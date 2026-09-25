"""
Order status for customers ("where is my order?").

Security:
- Orders placed from this same chat identity (channel + customer) are shown.
- Otherwise: order number + the phone used for the order must match.
- Anything else gets the SAME neutral answer, never confirming an order exists.
- Max FAILED_LIMIT failed number+phone lookups per conversation per hour.
- Revealed: number, items, status, total, tracking. Never address or phone.

Status source: live from Shopify/WooCommerce for pushed orders (tracking on
Shopify), otherwise StoreFlow's own status. Live failure -> StoreFlow status
with a note. Deterministic: no LLM involved.
"""

import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import httpx
from pymongo import MongoClient
from pymongo.collection import Collection

from app.commerce import repository as orders
from app.commerce.phones import normalize_phone
from app.config import settings
from app.ingestion.shopify_adapter import _SHOPIFY_API_VERSION
from app.schemas.models import IntentLabel, IntentResult
from app.settings.store_credentials import get_credentials

FAILED_LIMIT = 5
FAILED_WINDOW = timedelta(hours=1)
STATUS_CONFIDENCE = 0.3
LIVE_TIMEOUT = 8.0
MAX_ORDERS_SHOWN = 3

_ORDER_NUMBER = re.compile(r"\bSF[\s\-]?(\d{4,})\b", re.IGNORECASE)
_PHONE = re.compile(r"(\+?[\d٠-٩۰-۹][\d٠-٩۰-۹\s\-]{6,}[\d٠-٩۰-۹])")
_STATUS_PHRASES = re.compile(
    r"where('?s| is) my order|order status|track(ing)?( my)? order|when will (it|my order) (arrive|come)"
    r"|طلبي فين|الاوردر فين|الأوردر فين|اوردري|أوردري|حالة الطلب|فين الطلب|طلبي وصل|\b(order|talaby) fen\b",
    re.IGNORECASE,
)

NEUTRAL = ("ORDER LOOKUP: no order could be shown. Say you couldn't find an order matching those details, and ask "
           "for the order number (like SF-1234) and the phone number used when ordering. Do not say whether "
           "any order exists.")
LOCKED = ("ORDER LOOKUP: too many unsuccessful attempts. Politely say you can't look up more orders right now "
          "and that the store team will help. Do not reveal anything about any order.")

_STOREFLOW_STATUS = {
    "pending_approval": "received — waiting for the store to confirm it",
    "approved": "confirmed by the store — being prepared",
    "shipped": "shipped",
    "delivered": "delivered",
    "rejected": "not accepted — the store could not fulfil it",
    "cancelled": "cancelled",
}
_WC_STATUS = {
    "pending": "received — waiting for the store", "processing": "confirmed — being prepared",
    "on-hold": "on hold — the store will contact the customer", "completed": "completed (shipped/delivered)",
    "cancelled": "cancelled", "refunded": "refunded", "failed": "failed — the store will contact the customer",
}


# ---------------------------------------------------------------- trigger + parsing

def wants_status(intent: IntentResult, text: str) -> bool:
    if intent.label == IntentLabel.ORDER_STATUS and intent.confidence >= STATUS_CONFIDENCE:
        return True
    return bool(_STATUS_PHRASES.search(text or "") or _ORDER_NUMBER.search(text or ""))


def _order_number(text: str) -> str | None:
    m = _ORDER_NUMBER.search(text or "")
    return f"SF-{m.group(1)}" if m else None


def _phone(text: str, country: str) -> str | None:
    for candidate in _PHONE.findall(text or ""):
        result = normalize_phone(candidate, country)
        if result.valid:
            return result.e164
    return None


# ---------------------------------------------------------------- failed attempts

@lru_cache(maxsize=1)
def _attempts() -> Collection:
    col = MongoClient(settings.MONGO_URI, tz_aware=True)[settings.MONGO_DB]["order_lookup_attempts"]
    col.create_index("expires_at", expireAfterSeconds=0)
    col.create_index([("store_id", 1), ("conversation_id", 1)])
    return col


def _failed_count(store_id: str, cid: str) -> int:
    return _attempts().count_documents(
        {"store_id": store_id, "conversation_id": cid, "expires_at": {"$gt": datetime.now(timezone.utc)}})


def _record_failure(store_id: str, cid: str) -> None:
    now = datetime.now(timezone.utc)
    _attempts().insert_one({"store_id": store_id, "conversation_id": cid, "at": now, "expires_at": now + FAILED_WINDOW})


# ---------------------------------------------------------------- live platform status

_SHOPIFY_ORDER = """
query status($id: ID!) {
  node(id: $id) {
    ... on Order {
      name cancelledAt displayFulfillmentStatus
      fulfillments(first: 5) { displayStatus trackingInfo { company number url } }
    }
  }
}
"""


def _shopify_status(store_id: str, platform_order_id: str, http: httpx.Client) -> tuple[str, list[str]] | None:
    creds = get_credentials(store_id, "shopify")
    if creds is None:
        return None
    resp = http.post(f"https://{creds['shop']}/admin/api/{_SHOPIFY_API_VERSION}/graphql.json",
                     headers={"X-Shopify-Access-Token": creds["access_token"], "Content-Type": "application/json"},
                     json={"query": _SHOPIFY_ORDER, "variables": {"id": platform_order_id}})
    data = resp.json() if resp.status_code < 400 else {}
    node = (data.get("data") or {}).get("node") if not data.get("errors") else None
    if not node:
        return None
    fulfillments = node.get("fulfillments") or []
    tracking = [" ".join(x for x in (t.get("company"), t.get("number"), t.get("url")) if x)
                for f in fulfillments for t in (f.get("trackingInfo") or [])]
    if node.get("cancelledAt"):
        return "cancelled", []
    if any((f.get("displayStatus") or "") == "DELIVERED" for f in fulfillments):
        return "delivered", tracking
    fs = node.get("displayFulfillmentStatus") or ""
    if fs == "FULFILLED":
        return "shipped", tracking
    if fs == "PARTIALLY_FULFILLED":
        return "partly shipped", tracking
    return "confirmed — being prepared", tracking


def _wc_status(store_id: str, platform_order_id: str, http: httpx.Client) -> tuple[str, list[str]] | None:
    creds = get_credentials(store_id, "woocommerce")
    if creds is None:
        return None
    resp = http.get(f"{creds['site_url'].rstrip('/')}/wp-json/wc/v3/orders/{platform_order_id}",
                    auth=(creds["username"], creds["password"]))
    if resp.status_code >= 400:
        return None
    status = resp.json().get("status")
    return (_WC_STATUS.get(status, status), []) if status else None


def live_status(store_id: str, order: dict, client: httpx.Client | None = None) -> tuple[str, list[str]] | None:
    platform = order.get("platform") or {}
    if not platform.get("order_id"):
        return None
    http = client or httpx.Client(timeout=LIVE_TIMEOUT)
    try:
        if platform.get("type") == "shopify":
            return _shopify_status(store_id, platform["order_id"], http)
        if platform.get("type") == "woocommerce":
            return _wc_status(store_id, platform["order_id"], http)
        return None
    except (httpx.HTTPError, ValueError):
        return None
    finally:
        if client is None:
            http.close()


# ---------------------------------------------------------------- public

def identify_orders(*, store_id: str, conversation_id: str, channel: str, customer_external_id: str,
                    text: str, store_country: str) -> tuple[list[dict], str]:
    """
    Orders this requester may see, and how they were identified:
    own | verified (number + phone) | locked | none. Shared by order status and after-sales.
    """
    if _failed_count(store_id, conversation_id) >= FAILED_LIMIT:
        return [], "locked"

    own = orders.list_customer_orders(store_id, channel, customer_external_id, limit=10)
    number = _order_number(text)
    if number:
        order = orders.get_order_by_number(store_id, number)
        if order and any(o["id"] == order["id"] for o in own):
            return [order], "own"
        phone = _phone(_ORDER_NUMBER.sub(" ", text), store_country)
        if order and phone and phone == order["customer"]["phone"]:
            return [order], "verified"
        if phone:  # a real attempt with number + phone that didn't match
            _record_failure(store_id, conversation_id)
        return [], "none"
    return (own[:MAX_ORDERS_SHOWN], "own") if own else ([], "none")


def order_status_text(*, store_id: str, conversation_id: str, channel: str, customer_external_id: str,
                      text: str, store_country: str) -> str:
    """Facts block for the AI. Always safe to show to the requester."""
    found, how = identify_orders(store_id=store_id, conversation_id=conversation_id, channel=channel,
                                 customer_external_id=customer_external_id, text=text, store_country=store_country)
    if how == "locked":
        return LOCKED
    return _describe(found, store_id) if found else NEUTRAL


def current_status(store_id: str, order: dict) -> tuple[str, list[str], bool]:
    """(customer-facing status, tracking, shipped_or_later)."""
    live = live_status(store_id, order)
    if live:
        status, tracking = live
    else:
        status, tracking = _STOREFLOW_STATUS.get(order["status"], order["status"]), []
    shipped = any(w in status for w in ("shipped", "delivered", "completed")) or order["status"] in ("shipped", "delivered")
    return status, tracking, shipped


def _describe(found: list[dict], store_id: str) -> str:
    lines = ["ORDER STATUS — use ONLY these facts. Never reveal addresses or phone numbers. "
             "Do not promise delivery dates."]
    for o in found:
        live = live_status(store_id, o)
        if live:
            status, tracking = live
        else:
            status, tracking = _STOREFLOW_STATUS.get(o["status"], o["status"]), []
            if (o.get("platform") or {}).get("order_id"):
                status += " (couldn't check the latest update right now)"
        items = ", ".join(f"{i['quantity']} × {i['name']}" for i in o["items"])
        total = f"{o['total']:,.2f} {o['currency']}" if o.get("total") is not None else "shipping fee to be confirmed"
        lines.append(f"- Order {o['number']}: {status}. Items: {items}. Total: {total}.")
        if tracking:
            lines.append(f"  Tracking: {'; '.join(tracking)}")
    return "\n".join(lines)