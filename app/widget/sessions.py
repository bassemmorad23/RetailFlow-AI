"""
Web widget sessions.

The widget never chooses its own identity. The server issues a random
visitor id + secret token; only the SHA-256 of the token is stored.
Tokens last WIDGET_SESSION_TTL with a sliding expiry; Mongo TTL cleans up.
The widget sends the token in the Authorization header (not in URLs).
"""

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from pymongo import MongoClient
from pymongo.collection import Collection

from app.config import settings

WIDGET_SESSION_TTL = timedelta(days=30)
_SLIDE_AFTER = timedelta(days=1)


@dataclass(frozen=True)
class WidgetSession:
    store_id: str
    visitor_id: str


@lru_cache(maxsize=1)
def _col() -> Collection:
    col = MongoClient(settings.MONGO_URI, tz_aware=True)[settings.MONGO_DB]["widget_sessions"]
    col.create_index("token_hash", unique=True)
    col.create_index("expires_at", expireAfterSeconds=0)
    return col


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_session(store_id: str) -> tuple[str, datetime]:
    """Returns (raw_token, expires_at). The raw token is never stored."""
    raw = secrets.token_urlsafe(32)
    now = _now()
    expires = now + WIDGET_SESSION_TTL
    _col().insert_one({
        "token_hash": _hash(raw),
        "store_id": store_id,
        "visitor_id": f"v_{uuid.uuid4().hex}",
        "created_at": now,
        "expires_at": expires,
    })
    return raw, expires


def resolve_session(raw: str) -> WidgetSession | None:
    if not raw:
        return None
    col = _col()
    token_hash = _hash(raw)
    now = _now()
    doc = col.find_one({"token_hash": token_hash, "expires_at": {"$gt": now}})
    if doc is None:
        return None
    new_expiry = now + WIDGET_SESSION_TTL
    if new_expiry - doc["expires_at"] > _SLIDE_AFTER:
        col.update_one({"token_hash": token_hash}, {"$set": {"expires_at": new_expiry}})
    return WidgetSession(store_id=doc["store_id"], visitor_id=doc["visitor_id"])