"""
Login brute-force protection.

Fixed 15-minute window per key (email and IP). Only failed attempts
count. Stored in Mongo so limits survive restarts and work across
multiple workers. Expired docs are removed by a TTL index.
"""

from datetime import datetime, timedelta, timezone
from functools import lru_cache

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

from app.auth.repository import normalize_email
from app.config import settings

WINDOW = timedelta(minutes=15)
MAX_FAILURES_PER_EMAIL = 5
MAX_FAILURES_PER_IP = 20


@lru_cache(maxsize=1)
def _col() -> Collection:
    client = MongoClient(settings.MONGO_URI, tz_aware=True)
    col = client[settings.MONGO_DB]["login_attempts"]
    col.create_index("key", unique=True)
    col.create_index("expires_at", expireAfterSeconds=0)
    return col


def _keys(email: str, ip: str) -> list[tuple[str, int]]:
    return [
        (f"email:{normalize_email(email)}", MAX_FAILURES_PER_EMAIL),
        (f"ip:{ip}", MAX_FAILURES_PER_IP),
    ]


def retry_after_seconds(email: str, ip: str) -> int | None:
    """Seconds until login is allowed again, or None if not blocked."""
    now = datetime.now(timezone.utc)
    for key, limit in _keys(email, ip):
        doc = _col().find_one({"key": key, "expires_at": {"$gt": now}})
        if doc and doc["count"] >= limit:
            return max(1, int((doc["expires_at"] - now).total_seconds()))
    return None


def record_failure(email: str, ip: str) -> None:
    now = datetime.now(timezone.utc)
    col = _col()
    for key, _ in _keys(email, ip):
        # Reset a window that expired but hasn't been TTL-deleted yet
        col.update_one(
            {"key": key, "expires_at": {"$lte": now}},
            {"$set": {"count": 0, "expires_at": now + WINDOW}},
        )
        try:
            col.update_one(
                {"key": key},
                {"$inc": {"count": 1}, "$setOnInsert": {"expires_at": now + WINDOW}},
                upsert=True,
            )
        except DuplicateKeyError:  # two concurrent first-failures
            col.update_one({"key": key}, {"$inc": {"count": 1}})


def clear_email(email: str) -> None:
    _col().delete_one({"key": f"email:{normalize_email(email)}"})