"""
Auth data layer: users, sessions, store memberships (MongoDB).

- users:         one per merchant login, unique email
- sessions:      SHA-256 of a random token (raw token only lives in the cookie);
                 Mongo TTL index deletes expired sessions automatically
- store_members: (user_id, store_id, role); only "owner" for now
"""

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from pymongo import ASCENDING, MongoClient
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from app.config import settings

SESSION_TTL = timedelta(days=30)
_SLIDE_AFTER = timedelta(days=1)  # extend expiry at most once per day


class EmailAlreadyRegistered(Exception):
    pass


@lru_cache(maxsize=1)
def _db() -> Database:
    client = MongoClient(settings.MONGO_URI, tz_aware=True)
    db = client[settings.MONGO_DB]
    db["users"].create_index("email", unique=True)
    db["sessions"].create_index("token_hash", unique=True)
    db["sessions"].create_index("expires_at", expireAfterSeconds=0)
    db["store_members"].create_index(
        [("user_id", ASCENDING), ("store_id", ASCENDING)], unique=True
    )
    db["store_members"].create_index("store_id")
    return db


def _now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_email(email: str) -> str:
    return email.strip().lower()


def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


# ------------------------------------------------------------------ users

def create_user(email: str, password_hash: str, full_name: str = "") -> str:
    user_id = uuid.uuid4().hex
    try:
        _db()["users"].insert_one({
            "user_id": user_id,
            "email": normalize_email(email),
            "password_hash": password_hash,
            "full_name": full_name.strip(),
            "created_at": _now(),
        })
    except DuplicateKeyError:
        raise EmailAlreadyRegistered(email)
    return user_id


def get_user_by_email(email: str) -> dict | None:
    return _db()["users"].find_one({"email": normalize_email(email)}, {"_id": 0})


def get_user_by_id(user_id: str) -> dict | None:
    return _db()["users"].find_one({"user_id": user_id}, {"_id": 0})


def update_password_hash(user_id: str, password_hash: str) -> None:
    _db()["users"].update_one({"user_id": user_id}, {"$set": {"password_hash": password_hash}})


# ------------------------------------------------------------------ sessions

def create_session(user_id: str) -> str:
    """Create a session and return the RAW token (goes in the cookie only)."""
    raw = secrets.token_urlsafe(32)
    now = _now()
    _db()["sessions"].insert_one({
        "token_hash": _hash_token(raw),
        "user_id": user_id,
        "created_at": now,
        "expires_at": now + SESSION_TTL,
    })
    return raw


def resolve_session(raw: str) -> str | None:
    """Return user_id for a valid session, sliding the expiry forward."""
    if not raw:
        return None
    col = _db()["sessions"]
    token_hash = _hash_token(raw)
    now = _now()
    doc = col.find_one({"token_hash": token_hash, "expires_at": {"$gt": now}})
    if doc is None:
        return None
    new_expiry = now + SESSION_TTL
    if new_expiry - doc["expires_at"] > _SLIDE_AFTER:
        col.update_one({"token_hash": token_hash}, {"$set": {"expires_at": new_expiry}})
    return doc["user_id"]


def delete_session(raw: str) -> None:
    if raw:
        _db()["sessions"].delete_one({"token_hash": _hash_token(raw)})


# ------------------------------------------------------------------ memberships

def add_store_member(user_id: str, store_id: str, role: str = "owner") -> None:
    _db()["store_members"].update_one(
        {"user_id": user_id, "store_id": store_id},
        {"$setOnInsert": {"role": role, "created_at": _now()}},
        upsert=True,
    )


def is_store_member(user_id: str, store_id: str) -> bool:
    return _db()["store_members"].count_documents(
        {"user_id": user_id, "store_id": store_id}, limit=1
    ) > 0


def store_has_members(store_id: str) -> bool:
    return _db()["store_members"].count_documents({"store_id": store_id}, limit=1) > 0


def list_user_stores(user_id: str) -> list[dict]:
    return list(_db()["store_members"].find(
        {"user_id": user_id}, {"_id": 0, "store_id": 1, "role": 1}
    ))