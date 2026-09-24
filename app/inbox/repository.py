"""
Inbox storage layer (MongoDB).

Collections:
- inbox_conversations: one doc per (store_id, thread_key). thread_key is
  internal ("whatsapp:2010..."); the API only ever sees the random `id`.
- inbox_messages: one doc per message, ordered by per-conversation `seq`.
- inbox_events: live event log for SSE, per-store `seq`, auto-expires.
- inbox_event_counters: per-store event sequence counter.

TENANT ISOLATION: every read/write filters by store_id AND the object id.
Nothing is ever looked up by conversation/message id alone.

CONCURRENCY:
- conversation creation: atomic upsert on unique (store_id, thread_key)
- message seq: atomic $inc on the conversation
- dedupe: unique (store_id, external_message_id) -> Meta retries are ignored
- last_message / timestamps: only move forward ($max, seq-guarded $set)
- summary: compare-and-set on covers_through_seq (newer always wins)
"""

import base64
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from app.config import settings

EVENT_TTL = timedelta(hours=24)
MAX_PAGE = 100
_CONV_PROJECTION = {"_id": 0, "thread_key": 0}


@lru_cache(maxsize=1)
def _db() -> Database:
    db = MongoClient(settings.MONGO_URI, tz_aware=True)[settings.MONGO_DB]

    conv = db["inbox_conversations"]
    conv.create_index("id", unique=True)
    conv.create_index([("store_id", ASCENDING), ("thread_key", ASCENDING)], unique=True)
    conv.create_index([("store_id", ASCENDING), ("updated_at", DESCENDING), ("id", DESCENDING)])
    conv.create_index([("store_id", ASCENDING), ("channel", ASCENDING),
                       ("updated_at", DESCENDING), ("id", DESCENDING)])

    msgs = db["inbox_messages"]
    msgs.create_index("id", unique=True)
    msgs.create_index([("store_id", ASCENDING), ("conversation_id", ASCENDING), ("seq", ASCENDING)],
                      unique=True)
    msgs.create_index([("store_id", ASCENDING), ("external_message_id", ASCENDING)], unique=True,
                      partialFilterExpression={"external_message_id": {"$type": "string"}})

    events = db["inbox_events"]
    events.create_index([("store_id", ASCENDING), ("seq", ASCENDING)], unique=True)
    events.create_index("created_at", expireAfterSeconds=int(EVENT_TTL.total_seconds()))
    return db


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


# ---------------------------------------------------------------- conversations

def get_or_create_conversation(store_id: str, channel: str, customer_external_id: str) -> dict:
    """Return the conversation for this customer on this channel, creating it atomically."""
    col = _db()["inbox_conversations"]
    thread_key = f"{channel}:{customer_external_id}"
    flt = {"store_id": store_id, "thread_key": thread_key}
    now = _now()
    on_insert = {
        "id": _new_id("conv"),
        "channel": channel,
        "customer": {"external_id": customer_external_id, "display_name": None},
        "ai_mode": "auto",
        "ai_mode_changed_at": None,
        "status": "open",
        "last_message": None,
        "message_seq": 0,
        "unread_count": 0,
        "last_customer_message_at": None,
        "summary": {"text": "", "covers_through_seq": 0, "updated_at": None},
        "created_at": now,
        "updated_at": now,
    }
    try:
        return col.find_one_and_update(
            flt, {"$setOnInsert": on_insert}, upsert=True,
            return_document=ReturnDocument.AFTER, projection=_CONV_PROJECTION,
        )
    except DuplicateKeyError:  # two first-messages raced; the other one created it
        return col.find_one(flt, _CONV_PROJECTION)


def get_conversation(store_id: str, conversation_id: str) -> dict | None:
    return _db()["inbox_conversations"].find_one(
        {"store_id": store_id, "id": conversation_id}, _CONV_PROJECTION
    )


def _encode_cursor(doc: dict) -> str:
    raw = f"{doc['updated_at'].isoformat()}|{doc['id']}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def _decode_cursor(cursor: str) -> tuple[datetime, str]:
    """Raises ValueError on any malformed cursor."""
    try:
        ts, cid = base64.urlsafe_b64decode(cursor.encode()).decode().split("|", 1)
        return datetime.fromisoformat(ts), cid
    except Exception as exc:
        raise ValueError("Invalid cursor") from exc


def list_conversations(
    store_id: str,
    *,
    channel: str | None = None,
    ai_mode: str | None = None,
    unread_only: bool = False,
    cursor: str | None = None,
    limit: int = 30,
) -> tuple[list[dict], str | None]:
    """Newest-activity first. Returns (conversations, next_cursor)."""
    query: dict = {"store_id": store_id}
    if channel:
        query["channel"] = channel
    if ai_mode:
        query["ai_mode"] = ai_mode
    if unread_only:
        query["unread_count"] = {"$gt": 0}
    if cursor:
        ts, cid = _decode_cursor(cursor)
        query["$or"] = [{"updated_at": {"$lt": ts}}, {"updated_at": ts, "id": {"$lt": cid}}]

    limit = max(1, min(limit, MAX_PAGE))
    docs = list(
        _db()["inbox_conversations"].find(query, _CONV_PROJECTION)
        .sort([("updated_at", DESCENDING), ("id", DESCENDING)])
        .limit(limit + 1)
    )
    next_cursor = _encode_cursor(docs[limit - 1]) if len(docs) > limit else None
    return docs[:limit], next_cursor


# ---------------------------------------------------------------- messages

def add_message(
    store_id: str,
    conversation_id: str,
    sender_type: str,
    text: str,
    *,
    delivery_status: str,
    external_message_id: str | None = None,
    sent_by_user_id: str | None = None,
    ai_meta: dict | None = None,
) -> dict | None:
    """
    Append a message with the next seq. Returns the stored doc, or None if
    external_message_id was already stored (platform retry / echo of our send).
    Raises LookupError if the conversation doesn't belong to this store.
    """
    db = _db()
    msgs = db["inbox_messages"]

    if external_message_id and msgs.find_one(
        {"store_id": store_id, "external_message_id": external_message_id}, {"_id": 1}
    ):
        return None

    conv = db["inbox_conversations"].find_one_and_update(
        {"store_id": store_id, "id": conversation_id},
        {"$inc": {"message_seq": 1}},
        return_document=ReturnDocument.AFTER,
        projection={"_id": 0, "message_seq": 1},
    )
    if conv is None:
        raise LookupError("Conversation not found for this store")

    doc = {
        "id": _new_id("msg"),
        "store_id": store_id,
        "conversation_id": conversation_id,
        "seq": conv["message_seq"],
        "sender_type": sender_type,
        "text": text,
        "created_at": _now(),
        "delivery_status": delivery_status,
        "delivery_error": None,
        "external_message_id": external_message_id,
        "sent_by_user_id": sent_by_user_id,
        "ai_meta": ai_meta,
    }
    try:
        msgs.insert_one(doc)
    except DuplicateKeyError:  # same external id raced in; seq gap is harmless
        return None
    doc.pop("_id", None)

    _touch_conversation(store_id, conversation_id, doc)
    return doc


def _touch_conversation(store_id: str, conversation_id: str, msg: dict) -> None:
    """Move conversation state forward only — an older message never overwrites a newer one."""
    col = _db()["inbox_conversations"]
    flt = {"store_id": store_id, "id": conversation_id}
    at = msg["created_at"]

    update: dict = {"$max": {"updated_at": at}}
    if msg["sender_type"] == "customer":
        update["$max"]["last_customer_message_at"] = at
        update["$inc"] = {"unread_count": 1}
    col.update_one(flt, update)

    col.update_one(
        {**flt, "$or": [{"last_message": None}, {"last_message.seq": {"$lt": msg["seq"]}}]},
        {"$set": {"last_message": {
            "seq": msg["seq"], "text": msg["text"][:200],
            "sender_type": msg["sender_type"], "at": at,
        }}},
    )


def set_delivery(
    store_id: str,
    message_id: str,
    status: str,
    *,
    external_message_ids: list[str] | None = None,
    error: str | None = None,
) -> None:
    fields: dict = {"delivery_status": status, "delivery_error": error}
    if external_message_ids:
        fields["external_message_id"] = external_message_ids[0]
        fields["external_message_ids"] = external_message_ids
    _db()["inbox_messages"].update_one({"store_id": store_id, "id": message_id}, {"$set": fields})
    
    
def set_ai_mode(store_id: str, conversation_id: str, mode: str) -> bool:
    """Set auto/paused. True if the conversation exists for this store."""
    if mode not in ("auto", "paused"):
        raise ValueError(f"Invalid ai_mode '{mode}'")
    result = _db()["inbox_conversations"].update_one(
        {"store_id": store_id, "id": conversation_id},
        {"$set": {"ai_mode": mode, "ai_mode_changed_at": _now()}},
    )
    return result.matched_count == 1


def list_messages(
    store_id: str, conversation_id: str, *, before_seq: int | None = None, limit: int = 50
) -> tuple[list[dict], bool]:
    """Latest page (or the page before `before_seq`), returned oldest -> newest. Returns (messages, has_more)."""
    query: dict = {"store_id": store_id, "conversation_id": conversation_id}
    if before_seq is not None:
        query["seq"] = {"$lt": before_seq}
    limit = max(1, min(limit, MAX_PAGE))
    docs = list(
        _db()["inbox_messages"].find(query, {"_id": 0})
        .sort("seq", DESCENDING).limit(limit + 1)
    )
    has_more = len(docs) > limit
    return list(reversed(docs[:limit])), has_more


def messages_after(store_id: str, conversation_id: str, after_seq: int, limit: int = 30) -> list[dict]:
    """Messages with seq > after_seq, oldest first (used for AI context and summaries)."""
    return list(
        _db()["inbox_messages"].find(
            {"store_id": store_id, "conversation_id": conversation_id, "seq": {"$gt": after_seq}},
            {"_id": 0},
        ).sort("seq", ASCENDING).limit(limit)
    )


# ---------------------------------------------------------------- summary

def update_summary_cas(
    store_id: str, conversation_id: str, *, expected_covers: int, text: str, new_covers: int
) -> bool:
    """Write the summary only if nobody stored a newer one meanwhile. True if written."""
    if new_covers <= expected_covers:
        return False
    result = _db()["inbox_conversations"].update_one(
        {"store_id": store_id, "id": conversation_id, "summary.covers_through_seq": expected_covers},
        {"$set": {"summary": {"text": text, "covers_through_seq": new_covers, "updated_at": _now()}}},
    )
    return result.modified_count == 1


# ---------------------------------------------------------------- live events

def append_event(store_id: str, event_type: str, conversation_id: str, payload: dict) -> int:
    """Record a live-inbox event with the next per-store seq. Returns that seq."""
    db = _db()
    counter = db["inbox_event_counters"].find_one_and_update(
        {"_id": store_id}, {"$inc": {"seq": 1}},
        upsert=True, return_document=ReturnDocument.AFTER,
    )
    seq = counter["seq"]
    db["inbox_events"].insert_one({
        "store_id": store_id,
        "seq": seq,
        "type": event_type,
        "conversation_id": conversation_id,
        "payload": payload,
        "created_at": _now(),
    })
    return seq


def events_after(store_id: str, after_seq: int, limit: int = 100) -> list[dict]:
    return list(
        _db()["inbox_events"].find(
            {"store_id": store_id, "seq": {"$gt": after_seq}}, {"_id": 0}
        ).sort("seq", ASCENDING).limit(limit)
    )