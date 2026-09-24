"""
Live inbox event helpers (shared by the Conversation Service and merchant actions).

Best-effort: a failed event is logged and never breaks a conversation.
"""

import logging

from app.inbox import repository as repo
from app.inbox.windows import with_reply_window

logger = logging.getLogger(__name__)

_CONVERSATION_FIELDS = (
    "id", "channel", "ai_mode", "unread_count", "last_message", "updated_at",
    "reply_window_open", "reply_window_expires_at",
)


def public_message(msg: dict) -> dict:
    return {k: v for k, v in msg.items() if k not in ("store_id", "external_message_ids")}


def emit(store_id: str, event_type: str, conversation_id: str, payload: dict) -> None:
    try:
        repo.append_event(store_id, event_type, conversation_id, payload)
    except Exception:
        logger.exception("Inbox event emit failed", extra={"inbox_event_type": event_type})


def emit_message_created(store_id: str, msg: dict) -> None:
    emit(store_id, "message.created", msg["conversation_id"], {"message": public_message(msg)})


def emit_message_updated(store_id: str, msg: dict) -> None:
    emit(store_id, "message.updated", msg["conversation_id"], {
        "message_id": msg["id"],
        "delivery_status": msg["delivery_status"],
        "delivery_error": msg.get("delivery_error"),
    })


def emit_conversation_updated(store_id: str, conversation_id: str) -> None:
    conv = repo.get_conversation(store_id, conversation_id)
    if conv is None:
        return
    conv = with_reply_window(conv)
    emit(store_id, "conversation.updated", conversation_id,
         {"conversation": {k: conv.get(k) for k in _CONVERSATION_FIELDS}})