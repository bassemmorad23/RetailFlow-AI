"""
Merchant actions on a conversation (channel-agnostic):
pause AI, resume AI, send a manual reply, mark as read.

Rules (locked decisions):
- A manual reply auto-pauses the AI, so AI and human never both answer.
- Resume is manual only; the AI answers the NEXT customer message.
- Pause/resume leave an internal note in the transcript (never sent to the customer).
- Manual replies respect each channel's reply window.
- The same client_message_id is only ever sent once (double-click safe).
"""

import logging

from app.channels.dispatcher import send
from app.inbox import events
from app.inbox import repository as repo
from app.inbox.windows import reply_window

logger = logging.getLogger(__name__)

MAX_REPLY_CHARS = 4000

NOTE_PAUSED = "AI paused by merchant"
NOTE_AUTO_PAUSED = "AI paused automatically: merchant replied"
NOTE_RESUMED = "AI resumed by merchant"


class ConversationNotFound(Exception):
    pass


class ReplyWindowClosed(Exception):
    pass


def _get(store_id: str, conversation_id: str) -> dict:
    conv = repo.get_conversation(store_id, conversation_id)
    if conv is None:
        raise ConversationNotFound(conversation_id)
    return conv


def _note(store_id: str, conversation_id: str, text: str) -> None:
    msg = repo.add_message(store_id, conversation_id, "system", text, delivery_status="internal")
    if msg:
        events.emit_message_created(store_id, msg)


# ---------------------------------------------------------------- AI mode

def pause_ai(store_id: str, conversation_id: str, *, user_id: str, note: str = NOTE_PAUSED) -> dict:
    """Idempotent: pausing an already-paused conversation changes nothing."""
    _get(store_id, conversation_id)
    if repo.change_ai_mode(store_id, conversation_id, expected="auto", new="paused", changed_by=user_id):
        _note(store_id, conversation_id, note)
        events.emit_conversation_updated(store_id, conversation_id)
        logger.info("AI paused", extra={"inbox_conversation": conversation_id})
    return _get(store_id, conversation_id)


def resume_ai(store_id: str, conversation_id: str, *, user_id: str) -> dict:
    """Idempotent. The caller should refresh the summary afterwards (background)."""
    _get(store_id, conversation_id)
    if repo.change_ai_mode(store_id, conversation_id, expected="paused", new="auto", changed_by=user_id):
        _note(store_id, conversation_id, NOTE_RESUMED)
        events.emit_conversation_updated(store_id, conversation_id)
        logger.info("AI resumed", extra={"inbox_conversation": conversation_id})
    return _get(store_id, conversation_id)


# ---------------------------------------------------------------- read state

def mark_read(store_id: str, conversation_id: str) -> dict:
    _get(store_id, conversation_id)
    repo.mark_read(store_id, conversation_id)
    events.emit_conversation_updated(store_id, conversation_id)
    return _get(store_id, conversation_id)


# ---------------------------------------------------------------- manual reply

def send_manual_reply(
    store_id: str,
    conversation_id: str,
    text: str,
    *,
    user_id: str,
    client_message_id: str,
) -> dict:
    """
    Send a merchant reply through the conversation's own channel.
    Returns the stored message (with delivery status). A repeated
    client_message_id returns the original message without resending.
    """
    conv = _get(store_id, conversation_id)

    existing = repo.get_message_by_client_id(store_id, conversation_id, client_message_id)
    if existing:
        return existing

    is_open, _ = reply_window(conv)
    if not is_open:
        raise ReplyWindowClosed(conversation_id)

    if conv.get("ai_mode") == "auto":
        pause_ai(store_id, conversation_id, user_id=user_id, note=NOTE_AUTO_PAUSED)

    msg = repo.add_message(
        store_id, conversation_id, "human", text,
        delivery_status="pending", sent_by_user_id=user_id, client_message_id=client_message_id,
    )
    if msg is None:  # the same click arrived twice at the same moment
        return repo.get_message_by_client_id(store_id, conversation_id, client_message_id)
    events.emit_message_created(store_id, msg)

    result = send(conv["channel"], store_id, conv["customer"]["external_id"], text)
    status = "sent" if result.ok else "failed"
    repo.set_delivery(
        store_id, msg["id"], status,
        external_message_ids=result.external_message_ids, error=result.error_code,
    )
    msg.update(delivery_status=status, delivery_error=result.error_code)
    events.emit_message_updated(store_id, msg)
    events.emit_conversation_updated(store_id, conversation_id)

    logger.info("Manual reply sent" if result.ok else "Manual reply failed",
                extra={"inbox_conversation": conversation_id, "send_error": result.error_code})
    return msg