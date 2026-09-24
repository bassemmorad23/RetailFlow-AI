"""
Business echoes (Instagram, Messenger).

Meta sends an "echo" for every message sent FROM the business account:
- our own API sends (AI or StoreFlow manual replies) -> ignore
- the merchant replying in the native Instagram app / Meta Business Suite
  -> store as a merchant message and auto-pause the AI

Protection against mistaking our own sends for merchant replies:
1. echo carries one of our app ids          -> ours
2. wait ECHO_SETTLE_SECONDS before deciding  -> our send has saved its ids
3. echo id matches any id we stored          -> ours
Only if all say "not ours" is it treated as a native merchant reply.

Human messages from the native app have sent_by_user_id = None.
"""

import logging
import time

from app.config import settings
from app.inbox import events
from app.inbox import repository as repo
from app.inbox.summary import maybe_refresh_summary

logger = logging.getLogger(__name__)

ECHO_SETTLE_SECONDS = 5.0
_sleep = time.sleep  # replaced in tests

_CHANNEL_LABEL = {"instagram": "Instagram", "messenger": "Messenger"}


def _own_app_ids() -> set[str]:
    ids = {getattr(settings, "META_APP_ID", ""), getattr(settings, "INSTAGRAM_APP_ID", "")}
    return {str(i) for i in ids if i}


def handle_business_echo(
    store_id: str,
    channel: str,
    customer_external_id: str,
    text: str,
    *,
    external_message_id: str | None,
    app_id: str | int | None = None,
    settle: bool = True,
) -> str:
    """Returns "known" (our own / already stored) or "stored" (native merchant reply)."""
    if app_id is not None and str(app_id) in _own_app_ids():
        return "known"

    if settle:
        _sleep(ECHO_SETTLE_SECONDS)

    if external_message_id and repo.find_message_by_external_id(store_id, external_message_id):
        return "known"

    conv = repo.get_or_create_conversation(store_id, channel, customer_external_id)
    cid = conv["id"]

    msg = repo.add_message(
        store_id, cid, "human", text,
        delivery_status="sent", external_message_id=external_message_id,
    )
    if msg is None:
        return "known"  # retry of an echo we already stored
    events.emit_message_created(store_id, msg)

    if repo.change_ai_mode(store_id, cid, expected="auto", new="paused", changed_by=None):
        label = _CHANNEL_LABEL.get(channel, channel)
        note = repo.add_message(
            store_id, cid, "system",
            f"AI paused automatically: merchant replied from the {label} app",
            delivery_status="internal",
        )
        if note:
            events.emit_message_created(store_id, note)
    events.emit_conversation_updated(store_id, cid)

    maybe_refresh_summary(store_id, cid, force=True)
    logger.info("Native-app merchant reply stored", extra={"inbox_conversation": cid})
    return "stored"