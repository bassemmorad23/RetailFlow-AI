"""
Conversation Service — the single path every inbound customer message takes,
for every channel (Instagram, Messenger, WhatsApp, web).

  1. find/create the conversation (per store + channel + customer)
  2. store the customer message (platform retries are deduplicated)
  3. emit live events for the merchant's inbox
  4. if AI is paused -> stop here (no LLM call, no cost)
  5. run the AI pipeline
  6. re-check AI status (merchant may have paused meanwhile)
  7. store the reply, send it through the channel, record delivery

Event emission is best-effort: a failed event never breaks a conversation.
"""

import logging
from dataclasses import dataclass

from app.billing.plans import LIMIT_REACHED_REPLY
from app.channels.dispatcher import send
from app.core.orchestrator import PIPELINE_ERROR_REPLY, handle_message
from app.inbox import repository as repo
from app.response.response_generator import FALLBACK_REPLY
from app.schemas.models import AgentReply, CustomerMessage

logger = logging.getLogger(__name__)

# Automated non-AI replies are shown in the inbox as "system", not "ai".
_SYSTEM_REPLIES = {LIMIT_REACHED_REPLY, FALLBACK_REPLY, PIPELINE_ERROR_REPLY}


@dataclass
class IncomingResult:
    conversation_id: str
    duplicate: bool = False
    ai_replied: bool = False
    reply: AgentReply | None = None
    reply_message: dict | None = None


def handle_incoming_message(
    store_id: str,
    channel: str,
    customer_external_id: str,
    text: str,
    *,
    external_message_id: str | None = None,
) -> IncomingResult:
    conv = repo.get_or_create_conversation(store_id, channel, customer_external_id)
    cid = conv["id"]

    customer_msg = repo.add_message(
        store_id, cid, "customer", text,
        delivery_status="received", external_message_id=external_message_id,
    )
    if customer_msg is None:
        logger.info("Duplicate inbound message ignored", extra={"inbox_conversation": cid})
        return IncomingResult(conversation_id=cid, duplicate=True)

    _emit_message_created(store_id, customer_msg)
    _emit_conversation_updated(store_id, cid)

    if not _ai_enabled(store_id, cid):
        logger.info("AI paused: message stored for merchant", extra={"inbox_conversation": cid})
        return IncomingResult(conversation_id=cid)

    reply = handle_message(CustomerMessage(
        conversation_id=cid,
        customer_id=customer_external_id,
        store_id=store_id,
        text=text,
        channel=channel,
    ))

    sender_type = "system" if reply.reply_text in _SYSTEM_REPLIES else "ai"
    ai_meta = _ai_meta(reply) if sender_type == "ai" else None

    # Merchant may have paused while the AI was generating.
    still_enabled = _ai_enabled(store_id, cid)
    reply_msg = repo.add_message(
        store_id, cid, sender_type, reply.reply_text,
        delivery_status="pending" if still_enabled else "not_sent",
        ai_meta=ai_meta,
    )
    _emit_message_created(store_id, reply_msg)

    if not still_enabled:
        logger.info("AI reply suppressed: paused during generation", extra={"inbox_conversation": cid})
        _emit_conversation_updated(store_id, cid)
        return IncomingResult(conversation_id=cid, reply=reply, reply_message=reply_msg)

    result = send(channel, store_id, customer_external_id, reply.reply_text)
    status = "sent" if result.ok else "failed"
    repo.set_delivery(
        store_id, reply_msg["id"], status,
        external_message_ids=result.external_message_ids,
        error=result.error_code,
    )
    reply_msg.update(delivery_status=status, delivery_error=result.error_code)
    _emit(store_id, "message.updated", cid, {
        "message_id": reply_msg["id"], "delivery_status": status, "delivery_error": result.error_code,
    })
    _emit_conversation_updated(store_id, cid)

    return IncomingResult(
        conversation_id=cid, ai_replied=result.ok, reply=reply, reply_message=reply_msg,
    )


# ---------------------------------------------------------------- helpers

def _ai_enabled(store_id: str, conversation_id: str) -> bool:
    conv = repo.get_conversation(store_id, conversation_id)
    return conv is not None and conv.get("ai_mode") == "auto"


def _ai_meta(reply: AgentReply) -> dict:
    return {
        "emotion": reply.emotion.label.value,
        "intent": reply.intent.label.value,
        "intent_confidence": reply.intent.confidence,
        "product_ids": [r.product_id for r in reply.recommendations],
    }


def _public_message(msg: dict) -> dict:
    return {k: v for k, v in msg.items() if k != "store_id"}


def _emit(store_id: str, event_type: str, conversation_id: str, payload: dict) -> None:
    try:
        repo.append_event(store_id, event_type, conversation_id, payload)
    except Exception:
        logger.exception("Inbox event emit failed", extra={"inbox_event_type": event_type})


def _emit_message_created(store_id: str, msg: dict) -> None:
    _emit(store_id, "message.created", msg["conversation_id"], {"message": _public_message(msg)})


def _emit_conversation_updated(store_id: str, conversation_id: str) -> None:
    conv = repo.get_conversation(store_id, conversation_id)
    if conv is None:
        return
    fields = ("id", "channel", "ai_mode", "unread_count", "last_message", "updated_at")
    _emit(store_id, "conversation.updated", conversation_id,
          {"conversation": {k: conv.get(k) for k in fields}})   