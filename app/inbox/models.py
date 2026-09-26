"""
Omnichannel Inbox models (API-facing shapes).

Channel-agnostic: the same models describe Instagram, Messenger,
WhatsApp and web conversations. `extra="ignore"` because Mongo docs
carry internal fields (e.g. thread_key) that the API never exposes.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.models import Channel

SenderType = Literal["customer", "ai", "human", "system"]
AiMode = Literal["auto", "paused"]
# internal = note visible to the merchant only (e.g. "AI paused"), never sent
DeliveryStatus = Literal["received", "pending", "sent", "failed", "not_sent", "internal"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


class CustomerRef(_Base):
    external_id: str
    display_name: str | None = None


class LastMessage(_Base):
    seq: int
    text: str
    sender_type: SenderType
    at: datetime


class ConversationSummary(_Base):
    text: str = ""
    covers_through_seq: int = 0
    updated_at: datetime | None = None


class InboxConversation(_Base):
    id: str
    store_id: str
    channel: Channel
    customer: CustomerRef
    ai_mode: AiMode = "auto"
    ai_mode_changed_at: datetime | None = None
    ai_mode_changed_by: str | None = None
    status: Literal["open"] = "open"
    last_message: LastMessage | None = None
    message_seq: int = 0
    unread_count: int = 0
    needs_attention: bool = False
    last_read_at: datetime | None = None
    last_customer_message_at: datetime | None = None
    reply_window_open: bool = True
    reply_window_expires_at: datetime | None = None
    summary: ConversationSummary = Field(default_factory=ConversationSummary)
    created_at: datetime
    updated_at: datetime


class AiMeta(_Base):
    emotion: str | None = None
    intent: str | None = None
    intent_confidence: float | None = None
    product_ids: list[str] = Field(default_factory=list)
    order_event: str | None = None
    aftersales: str | None = None
    


class InboxMessage(_Base):
    id: str
    store_id: str
    conversation_id: str
    seq: int
    sender_type: SenderType
    text: str
    created_at: datetime
    delivery_status: DeliveryStatus
    delivery_error: str | None = None
    external_message_id: str | None = None
    sent_by_user_id: str | None = None
    client_message_id: str | None = None
    ai_meta: AiMeta | None = None