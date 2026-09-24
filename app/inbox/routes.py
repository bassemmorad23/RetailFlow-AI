"""
Unified Inbox API (read side). Channel-agnostic: the same endpoints and
shapes serve Instagram, Messenger, WhatsApp and web conversations.

Tenant isolation:
- require_store_member: caller must own store_id
- every lookup filters by store_id AND conversation_id
- "not found" and "belongs to another store" return the identical 404
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.auth.dependencies import require_store_member
from app.inbox import repository as repo
from app.inbox.models import AiMode, InboxConversation, InboxMessage
from app.schemas.models import Channel

router = APIRouter(prefix="/stores/{store_id}/conversations", tags=["inbox"])


class ConversationList(BaseModel):
    conversations: list[InboxConversation]
    next_cursor: str | None


class MessagePage(BaseModel):
    messages: list[InboxMessage]
    has_more: bool


def _conversation_or_404(store_id: str, conversation_id: str) -> dict:
    conv = repo.get_conversation(store_id, conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.get("", response_model=ConversationList)
def list_conversations(
    store_id: str = Depends(require_store_member),
    channel: Channel | None = None,
    ai_mode: AiMode | None = None,
    unread: bool = False,
    cursor: str | None = Query(default=None, max_length=300),
    limit: int = Query(default=30, ge=1, le=100),
) -> dict:
    """All channels by default; filter with ?channel=instagram|messenger|whatsapp|web."""
    try:
        docs, next_cursor = repo.list_conversations(
            store_id, channel=channel, ai_mode=ai_mode, unread_only=unread,
            cursor=cursor, limit=limit,
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid cursor")
    return {"conversations": docs, "next_cursor": next_cursor}


@router.get("/{conversation_id}", response_model=InboxConversation)
def get_conversation(conversation_id: str, store_id: str = Depends(require_store_member)) -> dict:
    return _conversation_or_404(store_id, conversation_id)


@router.get("/{conversation_id}/messages", response_model=MessagePage)
def get_messages(
    conversation_id: str,
    store_id: str = Depends(require_store_member),
    before_seq: int | None = Query(default=None, ge=1),
    limit: int = Query(default=50, ge=1, le=100),
) -> dict:
    """Latest page by default; pass before_seq to load older messages."""
    _conversation_or_404(store_id, conversation_id)
    msgs, has_more = repo.list_messages(store_id, conversation_id, before_seq=before_seq, limit=limit)
    return {"messages": msgs, "has_more": has_more}