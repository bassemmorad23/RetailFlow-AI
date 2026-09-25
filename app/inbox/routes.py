"""
Unified Inbox API. Channel-agnostic: the same endpoints and shapes serve
Instagram, Messenger, WhatsApp and web conversations.

Tenant isolation:
- require_store_member: caller must own store_id
- every lookup filters by store_id AND conversation_id
- "not found" and "belongs to another store" return the identical 404
"""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.auth.dependencies import get_current_user_id, require_store_member
from app.inbox import actions
from app.inbox import repository as repo
from app.inbox.models import AiMode, InboxConversation, InboxMessage
from app.inbox.summary import maybe_refresh_summary
from app.inbox.windows import with_reply_window
from app.schemas.models import Channel

router = APIRouter(prefix="/stores/{store_id}/conversations", tags=["inbox"])

_NOT_FOUND = "Conversation not found"


class ConversationList(BaseModel):
    conversations: list[InboxConversation]
    next_cursor: str | None


class MessagePage(BaseModel):
    messages: list[InboxMessage]
    has_more: bool


class ManualReplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=actions.MAX_REPLY_CHARS)
    client_message_id: str = Field(min_length=8, max_length=100,
                                   description="Unique per send attempt (e.g. a UUID). Prevents double sends.")

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty")
        return v


def _conversation_or_404(store_id: str, conversation_id: str) -> dict:
    conv = repo.get_conversation(store_id, conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return conv


# ---------------------------------------------------------------- read

@router.get("", response_model=ConversationList)
def list_conversations(
    store_id: str = Depends(require_store_member),
    channel: Channel | None = None,
    ai_mode: AiMode | None = None,
    unread: bool = False,
    needs_attention: bool = False,
    cursor: str | None = Query(default=None, max_length=300),
    limit: int = Query(default=30, ge=1, le=100),
) -> dict:
    """All channels by default; filter with ?channel=instagram|messenger|whatsapp|web."""
    try:
        docs, next_cursor = repo.list_conversations(
            store_id, channel=channel, ai_mode=ai_mode, unread_only=unread,
            needs_attention=needs_attention, cursor=cursor, limit=limit,
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid cursor")
    return {"conversations": [with_reply_window(d) for d in docs], "next_cursor": next_cursor}


@router.get("/{conversation_id}", response_model=InboxConversation)
def get_conversation(conversation_id: str, store_id: str = Depends(require_store_member)) -> dict:
    return with_reply_window(_conversation_or_404(store_id, conversation_id))


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


# ---------------------------------------------------------------- merchant actions

@router.post("/{conversation_id}/ai/pause", response_model=InboxConversation)
def pause_ai(
    conversation_id: str,
    store_id: str = Depends(require_store_member),
    user_id: str = Depends(get_current_user_id),
) -> dict:
    try:
        return with_reply_window(actions.pause_ai(store_id, conversation_id, user_id=user_id))
    except actions.ConversationNotFound:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)


@router.post("/{conversation_id}/ai/resume", response_model=InboxConversation)
def resume_ai(
    conversation_id: str,
    background_tasks: BackgroundTasks,
    store_id: str = Depends(require_store_member),
    user_id: str = Depends(get_current_user_id),
) -> dict:
    try:
        conv = actions.resume_ai(store_id, conversation_id, user_id=user_id)
    except actions.ConversationNotFound:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    # Bring the summary up to date with everything said while the AI was paused.
    background_tasks.add_task(maybe_refresh_summary, store_id, conversation_id, force=True)
    return with_reply_window(conv)


@router.post("/{conversation_id}/messages", response_model=InboxMessage, status_code=201)
def send_manual_reply(
    conversation_id: str,
    body: ManualReplyRequest,
    background_tasks: BackgroundTasks,
    store_id: str = Depends(require_store_member),
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """
    Merchant replies through the conversation's own channel. Auto-pauses the AI.
    Returns 201 even if the platform rejected delivery: check delivery_status.
    """
    try:
        msg = actions.send_manual_reply(
            store_id, conversation_id, body.text,
            user_id=user_id, client_message_id=body.client_message_id,
        )
    except actions.ConversationNotFound:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    except actions.ReplyWindowClosed:
        raise HTTPException(status_code=409, detail={
            "code": "reply_window_closed",
            "message": "The reply window for this channel has closed. "
                       "The customer needs to message you first.",
        })
    background_tasks.add_task(maybe_refresh_summary, store_id, conversation_id, force=True)
    return msg


@router.post("/{conversation_id}/read", response_model=InboxConversation)
def mark_read(conversation_id: str, store_id: str = Depends(require_store_member)) -> dict:
    try:
        return with_reply_window(actions.mark_read(store_id, conversation_id))
    except actions.ConversationNotFound:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)


@router.post("/{conversation_id}/attention/clear", response_model=InboxConversation)
def clear_attention(conversation_id: str, store_id: str = Depends(require_store_member)) -> dict:
    _conversation_or_404(store_id, conversation_id)
    repo.set_needs_attention(store_id, conversation_id, False)
    return with_reply_window(_conversation_or_404(store_id, conversation_id))