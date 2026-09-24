"""
Web widget API (public, used from merchants' websites).

POST /widget/sessions                 -> {token, expires_at}
POST /widget/messages   (Bearer token) -> {reply, ai_paused, duplicate}
GET  /widget/messages   (Bearer token) -> {messages, last_seq}

Security:
- Visitor identity comes only from the server-issued token.
- A visitor only ever sees their own conversation.
- Internal notes, undelivered drafts, AI metadata and ids are never exposed.
- Rate limited per client.
"""

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.inbox import repository as repo
from app.inbox.service import handle_incoming_message
from app.inbox.summary import maybe_refresh_summary
from app.rate_limiter import rate_limit
from app.settings.store_settings import store_exists
from app.widget import sessions

router = APIRouter(prefix="/widget", tags=["widget"])

MAX_WIDGET_MESSAGE_CHARS = 2000
_HIDDEN = {"internal", "not_sent"}
_AUTHOR = {"customer": "customer", "ai": "ai", "human": "merchant", "system": "system"}

Author = Literal["customer", "ai", "merchant", "system"]


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    store_id: str = Field(min_length=1, max_length=100)


class CreateSessionResponse(BaseModel):
    token: str
    expires_at: datetime


class WidgetMessage(BaseModel):
    id: str
    seq: int
    author: Author
    text: str
    created_at: datetime


class SendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=MAX_WIDGET_MESSAGE_CHARS)
    client_message_id: str = Field(min_length=8, max_length=100)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty")
        return v


class SendResponse(BaseModel):
    reply: WidgetMessage | None
    ai_paused: bool
    duplicate: bool = False


class MessagesResponse(BaseModel):
    messages: list[WidgetMessage]
    last_seq: int


def widget_session(authorization: str | None = Header(default=None)) -> sessions.WidgetSession:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing widget token")
    session = sessions.resolve_session(authorization[7:].strip())
    if session is None:
        raise HTTPException(status_code=401, detail="Invalid or expired widget token")
    return session


def _public(msg: dict) -> dict:
    return {
        "id": msg["id"],
        "seq": msg["seq"],
        "author": _AUTHOR[msg["sender_type"]],
        "text": msg["text"],
        "created_at": msg["created_at"],
    }


@router.post("/sessions", response_model=CreateSessionResponse, status_code=201,
             dependencies=[Depends(rate_limit)])
def create_widget_session(body: CreateSessionRequest) -> dict:
    if not store_exists(body.store_id):
        raise HTTPException(status_code=404, detail="Store not found")
    token, expires_at = sessions.create_session(body.store_id)
    return {"token": token, "expires_at": expires_at}


@router.post("/messages", response_model=SendResponse, dependencies=[Depends(rate_limit)])
def send_widget_message(
    body: SendRequest,
    background_tasks: BackgroundTasks,
    session: sessions.WidgetSession = Depends(widget_session),
) -> dict:
    result = handle_incoming_message(
        session.store_id, "web", session.visitor_id, body.text,
        external_message_id=f"widget:{session.visitor_id}:{body.client_message_id}",
        refresh_summary=False,
    )
    background_tasks.add_task(maybe_refresh_summary, session.store_id, result.conversation_id)

    if result.duplicate:
        return {"reply": None, "ai_paused": False, "duplicate": True}

    msg = result.reply_message
    visible = msg is not None and msg.get("delivery_status") not in _HIDDEN
    return {"reply": _public(msg) if visible else None, "ai_paused": not visible}


@router.get("/messages", response_model=MessagesResponse)
def poll_widget_messages(
    after_seq: int | None = Query(default=None, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    session: sessions.WidgetSession = Depends(widget_session),
) -> dict:
    """First load: omit after_seq (latest page). Then poll with the returned last_seq."""
    conv = repo.find_conversation_by_customer(session.store_id, "web", session.visitor_id)
    if conv is None:
        return {"messages": [], "last_seq": after_seq or 0}

    if after_seq is None:
        msgs, _ = repo.list_messages(session.store_id, conv["id"], limit=limit)
    else:
        msgs = repo.messages_after(session.store_id, conv["id"], after_seq, limit=limit)

    last_seq = max([m["seq"] for m in msgs], default=after_seq or 0)
    visible = [_public(m) for m in msgs if m.get("delivery_status") not in _HIDDEN]
    return {"messages": visible, "last_seq": last_seq}