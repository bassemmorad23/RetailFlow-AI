"""
Live inbox stream (Server-Sent Events).

GET /stores/{store_id}/inbox/events

- Pushes inbox events (message.created, message.updated, conversation.updated)
- Resume: the browser's EventSource sends Last-Event-ID on reconnect; we
  continue right after it. `?after=` works for the first connection.
- First connection without either: starts from "now" (no history replay).
- If the requested position has expired (events kept 24h): sends `resync`
  so the UI reloads its lists, then continues from now.
- Session + membership re-checked periodically: `auth_expired` then close.
- Heartbeat comments keep proxies from closing idle connections.
- Connections end after MAX_CONNECTION_SECONDS; EventSource reconnects
  automatically (bounded resources, fresh auth check).
- All Mongo calls run in the threadpool, never on the event loop.
"""

import asyncio
import json
import time
from datetime import datetime

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.auth.dependencies import require_store_member
from app.auth.repository import is_store_member, resolve_session
from app.config import settings
from app.inbox import repository as repo

router = APIRouter(prefix="/stores/{store_id}/inbox", tags=["inbox"])

POLL_INTERVAL_SECONDS = 1.0
HEARTBEAT_SECONDS = 15.0
AUTH_RECHECK_SECONDS = 60.0
MAX_CONNECTION_SECONDS = 30 * 60
RECONNECT_DELAY_MS = 3000
_BATCH = 100


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Not serializable: {type(value).__name__}")


def format_sse(event_type: str, data: dict, event_id: int | None = None) -> str:
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event_type}")
    lines.append("data: " + json.dumps(data, default=_json_default, ensure_ascii=False))
    return "\n".join(lines) + "\n\n"


def _parse_last_event_id(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _still_authorized(raw_session: str | None, store_id: str) -> bool:
    user_id = resolve_session(raw_session) if raw_session else None
    return user_id is not None and is_store_member(user_id, store_id)


@router.get("/events")
async def inbox_events(
    request: Request,
    store_id: str = Depends(require_store_member),
    after: int | None = Query(default=None, ge=0),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    start = _parse_last_event_id(last_event_id)
    if start is None:
        start = after
    if start is None:
        start = await run_in_threadpool(repo.latest_event_seq, store_id)

    raw_session = request.cookies.get(settings.SESSION_COOKIE_NAME)
    return StreamingResponse(
        _stream(request, store_id, start, raw_session),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _stream(request: Request, store_id: str, after_seq: int, raw_session: str | None):
    yield f"retry: {RECONNECT_DELAY_MS}\n\n"

    # Resume position older than what we still keep -> tell the UI to reload.
    if after_seq > 0:
        first = await run_in_threadpool(repo.events_after, store_id, after_seq, 1)
        if first and first[0]["seq"] > after_seq + 1:
            latest = await run_in_threadpool(repo.latest_event_seq, store_id)
            yield format_sse("resync", {"reason": "events_expired"}, event_id=latest)
            after_seq = latest

    started = last_beat = last_auth = time.monotonic()

    while True:
        if await request.is_disconnected():
            break
        now = time.monotonic()
        if now - started > MAX_CONNECTION_SECONDS:
            break  # EventSource reconnects with Last-Event-ID

        if now - last_auth > AUTH_RECHECK_SECONDS:
            if not await run_in_threadpool(_still_authorized, raw_session, store_id):
                yield format_sse("auth_expired", {})
                break
            last_auth = now

        events = await run_in_threadpool(repo.events_after, store_id, after_seq, _BATCH)
        for event in events:
            data = {"conversation_id": event["conversation_id"], **event["payload"]}
            yield format_sse(event["type"], data, event_id=event["seq"])
            after_seq = event["seq"]

        if events:
            last_beat = now
        elif now - last_beat > HEARTBEAT_SECONDS:
            yield ": ping\n\n"
            last_beat = now

        if len(events) < _BATCH:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)