"""
Per-request logging context using contextvars.

WHY THIS EXISTS:
Every log line during a single HTTP request should carry the same
request_id, store_id, and conversation_id. Passing them explicitly to
every logger call would be tedious and easy to forget. Instead, we store
them in Python's contextvars — which are automatically scoped to the
current request (even under FastAPI's threadpool concurrency) — and a
logging filter injects them into every log record automatically.

HOW IT WORKS:
1. Middleware in api.py calls set_request_context(...) at the start of
   each request
2. Anywhere downstream, any logger.info/warning/error call inherits
   those fields (via the ContextFilter installed in logging_config.py)
3. Middleware calls clear_request_context() when the request ends,
   so fields don't leak to the next request on the same worker

WHY contextvars, NOT threading.local:
FastAPI runs sync endpoints in a threadpool. threading.local works there
but breaks under async code. contextvars is the standard Python pattern
that works correctly for BOTH sync and async, so it's future-proof if we
ever switch to async endpoints.
"""

from contextvars import ContextVar

_request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)
_store_id_ctx: ContextVar[str | None] = ContextVar("store_id", default=None)
_conversation_id_ctx: ContextVar[str | None] = ContextVar("conversation_id", default=None)


def set_request_context(
    request_id: str | None = None,
    store_id: str | None = None,
    conversation_id: str | None = None,
) -> None:
    """Set the fields that every subsequent log line should carry."""
    if request_id is not None:
        _request_id_ctx.set(request_id)
    if store_id is not None:
        _store_id_ctx.set(store_id)
    if conversation_id is not None:
        _conversation_id_ctx.set(conversation_id)


def clear_request_context() -> None:
    """Reset all context fields so they don't leak to the next request."""
    _request_id_ctx.set(None)
    _store_id_ctx.set(None)
    _conversation_id_ctx.set(None)


def get_request_context() -> dict:
    """Return the current context as a dict (used by the logging filter)."""
    return {
        "request_id": _request_id_ctx.get(),
        "store_id": _store_id_ctx.get(),
        "conversation_id": _conversation_id_ctx.get(),
    }