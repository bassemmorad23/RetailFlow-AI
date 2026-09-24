"""
Shared channel sending primitives.

- SendResult: what every channel returns (ok, platform message ids, error reason)
- split_text: respects per-channel length limits (Instagram rejects > 1000 chars)
- classify_meta_error: maps Meta Graph errors to reasons the merchant can act on
- send_parts: sends parts in order, stops at the first failure

Tokens are sent in the Authorization header, never in the URL, so failed
requests can't leak them into logs or Sentry.
"""

from dataclasses import dataclass, field
from typing import Callable

import httpx

CHANNEL_TEXT_LIMITS: dict[str, int] = {
    "instagram": 1000,
    "messenger": 2000,
    "whatsapp": 4096,
}

# error_code values: window_closed | auth_expired | rate_limited |
#                    recipient_unavailable | not_configured | network |
#                    unsupported_channel | unknown


@dataclass(frozen=True)
class SendResult:
    ok: bool
    external_message_ids: list[str] = field(default_factory=list)
    error_code: str | None = None
    error_detail: str | None = None


def split_text(text: str, limit: int | None) -> list[str]:
    """Split on paragraph, then sentence, then word boundaries. Never returns empty parts."""
    text = (text or "").strip()
    if not text:
        return []
    if not limit or len(text) <= limit:
        return [text]

    parts: list[str] = []
    rest = text
    while len(rest) > limit:
        window = rest[:limit]
        cut = window.rfind("\n")
        if cut < limit * 0.5:
            cut = max(window.rfind(s) for s in (". ", "! ", "? ", "؟ "))
        if cut < limit * 0.5:
            cut = window.rfind(" ")
        cut = limit if cut <= 0 else cut + 1
        parts.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        parts.append(rest)
    return [p for p in parts if p]


_RATE_LIMIT_CODES = {4, 17, 32, 613, 130429}
_WINDOW_CODES = {2018278, 131047}           # IG/Messenger subcode, WhatsApp code
_UNAVAILABLE_CODES = {551, 1545041, 131026}


def classify_meta_error(resp: httpx.Response) -> tuple[str, str]:
    """Return (error_code, short_detail) for a failed Meta Graph response."""
    try:
        err = resp.json().get("error", {}) or {}
    except ValueError:
        return "unknown", f"HTTP {resp.status_code}"

    code = err.get("code")
    sub = err.get("error_subcode")
    detail = f"{code}/{sub}: {str(err.get('message', ''))[:150]}"

    if code in _WINDOW_CODES or sub in _WINDOW_CODES:
        return "window_closed", detail
    if code == 190:
        return "auth_expired", detail
    if code in _RATE_LIMIT_CODES:
        return "rate_limited", detail
    if code in _UNAVAILABLE_CODES or sub in _UNAVAILABLE_CODES:
        return "recipient_unavailable", detail
    return "unknown", detail


def send_parts(
    post_one: Callable[[str], httpx.Response],
    parts: list[str],
    extract_id: Callable[[dict], str | None],
) -> SendResult:
    """Send parts in order. Stops at the first failure (ids of sent parts are kept)."""
    if not parts:
        return SendResult(ok=False, error_code="unknown", error_detail="empty message")

    ids: list[str] = []
    for part in parts:
        try:
            resp = post_one(part)
        except httpx.HTTPError as exc:
            return SendResult(False, ids, "network", type(exc).__name__)
        if resp.status_code >= 400:
            code, detail = classify_meta_error(resp)
            return SendResult(False, ids, code, detail)
        try:
            mid = extract_id(resp.json())
        except ValueError:
            mid = None
        if mid:
            ids.append(mid)
    return SendResult(ok=True, external_message_ids=ids)