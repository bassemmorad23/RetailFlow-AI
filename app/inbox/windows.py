"""
Reply-window rules.

Messaging platforms only allow free-form business replies for a limited
time after the customer's last message:
- WhatsApp: 24h (after that, only pre-approved templates)
- Instagram / Messenger: 24h (up to 7 days with Meta's Human Agent
  permission, which requires App Review — not enabled yet)
- Web widget: no window
"""

from datetime import datetime, timedelta, timezone

REPLY_WINDOWS: dict[str, timedelta] = {
    "whatsapp": timedelta(hours=24),
    "instagram": timedelta(hours=24),
    "messenger": timedelta(hours=24),
}


def reply_window(conv: dict, now: datetime | None = None) -> tuple[bool, datetime | None]:
    """Return (is_open, expires_at). expires_at is None when the channel has no window."""
    window = REPLY_WINDOWS.get(conv.get("channel"))
    if window is None:
        return True, None

    last = conv.get("last_customer_message_at")
    if last is None:
        return False, None
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)

    expires = last + window
    now = now or datetime.now(timezone.utc)
    return now < expires, expires


def with_reply_window(conv: dict) -> dict:
    """Conversation dict enriched with reply_window_open / reply_window_expires_at."""
    is_open, expires = reply_window(conv)
    return {**conv, "reply_window_open": is_open, "reply_window_expires_at": expires}