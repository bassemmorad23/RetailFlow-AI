"""
Channel-agnostic send. The inbox and the AI never call a specific channel
directly — they call send(channel, ...) and get a SendResult back.
"""

from app.channels.base import SendResult
from app.channels.instagram_channel import send_dm
from app.channels.messenger_channel import send_message
from app.channels.whatsapp_channel import send_text


def send(channel: str, store_id: str, recipient_id: str, text: str) -> SendResult:
    if channel == "instagram":
        return send_dm(store_id, recipient_id, text)
    if channel == "messenger":
        return send_message(store_id, recipient_id, text)
    if channel == "whatsapp":
        return send_text(store_id, recipient_id, text)
    if channel == "web":
        # Web replies are stored in the inbox; the widget receives them
        # through its own live channel (Inbox Phase 3). Nothing to push here.
        return SendResult(ok=True)
    return SendResult(ok=False, error_code="unsupported_channel")