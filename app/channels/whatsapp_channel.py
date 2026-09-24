"""
WhatsApp Cloud API channel.

Free-form text is only allowed within 24h of the customer's last message;
outside that window Meta returns an error mapped to "window_closed".

Credentials are still global (single test number). store_id is already in
the signature so the multi-tenant switch later stays inside this module.
"""

import logging

import httpx

from app.channels.base import CHANNEL_TEXT_LIMITS, SendResult, send_parts, split_text
from app.config import settings

logger = logging.getLogger(__name__)


def send_text(store_id: str, recipient_phone: str, text: str) -> SendResult:
    if not (settings.WHATSAPP_PHONE_NUMBER_ID and settings.WHATSAPP_ACCESS_TOKEN):
        return SendResult(ok=False, error_code="not_configured")

    url = f"https://graph.facebook.com/v21.0/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}"}

    def post(part: str) -> httpx.Response:
        return httpx.post(
            url, headers=headers, timeout=30.0,
            json={
                "messaging_product": "whatsapp",
                "to": recipient_phone,
                "type": "text",
                "text": {"body": part},
            },
        )

    def extract(body: dict) -> str | None:
        msgs = body.get("messages") or []
        return msgs[0].get("id") if msgs else None

    result = send_parts(post, split_text(text, CHANNEL_TEXT_LIMITS["whatsapp"]), extract)
    if not result.ok:
        logger.warning("WhatsApp send failed", extra={"send_error": result.error_code,
                                                      "send_detail": result.error_detail})
    return result