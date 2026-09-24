"""Facebook Messenger channel: send replies via the Page (per-store credentials)."""

import logging

import httpx

from app.channels.base import CHANNEL_TEXT_LIMITS, SendResult, send_parts, split_text
from app.settings.store_credentials import get_credentials

logger = logging.getLogger(__name__)


def send_message(store_id: str, recipient_psid: str, text: str) -> SendResult:
    creds = get_credentials(store_id, "messenger")
    if creds is None:
        logger.error("No Messenger creds for store")
        return SendResult(ok=False, error_code="not_configured")

    url = f"https://graph.facebook.com/v21.0/{creds['page_id']}/messages"
    headers = {"Authorization": f"Bearer {creds['page_access_token']}"}

    def post(part: str) -> httpx.Response:
        return httpx.post(
            url, headers=headers, timeout=30.0,
            json={
                "recipient": {"id": recipient_psid},
                "message": {"text": part},
                "messaging_type": "RESPONSE",
            },
        )

    result = send_parts(post, split_text(text, CHANNEL_TEXT_LIMITS["messenger"]),
                        lambda body: body.get("message_id"))
    if not result.ok:
        logger.warning("Messenger send failed", extra={"send_error": result.error_code,
                                                       "send_detail": result.error_detail})
    return result