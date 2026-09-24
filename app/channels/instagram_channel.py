"""Instagram DM channel: send replies via the Instagram Graph API (per-store credentials)."""

import logging

import httpx

from app.channels.base import CHANNEL_TEXT_LIMITS, SendResult, send_parts, split_text
from app.settings.store_credentials import get_credentials

logger = logging.getLogger(__name__)


def send_dm(store_id: str, recipient_id: str, text: str) -> SendResult:
    creds = get_credentials(store_id, "instagram")
    if creds is None:
        logger.error("No IG creds for store")
        return SendResult(ok=False, error_code="not_configured")

    url = "https://graph.instagram.com/v21.0/me/messages"
    headers = {"Authorization": f"Bearer {creds['access_token']}"}

    def post(part: str) -> httpx.Response:
        return httpx.post(
            url, headers=headers, timeout=30.0,
            json={"recipient": {"id": recipient_id}, "message": {"text": part}},
        )

    result = send_parts(post, split_text(text, CHANNEL_TEXT_LIMITS["instagram"]),
                        lambda body: body.get("message_id"))
    if not result.ok:
        logger.warning("IG send failed", extra={"send_error": result.error_code,
                                                "send_detail": result.error_detail})
    return result