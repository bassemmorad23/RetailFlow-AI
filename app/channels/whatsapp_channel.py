"""WhatsApp Cloud API channel."""

import logging
import httpx
from app.config import settings

logger = logging.getLogger(__name__)


def send_text(recipient_phone: str, text: str) -> bool:
    """Send a text message via WhatsApp Cloud API.

    Note: WhatsApp requires the customer to message us first (opens 24h window)
    before we can send free-form text. Outside that window, only approved
    templates can be sent.
    """
    try:
        r = httpx.post(
            f"https://graph.facebook.com/v21.0/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages",
            headers={"Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}"},
            json={
                "messaging_product": "whatsapp",
                "to": recipient_phone,
                "type": "text",
                "text": {"body": text},
            },
            timeout=30.0,
        )
        r.raise_for_status()
        return True
    except Exception:
        logger.exception("WhatsApp send failed")
        return False