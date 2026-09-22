"""Facebook Messenger channel: send replies via Graph API."""

import logging
import httpx
from app.settings.store_credentials import get_credentials

logger = logging.getLogger(__name__)


def send_message(store_id: str, recipient_psid: str, text: str) -> bool:
    """Send a message to a customer via their Page-scoped ID (PSID)."""
    creds = get_credentials(store_id, "messenger")
    if creds is None:
        logger.error("No Messenger creds for store", extra={"store_id": store_id})
        return False

    try:
        resp = httpx.post(
            f"https://graph.facebook.com/v21.0/{creds['page_id']}/messages",
            params={"access_token": creds["page_access_token"]},
            json={
                "recipient": {"id": recipient_psid},
                "message": {"text": text},
                "messaging_type": "RESPONSE",
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        return True
    except Exception:
        logger.exception("Messenger send failed")
        return False