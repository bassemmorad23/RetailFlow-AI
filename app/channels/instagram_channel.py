"""
Instagram DM channel: send replies via IG Graph API.
Uses per-store credentials.
"""

import logging
import httpx

from app.settings.store_credentials import get_credentials

logger = logging.getLogger(__name__)


def send_dm(store_id: str, recipient_id: str, text: str) -> bool:
    """Send a DM to a customer. Returns True on success."""
    creds = get_credentials(store_id, "instagram")
    if creds is None:
        logger.error("No IG creds for store", extra={"store_id": store_id})
        return False

    try:
        resp = httpx.post(
            f"https://graph.instagram.com/v21.0/{creds['instagram_business_account_id']}/messages",
            params={"access_token": creds["access_token"]},
            json={
                "recipient": {"id": recipient_id},
                "message": {"text": text},
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        return True
    except Exception:
        logger.exception("IG DM send failed")
        return False