"""
Meta webhook signature verification.

Meta signs every webhook POST with X-Hub-Signature-256:
    "sha256=" + HMAC_SHA256(app_secret, raw_request_body)

We verify against the raw bytes BEFORE parsing or storing anything.
Fail closed: a channel with no configured secret rejects everything.
"""

import hashlib
import hmac

from app.config import settings


def verify_meta_signature(raw_body: bytes, header: str | None, secrets: list[str]) -> bool:
    if not header or not header.startswith("sha256="):
        return False
    received = header[len("sha256="):].strip().lower()
    for secret in secrets:
        if not secret:
            continue
        expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
        if hmac.compare_digest(expected, received):
            return True
    return False


def webhook_secrets(channel: str) -> list[str]:
    """App secrets that may legitimately sign this channel's webhooks."""
    candidates = {
        "instagram": [settings.INSTAGRAM_APP_SECRET, settings.WHATSAPP_APP_SECRET],
        "messenger": [settings.META_APP_SECRET],
        "whatsapp": [settings.WHATSAPP_APP_SECRET],
    }.get(channel, [])
    return [s for s in candidates if s]