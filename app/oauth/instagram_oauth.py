"""
Instagram Business OAuth flow.

Merchant clicks 'Connect Instagram' on StoreFlow AI → we redirect them
to Meta's authorize page → they approve → Meta redirects to our callback
with a code → we exchange for token → save per-store.

Uses Instagram Business Login (long-lived tokens, 60 days).
"""

import logging
from urllib.parse import urlencode

import httpx

from app.config import settings


logger = logging.getLogger(__name__)


_META_GRAPH_URL = "https://graph.instagram.com"
_META_OAUTH_URL = "https://api.instagram.com/oauth"
_SCOPES = "instagram_business_basic,instagram_business_manage_messages"


def build_authorize_url(store_id: str) -> str:
    """
    URL the merchant hits to start Instagram OAuth.
    We put store_id in the state param to know which store to save under
    when Meta calls our callback.
    """
    params = {
        "client_id": settings.INSTAGRAM_APP_ID,
        "redirect_uri": f"{settings.INSTAGRAM_REDIRECT_BASE_URL}/instagram/callback",
        "response_type": "code",
        "scope": _SCOPES,
        "state": store_id,  # simple — no CSRF for now, add later
    }
    return f"https://www.instagram.com/oauth/authorize?{urlencode(params)}"


def exchange_code_for_token(code: str) -> dict:
    """
    Exchange short-lived code for a short-lived access token.
    Returns: {access_token, user_id}
    """
    resp = httpx.post(
        f"{_META_OAUTH_URL}/access_token",
        data={
            "client_id": settings.INSTAGRAM_APP_ID,
            "client_secret": settings.INSTAGRAM_APP_SECRET,
            "grant_type": "authorization_code",
            "redirect_uri": f"{settings.INSTAGRAM_REDIRECT_BASE_URL}/instagram/callback",
            "code": code,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def exchange_for_long_lived_token(short_token: str) -> dict:
    """
    Swap short-lived (1h) token for long-lived (60d) token.
    Returns: {access_token, token_type, expires_in}
    """
    resp = httpx.get(
        f"{_META_GRAPH_URL}/access_token",
        params={
            "grant_type": "ig_exchange_token",
            "client_secret": settings.INSTAGRAM_APP_SECRET,
            "access_token": short_token,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def get_account_info(access_token: str) -> dict:
    """Fetch IG account info to store username + business account id."""
    resp = httpx.get(
        f"{_META_GRAPH_URL}/v21.0/me",
        params={
            "fields": "id,user_id,username,account_type",
            "access_token": access_token,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()