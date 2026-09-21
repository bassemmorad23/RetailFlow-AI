"""Messenger (Facebook Page) OAuth flow."""

import httpx
from urllib.parse import urlencode
from app.config import settings

_META_OAUTH = "https://www.facebook.com/v21.0/dialog/oauth"
_GRAPH = "https://graph.facebook.com/v21.0"
_SCOPES = "pages_messaging,pages_manage_metadata,pages_read_engagement,pages_show_list"


def build_authorize_url(store_id: str) -> str:
    params = {
        "client_id": settings.META_APP_ID,  # same Meta app
        "redirect_uri": f"{settings.META_REDIRECT_BASE_URL}/messenger/callback",
        "state": store_id,
        "config_id": settings.META_FB_LOGIN_CONFIG_ID,
        "response_type": "code",
    }
    return f"{_META_OAUTH}?{urlencode(params)}"


def exchange_code_for_user_token(code: str) -> str:
    """Exchange code for short-lived user token."""
    r = httpx.get(
        f"{_GRAPH}/oauth/access_token",
        params={
            "client_id": settings.META_APP_ID,
            "client_secret": settings.META_APP_SECRET,
            "redirect_uri": f"{settings.META_REDIRECT_BASE_URL}/messenger/callback",
            "code": code,
        },
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def get_user_pages(user_token: str) -> list[dict]:
    """List Pages the user manages, each with its own page token."""
    r = httpx.get(
        f"{_GRAPH}/me/accounts",
        params={"access_token": user_token, "fields": "id,name,access_token"},
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json().get("data", [])