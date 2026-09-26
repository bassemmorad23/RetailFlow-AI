"""
Channels & sources for the merchant dashboard (owner only). Never returns tokens.

GET    /stores/{id}/channels               connected yes/no + a safe label per channel
POST   /stores/{id}/channels/{name}/check  live health check
DELETE /stores/{id}/channels/{name}        disconnect (removes our credentials;
                                           Shopify: also revokes app access, best-effort)
"""

import logging
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth.dependencies import require_store_member
from app.ingestion.shopify_adapter import _SHOPIFY_API_VERSION
from app.settings.store_credentials import _get_collection, get_credentials

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/stores/{store_id}/channels", tags=["channels"])

ChannelName = Literal["shopify", "woocommerce", "instagram", "messenger", "whatsapp", "web"]
_CONNECTABLE = ("shopify", "woocommerce", "instagram", "messenger")
REQUIRED_SHOPIFY_SCOPES = {"read_products", "read_inventory", "read_orders", "write_orders", "write_draft_orders"}
CHECK_TIMEOUT = 10.0


class ChannelInfo(BaseModel):
    name: ChannelName
    kind: Literal["catalog", "messaging"]
    connected: bool
    label: str | None = None
    note: str | None = None


class ChannelList(BaseModel):
    channels: list[ChannelInfo]


class HealthResult(BaseModel):
    name: ChannelName
    ok: bool
    problem: str | None = None   # not_connected | invalid_credentials | missing_permissions | unreachable | not_supported
    detail: str | None = None


def _label(name: str, creds: dict) -> str | None:
    return {"shopify": creds.get("shop"), "woocommerce": creds.get("site_url"),
            "instagram": ("@" + creds["username"]) if creds.get("username") else None,
            "messenger": creds.get("page_name")}.get(name)


@router.get("", response_model=ChannelList)
def list_channels(store_id: str = Depends(require_store_member)) -> dict:
    out = []
    for name in _CONNECTABLE:
        creds = get_credentials(store_id, name)
        out.append({"name": name, "kind": "catalog" if name in ("shopify", "woocommerce") else "messaging",
                    "connected": creds is not None, "label": _label(name, creds) if creds else None})
    out.append({"name": "whatsapp", "kind": "messaging", "connected": False,
                "note": "WhatsApp is set up by the StoreFlow team during onboarding."})
    out.append({"name": "web", "kind": "messaging", "connected": True, "label": "Website chat widget"})
    return {"channels": out}


# ---------------------------------------------------------------- live health checks

def _check_shopify(creds: dict, http: httpx.Client) -> tuple[bool, str | None, str | None]:
    r = http.get(f"https://{creds['shop']}/admin/oauth/access_scopes.json",
                 headers={"X-Shopify-Access-Token": creds["access_token"]})
    if r.status_code in (401, 403):
        return False, "invalid_credentials", "Shopify rejected the connection. Reconnect the store."
    if r.status_code >= 400:
        return False, "unreachable", f"Shopify returned {r.status_code}."
    granted = {s.get("handle") for s in r.json().get("access_scopes", [])}
    missing = sorted(REQUIRED_SHOPIFY_SCOPES - granted)
    if missing:
        return False, "missing_permissions", "Reconnect to grant: " + ", ".join(missing)
    return True, None, None


def _check_woocommerce(creds: dict, http: httpx.Client) -> tuple[bool, str | None, str | None]:
    r = http.get(f"{creds['site_url'].rstrip('/')}/wp-json/wc/v3/products", params={"per_page": 1},
                 auth=(creds["username"], creds["password"]))
    if r.status_code in (401, 403):
        return False, "invalid_credentials", "WooCommerce rejected the API credentials."
    if r.status_code >= 400:
        return False, "unreachable", f"WooCommerce returned {r.status_code}."
    return True, None, None


def _check_instagram(creds: dict, http: httpx.Client) -> tuple[bool, str | None, str | None]:
    r = http.get("https://graph.instagram.com/me", params={"fields": "username"},
                 headers={"Authorization": f"Bearer {creds['access_token']}"})
    if r.status_code in (400, 401, 403):
        return False, "invalid_credentials", "Instagram connection expired or was revoked. Reconnect Instagram."
    if r.status_code >= 400:
        return False, "unreachable", f"Instagram returned {r.status_code}."
    return True, None, None


_CHECKS = {"shopify": _check_shopify, "woocommerce": _check_woocommerce, "instagram": _check_instagram}


def run_check(store_id: str, name: str, client: httpx.Client | None = None) -> dict:
    if name == "web":
        return {"name": name, "ok": True}
    if name not in _CHECKS:
        return {"name": name, "ok": False, "problem": "not_supported",
                "detail": "A live check isn't available for this channel yet."}
    creds = get_credentials(store_id, name)
    if creds is None:
        return {"name": name, "ok": False, "problem": "not_connected"}
    http = client or httpx.Client(timeout=CHECK_TIMEOUT)
    try:
        ok, problem, detail = _CHECKS[name](creds, http)
    except httpx.HTTPError:
        ok, problem, detail = False, "unreachable", "Couldn't reach the service right now."
    finally:
        if client is None:
            http.close()
    return {"name": name, "ok": ok, "problem": problem, "detail": detail}


@router.post("/{name}/check", response_model=HealthResult)
def check_channel(name: ChannelName, store_id: str = Depends(require_store_member)) -> dict:
    return run_check(store_id, name)


# ---------------------------------------------------------------- disconnect

def _revoke_shopify(creds: dict) -> None:
    try:
        with httpx.Client(timeout=CHECK_TIMEOUT) as http:
            http.delete(f"https://{creds['shop']}/admin/api/{_SHOPIFY_API_VERSION}/api_permissions/current.json",
                        headers={"X-Shopify-Access-Token": creds["access_token"]})
    except httpx.HTTPError:
        logger.warning("Shopify revoke failed (credentials removed anyway)")


@router.delete("/{name}", status_code=204)
def disconnect_channel(name: ChannelName, store_id: str = Depends(require_store_member)) -> None:
    if name not in _CONNECTABLE:
        raise HTTPException(status_code=409, detail={"code": "not_disconnectable",
                                                     "message": "This channel can't be disconnected here."})
    creds = get_credentials(store_id, name)
    if creds is None:
        raise HTTPException(status_code=404, detail="Channel not connected")
    if name == "shopify":
        _revoke_shopify(creds)
    _get_collection().delete_one({"store_id": store_id, "source": name})