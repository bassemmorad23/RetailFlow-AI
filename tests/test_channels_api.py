"""Channels API: status without secrets, live health checks (faked HTTP), disconnect, isolation."""

import uuid

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.channels import routes as ch
from app.settings.store_credentials import _get_collection

SHOPIFY = {"shop": "t.myshopify.com", "access_token": "shpat_SECRET"}
INSTAGRAM = {"access_token": "IGSECRET", "instagram_business_account_id": "1", "username": "shop_eg"}
WOO = {"site_url": "https://w.example", "auth_method": "basic", "username": "u", "password": "WOOSECRET"}


def _merchant():
    c = TestClient(app)
    c.post("/auth/register", json={"email": f"ch_{uuid.uuid4().hex[:8]}@example.com", "password": "Strong-pass-123"})
    sid = c.post("/stores", json={"display_name": "S", "industry": "fashion", "country": "EG"}).json()["store_id"]
    return c, sid


def _connect(sid, source, creds):
    _get_collection().insert_one({"store_id": sid, "source": source, "credentials": creds})


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_list_shows_status_and_labels_never_secrets():
    c, sid = _merchant()
    _connect(sid, "shopify", SHOPIFY)
    _connect(sid, "instagram", INSTAGRAM)
    r = c.get(f"/stores/{sid}/channels")
    by = {x["name"]: x for x in r.json()["channels"]}
    assert (by["shopify"]["connected"], by["shopify"]["label"]) == (True, "t.myshopify.com")
    assert (by["instagram"]["label"], by["woocommerce"]["connected"], by["messenger"]["connected"]) == ("@shop_eg", False, False)
    assert by["web"]["connected"] is True and "StoreFlow team" in by["whatsapp"]["note"]
    assert "SECRET" not in r.text


@pytest.mark.parametrize("response,ok,problem", [
    (httpx.Response(200, json={"access_scopes": [{"handle": s} for s in ch.REQUIRED_SHOPIFY_SCOPES]}), True, None),
    (httpx.Response(200, json={"access_scopes": [{"handle": "read_products"}]}), False, "missing_permissions"),
    (httpx.Response(401), False, "invalid_credentials"),
    (httpx.Response(503), False, "unreachable"),
])
def test_shopify_health(response, ok, problem):
    _, sid = _merchant()
    _connect(sid, "shopify", SHOPIFY)
    result = ch.run_check(sid, "shopify", client=_client(lambda r: response))
    assert (result["ok"], result["problem"]) == (ok, problem)


def test_missing_scopes_are_named():
    _, sid = _merchant()
    _connect(sid, "shopify", SHOPIFY)
    result = ch.run_check(sid, "shopify", client=_client(
        lambda r: httpx.Response(200, json={"access_scopes": [{"handle": "read_products"}]})))
    assert "write_orders" in result["detail"] and "read_inventory" in result["detail"]


def test_woocommerce_and_instagram_health():
    _, sid = _merchant()
    _connect(sid, "woocommerce", WOO)
    _connect(sid, "instagram", INSTAGRAM)
    assert ch.run_check(sid, "woocommerce", client=_client(lambda r: httpx.Response(200, json=[])))["ok"] is True
    assert ch.run_check(sid, "woocommerce", client=_client(lambda r: httpx.Response(401)))["problem"] == "invalid_credentials"
    assert ch.run_check(sid, "instagram", client=_client(lambda r: httpx.Response(400)))["problem"] == "invalid_credentials"


def test_network_error_and_special_channels():
    _, sid = _merchant()
    _connect(sid, "instagram", INSTAGRAM)

    def boom(request):
        raise httpx.ConnectError("down")
    assert ch.run_check(sid, "instagram", client=_client(boom))["problem"] == "unreachable"
    assert ch.run_check(sid, "shopify")["problem"] == "not_connected"
    assert ch.run_check(sid, "whatsapp")["problem"] == "not_supported"
    assert ch.run_check(sid, "web")["ok"] is True


def test_disconnect(monkeypatch):
    revoked = []
    monkeypatch.setattr(ch, "_revoke_shopify", lambda creds: revoked.append(creds["shop"]))
    c, sid = _merchant()
    _connect(sid, "shopify", SHOPIFY)
    _connect(sid, "instagram", INSTAGRAM)

    assert c.delete(f"/stores/{sid}/channels/instagram").status_code == 204
    assert c.delete(f"/stores/{sid}/channels/shopify").status_code == 204 and revoked == ["t.myshopify.com"]
    assert not any(x["connected"] for x in c.get(f"/stores/{sid}/channels").json()["channels"]
                   if x["name"] in ("shopify", "instagram"))
    assert c.delete(f"/stores/{sid}/channels/instagram").status_code == 404
    assert c.delete(f"/stores/{sid}/channels/web").status_code == 409


def test_isolation():
    c, sid = _merchant()
    other, other_sid = _merchant()
    _connect(sid, "instagram", INSTAGRAM)
    assert other.get(f"/stores/{sid}/channels").status_code == 404
    assert other.post(f"/stores/{sid}/channels/instagram/check").status_code == 404
    assert other.delete(f"/stores/{sid}/channels/instagram").status_code == 404
    assert other.delete(f"/stores/{other_sid}/channels/instagram").status_code == 404  # not theirs to remove
    assert c.get(f"/stores/{sid}/channels").json()["channels"][2]["connected"] is True