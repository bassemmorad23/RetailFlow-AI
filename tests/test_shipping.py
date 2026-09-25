"""Shipping configuration: validation, quoting for all methods, merchant API."""

import uuid

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api import app
from app.commerce.shipping import ShippingSettings, quote_shipping
from app.stores import routes as store_routes

RULES = {"rules": [{"country": "EG", "region": "EG-C", "fee": 50},
                   {"country": "EG", "fee": 80},
                   {"country": "SA", "region": "SA-01", "fee": 30}],
         "free_over": 2000}


def _s(**kw) -> ShippingSettings:
    return ShippingSettings(**kw)


# ---------------------------------------------------------------- validation

@pytest.mark.parametrize("cfg", [
    {"method": "platform"},                                            # no fallback
    {"method": "rules"},                                               # no rules
    {"method": "fixed"},                                               # no fee
    {"method": "fixed", "fixed": {"fee": 10}, "fallback": "manual"},   # fallback only for platform
    {"method": "platform", "fallback": "rules"},                       # fallback config missing
    {"method": "rules", "rules": {"rules": [{"country": "XX", "fee": 5}]}},
    {"method": "rules", "rules": {"rules": [{"country": "EG", "region": "SA-01", "fee": 5}]}},
    {"method": "rules", "rules": {"rules": [{"country": "EG", "fee": 5}, {"country": "eg", "fee": 6}]}},
    {"method": "rules", "rules": {"rules": []}},                       # nothing to charge
    {"method": "fixed", "fixed": {"fee": -1}},
])
def test_invalid_configs_rejected(cfg):
    with pytest.raises(ValidationError):
        _s(**cfg)


def test_codes_normalized():
    s = _s(method="rules", rules={"rules": [{"country": "eg", "region": "eg-gz", "fee": 60}]})
    assert (s.rules.rules[0].country, s.rules.rules[0].region) == ("EG", "EG-GZ")


# ---------------------------------------------------------------- quoting

def _q(settings, country="EG", region=None, subtotal=500, platform=None):
    return quote_shipping(settings, country=country, region=region, subtotal=subtotal, platform_quote=platform)


def test_rules_region_then_country_then_default():
    s = _s(method="rules", rules={**RULES, "default_fee": 120})
    assert _q(s, region="EG-C").fee == 50
    assert _q(s, region="EG-GZ").fee == 80          # country-wide rule
    assert _q(s, country="AE").fee == 120           # default fee
    assert _q(s, region="EG-C", subtotal=2500).fee == 0.0  # free over


def test_rules_not_covered_goes_to_merchant():
    q = _q(_s(method="rules", rules=RULES), country="AE")
    assert (q.fee, q.status, q.reason) == (None, "pending_merchant", "region_not_covered")


def test_fixed_and_manual():
    assert _q(_s(method="fixed", fixed={"fee": 45, "free_over": 1000})).fee == 45
    assert _q(_s(method="fixed", fixed={"fee": 45, "free_over": 1000}), subtotal=1000).fee == 0.0
    q = _q(_s(method="manual"))
    assert (q.fee, q.status) == (None, "pending_merchant")


def test_platform_rate_used_when_available():
    q = _q(_s(method="platform", fallback="manual"), platform=lambda: 37.5)
    assert (q.fee, q.method_used) == (37.5, "platform")


@pytest.mark.parametrize("platform", [None, lambda: None, lambda: 1 / 0])
def test_platform_unavailable_uses_configured_fallback(platform):
    q = _q(_s(method="platform", fallback="fixed", fixed={"fee": 60}), platform=platform)
    assert (q.fee, q.method_used, q.status) == (60, "fixed", "quoted")


def test_not_configured():
    q = _q(None)
    assert (q.status, q.fee) == ("not_configured", None)


# ---------------------------------------------------------------- merchant API

def _merchant():
    c = TestClient(app)
    c.post("/auth/register", json={"email": f"sh_{uuid.uuid4().hex[:8]}@example.com", "password": "Strong-pass-123"})
    sid = c.post("/stores", json={"display_name": "S", "industry": "fashion", "country": "EG"}).json()["store_id"]
    return c, sid


def test_owner_sets_and_reads_shipping():
    c, sid = _merchant()
    assert c.get(f"/stores/{sid}/shipping").json() == {"configured": False, "settings": None}
    body = {"method": "rules", "rules": RULES}
    assert c.put(f"/stores/{sid}/shipping", json=body).status_code == 200
    got = c.get(f"/stores/{sid}/shipping").json()
    assert got["configured"] and got["settings"]["rules"]["rules"][0]["region"] == "EG-C"


def test_shipping_api_validation_and_isolation(monkeypatch):
    c, sid = _merchant()
    other, _ = _merchant()
    assert c.put(f"/stores/{sid}/shipping", json={"method": "platform"}).status_code == 422
    monkeypatch.setattr(store_routes, "get_credentials", lambda s, src: None)
    r = c.put(f"/stores/{sid}/shipping", json={"method": "platform", "fallback": "manual"})
    assert r.status_code == 422 and "Shopify" in r.json()["detail"]
    assert other.get(f"/stores/{sid}/shipping").status_code == 404
    assert other.put(f"/stores/{sid}/shipping", json={"method": "manual"}).status_code == 404