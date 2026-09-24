"""CORS + CSRF middleware tests (real API, isolated test DB)."""

import hashlib
import hmac
import json
import uuid

import pytest
from fastapi.testclient import TestClient

from app import api
from app.api import app
from app.config import settings

DASH = "http://localhost:3000"
EVIL = "https://evil.example"
PW = "Strong-pass-123"


@pytest.fixture(autouse=True)
def origins(monkeypatch):
    monkeypatch.setattr(settings, "DASHBOARD_ORIGINS", DASH)


def _preflight(path: str, origin: str):
    return TestClient(app).options(path, headers={
        "Origin": origin, "Access-Control-Request-Method": "POST"})


def _merchant() -> TestClient:
    c = TestClient(app)
    assert c.post("/auth/register", json={"email": f"c_{uuid.uuid4().hex[:8]}@example.com",
                                          "password": PW}).status_code == 201
    return c


# ---------------------------------------------------------------- CORS

def test_dashboard_preflight_allowed_with_credentials():
    r = _preflight("/stores", DASH)
    assert r.status_code == 204
    assert r.headers["access-control-allow-origin"] == DASH
    assert r.headers["access-control-allow-credentials"] == "true"


@pytest.mark.parametrize("origin", [EVIL, "null"])
def test_dashboard_preflight_unknown_origin_rejected(origin):
    assert _preflight("/stores", origin).status_code == 403


def test_widget_preflight_any_origin_without_credentials():
    r = _preflight("/widget/messages", EVIL)
    assert r.status_code == 204
    assert r.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in r.headers
    assert "Authorization" in r.headers["access-control-allow-headers"]


def test_simple_response_headers_only_for_allowed_origin():
    c = TestClient(app)
    assert c.get("/health", headers={"Origin": DASH}).headers["access-control-allow-origin"] == DASH
    assert "access-control-allow-origin" not in c.get("/health", headers={"Origin": EVIL}).headers


# ---------------------------------------------------------------- CSRF

def test_cross_site_state_change_blocked():
    c = _merchant()
    body = {"display_name": "X", "industry": "fashion", "country": "EG"}
    assert c.post("/stores", json=body, headers={"Origin": EVIL}).status_code == 403
    assert c.get("/auth/me").json()["stores"] == []  # nothing was created


def test_same_site_and_originless_requests_allowed():
    c = _merchant()
    body = {"display_name": "X", "industry": "fashion", "country": "EG"}
    assert c.post("/stores", json=body, headers={"Origin": DASH}).status_code == 201
    assert c.post("/stores", json=body).status_code == 201  # scripts / server-to-server


def test_login_csrf_blocked():
    r = TestClient(app).post("/auth/login", headers={"Origin": EVIL},
                             json={"email": "x@example.com", "password": "whatever-123"})
    assert r.status_code == 403


def test_widget_callable_from_any_site():
    _merchant()
    r = TestClient(app).post("/widget/sessions", headers={"Origin": EVIL}, json={"store_id": "store_nope"})
    assert r.status_code == 404  # reached the endpoint (store check), not blocked by CSRF


def test_webhooks_not_subject_to_origin_check(monkeypatch):
    monkeypatch.setattr(api, "webhook_secrets", lambda ch: ["s"])
    monkeypatch.setattr(api, "_process_entries", lambda *a: None)
    body = json.dumps({"entry": []}).encode()
    sig = "sha256=" + hmac.new(b"s", body, hashlib.sha256).hexdigest()
    r = TestClient(app).post("/instagram/webhook", content=body, headers={
        "Origin": EVIL, "Content-Type": "application/json", "X-Hub-Signature-256": sig})
    assert r.status_code == 200