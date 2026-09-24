"""Webhook signature verification: unit tests + all three webhook endpoints."""

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from app import api
from app.api import app
from app.channels.signatures import verify_meta_signature

SECRET = "test_app_secret"


def _sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


# ---------------------------------------------------------------- unit

def test_valid_signature():
    body = b'{"entry": []}'
    assert verify_meta_signature(body, _sign(body), [SECRET])


def test_second_candidate_secret_accepted():
    body = b"{}"
    assert verify_meta_signature(body, _sign(body, "other"), [SECRET, "other"])


@pytest.mark.parametrize("header", [None, "", "sha1=abc", "sha256=", "sha256=deadbeef"])
def test_missing_or_malformed_header_rejected(header):
    assert not verify_meta_signature(b"{}", header, [SECRET])


def test_tampered_body_rejected():
    assert not verify_meta_signature(b'{"entry": [1]}', _sign(b'{"entry": []}'), [SECRET])


def test_no_configured_secret_fails_closed():
    body = b"{}"
    assert not verify_meta_signature(body, _sign(body), [])
    assert not verify_meta_signature(body, _sign(body, ""), [""])


# ---------------------------------------------------------------- endpoints

@pytest.fixture
def processed(monkeypatch):
    rec = []
    monkeypatch.setattr(api, "webhook_secrets", lambda channel: [SECRET])
    monkeypatch.setattr(api, "_process_entries", lambda entries, fn, name: rec.append((name, entries)))
    return rec


_PATHS = ["/instagram/webhook", "/messenger/webhook", "/whatsapp/webhook"]


@pytest.mark.parametrize("path", _PATHS)
def test_signed_webhook_processed(processed, path):
    body = json.dumps({"entry": [{"id": "x"}]}).encode()
    r = TestClient(app).post(path, content=body, headers={
        "Content-Type": "application/json", "X-Hub-Signature-256": _sign(body)})
    assert r.status_code == 200
    assert processed and processed[0][1] == [{"id": "x"}]


@pytest.mark.parametrize("path", _PATHS)
def test_unsigned_or_forged_webhook_rejected(processed, path):
    body = json.dumps({"entry": [{"id": "x"}]}).encode()
    c = TestClient(app)
    assert c.post(path, content=body, headers={"Content-Type": "application/json"}).status_code == 403
    assert c.post(path, content=body, headers={
        "Content-Type": "application/json", "X-Hub-Signature-256": _sign(body, "attacker")}).status_code == 403
    assert processed == []


def test_oversized_body_rejected(processed):
    body = b'{"entry": [], "pad": "' + b"x" * (1024 * 1024 + 10) + b'"}'
    r = TestClient(app).post("/instagram/webhook", content=body, headers={
        "Content-Type": "application/json", "X-Hub-Signature-256": _sign(body)})
    assert r.status_code == 413 and processed == []


def test_signed_but_invalid_json_400(processed):
    body = b"not json"
    r = TestClient(app).post("/whatsapp/webhook", content=body, headers={
        "Content-Type": "application/json", "X-Hub-Signature-256": _sign(body)})
    assert r.status_code == 400 and processed == []