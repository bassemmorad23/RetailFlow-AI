"""Auth + tenant isolation tests (real API, isolated test DB)."""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.auth.login_limiter import MAX_FAILURES_PER_EMAIL, _col as login_attempts
from app.config import settings

PW = "Strong-pass-123"
COOKIE = settings.SESSION_COOKIE_NAME


@pytest.fixture(autouse=True)
def _reset_login_attempts():
    login_attempts().delete_many({})


def _email() -> str:
    return f"t_{uuid.uuid4().hex[:8]}@example.com"


def _registered(email: str | None = None) -> tuple[TestClient, str]:
    client, email = TestClient(app), email or _email()
    r = client.post("/auth/register", json={"email": email, "password": PW})
    assert r.status_code == 201
    return client, email


def _store(client: TestClient) -> str:
    r = client.post("/stores", json={"display_name": "Shop", "industry": "fashion", "country": "EG"})
    assert r.status_code == 201
    return r.json()["store_id"]


# ---------------------------------------------------------------- register

def test_register_sets_hardened_cookie():
    r = TestClient(app).post("/auth/register", json={"email": _email(), "password": PW})
    cookie = r.headers["set-cookie"].lower()
    assert r.status_code == 201
    assert "httponly" in cookie and "samesite=lax" in cookie


def test_password_never_returned():
    r = TestClient(app).post("/auth/register", json={"email": _email(), "password": PW})
    assert "password" not in r.text.lower()


def test_duplicate_email_is_case_insensitive():
    _, email = _registered()
    r = TestClient(app).post("/auth/register", json={"email": email.upper(), "password": PW})
    assert r.status_code == 409


@pytest.mark.parametrize("payload", [
    {"email": "not-an-email", "password": PW},
    {"email": "ok@example.com", "password": "short"},
    {"email": "ok@example.com", "password": PW, "role": "admin"},  # extra field rejected
])
def test_register_validation(payload):
    assert TestClient(app).post("/auth/register", json=payload).status_code == 422


# ---------------------------------------------------------------- sessions

def test_me_logout_flow():
    client, _ = _registered()
    assert client.get("/auth/me").status_code == 200
    assert client.post("/auth/logout").status_code == 204
    assert client.get("/auth/me").status_code == 401


def test_old_cookie_rejected_after_logout():
    client, _ = _registered()
    old = client.cookies.get(COOKIE)
    client.post("/auth/logout")
    r = TestClient(app).get("/auth/me", headers={"Cookie": f"{COOKIE}={old}"})
    assert r.status_code == 401  # revoked server-side, not just cleared in the browser


def test_forged_cookie_rejected():
    r = TestClient(app).get("/auth/me", headers={"Cookie": f"{COOKIE}=forged-token"})
    assert r.status_code == 401


# ---------------------------------------------------------------- login

def test_wrong_password_and_unknown_email_look_identical():
    _, email = _registered()
    wrong_pw = TestClient(app).post("/auth/login", json={"email": email, "password": "wrong-pass-1"})
    no_user = TestClient(app).post("/auth/login", json={"email": _email(), "password": "wrong-pass-1"})
    assert wrong_pw.status_code == no_user.status_code == 401
    assert wrong_pw.json() == no_user.json()


def test_lockout_after_max_failures_blocks_even_correct_password():
    _, email = _registered()
    c = TestClient(app)
    for _ in range(MAX_FAILURES_PER_EMAIL):
        assert c.post("/auth/login", json={"email": email, "password": "wrong-pass-1"}).status_code == 401
    r = c.post("/auth/login", json={"email": email, "password": PW})
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) > 0


def test_successful_login_resets_counter():
    _, email = _registered()
    c = TestClient(app)
    for _ in range(MAX_FAILURES_PER_EMAIL - 1):
        c.post("/auth/login", json={"email": email, "password": "wrong-pass-1"})
    assert c.post("/auth/login", json={"email": email, "password": PW}).status_code == 200
    for _ in range(MAX_FAILURES_PER_EMAIL - 1):
        assert c.post("/auth/login", json={"email": email, "password": "wrong-pass-1"}).status_code == 401


# ---------------------------------------------------------------- tenant isolation

def test_store_owner_only():
    owner, _ = _registered()
    other, _ = _registered()
    sid = _store(owner)
    assert owner.get(f"/stores/{sid}").status_code == 200
    assert other.get(f"/stores/{sid}").status_code == 404
    assert TestClient(app).get(f"/stores/{sid}").status_code == 401


def test_other_merchant_cannot_upload_or_sync():
    owner, _ = _registered()
    other, _ = _registered()
    sid = _store(owner)
    csv = b"Product ID,Product Name,Price\np-1,Shirt,100\n"
    up = other.post(f"/stores/{sid}/products/upload", files={"file": ("p.csv", csv, "text/csv")})
    assert up.status_code == 404
    assert other.post(f"/stores/{sid}/sync/shopify").status_code == 404
    assert other.post(f"/stores/{sid}/sync/woocommerce").status_code == 404


def test_oauth_install_and_callback_owner_only():
    owner, _ = _registered()
    other, _ = _registered()
    sid = _store(owner)
    assert other.get("/instagram/install", params={"store_id": sid}, follow_redirects=False).status_code == 404
    assert owner.get("/instagram/install", params={"store_id": sid}, follow_redirects=False).status_code == 307
    assert other.get("/instagram/callback", params={"code": "x", "state": sid}).status_code == 404
    assert other.get("/messenger/callback", params={"code": "x", "state": sid}).status_code == 404


def test_store_creation_validation():
    client, _ = _registered()
    bad = {"display_name": "X", "industry": "fashion", "country": "FR"}
    assert client.post("/stores", json=bad).status_code == 422