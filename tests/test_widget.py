"""
Web widget tests: server-issued identity, send/poll, isolation, merchant
replies reaching the visitor. Real API + isolated test DB; AI is faked.
"""

import hashlib
import uuid

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.core import orchestrator
from app.inbox import service
from app.rate_limiter import rate_limit
from app.schemas.models import AgentReply
from app.widget import routes as widget_routes
from app.widget import sessions

PW = "Strong-pass-123"


@pytest.fixture(autouse=True)
def env(monkeypatch):
    app.dependency_overrides[rate_limit] = lambda: None
    calls = {"ai": 0, "reply": "Hello from AI"}

    def fake_ai(msg, context=None):
        calls["ai"] += 1
        return AgentReply(conversation_id="c", reply_text=calls["reply"],
                          emotion=orchestrator._FALLBACK_EMOTION, intent=orchestrator._FALLBACK_INTENT,
                          recommendations=[], retrieved_context=[])

    monkeypatch.setattr(service, "handle_message", fake_ai)
    monkeypatch.setattr(service, "maybe_refresh_summary", lambda *a, **k: False)
    monkeypatch.setattr(widget_routes, "maybe_refresh_summary", lambda *a, **k: False)
    yield calls
    app.dependency_overrides.pop(rate_limit, None)


def _merchant() -> tuple[TestClient, str]:
    client = TestClient(app)
    client.post("/auth/register", json={"email": f"w_{uuid.uuid4().hex[:8]}@example.com", "password": PW})
    sid = client.post("/stores", json={"display_name": "Shop", "industry": "fashion", "country": "EG"}).json()["store_id"]
    return client, sid


def _token(sid: str) -> str:
    r = TestClient(app).post("/widget/sessions", json={"store_id": sid})
    assert r.status_code == 201
    return r.json()["token"]


def _hdr(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _send(token: str, text: str = "hi", click: str | None = None):
    return TestClient(app).post("/widget/messages", headers=_hdr(token),
                                json={"text": text, "client_message_id": click or uuid.uuid4().hex})


def _poll(token: str, after: int | None = None) -> dict:
    params = {} if after is None else {"after_seq": after}
    return TestClient(app).get("/widget/messages", headers=_hdr(token), params=params).json()


# ---------------------------------------------------------------- sessions

def test_session_unknown_store_404():
    assert TestClient(app).post("/widget/sessions", json={"store_id": "store_nope"}).status_code == 404


def test_token_required_and_validated():
    _, sid = _merchant()
    c = TestClient(app)
    body = {"text": "hi", "client_message_id": uuid.uuid4().hex}
    assert c.post("/widget/messages", json=body).status_code == 401
    assert c.post("/widget/messages", json=body, headers=_hdr("forged")).status_code == 401
    assert c.get("/widget/messages", headers=_hdr("forged")).status_code == 401


def test_token_stored_only_as_hash():
    _, sid = _merchant()
    token = _token(sid)
    doc = sessions._col().find_one({"token_hash": hashlib.sha256(token.encode()).hexdigest()})
    assert doc is not None and token not in str(doc)


# ---------------------------------------------------------------- send / poll

def test_send_returns_ai_reply(env):
    _, sid = _merchant()
    r = _send(_token(sid)).json()
    assert r["reply"]["author"] == "ai" and r["reply"]["text"] == "Hello from AI"
    assert r["ai_paused"] is False


def test_double_submit_one_ai_call(env):
    _, sid = _merchant()
    token = _token(sid)
    click = uuid.uuid4().hex
    _send(token, click=click)
    second = _send(token, click=click).json()
    assert second["duplicate"] is True and env["ai"] == 1


def test_poll_returns_only_new_messages():
    _, sid = _merchant()
    token = _token(sid)
    _send(token, "first")
    first = _poll(token)
    assert [m["author"] for m in first["messages"]] == ["customer", "ai"]
    _send(token, "second")
    newer = _poll(token, after=first["last_seq"])
    assert [m["text"] for m in newer["messages"]] == ["second", "Hello from AI"]


def test_text_limit():
    _, sid = _merchant()
    assert _send(_token(sid), text="x" * (widget_routes.MAX_WIDGET_MESSAGE_CHARS + 1)).status_code == 422


# ---------------------------------------------------------------- isolation

def test_visitors_are_isolated():
    owner, sid = _merchant()
    a, b = _token(sid), _token(sid)
    _send(a, "my secret order")
    assert _poll(b)["messages"] == []
    convs = owner.get(f"/stores/{sid}/conversations").json()["conversations"]
    assert len(convs) == 1 and convs[0]["channel"] == "web"


# ---------------------------------------------------------------- merchant takeover

def _web_conversation(owner: TestClient, sid: str) -> str:
    return owner.get(f"/stores/{sid}/conversations").json()["conversations"][0]["id"]


def test_merchant_reply_reaches_widget_and_notes_hidden():
    owner, sid = _merchant()
    token = _token(sid)
    _send(token)
    cid = _web_conversation(owner, sid)
    r = owner.post(f"/stores/{sid}/conversations/{cid}/messages",
                   json={"text": "Hi, I'm Sara from the store", "client_message_id": uuid.uuid4().hex})
    assert r.status_code == 201

    msgs = _poll(token)["messages"]
    assert [(m["author"], m["text"]) for m in msgs][-1] == ("merchant", "Hi, I'm Sara from the store")
    assert all(m["author"] != "system" for m in msgs)  # "AI paused" note stays internal


def test_paused_ai_returns_no_reply(env):
    owner, sid = _merchant()
    token = _token(sid)
    _send(token)
    cid = _web_conversation(owner, sid)
    owner.post(f"/stores/{sid}/conversations/{cid}/ai/pause")

    r = _send(token, "anyone?").json()
    assert r["reply"] is None and r["ai_paused"] is True
    assert env["ai"] == 1


# ---------------------------------------------------------------- legacy

def test_insecure_chat_endpoint_removed():
    assert TestClient(app).post("/chat", json={}).status_code in (404, 405)