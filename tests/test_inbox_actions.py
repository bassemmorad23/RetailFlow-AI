"""
Inbox Phase 2: pause/resume, manual replies, reply windows, mark read.
Real API + isolated test DB. Channel sending and summary LLM are faked.
"""

import itertools
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.channels.base import SendResult
from app.inbox import actions, routes
from app.inbox import repository as repo
from app.inbox.summary import build_context

PW = "Strong-pass-123"


@pytest.fixture
def sent(monkeypatch):
    rec = {"calls": [], "result": None}
    ids = itertools.count(1)

    def fake_send(channel, store_id, recipient, text):
        rec["calls"].append((channel, recipient, text))
        return rec["result"] or SendResult(True, [f"out_{next(ids)}_{uuid.uuid4().hex[:6]}"])

    monkeypatch.setattr(actions, "send", fake_send)
    monkeypatch.setattr(routes, "maybe_refresh_summary", lambda *a, **k: False)
    return rec


def _merchant() -> tuple[TestClient, str]:
    client = TestClient(app)
    client.post("/auth/register", json={"email": f"a_{uuid.uuid4().hex[:8]}@example.com", "password": PW})
    sid = client.post("/stores", json={"display_name": "Shop", "industry": "fashion", "country": "EG"}).json()["store_id"]
    return client, sid


def _conv(store_id: str, channel: str = "whatsapp") -> str:
    cid = repo.get_or_create_conversation(store_id, channel, uuid.uuid4().hex[:10])["id"]
    repo.add_message(store_id, cid, "customer", "hi", delivery_status="received")
    return cid


def _base(sid: str, cid: str) -> str:
    return f"/stores/{sid}/conversations/{cid}"


def _reply(client, sid, cid, text="We'll ship tomorrow", client_id=None):
    return client.post(f"{_base(sid, cid)}/messages",
                       json={"text": text, "client_message_id": client_id or uuid.uuid4().hex})


def _transcript(client, sid, cid) -> list[dict]:
    return client.get(f"{_base(sid, cid)}/messages").json()["messages"]


# ---------------------------------------------------------------- pause / resume

def test_pause_resume_with_internal_notes(sent):
    client, sid = _merchant()
    cid = _conv(sid)

    r = client.post(f"{_base(sid, cid)}/ai/pause")
    assert r.status_code == 200 and r.json()["ai_mode"] == "paused"
    client.post(f"{_base(sid, cid)}/ai/pause")  # idempotent: no second note
    assert client.post(f"{_base(sid, cid)}/ai/resume").json()["ai_mode"] == "auto"

    notes = [(m["text"], m["delivery_status"]) for m in _transcript(client, sid, cid) if m["sender_type"] == "system"]
    assert notes == [(actions.NOTE_PAUSED, "internal"), (actions.NOTE_RESUMED, "internal")]
    assert sent["calls"] == []  # notes are never sent to the customer


def test_internal_notes_hidden_from_ai_context(sent):
    client, sid = _merchant()
    cid = _conv(sid)
    client.post(f"{_base(sid, cid)}/ai/pause")
    ctx = build_context(sid, cid)
    assert all(actions.NOTE_PAUSED not in t.text for t in ctx.turns)


# ---------------------------------------------------------------- manual reply

def test_manual_reply_sends_and_auto_pauses(sent):
    client, sid = _merchant()
    cid = _conv(sid, "instagram")

    r = _reply(client, sid, cid)
    assert r.status_code == 201
    body = r.json()
    assert (body["sender_type"], body["delivery_status"]) == ("human", "sent")
    assert sent["calls"][0][0] == "instagram"

    conv = client.get(_base(sid, cid)).json()
    assert conv["ai_mode"] == "paused"
    texts = [m["text"] for m in _transcript(client, sid, cid)]
    assert actions.NOTE_AUTO_PAUSED in texts


def test_manual_reply_in_ai_context_after_resume(sent):
    client, sid = _merchant()
    cid = _conv(sid)
    _reply(client, sid, cid, text="Free delivery for you")
    client.post(f"{_base(sid, cid)}/ai/resume")
    turns = [t.text for t in build_context(sid, cid).turns]
    assert "[Merchant] Free delivery for you" in turns


def test_double_click_sends_once(sent):
    client, sid = _merchant()
    cid = _conv(sid)
    click = uuid.uuid4().hex
    first = _reply(client, sid, cid, client_id=click).json()
    second = _reply(client, sid, cid, client_id=click).json()
    assert first["id"] == second["id"]
    assert len(sent["calls"]) == 1


def test_delivery_failure_reported(sent):
    client, sid = _merchant()
    cid = _conv(sid, "messenger")
    sent["result"] = SendResult(False, [], "auth_expired", "190")
    body = _reply(client, sid, cid).json()
    assert (body["delivery_status"], body["delivery_error"]) == ("failed", "auth_expired")


@pytest.mark.parametrize("text", ["   ", "x" * (actions.MAX_REPLY_CHARS + 1)])
def test_reply_validation(sent, text):
    client, sid = _merchant()
    cid = _conv(sid)
    assert _reply(client, sid, cid, text=text).status_code == 422
    assert sent["calls"] == []


# ---------------------------------------------------------------- reply window

def _age_last_customer_message(sid: str, cid: str, hours: int) -> None:
    repo._db()["inbox_conversations"].update_one(
        {"store_id": sid, "id": cid},
        {"$set": {"last_customer_message_at": datetime.now(timezone.utc) - timedelta(hours=hours)}},
    )


def test_window_closed_blocks_reply(sent):
    client, sid = _merchant()
    cid = _conv(sid, "whatsapp")
    _age_last_customer_message(sid, cid, 25)

    conv = client.get(_base(sid, cid)).json()
    assert conv["reply_window_open"] is False and conv["reply_window_expires_at"] is not None

    r = _reply(client, sid, cid)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "reply_window_closed"
    assert sent["calls"] == []
    assert client.get(_base(sid, cid)).json()["ai_mode"] == "auto"  # nothing changed


def test_web_has_no_window(sent):
    client, sid = _merchant()
    cid = _conv(sid, "web")
    _age_last_customer_message(sid, cid, 500)
    conv = client.get(_base(sid, cid)).json()
    assert conv["reply_window_open"] is True and conv["reply_window_expires_at"] is None
    assert _reply(client, sid, cid).status_code == 201


# ---------------------------------------------------------------- read

def test_mark_read(sent):
    client, sid = _merchant()
    cid = _conv(sid)
    assert client.get(_base(sid, cid)).json()["unread_count"] == 1
    r = client.post(f"{_base(sid, cid)}/read")
    assert r.json()["unread_count"] == 0 and r.json()["last_read_at"] is not None


# ---------------------------------------------------------------- tenant isolation

def test_actions_cross_store_always_404(sent):
    owner, owner_sid = _merchant()
    other, other_sid = _merchant()
    cid = _conv(owner_sid)

    for path in ("ai/pause", "ai/resume", "read"):
        assert other.post(f"{_base(other_sid, cid)}/{path}").status_code == 404
        assert other.post(f"{_base(owner_sid, cid)}/{path}").status_code == 404
    assert _reply(other, other_sid, cid).status_code == 404
    assert TestClient(app).post(f"{_base(owner_sid, cid)}/ai/pause").status_code == 401

    assert sent["calls"] == []
    assert owner.get(_base(owner_sid, cid)).json()["ai_mode"] == "auto"  # untouched