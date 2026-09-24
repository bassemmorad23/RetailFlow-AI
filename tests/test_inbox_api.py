"""
Unified Inbox API + web /chat tests (real API, isolated test DB).
Conversations are created directly in storage; the AI is never called.
"""

import time
import uuid

import pytest
from fastapi.testclient import TestClient

from app import api
from app.api import app
from app.core import orchestrator
from app.inbox import repository as repo
from app.inbox.service import IncomingResult
from app.schemas.models import AgentReply

PW = "Strong-pass-123"


def _merchant() -> tuple[TestClient, str]:
    client = TestClient(app)
    r = client.post("/auth/register", json={"email": f"i_{uuid.uuid4().hex[:8]}@example.com", "password": PW})
    assert r.status_code == 201
    r = client.post("/stores", json={"display_name": "Shop", "industry": "fashion", "country": "EG"})
    assert r.status_code == 201
    return client, r.json()["store_id"]


def _conversation(store_id: str, channel: str, customer: str, texts: list[str]) -> str:
    cid = repo.get_or_create_conversation(store_id, channel, customer)["id"]
    for t in texts:
        repo.add_message(store_id, cid, "customer", t, delivery_status="received")
        time.sleep(0.01)  # distinct updated_at for stable ordering
    return cid


# ---------------------------------------------------------------- list

def test_unified_list_all_channels_newest_first_and_filter():
    client, sid = _merchant()
    ig = _conversation(sid, "instagram", "u1", ["hi"])
    wa = _conversation(sid, "whatsapp", "2010", ["salam"])
    web = _conversation(sid, "web", "w1", ["hello"])

    body = client.get(f"/stores/{sid}/conversations").json()
    assert [c["id"] for c in body["conversations"]] == [web, wa, ig]
    assert {c["channel"] for c in body["conversations"]} == {"instagram", "whatsapp", "web"}

    only_wa = client.get(f"/stores/{sid}/conversations", params={"channel": "whatsapp"}).json()
    assert [c["id"] for c in only_wa["conversations"]] == [wa]


def test_list_validation():
    client, sid = _merchant()
    assert client.get(f"/stores/{sid}/conversations", params={"channel": "telegram"}).status_code == 422
    assert client.get(f"/stores/{sid}/conversations", params={"cursor": "garbage"}).status_code == 400


def test_pagination_no_overlap():
    client, sid = _merchant()
    ids = [_conversation(sid, "instagram", f"p{i}", ["x"]) for i in range(3)]
    p1 = client.get(f"/stores/{sid}/conversations", params={"limit": 2}).json()
    p2 = client.get(f"/stores/{sid}/conversations",
                    params={"limit": 2, "cursor": p1["next_cursor"]}).json()
    got = [c["id"] for c in p1["conversations"] + p2["conversations"]]
    assert sorted(got) == sorted(ids) and len(set(got)) == 3
    assert p2["next_cursor"] is None


def test_internal_fields_not_exposed():
    client, sid = _merchant()
    _conversation(sid, "whatsapp", "2011", ["hi"])
    conv = client.get(f"/stores/{sid}/conversations").json()["conversations"][0]
    assert "thread_key" not in conv and "_id" not in conv


# ---------------------------------------------------------------- tenant isolation

def test_cross_store_access_always_404():
    owner, owner_sid = _merchant()
    other, other_sid = _merchant()
    cid = _conversation(owner_sid, "instagram", "victim", ["my address is ..."])

    assert other.get(f"/stores/{owner_sid}/conversations").status_code == 404
    # Tampering: real conversation id, attacker's own store in the path
    assert other.get(f"/stores/{other_sid}/conversations/{cid}").status_code == 404
    assert other.get(f"/stores/{other_sid}/conversations/{cid}/messages").status_code == 404
    assert other.get(f"/stores/{other_sid}/conversations").json()["conversations"] == []
    assert TestClient(app).get(f"/stores/{owner_sid}/conversations").status_code == 401
    assert owner.get(f"/stores/{owner_sid}/conversations/{cid}").status_code == 200


# ---------------------------------------------------------------- transcript

def test_transcript_pagination():
    client, sid = _merchant()
    cid = _conversation(sid, "messenger", "m1", [f"msg {i}" for i in range(5)])

    latest = client.get(f"/stores/{sid}/conversations/{cid}/messages", params={"limit": 2}).json()
    assert [m["text"] for m in latest["messages"]] == ["msg 3", "msg 4"] and latest["has_more"]

    older = client.get(f"/stores/{sid}/conversations/{cid}/messages",
                       params={"limit": 10, "before_seq": latest["messages"][0]["seq"]}).json()
    assert [m["text"] for m in older["messages"]] == ["msg 0", "msg 1", "msg 2"]
    assert not older["has_more"]


