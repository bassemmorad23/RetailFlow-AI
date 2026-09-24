"""
Live inbox stream (SSE) tests. Real API + isolated test DB.
Timings are shortened so each stream closes by itself in < 1 second.
"""

import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.inbox import repository as repo
from app.inbox import stream

PW = "Strong-pass-123"


@pytest.fixture(autouse=True)
def fast_stream(monkeypatch):
    monkeypatch.setattr(stream, "POLL_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(stream, "MAX_CONNECTION_SECONDS", 0.6)
    monkeypatch.setattr(stream, "HEARTBEAT_SECONDS", 100)
    monkeypatch.setattr(stream, "AUTH_RECHECK_SECONDS", 100)


def _merchant() -> tuple[TestClient, str]:
    client = TestClient(app)
    client.post("/auth/register", json={"email": f"s_{uuid.uuid4().hex[:8]}@example.com", "password": PW})
    sid = client.post("/stores", json={"display_name": "Shop", "industry": "fashion", "country": "EG"}).json()["store_id"]
    return client, sid


def _seed(store_id: str, n: int) -> None:
    for i in range(n):
        repo.append_event(store_id, "message.created", "conv_test", {"message": {"text": f"m{i}"}})


def _parse(text: str) -> list[dict]:
    out = []
    for block in text.split("\n\n"):
        ev: dict = {}
        for line in block.splitlines():
            if line.startswith(":"):
                ev["comment"] = line
            elif ": " in line:
                key, value = line.split(": ", 1)
                ev[key] = value
        if ev:
            out.append(ev)
    return out


def _read(client: TestClient, sid: str, params=None, headers=None) -> list[dict]:
    with client.stream("GET", f"/stores/{sid}/inbox/events", params=params, headers=headers) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    return _parse(text)


def _events(parsed: list[dict]) -> list[dict]:
    return [e for e in parsed if "event" in e]


# ---------------------------------------------------------------- delivery

def test_streams_own_events_in_order():
    client, sid = _merchant()
    _, other_sid = _merchant()
    _seed(sid, 3)
    _seed(other_sid, 2)

    evs = _events(_read(client, sid, params={"after": 0}))
    assert [e["id"] for e in evs] == ["1", "2", "3"]
    assert all(e["event"] == "message.created" for e in evs)
    assert '"conversation_id": "conv_test"' in evs[0]["data"]


def test_last_event_id_header_resumes_and_wins_over_query():
    client, sid = _merchant()
    _seed(sid, 3)
    evs = _events(_read(client, sid, params={"after": 0}, headers={"Last-Event-ID": "2"}))
    assert [e["id"] for e in evs] == ["3"]


def test_first_connect_starts_from_now():
    client, sid = _merchant()
    _seed(sid, 2)
    assert _events(_read(client, sid)) == []


def test_resync_when_history_expired():
    client, sid = _merchant()
    _seed(sid, 3)
    repo._db()["inbox_events"].delete_many({"store_id": sid, "seq": {"$in": [1, 2]}})  # simulate TTL
    evs = _events(_read(client, sid, params={"after": 1}))
    assert evs[0]["event"] == "resync" and evs[0]["id"] == "3"


def test_datetimes_serialized():
    client, sid = _merchant()
    repo.append_event(sid, "conversation.updated", "c1", {"at": datetime(2026, 9, 24, tzinfo=timezone.utc)})
    evs = _events(_read(client, sid, params={"after": 0}))
    assert "2026-09-24T00:00:00+00:00" in evs[0]["data"]


def test_heartbeat_when_idle(monkeypatch):
    monkeypatch.setattr(stream, "HEARTBEAT_SECONDS", 0.1)
    client, sid = _merchant()
    parsed = _read(client, sid)
    assert any(e.get("comment") == ": ping" for e in parsed)


# ---------------------------------------------------------------- security

def test_stream_access_control():
    owner, sid = _merchant()
    other, _ = _merchant()
    assert other.get(f"/stores/{sid}/inbox/events").status_code == 404
    assert TestClient(app).get(f"/stores/{sid}/inbox/events").status_code == 401


def test_auth_expired_closes_stream(monkeypatch):
    monkeypatch.setattr(stream, "AUTH_RECHECK_SECONDS", 0)
    client, sid = _merchant()
    monkeypatch.setattr(stream, "resolve_session", lambda raw: None)  # session revoked mid-stream
    evs = _events(_read(client, sid))
    assert [e["event"] for e in evs] == ["auth_expired"]


def test_logout_revokes_stream_authorization():
    client, sid = _merchant()
    raw = client.cookies.get("sf_session")
    assert stream._still_authorized(raw, sid) is True
    client.post("/auth/logout")
    assert stream._still_authorized(raw, sid) is False