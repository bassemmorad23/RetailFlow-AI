"""Business echo tests (isolated test DB). No waiting: the settle delay is faked."""

import uuid

import pytest

from app import api
from app.inbox import echoes
from app.inbox import repository as repo


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(echoes, "_sleep", lambda s: None)
    monkeypatch.setattr(echoes, "maybe_refresh_summary", lambda *a, **k: False)


def _store() -> str:
    return f"store_echo_{uuid.uuid4().hex[:8]}"


def _our_sent_reply(sid: str, customer: str, ids: list[str]) -> str:
    cid = repo.get_or_create_conversation(sid, "instagram", customer)["id"]
    repo.add_message(sid, cid, "customer", "hi", delivery_status="received")
    msg = repo.add_message(sid, cid, "ai", "Hello!", delivery_status="pending")
    repo.set_delivery(sid, msg["id"], "sent", external_message_ids=ids)
    return cid


def _texts(sid: str, cid: str) -> list[tuple[str, str]]:
    msgs, _ = repo.list_messages(sid, cid)
    return [(m["sender_type"], m["text"]) for m in msgs]


def test_own_send_by_message_id_ignored():
    sid = _store()
    cid = _our_sent_reply(sid, "c1", ["mid.ours"])
    assert echoes.handle_business_echo(sid, "instagram", "c1", "Hello!", external_message_id="mid.ours") == "known"
    assert len(_texts(sid, cid)) == 2
    assert repo.get_conversation(sid, cid)["ai_mode"] == "auto"


def test_own_multipart_send_ignored():
    sid = _store()
    _our_sent_reply(sid, "c2", ["mid.p1", "mid.p2"])
    assert echoes.handle_business_echo(sid, "instagram", "c2", "part 2", external_message_id="mid.p2") == "known"


def test_own_app_id_ignored_without_lookup(monkeypatch):
    monkeypatch.setattr(echoes, "_own_app_ids", lambda: {"123"})
    monkeypatch.setattr(repo, "find_message_by_external_id", lambda *a: pytest.fail("should not look up"))
    assert echoes.handle_business_echo(_store(), "messenger", "c3", "x",
                                       external_message_id="m", app_id=123) == "known"


def test_native_reply_stored_and_pauses_ai():
    sid = _store()
    cid = _our_sent_reply(sid, "c4", ["mid.ai"])
    res = echoes.handle_business_echo(sid, "instagram", "c4", "I'll send photos", external_message_id="mid.native")
    assert res == "stored"
    msgs, _ = repo.list_messages(sid, cid)
    human = [m for m in msgs if m["sender_type"] == "human"][0]
    assert human["text"] == "I'll send photos" and human["sent_by_user_id"] is None
    assert repo.get_conversation(sid, cid)["ai_mode"] == "paused"
    assert "Instagram app" in msgs[-1]["text"] and msgs[-1]["delivery_status"] == "internal"


def test_native_echo_retry_stored_once():
    sid = _store()
    cid = _our_sent_reply(sid, "c5", ["mid.ai"])
    echoes.handle_business_echo(sid, "instagram", "c5", "ok", external_message_id="mid.n1")
    assert echoes.handle_business_echo(sid, "instagram", "c5", "ok", external_message_id="mid.n1") == "known"
    assert [t for t in _texts(sid, cid) if t[0] == "human"] == [("human", "ok")]


def test_already_paused_adds_no_second_note():
    sid = _store()
    cid = _our_sent_reply(sid, "c6", ["mid.ai"])
    echoes.handle_business_echo(sid, "instagram", "c6", "one", external_message_id="mid.a")
    echoes.handle_business_echo(sid, "instagram", "c6", "two", external_message_id="mid.b")
    notes = [t for t in _texts(sid, cid) if t[0] == "system"]
    assert len(notes) == 1


def test_echo_arriving_before_our_ids_are_saved(monkeypatch):
    """Race: echo lands before set_delivery; the settle wait lets our ids land first."""
    sid = _store()
    cid = repo.get_or_create_conversation(sid, "instagram", "c7")["id"]
    repo.add_message(sid, cid, "customer", "hi", delivery_status="received")
    ours = repo.add_message(sid, cid, "ai", "Hello!", delivery_status="pending")

    monkeypatch.setattr(echoes, "_sleep",
                        lambda s: repo.set_delivery(sid, ours["id"], "sent", external_message_ids=["mid.late"]))
    assert echoes.handle_business_echo(sid, "instagram", "c7", "Hello!", external_message_id="mid.late") == "known"
    assert repo.get_conversation(sid, cid)["ai_mode"] == "auto"


# ---------------------------------------------------------------- webhook parsing

@pytest.fixture
def routed(monkeypatch):
    rec = {"echo": [], "incoming": []}
    monkeypatch.setattr(api, "handle_business_echo", lambda *a, **k: rec["echo"].append((a, k)))
    monkeypatch.setattr(api, "handle_incoming_message", lambda *a, **k: rec["incoming"].append((a, k)))
    monkeypatch.setattr(api, "find_store_by_ig_account", lambda i: "store_x" if i == "ig_biz" else None)
    monkeypatch.setattr(api, "find_store_by_fb_page", lambda p: "store_x" if p == "page1" else None)
    return rec


def test_instagram_echo_routed_with_customer_as_recipient(routed):
    api._process_instagram_entry({"messaging": [{
        "sender": {"id": "ig_biz"}, "recipient": {"id": "cust1"},
        "message": {"mid": "m1", "text": "hello", "is_echo": True},
    }]})
    assert routed["echo"] == [(("store_x", "instagram", "cust1", "hello"),
                               {"external_message_id": "m1", "app_id": None})]
    assert routed["incoming"] == []


def test_messenger_echo_routed(routed):
    api._process_messenger_entry({"id": "page1", "messaging": [{
        "sender": {"id": "page1"}, "recipient": {"id": "cust2"},
        "message": {"mid": "m2", "text": "hey", "is_echo": True, "app_id": 555},
    }]})
    assert routed["echo"] == [(("store_x", "messenger", "cust2", "hey"),
                               {"external_message_id": "m2", "app_id": 555})]