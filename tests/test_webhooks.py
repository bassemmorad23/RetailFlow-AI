"""
Webhook processors must build VALID CustomerMessages and send a reply.
Regression test: these once failed validation silently inside try/except.
No Meta, no LLM: store lookup, orchestrator and senders are faked.
"""

from app import api
from app.schemas.models import AgentReply

_FAKE_REPLY = AgentReply(
    conversation_id="c", reply_text="Hello!",
    emotion=api.handle_message.__globals__["_FALLBACK_EMOTION"],
    intent=api.handle_message.__globals__["_FALLBACK_INTENT"],
    recommendations=[], retrieved_context=[],
)


def _capture(monkeypatch, sender_name: str):
    got = {"messages": [], "sent": []}

    def fake_handle(msg):
        got["messages"].append(msg)
        return _FAKE_REPLY

    monkeypatch.setattr(api, "handle_message", fake_handle)
    monkeypatch.setattr(api, sender_name, lambda *a: got["sent"].append(a))
    return got


def test_instagram_entry_builds_valid_message(monkeypatch):
    got = _capture(monkeypatch, "send_dm")
    monkeypatch.setattr(api, "find_store_by_ig_account", lambda rid: "store_x")
    api._process_instagram_entry({"messaging": [{
        "sender": {"id": "cust1"}, "recipient": {"id": "ig_biz"}, "message": {"text": "hi"},
    }]})
    msg = got["messages"][0]
    assert (msg.channel, msg.customer_id, msg.store_id) == ("instagram", "cust1", "store_x")
    assert got["sent"] == [("store_x", "cust1", "Hello!")]


def test_messenger_entry_builds_valid_message(monkeypatch):
    got = _capture(monkeypatch, "send_message")
    monkeypatch.setattr(api, "find_store_by_fb_page", lambda pid: "store_x")
    api._process_messenger_entry({"id": "page1", "messaging": [{
        "sender": {"id": "cust2"}, "message": {"text": "hi"},
    }]})
    msg = got["messages"][0]
    assert (msg.channel, msg.customer_id) == ("messenger", "cust2")
    assert got["sent"] == [("store_x", "cust2", "Hello!")]


def test_whatsapp_entry_builds_valid_message(monkeypatch):
    got = _capture(monkeypatch, "send_text")
    api._process_whatsapp_entry({"changes": [{"value": {"messages": [
        {"type": "text", "from": "201000000000", "text": {"body": "hi"}},
    ]}}]})
    msg = got["messages"][0]
    assert (msg.channel, msg.customer_id) == ("whatsapp", "201000000000")
    assert got["sent"] == [("201000000000", "Hello!")]


def test_echo_and_non_text_are_ignored(monkeypatch):
    got = _capture(monkeypatch, "send_dm")
    monkeypatch.setattr(api, "find_store_by_ig_account", lambda rid: "store_x")
    api._process_instagram_entry({"messaging": [
        {"sender": {"id": "a"}, "recipient": {"id": "b"}, "message": {"text": "x", "is_echo": True}},
        {"sender": {"id": "a"}, "recipient": {"id": "b"}, "message": {"attachments": []}},
    ]})
    assert got["messages"] == [] and got["sent"] == []