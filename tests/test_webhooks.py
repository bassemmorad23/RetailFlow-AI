"""
Webhook parsing: each channel hands the right values to the Conversation
Service (store, channel, customer, text, platform message id).
No Meta, no AI: the service is faked.
"""

import pytest

from app import api


@pytest.fixture
def calls(monkeypatch):
    rec = []
    monkeypatch.setattr(api, "handle_incoming_message", lambda *a, **k: rec.append((a, k)))
    monkeypatch.setattr(api, "find_store_by_ig_account", lambda rid: "store_x" if rid == "ig_biz" else None)
    monkeypatch.setattr(api, "find_store_by_fb_page", lambda pid: "store_x" if pid == "page1" else None)
    return rec


def test_instagram(calls):
    api._process_instagram_entry({"messaging": [{
        "sender": {"id": "cust1"}, "recipient": {"id": "ig_biz"},
        "message": {"mid": "mid.1", "text": "hi"},
    }]})
    assert calls == [(("store_x", "instagram", "cust1", "hi"), {"external_message_id": "mid.1"})]


def test_messenger(calls):
    api._process_messenger_entry({"id": "page1", "messaging": [{
        "sender": {"id": "cust2"}, "message": {"mid": "mid.2", "text": "hello"},
    }]})
    assert calls == [(("store_x", "messenger", "cust2", "hello"), {"external_message_id": "mid.2"})]


def test_whatsapp(calls):
    api._process_whatsapp_entry({"changes": [{"value": {"messages": [
        {"id": "wamid.3", "type": "text", "from": "201000000000", "text": {"body": "salam"}},
    ]}}]})
    assert calls == [(("store_wa_test", "whatsapp", "201000000000", "salam"),
                      {"external_message_id": "wamid.3"})]


def test_echo_and_non_text_ignored(calls):
    api._process_instagram_entry({"messaging": [
        {"sender": {"id": "a"}, "recipient": {"id": "ig_biz"}, "message": {"text": "x", "is_echo": True}},
        {"sender": {"id": "a"}, "recipient": {"id": "ig_biz"}, "message": {"attachments": []}},
    ]})
    api._process_whatsapp_entry({"changes": [{"value": {"messages": [
        {"id": "w", "type": "image", "from": "2010"},
    ]}}]})
    assert calls == []


def test_unknown_store_ignored(calls):
    api._process_instagram_entry({"messaging": [{
        "sender": {"id": "c"}, "recipient": {"id": "someone_else"}, "message": {"text": "hi"},
    }]})
    api._process_messenger_entry({"id": "unknown_page", "messaging": [{
        "sender": {"id": "c"}, "message": {"text": "hi"},
    }]})
    assert calls == []


def test_one_bad_entry_does_not_block_others(monkeypatch):
    seen = []

    def process(entry):
        if entry == "bad":
            raise RuntimeError("boom")
        seen.append(entry)

    api._process_entries(["bad", "good"], process, "Test")
    assert seen == ["good"]