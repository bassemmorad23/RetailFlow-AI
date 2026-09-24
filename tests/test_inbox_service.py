"""
Conversation Service tests. Real inbox storage (isolated test DB),
fake AI and fake channel sending.
"""

import uuid

import pytest

from app.billing.plans import LIMIT_REACHED_REPLY
from app.channels.base import SendResult
from app.core import orchestrator
from app.inbox import repository as repo
from app.inbox import service
from app.schemas.models import AgentReply


def _reply(text: str) -> AgentReply:
    return AgentReply(
        conversation_id="c", reply_text=text,
        emotion=orchestrator._FALLBACK_EMOTION, intent=orchestrator._FALLBACK_INTENT,
        recommendations=[], retrieved_context=[],
    )


@pytest.fixture
def env(monkeypatch):
    """Fake AI + fake channel send. Returns call recorders."""
    calls = {"ai": [], "sent": [], "reply_text": "We have size M.", "send_result": SendResult(True, ["mid_1"])}

    def fake_ai(msg):
        calls["ai"].append(msg)
        hook = calls.get("during_ai")
        if hook:
            hook(msg)
        return _reply(calls["reply_text"])

    def fake_send(channel, store_id, recipient, text):
        calls["sent"].append((channel, store_id, recipient, text))
        return calls["send_result"]

    monkeypatch.setattr(service, "handle_message", fake_ai)
    monkeypatch.setattr(service, "send", fake_send)
    return calls


def _store() -> str:
    return f"store_svc_{uuid.uuid4().hex[:8]}"


def _messages(store_id: str, cid: str) -> list[dict]:
    msgs, _ = repo.list_messages(store_id, cid)
    return msgs


def test_happy_path_stores_both_messages_and_sends(env):
    sid = _store()
    res = service.handle_incoming_message(sid, "whatsapp", "2010001", "size M?", external_message_id="wamid.A")

    msgs = _messages(sid, res.conversation_id)
    assert [(m["sender_type"], m["seq"]) for m in msgs] == [("customer", 1), ("ai", 2)]
    assert msgs[1]["delivery_status"] == "sent"
    assert msgs[1]["ai_meta"]["intent"] is not None
    assert env["sent"] == [("whatsapp", sid, "2010001", "We have size M.")]
    assert env["ai"][0].conversation_id == res.conversation_id  # AI memory keyed by inbox id
    assert res.ai_replied


def test_platform_retry_does_not_reply_or_bill_twice(env):
    sid = _store()
    service.handle_incoming_message(sid, "instagram", "igu1", "hi", external_message_id="mid.X")
    second = service.handle_incoming_message(sid, "instagram", "igu1", "hi", external_message_id="mid.X")
    assert second.duplicate
    assert len(env["ai"]) == 1 and len(env["sent"]) == 1


def test_paused_conversation_skips_ai(env):
    sid = _store()
    first = service.handle_incoming_message(sid, "messenger", "psid1", "hello")
    repo.set_ai_mode(sid, first.conversation_id, "paused")

    res = service.handle_incoming_message(sid, "messenger", "psid1", "anyone there?")
    assert len(env["ai"]) == 1  # only the first message reached the AI
    assert _messages(sid, res.conversation_id)[-1]["sender_type"] == "customer"


def test_paused_during_generation_reply_not_sent(env):
    sid = _store()
    env["during_ai"] = lambda msg: repo.set_ai_mode(sid, msg.conversation_id, "paused")

    res = service.handle_incoming_message(sid, "whatsapp", "2010002", "price?")
    last = _messages(sid, res.conversation_id)[-1]
    assert last["sender_type"] == "ai" and last["delivery_status"] == "not_sent"
    assert env["sent"] == []


def test_send_failure_recorded(env):
    sid = _store()
    env["send_result"] = SendResult(False, [], "window_closed", "131047")
    res = service.handle_incoming_message(sid, "whatsapp", "2010003", "hi")
    last = _messages(sid, res.conversation_id)[-1]
    assert (last["delivery_status"], last["delivery_error"]) == ("failed", "window_closed")
    assert not res.ai_replied


def test_limit_reply_is_system_message(env):
    sid = _store()
    env["reply_text"] = LIMIT_REACHED_REPLY
    res = service.handle_incoming_message(sid, "instagram", "igu2", "hi")
    last = _messages(sid, res.conversation_id)[-1]
    assert last["sender_type"] == "system" and last["ai_meta"] is None


def test_same_customer_two_stores_are_separate(env):
    a, b = _store(), _store()
    ra = service.handle_incoming_message(a, "whatsapp", "2010004", "hi")
    rb = service.handle_incoming_message(b, "whatsapp", "2010004", "hi")
    assert ra.conversation_id != rb.conversation_id
    assert repo.get_conversation(b, ra.conversation_id) is None


def test_events_emitted_in_order(env):
    sid = _store()
    service.handle_incoming_message(sid, "whatsapp", "2010005", "hi")
    types = [e["type"] for e in repo.events_after(sid, 0)]
    assert types[0] == "message.created" and "message.updated" in types
    assert types.count("message.created") == 2