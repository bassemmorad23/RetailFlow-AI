"""Conversation summary + AI context tests. Real inbox storage (test DB), fake LLM."""

import uuid

import pytest

from app.inbox import repository as repo
from app.inbox import summary
from app.inbox.summary import SUMMARY_REFRESH_THRESHOLD, build_context, maybe_refresh_summary


@pytest.fixture
def llm(monkeypatch):
    rec = {"prompts": [], "reply": "Customer wants a black T-shirt size M.", "during": None}

    def fake(system_prompt, prompt):
        rec["prompts"].append(prompt)
        if rec["during"]:
            rec["during"]()
        return rec["reply"]

    monkeypatch.setattr(summary, "_call_llm", fake)
    return rec


def _conv(store_id: str | None = None) -> tuple[str, str]:
    sid = store_id or f"store_sum_{uuid.uuid4().hex[:8]}"
    conv = repo.get_or_create_conversation(sid, "whatsapp", uuid.uuid4().hex[:10])
    return sid, conv["id"]


def _add(sid, cid, n, sender="customer", status="received"):
    return [repo.add_message(sid, cid, sender, f"{sender} msg {i}", delivery_status=status) for i in range(n)]


def _summary(sid, cid) -> dict:
    return repo.get_conversation(sid, cid)["summary"]


# ---------------------------------------------------------------- refresh

def test_no_refresh_below_threshold(llm):
    sid, cid = _conv()
    _add(sid, cid, SUMMARY_REFRESH_THRESHOLD - 1)
    assert maybe_refresh_summary(sid, cid) is False
    assert llm["prompts"] == []


def test_refresh_at_threshold_covers_all(llm):
    sid, cid = _conv()
    msgs = _add(sid, cid, SUMMARY_REFRESH_THRESHOLD)
    assert maybe_refresh_summary(sid, cid) is True
    s = _summary(sid, cid)
    assert s["covers_through_seq"] == msgs[-1]["seq"]
    assert s["text"] == llm["reply"]


def test_incremental_uses_previous_summary_and_only_new_messages(llm):
    sid, cid = _conv()
    _add(sid, cid, SUMMARY_REFRESH_THRESHOLD)
    maybe_refresh_summary(sid, cid)
    _add(sid, cid, SUMMARY_REFRESH_THRESHOLD, sender="human", status="sent")
    maybe_refresh_summary(sid, cid)

    second = llm["prompts"][1]
    assert llm["reply"] in second            # previous summary passed in
    assert "[1]" not in second               # old messages not re-sent
    assert "Merchant:" in second             # manual replies labelled for the summary


def test_newer_summary_never_overwritten(llm):
    sid, cid = _conv()
    _add(sid, cid, SUMMARY_REFRESH_THRESHOLD)
    # Another worker finishes first while ours is generating.
    llm["during"] = lambda: repo.update_summary_cas(
        sid, cid, expected_covers=0, text="NEWER", new_covers=SUMMARY_REFRESH_THRESHOLD)
    assert maybe_refresh_summary(sid, cid) is False
    assert _summary(sid, cid)["text"] == "NEWER"


def test_llm_failure_keeps_old_summary(llm):
    sid, cid = _conv()
    _add(sid, cid, SUMMARY_REFRESH_THRESHOLD)
    llm["reply"] = None
    assert maybe_refresh_summary(sid, cid) is False
    assert _summary(sid, cid)["covers_through_seq"] == 0


def test_force_refresh_after_key_event(llm):
    sid, cid = _conv()
    _add(sid, cid, 1, sender="human", status="sent")
    assert maybe_refresh_summary(sid, cid, force=True) is True


def test_not_sent_replies_excluded_but_covered(llm):
    sid, cid = _conv()
    _add(sid, cid, SUMMARY_REFRESH_THRESHOLD - 1)
    last = repo.add_message(sid, cid, "ai", "SECRET DRAFT", delivery_status="not_sent")
    maybe_refresh_summary(sid, cid)
    assert "SECRET DRAFT" not in llm["prompts"][0]
    assert _summary(sid, cid)["covers_through_seq"] == last["seq"]


# ---------------------------------------------------------------- context

def test_context_is_summary_plus_uncovered_messages(llm):
    sid, cid = _conv()
    _add(sid, cid, SUMMARY_REFRESH_THRESHOLD)
    maybe_refresh_summary(sid, cid)
    repo.add_message(sid, cid, "human", "Delivery in 3 days", delivery_status="sent")
    current = repo.add_message(sid, cid, "customer", "ok thanks", delivery_status="received")

    ctx = build_context(sid, cid, before_seq=current["seq"])
    assert ctx.summary == llm["reply"]
    assert [(t.role, t.text) for t in ctx.turns] == [("agent", "[Merchant] Delivery in 3 days")]


def test_context_for_unknown_conversation_is_empty():
    ctx = build_context("store_nope", "conv_nope")
    assert ctx.summary == "" and ctx.turns == []