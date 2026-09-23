"""
Billing / usage-limit tests. No Mongo, no LLM: all I/O is monkeypatched.
Run: python -m pytest tests/test_billing.py -q
"""

from datetime import date

import pytest

from app.billing import usage
from app.billing.plans import DAILY_MESSAGE_CAP_PER_STORE, LIMIT_REACHED_REPLY, get_monthly_limit
from app.billing.usage import UsageStatus, check_usage, current_period_start
from app.core import orchestrator
from app.response.response_generator import FALLBACK_REPLY
from app.schemas.models import CustomerMessage, StoreSettings
from app.settings.store_settings import set_plan


# ---------------------------------------------------------------- periods

@pytest.mark.parametrize("anchor,today,expected", [
    (23, date(2026, 9, 25), date(2026, 9, 23)),   # after anchor
    (23, date(2026, 9, 23), date(2026, 9, 23)),   # on anchor day
    (23, date(2026, 9, 22), date(2026, 8, 23)),   # day before anchor
    (5, date(2026, 1, 4), date(2025, 12, 5)),     # year rollover
    (1, date(2026, 3, 1), date(2026, 3, 1)),      # anchor = 1st
])
def test_period_start(anchor, today, expected):
    assert current_period_start(anchor, today) == expected


# ---------------------------------------------------------------- plans

def test_monthly_limits_per_plan():
    assert get_monthly_limit("starter") == 16000
    assert get_monthly_limit("pro") == 25000
    assert get_monthly_limit("enterprise", 100000) == 100000
    assert get_monthly_limit("enterprise") == 0          # no limit set -> 0 (blocked by fail-closed)
    assert get_monthly_limit("unknown_plan") == 16000    # unknown -> starter


def test_set_plan_rejects_unknown_plan():
    with pytest.raises(ValueError):
        set_plan("any_store", "gold")


@pytest.mark.parametrize("limit", [None, 0, -5])
def test_set_plan_enterprise_requires_positive_limit(limit):
    with pytest.raises(ValueError):
        set_plan("any_store", "enterprise", limit)


# ---------------------------------------------------------------- check_usage

def _fake_store(monkeypatch, plan="starter", enterprise_limit=None, monthly=0, daily=0):
    monkeypatch.setattr(usage, "get_settings", lambda sid: StoreSettings(
        store_id=sid, industry=None, plan=plan,
        enterprise_monthly_limit=enterprise_limit, billing_anchor_day=1,
    ))
    monkeypatch.setattr(usage, "_keys", lambda sid: ("2026-09-01", "2026-09-23"))
    monkeypatch.setattr(usage, "_read_count",
                        lambda sid, kind, key: monthly if kind == "monthly" else daily)


def test_allowed_under_limits(monkeypatch):
    _fake_store(monkeypatch, monthly=100, daily=10)
    s = check_usage("s1")
    assert s.allowed and s.reason is None


def test_blocked_exactly_at_monthly_limit(monkeypatch):
    _fake_store(monkeypatch, monthly=16000)
    s = check_usage("s1")
    assert not s.allowed and s.reason == "monthly_limit"


def test_blocked_at_daily_cap_even_with_monthly_room(monkeypatch):
    _fake_store(monkeypatch, monthly=10, daily=DAILY_MESSAGE_CAP_PER_STORE)
    s = check_usage("s1")
    assert not s.allowed and s.reason == "daily_cap"


def test_upgrade_raises_limit_immediately(monkeypatch):
    _fake_store(monkeypatch, plan="starter", monthly=20000)
    assert not check_usage("s1").allowed
    _fake_store(monkeypatch, plan="pro", monthly=20000)
    assert check_usage("s1").allowed


def test_enterprise_custom_limit(monkeypatch):
    _fake_store(monkeypatch, plan="enterprise", enterprise_limit=50, monthly=49)
    assert check_usage("s1").allowed
    _fake_store(monkeypatch, plan="enterprise", enterprise_limit=50, monthly=50)
    assert not check_usage("s1").allowed


def test_enterprise_without_limit_fails_closed(monkeypatch):
    _fake_store(monkeypatch, plan="enterprise", enterprise_limit=None)
    s = check_usage("s1")
    assert not s.allowed and s.reason == "monthly_limit"


# ---------------------------------------------------------------- orchestrator

_MSG = CustomerMessage(
    store_id="s1",
    conversation_id="c1",
    customer_id="test_customer",
    channel="web",
    text="hello",
)


def test_blocked_store_makes_no_pipeline_calls(monkeypatch):
    calls = []
    blocked = UsageStatus(False, "monthly_limit", 16000, 16000, 0, 300, "2026-09-01")
    monkeypatch.setattr(orchestrator, "check_usage", lambda sid: blocked)
    monkeypatch.setattr(orchestrator, "add_turn", lambda *a, **k: None)
    for name in ("detect_emotion", "detect_intent", "retrieve_context",
                 "generate_response", "extract_facts", "compare_products",
                 "record_ai_message"):
        monkeypatch.setattr(orchestrator, name, lambda *a, _n=name, **k: calls.append(_n))

    reply = orchestrator.handle_message(_MSG)

    assert reply.reply_text == LIMIT_REACHED_REPLY
    assert calls == []


def _allowed_pipeline(monkeypatch, response_fn):
    recorded = []
    allowed = UsageStatus(True, None, 0, 16000, 0, 300, "2026-09-01")
    monkeypatch.setattr(orchestrator, "check_usage", lambda sid: allowed)
    monkeypatch.setattr(orchestrator, "record_ai_message", lambda sid: recorded.append(sid))
    monkeypatch.setattr(orchestrator, "detect_emotion", lambda t: orchestrator._FALLBACK_EMOTION)
    monkeypatch.setattr(orchestrator, "detect_intent", lambda t: orchestrator._FALLBACK_INTENT)
    monkeypatch.setattr(orchestrator, "get_memory", lambda s, c: orchestrator._fallback_memory(c))
    monkeypatch.setattr(orchestrator, "retrieve_context", lambda s, t: [])
    monkeypatch.setattr(orchestrator, "get_industry", lambda s: None)
    monkeypatch.setattr(orchestrator, "recommend_products", lambda **k: [])
    monkeypatch.setattr(orchestrator, "generate_response", response_fn)
    monkeypatch.setattr(orchestrator, "add_turn", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator, "extract_facts", lambda *a, **k: {})
    monkeypatch.setattr(orchestrator, "log_turn", lambda *a, **k: None)
    return recorded


def _raise(**k):
    raise RuntimeError("LLM down")


@pytest.mark.parametrize("response_fn,should_count", [
    (lambda **k: "Real AI answer", True),
    (lambda **k: FALLBACK_REPLY, False),      # all models failed
    (_raise, False),                          # pipeline error fallback
])
def test_only_real_replies_are_counted(monkeypatch, response_fn, should_count):
    recorded = _allowed_pipeline(monkeypatch, response_fn)
    orchestrator.handle_message(_MSG)
    assert (len(recorded) == 1) == should_count