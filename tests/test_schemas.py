"""
Tests for schemas/models.py — the input validation layer.

These test the CustomerMessage validators we added in production step 3:
blank-identifier rejection, empty-text rejection, truncation of over-long
text, and control-character stripping. Pure logic, no models or network.
"""

import pytest
from pydantic import ValidationError

from app.schemas.models import (
    CustomerMessage,
    MAX_MESSAGE_LENGTH,
    ConversationTurn,
)


# ── CustomerMessage.text validation ─────────────────────────────────────────

def test_valid_message_is_stripped():
    msg = CustomerMessage(
        conversation_id="c1",
        customer_id="u1",
        text="  hello there  ",
        channel="web",
        store_id="store_001"
    )
    assert msg.text == "hello there"


def test_empty_text_is_rejected():
    with pytest.raises(ValidationError):
        CustomerMessage(
            conversation_id="c1",
            customer_id="u1",
            text="   ",
            channel="web",
            store_id="store_001"
        )


def test_overlong_text_is_truncated():
    msg = CustomerMessage(
        conversation_id="c1",
        customer_id="u1",
        text="x" * (MAX_MESSAGE_LENGTH + 500),
        channel="web",
        store_id="store_001"
    )
    assert len(msg.text) == MAX_MESSAGE_LENGTH


def test_control_characters_are_stripped():
    msg = CustomerMessage(
        conversation_id="c1",
        customer_id="u1",
        text="hi\x00\x07 there",
        channel="web",
        store_id="store_001"
    )
    assert msg.text == "hi there"


def test_newlines_and_tabs_are_preserved():
    msg = CustomerMessage(
        conversation_id="c1",
        customer_id="u1",
        text="line1\nline2\ttabbed",
        channel="web",
        store_id="store_001"
    )
    assert "\n" in msg.text
    assert "\t" in msg.text


# ── CustomerMessage identifier validation ───────────────────────────────────

@pytest.mark.parametrize("field", ["conversation_id", "customer_id", "channel"])
def test_blank_identifier_is_rejected(field):
    kwargs = {
        "conversation_id": "c1",
        "customer_id": "u1",
        "text": "hello",
        "channel": "web",
        "store_id": "store_001"
    }
    kwargs[field] = "   "
    with pytest.raises(ValidationError):
        CustomerMessage(**kwargs)


# ── ConversationTurn.role validation ────────────────────────────────────────

def test_valid_roles_accepted():
    assert ConversationTurn(role="customer", text="hi").role == "customer"
    assert ConversationTurn(role="agent", text="hi").role == "agent"


def test_invalid_role_rejected():
    with pytest.raises(ValidationError):
        ConversationTurn(role="robot", text="hi")