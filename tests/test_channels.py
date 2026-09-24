"""Channel send primitives: splitting, Meta error mapping, multi-part sending. No network."""

import httpx
import pytest

from app.channels.base import SendResult, classify_meta_error, send_parts, split_text
from app.channels.dispatcher import send


def _resp(status: int, body: dict) -> httpx.Response:
    return httpx.Response(status, json=body)


# ---------------------------------------------------------------- split_text

def test_short_text_single_part():
    assert split_text("hello", 1000) == ["hello"]


def test_empty_text_no_parts():
    assert split_text("   ", 1000) == []


def test_long_text_respects_limit_and_keeps_content():
    text = ("This is a sentence. " * 120).strip()
    parts = split_text(text, 1000)
    assert len(parts) > 1
    assert all(len(p) <= 1000 for p in parts)
    assert " ".join(parts).split() == text.split()


def test_unbreakable_text_hard_cut():
    parts = split_text("x" * 2500, 1000)
    assert [len(p) for p in parts] == [1000, 1000, 500]


# ---------------------------------------------------------------- error mapping

@pytest.mark.parametrize("body,expected", [
    ({"error": {"code": 10, "error_subcode": 2018278}}, "window_closed"),
    ({"error": {"code": 131047}}, "window_closed"),
    ({"error": {"code": 190}}, "auth_expired"),
    ({"error": {"code": 613}}, "rate_limited"),
    ({"error": {"code": 551}}, "recipient_unavailable"),
    ({"error": {"code": 999}}, "unknown"),
])
def test_meta_error_mapping(body, expected):
    code, _ = classify_meta_error(_resp(400, body))
    assert code == expected


def test_non_json_error_is_unknown():
    code, detail = classify_meta_error(httpx.Response(502, text="bad gateway"))
    assert code == "unknown" and "502" in detail


# ---------------------------------------------------------------- send_parts

def test_all_parts_sent_collects_ids():
    counter = iter(["m1", "m2"])
    result = send_parts(lambda p: _resp(200, {"message_id": next(counter)}), ["a", "b"],
                        lambda b: b.get("message_id"))
    assert result == SendResult(ok=True, external_message_ids=["m1", "m2"])


def test_stops_at_first_failure_and_keeps_sent_ids():
    responses = iter([_resp(200, {"message_id": "m1"}),
                      _resp(400, {"error": {"code": 190}}),
                      _resp(200, {"message_id": "m3"})])
    calls = []

    def post(part):
        calls.append(part)
        return next(responses)

    result = send_parts(post, ["a", "b", "c"], lambda b: b.get("message_id"))
    assert not result.ok and result.error_code == "auth_expired"
    assert result.external_message_ids == ["m1"]
    assert calls == ["a", "b"]


def test_network_error_mapped():
    def post(part):
        raise httpx.ConnectError("boom")

    result = send_parts(post, ["a"], lambda b: None)
    assert not result.ok and result.error_code == "network"


def test_error_detail_never_contains_token():
    result = send_parts(lambda p: _resp(400, {"error": {"code": 190, "message": "Invalid OAuth"}}),
                        ["a"], lambda b: None)
    assert "access_token" not in (result.error_detail or "")


# ---------------------------------------------------------------- dispatcher

def test_web_channel_needs_no_push():
    assert send("web", "store_x", "cust", "hi").ok


def test_unknown_channel_rejected():
    assert send("telegram", "store_x", "cust", "hi").error_code == "unsupported_channel"