"""
Tests for memory/fact_extractor.py — the _parse_facts logic.

We test _parse_facts (pure logic) directly, NOT extract_facts (which makes
a real network call). These cases mirror the real failures seen in live
testing: free models returning prose instead of JSON, wrapping JSON in
code fences, inventing keys, and returning wrong types.

The core guarantee under test: bad model output NEVER corrupts memory —
it yields {} or a cleaned subset, never a crash and never garbage facts.
"""

from app.memory.fact_extractor import _parse_facts


# ── happy path ──────────────────────────────────────────────────────────────

def test_clean_json_parses():
    raw = '{"preferred_size": "M", "preferred_color": "red"}'
    assert _parse_facts(raw) == {"preferred_size": "M", "preferred_color": "red"}


def test_empty_json_object():
    assert _parse_facts("{}") == {}


def test_budget_coerced_to_float():
    assert _parse_facts('{"budget_max": 1500}') == {"budget_max": 1500.0}


def test_mentioned_products_list():
    assert _parse_facts('{"mentioned_products": ["dress", "jacket"]}') == {
        "mentioned_products": ["dress", "jacket"]
    }


# ── model returns garbage (the real-world failures) ─────────────────────────

def test_prose_instead_of_json_returns_empty():
    # This is the exact failure seen live: the model "thinks out loud".
    raw = 'The user just said "casual". This is likely a style preference...'
    assert _parse_facts(raw) == {}


def test_json_wrapped_in_code_fences():
    raw = '```json\n{"preferred_size": "L"}\n```'
    assert _parse_facts(raw) == {"preferred_size": "L"}


def test_non_object_json_returns_empty():
    # Valid JSON, but a list, not an object.
    assert _parse_facts('["dress", "jacket"]') == {}


def test_invented_keys_are_discarded():
    raw = '{"preferred_size": "M", "favorite_food": "pizza"}'
    assert _parse_facts(raw) == {"preferred_size": "M"}


def test_null_values_are_skipped():
    # A null means "not mentioned" — must NOT overwrite an existing fact,
    # so it should be absent from the result, not present as None.
    raw = '{"preferred_size": "M", "preferred_color": null}'
    result = _parse_facts(raw)
    assert result == {"preferred_size": "M"}
    assert "preferred_color" not in result


def test_bad_budget_value_skipped():
    raw = '{"budget_max": "not a number"}'
    assert _parse_facts(raw) == {}


def test_empty_mentioned_products_skipped():
    assert _parse_facts('{"mentioned_products": []}') == {}


def test_whitespace_stripped_from_values():
    assert _parse_facts('{"preferred_size": "  M  "}') == {"preferred_size": "M"}