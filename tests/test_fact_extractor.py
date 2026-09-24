"""
Fact extractor tests: parsing, hard/soft split, normalization (incl. Arabic
numbers and currency), lists, budget, merge. No LLM, no registry.
"""

import pytest

from app.memory import fact_extractor as fx
from app.schemas.models import FieldDefinition


def _field(name, ftype, *, allowed=None, norm=None, valid=None):
    return FieldDefinition.model_construct(
        canonical_name=name, display_name=name, field_type=ftype, comparison_operator="==",
        unit=None, allowed_values=allowed, extraction_hint=None,
        normalization_rules=norm or {}, validation_rules=valid or {},
    )


FIELDS = [
    _field("size", "enum", allowed=["S", "M", "L", "XL"],
           norm={"case": "upper", "aliases": {"large": "L", "medium": "M"}}),
    _field("color", "string", norm={"case": "lower"}),
    _field("ram_gb", "int", norm={"strip_units": True}, valid={"min": 1, "max": 64}),
    _field("price", "float", norm={"strip_units": True}, valid={"min": 0}),
]


@pytest.fixture(autouse=True)
def fields(monkeypatch):
    monkeypatch.setattr(fx, "get_fields_for_industry", lambda industry_id: FIELDS)


def _parse(raw: str) -> dict:
    return fx._parse_facts(raw, "fashion")


# ---------------------------------------------------------------- parsing

def test_clean_json_hard_soft_split():
    out = _parse('{"hard_constraints": {"size": "M"}, "soft_preferences": {"color": "Black"}}')
    assert out == {"hard_constraints": {"size": "M"}, "soft_preferences": {"color": "black"}}


def test_code_fences_tolerated():
    out = _parse('```json\n{"hard_constraints": {"size": "L"}, "soft_preferences": {}}\n```')
    assert out["hard_constraints"] == {"size": "L"}


@pytest.mark.parametrize("raw", ["", "Sorry, I can't help", "[1, 2]", '{"hard_constraints": oops}'])
def test_garbage_returns_empty(raw):
    assert _parse(raw) == {"hard_constraints": {}, "soft_preferences": {}}


def test_invented_keys_and_nulls_dropped():
    out = _parse('{"hard_constraints": {"sleeve": "long", "size": null}, "soft_preferences": {"mood": "happy"}}')
    assert out == {"hard_constraints": {}, "soft_preferences": {}}


# ---------------------------------------------------------------- normalization

def test_enum_alias_and_invalid_enum():
    assert _parse('{"hard_constraints": {"size": "large"}}')["hard_constraints"] == {"size": "L"}
    assert _parse('{"hard_constraints": {"size": "XXXL"}}')["hard_constraints"] == {}


@pytest.mark.parametrize("raw_value,expected", [
    ('"8 GB"', 8), ("8", 8), ('"٨ جيجا"', 8), ('"128"', None),  # 128 > max 64
])
def test_int_parsing_with_units_and_arabic_digits(raw_value, expected):
    out = _parse(f'{{"hard_constraints": {{"ram_gb": {raw_value}}}}}')
    assert out["hard_constraints"].get("ram_gb") == expected


@pytest.mark.parametrize("raw_value,expected", [
    ('"500 EGP"', 500.0), ('"1,500"', 1500.0), ('"٥٠٠ جنيه"', 500.0), ("750", 750.0), ('"EGP 300"', 300.0),
])
def test_budget_parsing(raw_value, expected):
    out = _parse(f'{{"hard_constraints": {{"price": {raw_value}}}}}')
    assert out["hard_constraints"]["price"] == expected


def test_no_number_budget_dropped():
    assert _parse('{"hard_constraints": {"price": "cheap"}}')["hard_constraints"] == {}


def test_list_values_normalized_and_deduped():
    out = _parse('{"soft_preferences": {"color": ["Red", "blue", "RED"]}}')
    assert out["soft_preferences"]["color"] == ["red", "blue"]


def test_list_with_one_valid_item_becomes_scalar():
    out = _parse('{"hard_constraints": {"size": ["medium", "XXXL"]}}')
    assert out["hard_constraints"]["size"] == "M"


# ---------------------------------------------------------------- extract_facts

def test_no_industry_skips_llm(monkeypatch):
    monkeypatch.setattr(fx, "_call_single_model", lambda *a: pytest.fail("LLM must not be called"))
    assert fx.extract_facts("size M", None) == {"hard_constraints": {}, "soft_preferences": {}}


def test_all_models_failing_returns_empty(monkeypatch):
    def boom(*a):
        raise ValueError("no choices")
    monkeypatch.setattr(fx, "_call_single_model", boom)
    assert fx.extract_facts("size M", "fashion") == {"hard_constraints": {}, "soft_preferences": {}}


# ---------------------------------------------------------------- merge

def test_merge_customer_changes_mind():
    hard, soft = fx.merge_facts({"size": "M", "price": 500.0}, {"color": "black"},
                                {"hard_constraints": {}, "soft_preferences": {"size": "L"}})
    assert hard == {"price": 500.0}
    assert soft == {"color": "black", "size": "L"}


def test_merge_keeps_unrelated_facts():
    hard, soft = fx.merge_facts({"size": "M"}, {"color": "black"},
                                {"hard_constraints": {"price": 400.0}, "soft_preferences": {}})
    assert hard == {"size": "M", "price": 400.0} and soft == {"color": "black"}