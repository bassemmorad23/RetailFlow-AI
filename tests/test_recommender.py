"""
Recommender tests: hard filter, soft ranking, variants, honest fallback.
No Mongo, no industry registry: products and field operators are faked.
"""

import pytest

from app.recommendation import product_recommender as rec
from app.schemas.models import (
    IntentLabel, IntentResult, KnownFacts, MemoryState, Product, RetrievedChunk, Variant,
)

_OPERATORS = {"size": "==", "color": "==", "material": "==", "price": "<=", "ram_gb": ">="}


class _Field:
    def __init__(self, op: str):
        self.comparison_operator = op


def _fake_get_field(industry_id, name):
    if name not in _OPERATORS:
        raise KeyError(name)
    return _Field(_OPERATORS[name])


def _variant(sku, price, **attrs):
    return Variant.model_construct(sku=sku, price=price, attributes=attrs, specifications={})


def _product(pid, name, price, variants=None, specs=None, **attrs):
    return Product.model_construct(product_id=pid, name=name, price=price, attributes=attrs,
                                   specifications=specs or {}, variants=variants or [])


CATALOG = {
    "shirt": _product("shirt", "Cotton Shirt", 450, material="cotton", variants=[
        _variant("SH-S-RED", 400, size="S", color="red"),
        _variant("SH-M-BLK", 450, size="M", color="black"),
        _variant("SH-L-BLK", 450, size="L", color="black"),
    ]),
    "jacket": _product("jacket", "Denim Jacket", 900, size="M", color="blue"),
    "hoodie": _product("hoodie", "Hoodie", 600, size="L", color="green"),
    "cap": _product("cap", "Cap", 150, size="M", color="black"),
    "phone8": _product("phone8", "Phone 8GB", 9000, specs={"ram_gb": 8}),
    "phone4": _product("phone4", "Phone 4GB", 6000, specs={"ram_gb": 4}),
}


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr(rec, "get_field", _fake_get_field)
    monkeypatch.setattr(rec, "fetch_products",
                        lambda store_id, ids: [CATALOG[i] for i in ids if i in CATALOG])


def _run(ids, hard=None, soft=None, intent=IntentLabel.WANTS_RECOMMENDATION, score=0.8, industry="fashion"):
    memory = MemoryState(conversation_id="c", history=[],
                         known_facts=KnownFacts(hard_constraints=hard or {}, soft_preferences=soft or {}))
    chunks = [RetrievedChunk.model_construct(source=i, score=score) for i in ids]
    return rec.recommend_products(IntentResult(label=intent, confidence=0.9), memory, chunks, "store_x", industry)


# ---------------------------------------------------------------- gating

def test_non_recommendation_intent_returns_nothing():
    assert _run(["jacket"], intent=IntentLabel.OTHER) == []


def test_low_relevance_chunks_ignored():
    assert _run(["jacket"], score=0.2) == []


def test_max_three_results():
    assert len(_run(["shirt", "jacket", "hoodie", "cap"])) == 3


# ---------------------------------------------------------------- hard filter

def test_hard_size_keeps_matching_products_and_variant():
    out = {r.product_id: r for r in _run(["shirt", "jacket", "hoodie"], hard={"size": "M"})}
    assert set(out) == {"shirt", "jacket"}
    assert out["shirt"].variant_sku == "SH-M-BLK" and out["shirt"].variant_attrs["size"] == "M"
    assert out["shirt"].reason.startswith("Meets the customer's requirements")


def test_numeric_spec_filter():
    out = _run(["phone8", "phone4"], hard={"ram_gb": 8}, industry="smartphones")
    assert [r.product_id for r in out] == ["phone8"]


def test_budget_is_applied():
    out = _run(["shirt", "jacket"], hard={"price": 500})
    assert [r.product_id for r in out] == ["shirt"]


def test_variant_price_used_for_budget():
    out = _run(["shirt"], hard={"price": 420})
    assert out[0].variant_sku == "SH-S-RED" and out[0].price == 400


def test_unknown_field_ignored_even_if_missing_on_product():
    out = _run(["jacket"], hard={"sleeve_style": "raglan"})
    assert [r.product_id for r in out] == ["jacket"]
    assert "Closest alternative" not in out[0].reason


# ---------------------------------------------------------------- honest fallback

def test_no_match_falls_back_and_says_so():
    out = _run(["shirt", "jacket"], hard={"size": "XL"})
    assert {r.product_id for r in out} == {"shirt", "jacket"}
    assert all("does NOT meet" in r.reason for r in out)


# ---------------------------------------------------------------- soft ranking

def test_soft_preference_ranks_and_picks_variant():
    out = _run(["jacket", "shirt"], soft={"color": "black"})
    assert out[0].product_id == "shirt" and out[0].variant_attrs["color"] == "black"


def test_list_preference_matches_any_value():
    out = _run(["cap", "jacket"], soft={"color": ["blue", "green"]})
    assert out[0].product_id == "jacket"


def test_hard_then_soft_combined():
    out = _run(["shirt", "jacket", "cap"], hard={"size": "M"}, soft={"color": "black"})
    assert [r.product_id for r in out][:2] in (["shirt", "cap"], ["cap", "shirt"])
    assert out[-1].product_id == "jacket"


def test_no_industry_means_no_filtering():
    out = _run(["shirt", "jacket"], hard={"size": "XL"}, industry=None)
    assert len(out) == 2