"""Order building blocks: extraction parsing, item resolution, customer details. No LLM, no DB."""

import pytest

from app.commerce import item_resolver as ir
from app.commerce.customer_fields import apply_customer_details, missing_fields, resolve_country, to_customer_details
from app.commerce.models import DraftCustomer
from app.commerce.order_extractor import parse_extraction
from app.commerce.stock import StockCheck
from app.schemas.models import Product, ProductRecommendation, RetrievedChunk, Variant


# ---------------------------------------------------------------- extraction parsing

def test_parse_full_extraction():
    raw = ('```json\n{"items": [{"product": "the black shirt", "attributes": {"Size": "M", "color": "black"},'
           ' "quantity": 2}], "customer": {"name": " Sara ", "phone": "01012345678", "region": "Giza",'
           ' "city": null}, "cancel": false}\n```')
    ex = parse_extraction(raw)
    assert ex.items == [{"product": "the black shirt", "attributes": {"size": "M", "color": "black"}, "quantity": 2}]
    assert ex.customer == {"name": "Sara", "phone": "01012345678", "region": "Giza"}
    assert ex.cancel is False


@pytest.mark.parametrize("raw", ["nope", "[]", '{"items": "x", "customer": 5}', '{"cancel": "yes"}'])
def test_parse_garbage_is_empty(raw):
    ex = parse_extraction(raw)
    assert ex.items == [] and ex.customer == {} and ex.cancel is False


def test_parse_bad_quantity_and_cancel():
    ex = parse_extraction('{"items": [{"product": "cap", "quantity": -3}, {"product": "", "attributes": {}}],'
                          ' "cancel": true}')
    assert ex.items == [{"product": "cap", "attributes": {}, "quantity": None}] and ex.cancel is True


# ---------------------------------------------------------------- item resolution

TEE = Product.model_construct(
    product_id="tee", name="Cotton T-Shirt", price=300.0, attributes={"material": "cotton"}, specifications={},
    variants=[Variant.model_construct(sku=f"tee-{s}-{c}", price=p, attributes={"size": s, "color": c}, specifications={})
              for s, c, p in [("M", "black", 300.0), ("L", "black", 320.0), ("M", "white", 300.0)]])
CAP = Product.model_construct(product_id="cap", name="Baseball Cap", price=150.0,
                              attributes={}, specifications={}, variants=[])


@pytest.fixture
def catalog(monkeypatch):
    state = {"stock": "in_stock", "rag": None}
    monkeypatch.setattr(ir, "fetch_products", lambda sid, ids: [p for p in (TEE, CAP) if p.product_id in ids])
    monkeypatch.setattr(ir, "retrieve_context", lambda sid, q, top_k=1: (
        [RetrievedChunk.model_construct(source=state["rag"][0], score=state["rag"][1])] if state["rag"] else []))
    monkeypatch.setattr(ir, "check_stock", lambda sid, reqs: [StockCheck(state["stock"], "live")])
    return state


def _rec(p):
    return ProductRecommendation(product_id=p.product_id, name=p.name, price=p.price, reason="r")


def test_variant_resolved_with_db_price_and_quantity(catalog):
    r = ir.resolve_item("s", {"product": "the t-shirt", "attributes": {"size": "l", "color": "Black"}, "quantity": 2},
                        [_rec(TEE), _rec(CAP)])
    assert r.status == "ok"
    assert (r.item.variant_sku, r.item.unit_price, r.item.quantity) == ("tee-L-black", 320.0, 2)


def test_ambiguous_variant_asks_with_options(catalog):
    r = ir.resolve_item("s", {"product": "shirt", "attributes": {"size": "M"}}, [_rec(TEE)])
    assert r.status == "needs_variant" and r.options == {"color": ["black", "white"], "size": ["M"]}


def test_simple_product_default_quantity_one(catalog):
    r = ir.resolve_item("s", {"product": "it", "attributes": {}}, [_rec(CAP)])
    assert (r.status, r.item.quantity, r.item.unit_price) == ("ok", 1, 150.0)


def test_out_of_stock_never_added(catalog):
    catalog["stock"] = "out_of_stock"
    r = ir.resolve_item("s", {"product": "cap"}, [_rec(CAP)])
    assert r.status == "out_of_stock" and r.item is None


def test_unknown_stock_added_but_flagged(catalog):
    catalog["stock"] = "unknown"
    r = ir.resolve_item("s", {"product": "cap"}, [_rec(CAP)])
    assert r.status == "ok" and r.stock == "unknown"


def test_catalog_search_needs_strong_match(catalog):
    catalog["rag"] = ("cap", 0.3)
    assert ir.resolve_item("s", {"product": "hat"}, []).status == "not_found"
    catalog["rag"] = ("cap", 0.8)
    assert ir.resolve_item("s", {"product": "baseball cap"}, []).status == "ok"


def test_quantity_capped(catalog):
    r = ir.resolve_item("s", {"product": "cap", "quantity": 500}, [_rec(CAP)])
    assert r.item.quantity == 20


# ---------------------------------------------------------------- customer details

def _apply(stated, current=None, channel="web", sender="v1", store_country="EG"):
    return apply_customer_details(current or DraftCustomer(), stated,
                                  store_country=store_country, channel=channel, sender_id=sender)


def test_full_details_valid_and_complete():
    u = _apply({"name": "Sara", "phone": "01012345678", "region": "الجيزة",
                "city": "Dokki", "address_line": "12 Tahrir St"})
    assert u.problems == [] and missing_fields(u.customer) == []
    d = to_customer_details(u.customer)
    assert (d.phone, d.country, d.region_code, d.region_name) == ("+201012345678", "EG", "EG-GZ", "Giza")


def test_region_suggestion_not_filled():
    u = _apply({"region": "Gizza"})
    assert u.customer.region_code is None and u.customer.region_suggestions == ["Giza"]
    assert "region_suggest" in u.problems


def test_unknown_region_and_bad_phone_reported():
    u = _apply({"region": "Narnia", "phone": "123"})
    assert set(u.problems) == {"region_unknown", "phone_invalid"} and u.customer.phone is None


def test_whatsapp_number_offered_automatically():
    u = _apply({}, channel="whatsapp", sender="201012345678")
    assert (u.customer.phone, u.customer.phone_source) == ("+201012345678", "channel")
    u2 = _apply({"phone": "01099999999"}, current=u.customer, channel="whatsapp", sender="201012345678")
    assert (u2.customer.phone, u2.customer.phone_source) == ("+201099999999", "customer")


def test_other_country_changes_phone_and_region_rules():
    u = _apply({"country": "الإمارات", "phone": "0501234567", "region": "دبي"})
    assert (u.customer.country, u.customer.phone, u.customer.region_code) == ("AE", "+971501234567", "AE-DU")


def test_changing_country_clears_old_region():
    first = _apply({"region": "Giza"})
    second = _apply({"country": "Saudi Arabia"}, current=first.customer)
    assert second.customer.country == "SA" and second.customer.region_code is None


@pytest.mark.parametrize("text,code", [("Egypt", "EG"), ("مصر", "EG"), ("KSA", "SA"), ("uae", "AE"), ("France", "FR")])
def test_resolve_country(text, code):
    assert resolve_country(text) == code


def test_resolve_country_unknown():
    assert resolve_country("Atlantis") is None