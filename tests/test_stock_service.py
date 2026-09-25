"""Stock service: live vs synced, failures never become 'available', levels, ordering, prompt labels."""

import pytest

from app.commerce import stock
from app.commerce.stock_live import LiveStockError
from app.response.response_generator import stock_label
from app.schemas.models import Product, ProductRecommendation, Variant


def _variant(sku, ext=None, status="unknown", qty=None):
    return Variant.model_construct(sku=sku, price=100.0, attributes={}, specifications={},
                                   external_id=ext, stock_status=status, stock_quantity=qty)


def _product(pid, ext=None, parent=None, status="unknown", qty=None, variants=None):
    return Product.model_construct(product_id=pid, name=pid, price=100.0, attributes={}, specifications={},
                                   variants=variants or [], external_id=ext, external_parent_id=parent,
                                   stock_status=status, stock_quantity=qty)


@pytest.fixture
def env(monkeypatch):
    state = {"platform": None, "catalog": {}, "live": {}, "fail": False, "calls": []}

    def live(store_id, arg):
        state["calls"].append(arg)
        if state["fail"]:
            raise LiveStockError("down")
        return state["live"]

    monkeypatch.setattr(stock, "store_platform", lambda sid: state["platform"])
    monkeypatch.setattr(stock, "fetch_products", lambda sid, ids: [state["catalog"][i] for i in ids if i in state["catalog"]])
    monkeypatch.setattr(stock, "shopify_live_stock", live)
    monkeypatch.setattr(stock, "woocommerce_live_stock", live)
    return state


def _levels(checks):
    return [c.level for c in checks]


# ---------------------------------------------------------------- synced (CSV)

def test_synced_levels(env):
    env["catalog"] = {
        "a": _product("a", status="in_stock", qty=10), "b": _product("b", status="in_stock", qty=2),
        "c": _product("c", status="out_of_stock", qty=0), "d": _product("d"),
    }
    checks = stock.check_stock("s", [("a", None), ("b", None), ("c", None), ("d", None), ("missing", None)])
    assert _levels(checks) == ["in_stock", "low_stock", "out_of_stock", "unknown", "unknown"]
    assert checks[0].source == "synced" and checks[3].reason == "no_data"


# ---------------------------------------------------------------- live

def test_live_overrides_synced_and_is_batched(env):
    env["platform"] = "shopify"
    env["catalog"] = {
        "a": _product("a", ext="gid://V/1", status="in_stock", qty=9),   # synced says in stock...
        "b": _product("b", ext="gid://V/2"),
    }
    env["live"] = {"gid://V/1": ("out_of_stock", 0), "gid://V/2": ("in_stock", 1)}  # ...live says sold out
    checks = stock.check_stock("s", [("a", None), ("b", None)])
    assert _levels(checks) == ["out_of_stock", "low_stock"]
    assert all(c.source == "live" for c in checks)
    assert len(env["calls"]) == 1


def test_live_failure_is_never_available(env):
    env["platform"] = "shopify"
    env["fail"] = True
    env["catalog"] = {"a": _product("a", ext="gid://V/1", status="in_stock", qty=50)}
    check = stock.check_stock("s", [("a", None)])[0]
    assert (check.level, check.reason) == ("unknown", "live_check_failed")


def test_not_found_on_platform_is_unknown(env):
    env["platform"] = "shopify"
    env["catalog"] = {"a": _product("a", ext="gid://V/deleted")}
    assert stock.check_stock("s", [("a", None)])[0].reason == "not_found_on_platform"


def test_platform_store_product_without_id_uses_synced(env):
    env["platform"] = "shopify"
    env["catalog"] = {"a": _product("a", status="in_stock", qty=8)}
    check = stock.check_stock("s", [("a", None)])[0]
    assert (check.level, check.source) == ("in_stock", "synced")


def test_woocommerce_variation_sends_parent_id(env):
    env["platform"] = "woocommerce"
    env["catalog"] = {"tee": _product("tee", parent="20", variants=[_variant("tee-M", ext="21")])}
    env["live"] = {"21": ("in_stock", None)}
    assert stock.check_stock("s", [("tee", "tee-M")])[0].level == "in_stock"
    assert env["calls"] == [[("20", "21")]]


def test_product_level_check_aggregates_variants(env):
    env["catalog"] = {
        "tee": _product("tee", variants=[_variant("M", status="out_of_stock", qty=0),
                                        _variant("L", status="in_stock", qty=5)]),
        "cap": _product("cap", variants=[_variant("S", status="out_of_stock", qty=0)] * 2),
    }
    assert _levels(stock.check_stock("s", [("tee", None), ("cap", None)])) == ["in_stock", "out_of_stock"]


# ---------------------------------------------------------------- recommendations

def _rec(pid):
    return ProductRecommendation(product_id=pid, name=pid, price=100.0, reason="r")


def test_attach_stock_orders_available_first(env):
    env["catalog"] = {"out": _product("out", status="out_of_stock", qty=0),
                      "unk": _product("unk"), "ok": _product("ok", status="in_stock", qty=9)}
    recs = stock.attach_stock("s", [_rec("out"), _rec("unk"), _rec("ok")])
    assert [(r.product_id, r.stock) for r in recs] == [("ok", "in_stock"), ("unk", "unknown"), ("out", "out_of_stock")]


@pytest.mark.parametrize("level,must_contain", [
    ("in_stock", "in stock"), ("low_stock", "only a few left"),
    ("out_of_stock", "OUT OF STOCK"), ("unknown", "do not say it is available"),
])
def test_prompt_labels(level, must_contain):
    assert must_contain in stock_label(level)