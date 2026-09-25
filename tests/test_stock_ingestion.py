"""Stock ingestion: parser, CSV conversion (incl. variants), Shopify and WooCommerce rows."""

import pytest

from app.ingestion.canonical_converter import to_canonical_grouped
from app.ingestion.deterministic_mapper import map_columns
from app.ingestion.shopify_adapter import _flatten_product_with_variants
from app.ingestion.stock_parser import aggregate_status, parse_stock
from app.ingestion.woocommerce_adapter import _flatten_product, _flatten_variation


# ---------------------------------------------------------------- parser

@pytest.mark.parametrize("raw,expected", [
    (None, ("unknown", None)), ("", ("unknown", None)), ("maybe", ("unknown", None)),
    (12, ("in_stock", 12)), (0, ("out_of_stock", 0)), (-3, ("out_of_stock", 0)),
    ("7", ("in_stock", 7)), ("٥", ("in_stock", 5)), ("0", ("out_of_stock", 0)),
    (True, ("in_stock", None)), (False, ("out_of_stock", None)),
    ("In Stock", ("in_stock", None)), ("onbackorder", ("in_stock", None)), ("متوفر", ("in_stock", None)),
    ("Sold Out", ("out_of_stock", None)), ("غير متوفر", ("out_of_stock", None)), ("خلصان", ("out_of_stock", None)),
])
def test_parse_stock(raw, expected):
    assert parse_stock(raw) == expected


def test_aggregate_status():
    assert aggregate_status(["out_of_stock", "in_stock"]) == "in_stock"
    assert aggregate_status(["out_of_stock", "out_of_stock"]) == "out_of_stock"
    assert aggregate_status(["unknown", "out_of_stock"]) == "unknown"
    assert aggregate_status([]) == "unknown"


# ---------------------------------------------------------------- CSV conversion

def _convert(rows: list[dict]):
    cols = sorted({k for r in rows for k in r})
    return to_canonical_grouped(rows, map_columns(cols, "fashion"), "fashion", "store_x")


def test_csv_without_stock_column_is_unknown_not_available():
    products = _convert([
        {"Product ID": "cap", "Product Name": "Cap", "Price": "150"},
        {"Product ID": "tee", "Product Name": "Tee", "Price": "300", "Size": "M"},
        {"Product ID": "tee", "Product Name": "Tee", "Price": "300", "Size": "L"},
    ])
    by_id = {p.product_id: p for p in products}
    assert by_id["cap"].stock_status == "unknown"
    assert by_id["tee"].stock_status == "unknown"
    assert {v.stock_status for v in by_id["tee"].variants} == {"unknown"}


def test_csv_quantities_per_variant_and_parent_aggregate():
    products = _convert([
        {"Product ID": "tee", "Product Name": "Tee", "Price": "300", "Size": "M", "Stock": "3"},
        {"Product ID": "tee", "Product Name": "Tee", "Price": "300", "Size": "L", "Stock": "0"},
    ])
    tee = products[0]
    by_size = {v.attributes["size"]: v for v in tee.variants}
    assert (by_size["M"].stock_status, by_size["M"].stock_quantity) == ("in_stock", 3)
    assert (by_size["L"].stock_status, by_size["L"].stock_quantity) == ("out_of_stock", 0)
    assert tee.stock_status == "in_stock"


def test_csv_arabic_yes_no_column():
    products = _convert([
        {"Product ID": "a", "Product Name": "A", "Price": "100", "Stock": "متوفر"},
        {"Product ID": "b", "Product Name": "B", "Price": "100", "Stock": "نفذ"},
    ])
    by_id = {p.product_id: p.stock_status for p in products}
    assert by_id == {"a": "in_stock", "b": "out_of_stock"}


# ---------------------------------------------------------------- Shopify

def _shopify_product(**variant):
    node = {"id": "gid://shopify/ProductVariant/1", "sku": "S1", "price": "100",
            "selectedOptions": [], **variant}
    return {"handle": "tee", "title": "Tee", "variants": {"edges": [{"node": node}]}}


@pytest.mark.parametrize("variant,expected", [
    ({"inventoryQuantity": 4, "inventoryPolicy": "DENY", "inventoryItem": {"tracked": True}}, 4),
    ({"inventoryQuantity": 0, "inventoryPolicy": "DENY", "inventoryItem": {"tracked": True}}, 0),
    ({"inventoryQuantity": 0, "inventoryPolicy": "DENY", "inventoryItem": {"tracked": False}}, "in stock"),
    ({"inventoryQuantity": 0, "inventoryPolicy": "CONTINUE", "inventoryItem": {"tracked": True}}, "in stock"),
    ({}, None),
])
def test_shopify_stock_value(variant, expected):
    assert _flatten_product_with_variants(_shopify_product(**variant))[0]["stock"] == expected


# ---------------------------------------------------------------- WooCommerce

@pytest.mark.parametrize("raw,expected", [
    ({"manage_stock": True, "stock_quantity": 5, "stock_status": "instock"}, 5),
    ({"manage_stock": True, "stock_quantity": 0, "stock_status": "outofstock"}, 0),
    ({"manage_stock": False, "stock_status": "instock"}, "in stock"),
    ({"manage_stock": False, "stock_status": "outofstock"}, "out of stock"),
    ({"manage_stock": False, "stock_status": "onbackorder"}, "in stock"),
    ({}, None),
])
def test_woocommerce_stock_value(raw, expected):
    product = {"id": 1, "name": "Tee", "price": "100", **raw}
    assert _flatten_product(product)["stock"] == expected
    assert _flatten_variation({"id": 1, "name": "Tee"}, {"regular_price": "100", **raw})["stock"] == expected