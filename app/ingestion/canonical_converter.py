"""
Convert raw source rows into canonical Product objects.

WHY THIS EXISTS:
Ties the pieces together. Given raw merchant data + column mapping +
industry, produces validated Product records ready for MongoDB.

Preserves original data in Product.original_data for debugging and
merchant transparency.
"""

from typing import Any
from collections import defaultdict
from app.industries.registry import get_field
from app.ingestion.deterministic_mapper import MappingResult
from app.ingestion.normalizer import normalize
from app.schemas.models import Product , Variant
from app.ingestion.stock_parser import aggregate_status, parse_stock


def _as_str(value) -> str | None:
    return str(value) if value not in (None, "") else None


def to_canonical(
    raw_row: dict[str, Any],
    mapping: MappingResult,
    industry_id: str,
    store_id: str,
) -> Product | None:
    """
    Turn one raw source row into a Product. Returns None if
    required fields (name, price) can't be resolved.
    """
    # Build the Product init dict
    product_data: dict[str, Any] = {
        "store_id": store_id,
        "attributes": {},
        "specifications": {},
        "original_data": dict(raw_row),  # preserve merchant's raw values
    }

    # 1. Built-in fields (name, price, description, etc.) — direct copy
    for source_col, product_field in mapping.builtin.items():
        if source_col in raw_row:
            value = raw_row[source_col]
            if value is not None and str(value).strip():
                product_data[product_field] = value
                
    # Stock: interpret whatever the source gave; missing -> unknown (never "available")
    status, qty = parse_stock(product_data.pop("stock_available", None))
    product_data["stock_status"] = status
    product_data["stock_quantity"] = qty
    product_data["stock_available"] = status == "in_stock"  # legacy field, no longer trusted
    
    product_data["external_id"] = _as_str(raw_row.get("external_id"))
    product_data["external_parent_id"] = _as_str(raw_row.get("external_parent_id"))
    

    # 2. Industry canonical fields — normalize before storing
    for source_col, canonical_name in mapping.mapped.items():
        if source_col not in raw_row:
            continue
        raw_value = raw_row[source_col]
        if raw_value is None or str(raw_value).strip() == "":
            continue

        field = get_field(industry_id, canonical_name)
        normalized = normalize(raw_value, field)
        if normalized is None:
            continue  # skip unnormalizable values

        # Route to attributes vs specifications based on field metadata
        if field.is_specification:
            product_data["specifications"][canonical_name] = normalized
        else:
            product_data["attributes"][canonical_name] = normalized

    # Required fields check — Product needs name + price + product_id
    if "name" not in product_data or "price" not in product_data:
        return None
    if "product_id" not in product_data:
        return None

    # Coerce price to float
    try:
        product_data["price"] = float(product_data["price"])
    except (ValueError, TypeError):
        return None

    try:
        return Product(**product_data)
    except Exception:
        return None
    

# ---------------------------------------------------------------------------
# GROUPED CONVERSION — handles variants
# ---------------------------------------------------------------------------

# Fields that never become variant attributes (always parent-level or per-variant slot).
_NEVER_VARIANT_ATTR = {"product_id", "name", "description", "sku"}
# Fields that live directly on Variant (not in attributes dict).
_VARIANT_SLOT_FIELDS = {"price", "stock_available", "stock_status", "stock_quantity", "image_url"}


def to_canonical_grouped(
    raw_rows: list[dict[str, Any]],
    mapping: MappingResult,
    industry_id: str,
    store_id: str,
) -> list[Product]:
    """
    Convert raw rows to Products, detecting variants.

    Rows sharing the same product_id become one Product with multiple
    Variants. Rows with unique product_id become simple Products with
    empty variants list.

    For variant groups, fields that DIFFER across rows become variant
    attributes; fields that SHARE across rows stay on the parent.
    """
    if not raw_rows:
        return []

    # Group by resolved product_id (built-in field, always present)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        pid = _extract_product_id(row, mapping)
        if pid is None:
            continue  # skip rows without a product_id
        groups[pid].append(row)

    products: list[Product] = []
    for product_id, rows in groups.items():
        if len(rows) == 1:
            # Simple product — no variants
            p = to_canonical(rows[0], mapping, industry_id, store_id)
            if p is not None:
                products.append(p)
        else:
            # Variant product
            p = _build_variant_product(product_id, rows, mapping, industry_id, store_id)
            if p is not None:
                products.append(p)

    return products


def _extract_product_id(row: dict[str, Any], mapping: MappingResult) -> str | None:
    """Find product_id in a row using the mapping."""
    for source_col, target in mapping.builtin.items():
        if target == "product_id" and source_col in row:
            val = row[source_col]
            if val is not None and str(val).strip():
                return str(val).strip()
    return None


def _build_variant_product(
    product_id: str,
    rows: list[dict[str, Any]],
    mapping: MappingResult,
    industry_id: str,
    store_id: str,
) -> Product | None:
    """
    Build a parent Product with Variants from multiple rows sharing product_id.

    Fields that share the same value across all rows stay on the parent.
    Fields that differ become variant attributes.
    """
    # First, convert every row to a Product-shaped dict (already normalized).
    canonical_rows: list[Product] = []
    for row in rows:
        p = to_canonical(row, mapping, industry_id, store_id)
        if p is not None:
            canonical_rows.append(p)

    if not canonical_rows:
        return None

    # Parent Product: base values from first row.
    parent_row = canonical_rows[0]
    parent_data = parent_row.model_dump()

    # Detect which attributes DIFFER across rows → move to variants.
    all_attrs = _collect_attr_keys(canonical_rows)
    variant_attrs = _find_differing_keys(canonical_rows, all_attrs, "attributes")

    all_specs = _collect_attr_keys(canonical_rows, field="specifications")
    variant_specs = _find_differing_keys(canonical_rows, all_specs, "specifications")

    # Remove variant-level attrs/specs from parent (they belong to variants only).
    parent_data["attributes"] = {
        k: v for k, v in parent_data["attributes"].items() if k not in variant_attrs
    }
    parent_data["specifications"] = {
        k: v for k, v in parent_data["specifications"].items() if k not in variant_specs
    }

    # Build one Variant per row.
    variants: list[Variant] = []
    for cr in canonical_rows:
        cr_dump = cr.model_dump()
        sku = _generate_variant_sku(product_id, cr_dump, variant_attrs, variant_specs)
        variant = Variant(
            sku=sku,
            price=cr.price,
            stock_available=cr.stock_status == "in_stock",
            stock_status=cr.stock_status,
            stock_quantity=cr.stock_quantity,
            attributes={k: cr_dump["attributes"].get(k) for k in variant_attrs if k in cr_dump["attributes"]},
            specifications={k: cr_dump["specifications"].get(k) for k in variant_specs if k in cr_dump["specifications"]},
            image_url=cr.image_url,
            external_id=cr.external_id,
        )
        variants.append(variant)
        
    parent_data["stock_status"] = aggregate_status([v.stock_status for v in variants])
    parent_data["stock_quantity"] = None
    parent_data["stock_available"] = parent_data["stock_status"] == "in_stock"
    parent_data["variants"] = [v.model_dump() for v in variants]
    parent_data["external_id"] = None  # variants carry their own ids

    try:
        return Product(**parent_data)
    except Exception:
        return None


def _collect_attr_keys(rows: list[Product], field: str = "attributes") -> set[str]:
    keys: set[str] = set()
    for r in rows:
        keys.update(getattr(r, field).keys())
    return keys


def _find_differing_keys(rows: list[Product], keys: set[str], field: str) -> set[str]:
    """Return the subset of keys whose values differ across rows."""
    differing: set[str] = set()
    for key in keys:
        values = {getattr(r, field).get(key) for r in rows}
        if len(values) > 1:
            differing.add(key)
    return differing


def _generate_variant_sku(
    product_id: str,
    row_data: dict[str, Any],
    variant_attrs: set[str],
    variant_specs: set[str],
) -> str:
    """Auto-generate variant SKU: parent-attr1-attr2-spec1..."""
    parts = [product_id]
    for k in sorted(variant_attrs):
        v = row_data["attributes"].get(k)
        if v is not None:
            parts.append(str(v))
    for k in sorted(variant_specs):
        v = row_data["specifications"].get(k)
        if v is not None:
            parts.append(str(v))
    return "-".join(parts)