"""
Convert raw source rows into canonical Product objects.

WHY THIS EXISTS:
Ties the pieces together. Given raw merchant data + column mapping +
industry, produces validated Product records ready for MongoDB.

Preserves original data in Product.original_data for debugging and
merchant transparency.
"""

from typing import Any

from app.industries.registry import get_field
from app.ingestion.deterministic_mapper import MappingResult
from app.ingestion.normalizer import normalize
from app.schemas.models import Product


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