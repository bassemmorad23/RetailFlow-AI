"""
Deterministic column-to-canonical mapping.

Given a merchant's source column names + industry, produce a mapping
using FieldDefinition.aliases. No LLM. Case-insensitive.

WHY DETERMINISTIC FIRST:
Covers 80% of real cases. AI-assisted mapping (Week 4) only kicks in
for columns this can't resolve.
"""

from pydantic import BaseModel, Field
from app.industries.registry import get_fields_for_industry



# Built-in Product fields — always present, not industry-scoped.
# Maps common column names → canonical Product field name.
_BUILTIN_ALIASES: dict[str, str] = {
    # name
    "name": "name",
    "product name": "name",
    "product_name": "name",
    "title": "name",
    "product title": "name",
    "product_title": "name",

    # description
    "description": "description",
    "desc": "description",
    "product description": "description",
    "product_description": "description",
    "short_description": "description",

    # price
    "price": "price",
    "cost": "price",
    "product price": "price",
    "product_price": "price",
    "unit price": "price",
    "regular_price": "price",

    # currency
    "currency": "currency",

    # stock
    "stock": "stock_available",
    "in stock": "stock_available",
    "in_stock": "stock_available",
    "available": "stock_available",
    "availability": "stock_available",

    # image
    "image": "image_url",
    "image url": "image_url",
    "image_url": "image_url",
    "picture": "image_url",
    "photo": "image_url",

    # product_id
    "id": "product_id",
    "product id": "product_id",
    "product_id": "product_id",
    
    
    # category
    "category": "category",
    "product_type": "category",
    "product type": "category",
    "type": "category",
    
}


class MappingResult(BaseModel):
    """Output of column mapping — deterministic step only."""

    builtin: dict[str, str] = Field(default_factory=dict)
    """source_column -> Product built-in field (name, price, ...)"""

    mapped: dict[str, str] = Field(default_factory=dict)
    """source_column -> canonical FieldDefinition name (1-to-1)"""

    conflicts: dict[str, list[str]] = Field(default_factory=dict)
    """canonical_name -> [multiple source columns mapping to it]"""

    unmapped: list[str] = Field(default_factory=list)
    """source columns that matched nothing (candidates for AI mapping)"""


def map_columns(source_columns: list[str], industry_id: str) -> MappingResult:
    """
    Map merchant column names to canonical fields.

    Built-in Product fields (name, price, etc.) are matched first.
    Remaining columns are matched against the industry's FieldDefinition
    aliases. Case-insensitive. Duplicates → conflicts, not silent overwrite.
    """
    result = MappingResult()

    # Build lookup: alias (lowercased) → canonical_name
    industry_fields = get_fields_for_industry(industry_id)
    alias_to_canonical: dict[str, str] = {}
    for field in industry_fields:
        # canonical_name itself is an alias
        alias_to_canonical[field.canonical_name.lower()] = field.canonical_name
        for alias in field.aliases:
            alias_to_canonical[alias.lower()] = field.canonical_name

    # Track which source cols mapped to which canonical (for conflict detection)
    canonical_to_sources: dict[str, list[str]] = {}

    for source_col in source_columns:
        key = source_col.strip().lower()

        # 1. Built-in match
        if key in _BUILTIN_ALIASES:
            result.builtin[source_col] = _BUILTIN_ALIASES[key]
            continue

        # 2. Industry FieldDefinition alias match
        if key in alias_to_canonical:
            canonical = alias_to_canonical[key]
            canonical_to_sources.setdefault(canonical, []).append(source_col)
            continue

        # 3. No match
        result.unmapped.append(source_col)

    # Split canonical mappings: 1-to-1 → mapped, N-to-1 → conflicts
    for canonical, sources in canonical_to_sources.items():
        if len(sources) == 1:
            result.mapped[sources[0]] = canonical
        else:
            result.conflicts[canonical] = sources

    return result