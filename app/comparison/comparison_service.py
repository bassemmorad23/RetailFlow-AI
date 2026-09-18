"""
Product comparison service.

Handles the COMPARE_PRODUCTS intent flow:
1. Extract product mentions from customer message (LLM)
2. Resolve each mention against store catalog via RAG
3. Build structured comparison from real product data
4. Suggest alternatives for mentions that didn't match

DESIGN CONSTRAINTS (from spec):
- Max 3 products
- Database is source of truth for facts (LLM never invents specs)
- Industry-agnostic — comparison fields come from IndustryConfig
- Missing values shown as None, never guessed
- Alternatives come from same store's catalog only
"""

import json
import logging
import re
from typing import Any

from app.config import settings
from app.industries.registry import get_fields_for_industry
from app.products.product_store import fetch_products
from app.rag.retriever import MIN_SIMILARITY_SCORE, retrieve_context
from app.schemas.models import (
    ComparisonResult,
    ComparisonRow,
    IntentLabel,
    IntentResult,
    KnownFacts,
    MemoryState,
    Product,
    RetrievedChunk,
)

# LLM client — reuse the same pattern as fact_extractor
from openai import OpenAI


logger = logging.getLogger(__name__)


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_client = OpenAI(
    base_url=_OPENROUTER_BASE_URL,
    api_key=settings.OPENROUTER_API_KEY,
    timeout=30.0,
)


# Comparison-specific threshold — stricter than general RAG (0.35)
# because comparison needs precise product matching, not fuzzy relevance.
_COMPARISON_MATCH_THRESHOLD = 0.50

_MAX_COMPARISON_PRODUCTS = 3


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------


def compare_products(
    message_text: str,
    store_id: str,
    industry_id: str | None,
    memory: MemoryState,
) -> ComparisonResult:
    """
    Full comparison flow. Returns ComparisonResult even if no products
    matched — response generator handles all the framing.
    """
    if industry_id is None:
        # Can't build industry-specific comparison without industry
        return ComparisonResult(products=[], rows=[], not_found=[], alternatives=[])

    # 1. Extract mentions from customer message
    mentions = _extract_product_mentions(message_text)
    if not mentions:
        return ComparisonResult(products=[], rows=[], not_found=[], alternatives=[])

    # 2. Cap at max products
    mentions = mentions[:_MAX_COMPARISON_PRODUCTS]

    # 3. Resolve each mention against catalog
    found_products: list[Product] = []
    not_found: list[str] = []
    for mention in mentions:
        product = _resolve_mention(mention, store_id)
        if product is not None:
            found_products.append(product)
        else:
            not_found.append(mention)

    # 4. Alternatives for not-found (use RAG semantic search directly)
    alternatives: list[Product] = []
    for mention in not_found:
        alt = _find_alternative(mention, store_id, exclude_ids={p.product_id for p in found_products})
        if alt is not None:
            alternatives.append(alt)

    # 5. Build comparison rows for found products
    rows = _build_comparison_rows(found_products, industry_id)

    return ComparisonResult(
        products=found_products,
        rows=rows,
        not_found=not_found,
        alternatives=alternatives,
    )


# ---------------------------------------------------------------------------
# STEP 1 — Extract product mentions from message
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """You extract product names/models from customer messages.

The customer wants to compare products. Extract the specific product names or models they mentioned.

Rules:
- Return a JSON array of strings, e.g. ["iPhone 15", "Samsung Galaxy S24"]
- Include brand + model when both are stated
- Do NOT invent products the customer didn't mention
- Do NOT include generic terms like "phone" or "shirt" alone
- If nothing specific is mentioned, return: []
- Output ONLY the JSON array. No prose, no code fences.
"""


def _extract_product_mentions(text: str) -> list[str]:
    """LLM extracts product names/models from a comparison message."""
    if not text or not text.strip():
        return []

    for model in settings.response_model_chain:
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()
            return _parse_mentions(raw)
        except Exception as exc:
            logger.warning(
                "Product mention extraction failed, trying next model",
                extra={"failed_model": model, "error": str(exc)[:200]},
            )
            continue

    logger.error("All models failed for mention extraction; returning empty list")
    return []


def _parse_mentions(raw: str) -> list[str]:
    """Extract JSON array from LLM output. Handles code fences, prose."""
    if not raw:
        return []

    cleaned = re.sub(r"^```(?:json)?", "", raw.strip()).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match is None:
        return []

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    # Sanitize: only strings, trim whitespace, drop empties
    result = []
    for item in parsed:
        if isinstance(item, str):
            s = item.strip()
            if s:
                result.append(s)
    return result


# ---------------------------------------------------------------------------
# STEP 2 — Resolve mention → real Product
# ---------------------------------------------------------------------------


def _resolve_mention(mention: str, store_id: str) -> Product | None:
    """
    Search RAG for the mention. If top hit >= threshold, fetch the
    full Product from Mongo. Otherwise, return None (mark as not_found).
    """
    chunks = retrieve_context(store_id, mention, top_k=1)
    if not chunks:
        return None

    top = chunks[0]
    if top.score < _COMPARISON_MATCH_THRESHOLD:
        logger.debug(
            "Mention '%s' below comparison threshold (%.2f < %.2f)",
            mention, top.score, _COMPARISON_MATCH_THRESHOLD,
        )
        return None

    # Fetch full Product data
    products = fetch_products(store_id, [top.source])
    if not products:
        return None

    return products[0]


# ---------------------------------------------------------------------------
# STEP 3 — Find alternative for not-found mention
# ---------------------------------------------------------------------------


def _find_alternative(
    mention: str, store_id: str, exclude_ids: set[str]
) -> Product | None:
    """
    Return the best alternative for a not-found mention.
    Uses RAG with a lower threshold and excludes already-found products.
    """
    chunks = retrieve_context(store_id, mention, top_k=3)
    for chunk in chunks:
        if chunk.source in exclude_ids:
            continue
        # Accept any chunk that met RAG's own threshold (0.35)
        products = fetch_products(store_id, [chunk.source])
        if products:
            return products[0]
    return None


# ---------------------------------------------------------------------------
# STEP 4 — Build comparison rows (industry-agnostic)
# ---------------------------------------------------------------------------


def _build_comparison_rows(
    products: list[Product], industry_id: str
) -> list[ComparisonRow]:
    """
    For each industry field, extract that field's value from each product.
    Missing values = None (never invented).
    Always includes price as first row (built-in).
    """
    if not products:
        return []

    rows: list[ComparisonRow] = []

    # Price row (built-in, always included)
    price_values = {p.product_id: p.price for p in products}
    rows.append(ComparisonRow(
        field_name="price",
        display_name="Price",
        values=price_values,
    ))

    # Industry-specific fields
    industry_fields = get_fields_for_industry(industry_id)
    for field in industry_fields:
        values: dict[str, Any] = {}
        for product in products:
            values[product.product_id] = _get_field_value(product, field.canonical_name)
        rows.append(ComparisonRow(
            field_name=field.canonical_name,
            display_name=field.display_name,
            values=values,
        ))

    return rows


def _get_field_value(product: Product, canonical_name: str) -> Any:
    """
    Look up a field value on a Product.
    Checks specifications, attributes, and Product schema built-ins.
    Returns None if not present (never invented).
    Phase 1: only checks parent, not variants.
    """
    if canonical_name in product.specifications:
        return product.specifications[canonical_name]
    if canonical_name in product.attributes:
        return product.attributes[canonical_name]
    # Also check direct Product fields (rare — most industry fields live in attrs/specs)
    return getattr(product, canonical_name, None)