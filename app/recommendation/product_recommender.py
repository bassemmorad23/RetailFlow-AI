"""
Product recommendation with hard filtering and soft ranking.

WHY THIS EXISTS:
RAG returns semantically-relevant products. The recommender then:
  1. Fetches full Product data (needed for spec filtering)
  2. Applies HARD constraints (must-match — filters out products)
  3. Ranks by SOFT preferences (nice-to-have — influences order)
  4. Returns top N

WHY HARD vs SOFT SPLIT:
Customer saying "must have 16GB RAM" is different from "prefer black".
The first excludes non-matching products. The second just prefers.
Extraction step classifies each fact. Here we act on that classification.

WHY LENIENT FALLBACK (Option Z):
If hard filter removes everything, we return unfiltered candidates
ranked by soft preferences. Better to show close matches than nothing.
"""

from typing import Any

from app.industries.registry import get_field
from app.products.product_store import fetch_products
from app.schemas.models import (
    IntentLabel,
    IntentResult,
    KnownFacts,
    MemoryState,
    Product,
    ProductRecommendation,
    RetrievedChunk,
)


_RECOMMENDATION_INTENTS = {
    IntentLabel.WANTS_RECOMMENDATION,
    IntentLabel.BROWSING,
    IntentLabel.ASKING_DETAILS,
    IntentLabel.ASKING_AVAILABILITY,
    IntentLabel.READY_TO_BUY,
}

_MIN_SCORE_THRESHOLD = 0.35
_PRODUCT_SOURCE_PREFIX = ""  # legacy field, no longer required
_MAX_RECOMMENDATIONS = 3


def recommend_products(
    intent: IntentResult,
    memory: MemoryState,
    retrieved_context: list[RetrievedChunk],
    store_id: str,
    industry_id: str | None,
) -> list[ProductRecommendation]:
    """
    Turn RAG candidates into filtered, ranked recommendations.
    """
    if intent.label not in _RECOMMENDATION_INTENTS:
        return []

    if not retrieved_context:
        return []

    # Get candidate product_ids from RAG chunks (skip non-product chunks)
    candidate_ids = [
        chunk.source
        for chunk in retrieved_context
        if chunk.score >= _MIN_SCORE_THRESHOLD
    ]

    if not candidate_ids:
        return []

    # Fetch full Products from Mongo (specs needed for filtering)
    products = fetch_products(store_id, candidate_ids)

    if not products:
        return []

    facts = memory.known_facts
    hard = facts.hard_constraints or {}
    soft = facts.soft_preferences or {}

    # Apply hard filter
    filtered = [p for p in products if _matches_hard(p, hard, industry_id)]

    # Lenient fallback: if nothing passes hard filter, use all candidates
    if not filtered:
        filtered = products

    # Rank by soft preferences (higher score = better)
    ranked = sorted(
        filtered,
        key=lambda p: _score_soft(p, soft, industry_id),
        reverse=True,
    )

    # Build recommendations from top N
    return [
        _to_recommendation(p, facts) for p in ranked[:_MAX_RECOMMENDATIONS]
    ]


# ---------------------------------------------------------------------------
# Hard filter
# ---------------------------------------------------------------------------

def _matches_hard(
    product: Product, constraints: dict[str, Any], industry_id: str | None
) -> bool:
    """Product must satisfy ALL hard constraints."""
    if not constraints:
        return True
    if industry_id is None:
        return True

    for canonical_name, required_value in constraints.items():
        product_value = _get_product_value(product, canonical_name)
        if product_value is None:
            return False

        try:
            field = get_field(industry_id, canonical_name)
        except KeyError:
            continue  # unknown field — ignore constraint

        if not _compare(product_value, required_value, field.comparison_operator):
            return False

    return True


def _compare(product_value: Any, required: Any, operator: str) -> bool:
    """Apply the field's comparison operator."""
    try:
        if operator == "==":
            return str(product_value).strip().lower() == str(required).strip().lower()
        if operator == ">=":
            return float(product_value) >= float(required)
        if operator == "<=":
            return float(product_value) <= float(required)
        if operator == "range":
            # Within ±20% of required
            r = float(required)
            return abs(float(product_value) - r) <= r * 0.2
    except (ValueError, TypeError):
        return False
    return False


# ---------------------------------------------------------------------------
# Soft ranking
# ---------------------------------------------------------------------------

def _score_soft(
    product: Product, preferences: dict[str, Any], industry_id: str | None
) -> int:
    """+1 point per soft preference the product matches."""
    if not preferences or industry_id is None:
        return 0

    score = 0
    for canonical_name, preferred_value in preferences.items():
        product_value = _get_product_value(product, canonical_name)
        if product_value is None:
            continue

        try:
            field = get_field(industry_id, canonical_name)
        except KeyError:
            continue

        if _compare(product_value, preferred_value, field.comparison_operator):
            score += 1

    return score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_product_value(product: Product, canonical_name: str) -> Any:
    """Look up a value in specifications or attributes."""
    if canonical_name in product.specifications:
        return product.specifications[canonical_name]
    if canonical_name in product.attributes:
        return product.attributes[canonical_name]
    return None


def _to_recommendation(product: Product, facts: KnownFacts) -> ProductRecommendation:
    reason_parts = ["Matches your criteria."]

    # Show any hard constraint matches
    for key in facts.hard_constraints:
        value = _get_product_value(product, key)
        if value is not None:
            reason_parts.append(f"{key}: {value}.")

    return ProductRecommendation(
        product_id=product.product_id,
        name=product.name,
        price=product.price,
        reason=" ".join(reason_parts),
    )