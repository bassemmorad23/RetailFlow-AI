"""
Product recommendation with hard filtering, soft ranking, and variant support.

WHY THIS EXISTS:
RAG returns semantically-relevant products. The recommender then:
  1. Fetches full Product data (needed for spec filtering)
  2. Applies HARD constraints (must-match — filters out products)
  3. Ranks by SOFT preferences (nice-to-have — influences order)
  4. Returns top N with the best-matching variant (if applicable)

WHY HARD vs SOFT SPLIT:
Customer saying "must have 16GB RAM" is different from "prefer black".
The first excludes non-matching products. The second just prefers.
Extraction step classifies each fact. Here we act on that classification.

WHY LENIENT FALLBACK (Option Z):
If hard filter removes everything, we return unfiltered candidates
ranked by soft preferences. Better to show close matches than nothing.

VARIANT SUPPORT:
Products with variants (fashion size+color combos, phones with different
storage options) get filtered/ranked at the VARIANT level. The returned
ProductRecommendation carries both the parent product info AND the specific
matching variant's SKU + attrs. Simple products (no variants) work as before.
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

    # Get candidate product_ids from RAG chunks (skip low-relevance)
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

    # Apply hard filter (at variant level for variant products)
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
        _to_recommendation(p, facts, industry_id)
        for p in ranked[:_MAX_RECOMMENDATIONS]
    ]


# ---------------------------------------------------------------------------
# Hard filter — variant-aware
# ---------------------------------------------------------------------------

def _matches_hard(
    product: Product, constraints: dict[str, Any], industry_id: str | None
) -> bool:
    """
    Product-level hard filter. For products WITH variants, we check
    if ANY variant satisfies constraints (parent qualifies if any child does).
    Simple products check parent attrs only.
    """
    if not constraints:
        return True
    if industry_id is None:
        return True

    # If product has variants, at least one must match all hard constraints
    if product.variants:
        return any(
            _matches_hard_variant(product, v, constraints, industry_id)
            for v in product.variants
        )

    # Simple product — check parent attrs/specs directly
    return _check_constraints_against(
        constraints,
        {**product.attributes, **product.specifications},
        industry_id,
    )


def _matches_hard_variant(
    product: Product, variant, constraints: dict[str, Any], industry_id: str
) -> bool:
    """Check if a specific variant (with parent's shared attrs) matches constraints."""
    # Merge parent + variant attrs/specs — variant overrides parent
    merged = {
        **product.attributes,
        **product.specifications,
        **variant.attributes,
        **variant.specifications,
    }
    return _check_constraints_against(constraints, merged, industry_id)


def _check_constraints_against(
    constraints: dict[str, Any],
    values: dict[str, Any],
    industry_id: str,
) -> bool:
    """Apply each hard constraint against a values dict."""
    for canonical_name, required_value in constraints.items():
        product_value = values.get(canonical_name)
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
    """Apply the field's comparison operator to a value against a required value."""
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
# Soft ranking — variant-aware
# ---------------------------------------------------------------------------

def _score_soft(
    product: Product, preferences: dict[str, Any], industry_id: str | None
) -> int:
    """Product-level soft score. For variants, use best variant's score."""
    if not preferences or industry_id is None:
        return 0
    if product.variants:
        return max(
            _score_soft_variant(product, v, preferences, industry_id)
            for v in product.variants
        )
    return _score_values(
        preferences,
        {**product.attributes, **product.specifications},
        industry_id,
    )


def _score_soft_variant(
    product: Product, variant, preferences: dict[str, Any], industry_id: str
) -> int:
    """Score a specific variant against soft preferences."""
    merged = {
        **product.attributes,
        **product.specifications,
        **variant.attributes,
        **variant.specifications,
    }
    return _score_values(preferences, merged, industry_id)


def _score_values(
    preferences: dict[str, Any],
    values: dict[str, Any],
    industry_id: str,
) -> int:
    """+1 point per soft preference the values dict matches."""
    score = 0
    for canonical_name, preferred_value in preferences.items():
        product_value = values.get(canonical_name)
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
# Recommendation builder
# ---------------------------------------------------------------------------

def _to_recommendation(
    product: Product,
    facts: KnownFacts,
    industry_id: str | None,
) -> ProductRecommendation:
    """
    Build recommendation. For variant products, pick the best-matching
    variant and include its SKU + attrs.
    """
    hard = facts.hard_constraints or {}
    soft = facts.soft_preferences or {}

    best_variant = None
    best_variant_price = product.price

    if product.variants and industry_id is not None:
        # Pick the variant that matches hard constraints AND has highest soft score
        matching = [
            v for v in product.variants
            if _matches_hard_variant(product, v, hard, industry_id)
        ]
        candidates = matching if matching else product.variants

        best_variant = max(
            candidates,
            key=lambda v: _score_soft_variant(product, v, soft, industry_id),
        )
        best_variant_price = best_variant.price

    reason_parts = ["Matches your criteria."]
    for key in hard:
        if best_variant and key in best_variant.attributes:
            reason_parts.append(f"{key}: {best_variant.attributes[key]}.")
        elif key in product.attributes:
            reason_parts.append(f"{key}: {product.attributes[key]}.")

    return ProductRecommendation(
        product_id=product.product_id,
        name=product.name,
        price=best_variant_price,
        reason=" ".join(reason_parts),
        variant_sku=best_variant.sku if best_variant else None,
        variant_attrs=dict(best_variant.attributes) if best_variant else {},
    )