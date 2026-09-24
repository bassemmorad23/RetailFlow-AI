"""
Product recommendation with hard filtering, soft ranking, and variant support.

  1. Fetch full Products for relevant RAG candidates
  2. HARD constraints filter (must match)
  3. SOFT preferences rank (nice to have)
  4. Return top N with the best-matching variant

LENIENT FALLBACK:
If the hard filter removes everything, we still return the closest
candidates, but they are explicitly marked as NOT meeting the customer's
requirements, so the AI presents them honestly as alternatives.

VALUES CHECKED:
Product attributes + specifications, plus built-in `price` (the variant's
price when a variant is evaluated). Variant values override the parent's.

MATCHING:
- Unknown fields (not in the industry config) are always ignored.
- A list value matches if ANY of its items matches (e.g. colour red or blue).
"""

from typing import Any
from app.industries.fields import get_field
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
    if intent.label not in _RECOMMENDATION_INTENTS or not retrieved_context:
        return []

    candidate_ids = [c.source for c in retrieved_context if c.score >= _MIN_SCORE_THRESHOLD]
    if not candidate_ids:
        return []

    products = fetch_products(store_id, candidate_ids)
    if not products:
        return []

    facts = memory.known_facts
    hard = facts.hard_constraints or {}
    soft = facts.soft_preferences or {}

    hard_matches = [p for p in products if _matches_hard(p, hard, industry_id)]
    constraints_met = bool(hard_matches)
    candidates = hard_matches or products  # lenient fallback, marked honestly below

    ranked = sorted(candidates, key=lambda p: _score_soft(p, soft, industry_id), reverse=True)
    return [
        _to_recommendation(p, facts, industry_id, constraints_met)
        for p in ranked[:_MAX_RECOMMENDATIONS]
    ]


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------

def _values(product: Product, variant=None) -> dict[str, Any]:
    price = product.price
    if variant is not None and getattr(variant, "price", None) is not None:
        price = variant.price
    values: dict[str, Any] = {"price": price}
    values.update(product.attributes)
    values.update(product.specifications)
    if variant is not None:
        values.update(variant.attributes)
        values.update(variant.specifications)
    return values


# ---------------------------------------------------------------------------
# Hard filter — variant-aware
# ---------------------------------------------------------------------------

def _matches_hard(product: Product, constraints: dict[str, Any], industry_id: str | None) -> bool:
    """Parent qualifies if the product (or ANY of its variants) meets all constraints."""
    if not constraints or industry_id is None:
        return True
    if product.variants:
        return any(_matches_hard_variant(product, v, constraints, industry_id) for v in product.variants)
    return _check_constraints_against(constraints, _values(product), industry_id)


def _matches_hard_variant(product: Product, variant, constraints: dict[str, Any], industry_id: str) -> bool:
    return _check_constraints_against(constraints, _values(product, variant), industry_id)


def _check_constraints_against(constraints: dict[str, Any], values: dict[str, Any], industry_id: str) -> bool:
    for canonical_name, required_value in constraints.items():
        try:
            field = get_field(industry_id, canonical_name)
        except KeyError:
            continue  # unknown field: always ignored
        product_value = values.get(canonical_name)
        if product_value is None:
            return False
        if not _compare(product_value, required_value, field.comparison_operator):
            return False
    return True


def _compare(product_value: Any, required: Any, operator: str) -> bool:
    """Apply the field's operator. A list `required` matches if any item matches."""
    if isinstance(required, (list, tuple, set)):
        return any(_compare(product_value, r, operator) for r in required)
    try:
        if operator == "==":
            return str(product_value).strip().lower() == str(required).strip().lower()
        if operator == ">=":
            return float(product_value) >= float(required)
        if operator == "<=":
            return float(product_value) <= float(required)
        if operator == "range":
            r = float(required)
            return abs(float(product_value) - r) <= r * 0.2
    except (ValueError, TypeError):
        return False
    return False


# ---------------------------------------------------------------------------
# Soft ranking — variant-aware
# ---------------------------------------------------------------------------

def _score_soft(product: Product, preferences: dict[str, Any], industry_id: str | None) -> int:
    if not preferences or industry_id is None:
        return 0
    if product.variants:
        return max(_score_soft_variant(product, v, preferences, industry_id) for v in product.variants)
    return _score_values(preferences, _values(product), industry_id)


def _score_soft_variant(product: Product, variant, preferences: dict[str, Any], industry_id: str) -> int:
    return _score_values(preferences, _values(product, variant), industry_id)


def _score_values(preferences: dict[str, Any], values: dict[str, Any], industry_id: str) -> int:
    """+1 per matched soft preference."""
    score = 0
    for canonical_name, preferred_value in preferences.items():
        try:
            field = get_field(industry_id, canonical_name)
        except KeyError:
            continue
        product_value = values.get(canonical_name)
        if product_value is not None and _compare(product_value, preferred_value, field.comparison_operator):
            score += 1
    return score


# ---------------------------------------------------------------------------
# Recommendation builder
# ---------------------------------------------------------------------------

def _describe(constraints: dict[str, Any]) -> str:
    def fmt(v: Any) -> str:
        return " or ".join(map(str, v)) if isinstance(v, (list, tuple, set)) else str(v)
    return ", ".join(f"{k.replace('_', ' ')}: {fmt(v)}" for k, v in constraints.items())


def _to_recommendation(
    product: Product,
    facts: KnownFacts,
    industry_id: str | None,
    constraints_met: bool,
) -> ProductRecommendation:
    hard = facts.hard_constraints or {}
    soft = facts.soft_preferences or {}

    best_variant = None
    price = product.price

    if product.variants and industry_id is not None:
        matching = [v for v in product.variants if _matches_hard_variant(product, v, hard, industry_id)]
        pool = matching or product.variants
        best_variant = max(pool, key=lambda v: _score_soft_variant(product, v, soft, industry_id))
        if getattr(best_variant, "price", None) is not None:
            price = best_variant.price

    if not hard:
        reason = "Relevant to the customer's request."
    elif constraints_met:
        reason = f"Meets the customer's requirements ({_describe(hard)})."
    else:
        reason = (f"Closest alternative: does NOT meet all of the customer's requirements "
                  f"({_describe(hard)}). Say so honestly.")

    return ProductRecommendation(
        product_id=product.product_id,
        name=product.name,
        price=price,
        reason=reason,
        variant_sku=best_variant.sku if best_variant else None,
        variant_attrs=dict(best_variant.attributes) if best_variant else {},
    )