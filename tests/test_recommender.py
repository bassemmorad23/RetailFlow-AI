"""
Tests for recommendation/product_recommender.py — filtering logic.

Pure logic, no models or network. Covers the four behaviors that matter:
  - intent gating (non-recommendation intents return nothing)
  - score threshold (low-similarity chunks filtered out)
  - source filtering (FAQ/policy chunks excluded, only products kept)
  - known-facts injection into the recommendation reason
"""

from app.schemas.models import (
    IntentResult,
    IntentLabel,
    MemoryState,
    KnownFacts,
    RetrievedChunk,
)
from app.recommendation.product_recommender import recommend_products


def _memory(size=None, color=None):
    return MemoryState(
        conversation_id="c1",
        history=[],
        known_facts=KnownFacts(preferred_size=size, preferred_color=color),
    )


def _chunks():
    return [
        RetrievedChunk(source="prod_red_dress", content="Red dress", score=0.66),
        RetrievedChunk(source="prod_denim_jacket", content="Denim jacket", score=0.21),
        RetrievedChunk(source="faq_returns", content="Return policy", score=0.80),
    ]


# ── intent gating ───────────────────────────────────────────────────────────

def test_recommendation_intent_produces_results():
    recs = recommend_products(
        intent=IntentResult(label=IntentLabel.WANTS_RECOMMENDATION, confidence=0.8),
        memory=_memory(),
        retrieved_context=_chunks(),
    )
    assert len(recs) == 1
    assert recs[0].product_id == "prod_red_dress"


def test_complaint_intent_returns_empty():
    recs = recommend_products(
        intent=IntentResult(label=IntentLabel.COMPLAINT, confidence=0.9),
        memory=_memory(),
        retrieved_context=_chunks(),
    )
    assert recs == []


# ── score threshold ─────────────────────────────────────────────────────────

def test_low_score_chunks_filtered_out():
    low = [RetrievedChunk(source="prod_red_dress", content="Red dress", score=0.10)]
    recs = recommend_products(
        intent=IntentResult(label=IntentLabel.WANTS_RECOMMENDATION, confidence=0.8),
        memory=_memory(),
        retrieved_context=low,
    )
    assert recs == []


# ── source filtering ────────────────────────────────────────────────────────

def test_faq_chunks_are_excluded():
    # faq_returns has the highest score (0.80) but must NOT be recommended,
    # because it isn't a product.
    recs = recommend_products(
        intent=IntentResult(label=IntentLabel.WANTS_RECOMMENDATION, confidence=0.8),
        memory=_memory(),
        retrieved_context=_chunks(),
    )
    sources = [r.product_id for r in recs]
    assert "faq_returns" not in sources


def test_empty_context_returns_empty():
    recs = recommend_products(
        intent=IntentResult(label=IntentLabel.WANTS_RECOMMENDATION, confidence=0.8),
        memory=_memory(),
        retrieved_context=[],
    )
    assert recs == []


# ── known-facts injection ───────────────────────────────────────────────────

def test_known_size_appears_in_reason():
    recs = recommend_products(
        intent=IntentResult(label=IntentLabel.WANTS_RECOMMENDATION, confidence=0.8),
        memory=_memory(size="M"),
        retrieved_context=_chunks(),
    )
    assert "M" in recs[0].reason


def test_known_color_appears_in_reason():
    recs = recommend_products(
        intent=IntentResult(label=IntentLabel.WANTS_RECOMMENDATION, confidence=0.8),
        memory=_memory(color="red"),
        retrieved_context=_chunks(),
    )
    assert "red" in recs[0].reason


def test_no_known_facts_still_produces_reason():
    recs = recommend_products(
        intent=IntentResult(label=IntentLabel.WANTS_RECOMMENDATION, confidence=0.8),
        memory=_memory(),
        retrieved_context=_chunks(),
    )
    assert recs[0].reason  # non-empty reason even with no known facts