"""
Orchestrator: the SINGLE place that owns the pipeline shape.

WHY THIS EXISTS:
Every downstream module (emotion, intent, memory, RAG, recommendation,
response) has one job. This file is the ONLY place that knows the order
they run in, how their results connect, and how to recover when one of
them fails. That means schemas can change, storage can be swapped
(dict -> Mongo), models can be replaced, and none of that ever forces a
change in more than one file at a time.

WHY EVERY STEP HAS ITS OWN try/except INSTEAD OF ONE BIG ONE:
A single try/except around the whole pipeline would mean any single
failure short-circuits everything else and returns a generic error.
Real pipelines aren't that fragile: if emotion detection fails, we can
still recommend a product; if RAG fails, we can still generate a reply
from history alone. Isolating failure per step preserves as much of the
pipeline as possible on partial failure.

WHY SAFE FALLBACK VALUES INSTEAD OF PROPAGATED EXCEPTIONS:
Each step returns a valid-but-empty result on failure (e.g. Neutral
emotion, empty retrieval list) so downstream code never has to guard
against None. Downstream modules are simpler because they can trust the
shape they receive.

WHY WE UPDATE MEMORY LAST:
add_turn and update_known_facts are the operations that MUTATE state.
If they run early and something later fails, the conversation memory
gets a message that never actually produced a reply — a phantom turn.
Doing all writes at the end means memory only records what actually
happened.

WHY analytics_logger IS ALSO IN try/except:
Analytics is best-effort. If disk is full or the JSONL file is locked,
we do NOT want the CUSTOMER to see an error. Log the failure, keep
serving the reply.

STRUCTURED STEP TIMING:
Every pipeline step is wrapped in _time_step(), which measures wall-clock
duration and logs it with `pipeline_step` + `latency_ms` + `status`. This
is what lets you later ask "which step is slow?" without instrumenting
each one manually.
"""

import logging
import time
from typing import Any, Callable

from app.emotion.emotion_detector import detect_emotion
from app.intent.intent_detector import detect_intent
from app.memory.conversation_memory import get_memory, add_turn, update_known_facts
from app.memory.fact_extractor import extract_facts
from app.settings.store_settings import get_industry
from app.rag.retriever import retrieve_context
from app.recommendation.product_recommender import recommend_products
from app.response.response_generator import generate_response
from app.analytics.conversation_logger import log_turn
from app.comparison.comparison_service import compare_products
from app.schemas.models import ComparisonResult
from app.billing.plans import LIMIT_REACHED_REPLY
from app.billing.usage import check_usage, record_ai_message
from app.response.response_generator import FALLBACK_REPLY



from app.schemas.models import (
    AgentReply,
    CustomerMessage,
    EmotionResult,
    EmotionLabel,
    IntentResult,
    IntentLabel,
    KnownFacts,
    MemoryState,
)



logger = logging.getLogger(__name__)


# --- Safe fallback values for each step --------------------------------------

_FALLBACK_EMOTION = EmotionResult(
    label=EmotionLabel.NEUTRAL,
    confidence=0.0,
    scores={},
)

_FALLBACK_INTENT = IntentResult(
    label=IntentLabel.OTHER,
    confidence=0.0,
)
_PIPELINE_ERROR_REPLY = "Sorry, I could not process your message."


def _fallback_memory(conversation_id: str) -> MemoryState:
    """
    Best-effort memory when Mongo is unreachable. Empty history and facts,
    so the pipeline still returns a reply, just without personalization.
    """
    return MemoryState(
        conversation_id=conversation_id,
        history=[],
        known_facts=KnownFacts(),
    )


# --- Timing helper -----------------------------------------------------------

def _time_step(step_name: str, fn: Callable[[], Any], fallback: Any) -> Any:
    """
    Run a pipeline step, time it, log the outcome as one structured line.

    On success: logs pipeline_step + latency_ms + status="ok"
    On failure: logs pipeline_step + latency_ms + status="error"
                + exc_info, then returns the provided fallback so the
                pipeline can continue.

    Keeps timing/logging boilerplate out of the pipeline body itself,
    which stays readable as a top-to-bottom list of steps.
    """
    started = time.monotonic()
    try:
        result = fn()
        latency_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "pipeline step complete",
            extra={
                "pipeline_step": step_name,
                "latency_ms": latency_ms,
                "status": "ok",
            },
        )
        return result
    except Exception:
        latency_ms = int((time.monotonic() - started) * 1000)
        logger.exception(
            "pipeline step failed",
            extra={
                "pipeline_step": step_name,
                "latency_ms": latency_ms,
                "status": "error",
            },
        )
        return fallback


# --- Main pipeline -----------------------------------------------------------

def handle_message(message: CustomerMessage) -> AgentReply:
    """
    Run the full pipeline for one incoming message.
    """
    total_started = time.monotonic()

    # --- usage gate: before ANY LLM call ---
    usage = _time_step(
        "usage_check",
        lambda: check_usage(message.store_id),
        None,  # check failed (e.g. Mongo blip) -> fail open, logged as error
    )
    if usage is not None and not usage.allowed:
        logger.warning(
            "AI usage limit reached",
            extra={
                "limit_reason": usage.reason,
                "monthly_used": usage.monthly_used,
                "monthly_limit": usage.monthly_limit,
                "daily_used": usage.daily_used,
            },
        )
        _time_step(
            "memory_add_turn_customer",
            lambda: add_turn(message.store_id, message.conversation_id, "customer", message.text),
            None,
        )
        return AgentReply(
            conversation_id=message.conversation_id,
            reply_text=LIMIT_REACHED_REPLY,
            emotion=_FALLBACK_EMOTION,
            intent=_FALLBACK_INTENT,
            recommendations=[],
            retrieved_context=[],
        )

    emotion = _time_step(
        "emotion_detection",
        lambda: detect_emotion(message.text),
        _FALLBACK_EMOTION,
    )

    intent = _time_step(
        "intent_detection",
        lambda: detect_intent(message.text),
        _FALLBACK_INTENT,
    )

    memory = _time_step(
        "memory_load",
        lambda: get_memory(message.store_id, message.conversation_id),
        _fallback_memory(message.conversation_id),
    )

    retrieved_context = _time_step(
        "rag_retrieval",
        lambda: retrieve_context(message.store_id, message.text),
        [],
    )

    industry_id = _time_step(
        "industry_lookup",
        lambda: get_industry(message.store_id),
        None,
    )

    recommendations = _time_step(
        "recommendation",
        lambda: recommend_products(
            intent=intent,
            memory=memory,
            retrieved_context=retrieved_context,
            store_id=message.store_id,
            industry_id=industry_id,
        ),
        [],
    )

    # Comparison flow: intercept COMPARE_PRODUCTS intent (confidence-gated)
    # to prevent BART false positives from hijacking the response.
    comparison: ComparisonResult | None = None
    if intent.label == IntentLabel.COMPARE_PRODUCTS and intent.confidence >= 0.5:
        comparison = _time_step(
            "comparison",
            lambda: compare_products(
                message_text=message.text,
                store_id=message.store_id,
                industry_id=industry_id,
                memory=memory,
            ),
            None,
        )

    reply_text = _time_step(
        "response_generation",
        lambda: generate_response(
            message=message,
            emotion=emotion,
            intent=intent,
            memory=memory,
            retrieved_context=retrieved_context,
            recommendations=recommendations,
            industry_id=industry_id,
            comparison=comparison,
        ),
        _PIPELINE_ERROR_REPLY,
    )

    # Count only real AI replies, never error fallbacks
    if reply_text not in (FALLBACK_REPLY, _PIPELINE_ERROR_REPLY):
        _time_step(
            "usage_record",
            lambda: record_ai_message(message.store_id),
            None,
        )

    # --- writes: happen last, so nothing is persisted for a failed request ---

    _time_step(
        "memory_add_turn_customer",
        lambda: add_turn(message.store_id, message.conversation_id, "customer", message.text),
        None,
    )

    _time_step(
        "memory_add_turn_agent",
        lambda: add_turn(message.store_id, message.conversation_id, "agent", reply_text),
        None,
    )

    new_facts = _time_step(
        "fact_extraction",
        lambda: extract_facts(message.text, industry_id),
        {},
    )

    if new_facts:
        hard = {**memory.known_facts.hard_constraints, **new_facts.get("hard_constraints", {})}
        soft = {**memory.known_facts.soft_preferences, **new_facts.get("soft_preferences", {})}

        _time_step(
            "memory_update_facts",
            lambda: update_known_facts(
                message.store_id,
                message.conversation_id,
                hard_constraints=hard,
                soft_preferences=soft,
            ),
            None,
        )

    reply = AgentReply(
        conversation_id=message.conversation_id,
        reply_text=reply_text,
        emotion=emotion,
        intent=intent,
        recommendations=recommendations,
        retrieved_context=retrieved_context,
    )

    _time_step(
        "analytics_log",
        lambda: log_turn(message, reply),
        None,
    )

    total_latency_ms = int((time.monotonic() - total_started) * 1000)
    logger.info(
        "pipeline complete",
        extra={
            "pipeline_step": "TOTAL",
            "latency_ms": total_latency_ms,
            "status": "ok",
        },
    )

    return reply