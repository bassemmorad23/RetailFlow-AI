"""
Orchestrator — production-hardened version.

FAILURE STRATEGY:
Each pipeline step is independently wrapped. A failure in emotion detection
does not stop intent detection. A failure in RAG does not stop response
generation. The customer always gets a reply — degraded if necessary, but
never a raw crash or empty response.

Failure hierarchy:
  - Steps 1-5 (emotion, intent, memory, rag, recommendations):
      Catch exception, log warning, use sensible default, continue.
  - Step 6 (response generation):
      Catch exception, log error, return a polite fallback message.
      This is the only customer-facing failure mode.
  - Steps 7-9 (memory update, logging):
      Catch exception, log warning, never block the return.
"""

import logging

from app.analytics.conversation_logger import log_turn
from app.emotion.emotion_detector import detect_emotion
from app.intent.intent_detector import detect_intent
from app.memory.conversation_memory import add_turn, get_memory
from app.rag.retriever import retrieve_context
from app.recommendation.product_recommender import recommend_products
from app.response.response_generator import generate_response

from app.memory.fact_extractor import extract_facts
from app.memory.conversation_memory import add_turn, get_memory, update_known_facts

from app.schemas.models import (
    AgentReply,
    CustomerMessage,
    EmotionLabel,
    EmotionResult,
    IntentLabel,
    IntentResult,
    KnownFacts,
    MemoryState,
)

logger = logging.getLogger(__name__)

_FALLBACK_REPLY = (
    "I am sorry, I am having a little trouble right now. "
    "Please try again in a moment."
)


def handle_message(message: CustomerMessage) -> AgentReply:
    """
    Run a customer message through the full agent pipeline.
    Every step is independently fault-tolerant.
    """

    # Step 1 — Emotion
    try:
        emotion = detect_emotion(message.text)
    except Exception:
        logger.warning("Emotion detection failed, defaulting to NEUTRAL.", exc_info=True)
        emotion = EmotionResult(
            label=EmotionLabel.NEUTRAL,
            confidence=0.0,
            scores={},
        )

    # Step 2 — Intent
    try:
        intent = detect_intent(message.text)
    except Exception:
        logger.warning("Intent detection failed, defaulting to OTHER.", exc_info=True)
        intent = IntentResult(label=IntentLabel.OTHER, confidence=0.0)

    # Step 3 — Memory
    try:
        memory = get_memory(message.conversation_id)
    except Exception:
        logger.warning("Memory retrieval failed, using empty state.", exc_info=True)
        memory = MemoryState(
            conversation_id=message.conversation_id,
            history=[],
            known_facts=KnownFacts(),
        )

    # Step 4 — RAG retrieval (retriever already returns [] on failure)
    retrieved_context = retrieve_context(message.text)

    # Step 5 — Recommendations
    try:
        recommendations = recommend_products(
            intent=intent,
            memory=memory,
            retrieved_context=retrieved_context,
        )
    except Exception:
        logger.warning("Recommendation failed, returning empty list.", exc_info=True)
        recommendations = []

    # Step 6 — Response generation (critical, customer-facing)
    try:
        reply_text = generate_response(
            message=message,
            emotion=emotion,
            intent=intent,
            memory=memory,
            retrieved_context=retrieved_context,
            recommendations=recommendations,
        )
    except Exception:
        logger.error("Response generation failed, using fallback reply.", exc_info=True)
        reply_text = _FALLBACK_REPLY

    # Step 7 — Memory update (non-critical)
    try:
        add_turn(message.conversation_id, "customer", message.text)
        add_turn(message.conversation_id, "agent", reply_text)
    except Exception:
        logger.warning("Memory update failed, turn not persisted.", exc_info=True)

    # Step 7b — Fact extraction (non-critical, runs AFTER the reply so the
    # customer never waits on it. Facts land in time for the NEXT turn.)
    try:
        new_facts = extract_facts(message.text)
        if new_facts:
            # Only keys actually found are passed — existing facts not
            # mentioned in this message are left untouched.
            update_known_facts(message.conversation_id, **new_facts)
            logger.info("Extracted facts: %s", new_facts)
    except Exception:
        logger.warning("Fact extraction failed.", exc_info=True)
        
        

    # Step 8 — Assemble reply
    reply = AgentReply(
        conversation_id=message.conversation_id,
        reply_text=reply_text,
        emotion=emotion,
        intent=intent,
        recommendations=recommendations,
        retrieved_context=retrieved_context,
    )

    # Step 9 — Analytics (log_turn has its own try/except inside)
    log_turn(message, reply)

    return reply