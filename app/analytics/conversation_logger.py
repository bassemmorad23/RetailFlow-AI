"""
Conversation analytics logger.

WHY THIS MODULE EXISTS:
Every turn the agent handles is a data point -- what emotion did the
customer have, what did they intend, what did we recommend, what did we
reply? At prototype stage this data has no consumer yet, but capturing it
NOW means that by the time we reach MVP we'll have real conversation data
to:
  - Fine-tune the intent classifier (replacing zero-shot with a trained model)
  - Evaluate response quality
  - Build store analytics dashboards (most asked questions, common complaints)
  - Train a real recommendation ranker

WHY JSONL (JSON Lines), NOT A DATABASE:
One JSON object per line, appended to a flat file. This is the simplest
possible format that:
  - Requires zero infrastructure (no DB server to run)
  - Is human-readable and inspectable with any text editor
  - Is directly ingestible by pandas, BigQuery, MongoDB, and every
    analytics tool that exists -- so we're not locked into anything
  - Survives process restarts (file is persistent, unlike the memory dict)

LIMITATION (explicit): a flat file doesn't scale past a few thousand
conversations and isn't queryable in real-time. At MVP, this swaps to
MongoDB. The log_turn() signature won't change -- only the storage backend.

WHY THIS IS NAMED analytics/, NOT logging/:
Python's standard library has a module called `logging`. A folder with
that name shadows it and causes import errors across the entire codebase.
analytics/ is also more honest about what this module does: it's not
debug/console logging, it's business-level conversation analytics.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.models import AgentReply, CustomerMessage


logger = logging.getLogger(__name__)

_LOG_PATH = Path("data/analytics/conversations.jsonl")


def log_turn(message: CustomerMessage, reply: AgentReply) -> None:
    """
    Append one conversation turn to the analytics log.

    Failures here are logged as warnings but never allowed to crash the
    agent -- a logging failure should never break a customer conversation.
    """
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "conversation_id": message.conversation_id,
            "customer_id": message.customer_id,
            "channel": message.channel,
            "customer_message": message.text,
            "emotion": {
                "label": reply.emotion.label.value,
                "confidence": reply.emotion.confidence,
            },
            "intent": {
                "label": reply.intent.label.value,
                "confidence": reply.intent.confidence,
            },
            "recommendations": [
                {"product_id": r.product_id, "name": r.name}
                for r in reply.recommendations
            ],
            "reply_text": reply.reply_text,
        }

        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    except Exception as e:
        logger.warning("Failed to log conversation turn: %s", e)