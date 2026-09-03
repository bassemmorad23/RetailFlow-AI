"""
Known-facts extraction module.

WHY LLM EXTRACTION AND NOT REGEX RULES:
Customers write in Arabic and English, in unpredictable phrasing
("مقاس ميديم", "I'm a medium", "M size pls", "budget around 2k"). A
rule-based extractor would need a separate hand-written rule set per
language and would still miss most natural phrasing. An LLM handles all
of it with no per-language work. Cost is one extra model call per turn —
acceptable at current volume, revisit if latency or rate limits bite.

WHY THIS IS A SEPARATE MODULE FROM conversation_memory.py:
conversation_memory owns STORAGE (get/write/trim). This module owns
UNDERSTANDING (what facts are in this text). Different responsibilities,
different failure modes — a Mongo timeout and a model hallucination need
different handling. Keeping them apart means swapping the extraction
strategy later (to rules, or a fine-tuned model) touches only this file.

WHY IT NEVER RETURNS NULLS TO OVERWRITE EXISTING FACTS:
If the model finds no size in this message, that means "not mentioned
here" — NOT "the customer no longer has a size preference". Returning
None for it would wipe a fact we already knew. So we return only the
keys that were actually found, and the caller merges rather than replaces.

WHY THIS NOW HAS THE SAME RETRY + FAILOVER CHAIN AS response_generator.py:
Originally this called only the FIRST model in the chain, once, with no
retry. That meant a rate-limited primary model made fact extraction fail
for that turn even though a backup model was available and working —
exactly the gap response_generator.py's failover chain already solves
for response generation. This module now uses the identical two-layer
pattern: retry transient errors on the current model with backoff, then
move to the next model in the chain if it's still failing.

FAILURE MODE:
Extraction is best-effort. If every model in the chain fails, returns
malformed JSON, or times out, we return an empty dict and the conversation
continues with whatever facts we already had. A failed extraction must
never break a customer conversation.
"""

import json
import logging

from openai import OpenAI
from openai import APIStatusError, APITimeoutError, APIConnectionError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_client = OpenAI(
    base_url=_OPENROUTER_BASE_URL,
    api_key=settings.OPENROUTER_API_KEY,
    timeout=30.0,
)


_EXTRACTION_PROMPT = """You extract structured shopping preferences from a customer message.

Return ONLY a JSON object. No markdown, no code fences, no explanation.

Include a key ONLY if the customer explicitly stated it in THIS message.
Omit any key that was not mentioned. If nothing was stated, return {}.

Possible keys:
  "preferred_size"   - clothing size as a string, e.g. "M", "XL", "42"
  "preferred_color"  - a single color in English, lowercase, e.g. "red"
  "budget_max"       - maximum budget as a number only, no currency symbol
  "mentioned_products" - array of product names/types the customer referred to

Do NOT infer or guess. Do NOT include a key just because it seems likely.
The message may be in Arabic or English; always return the JSON keys in English.

Examples:
Message: "عايز فستان أحمر مقاس ميديم"
{"preferred_size": "M", "preferred_color": "red", "mentioned_products": ["dress"]}

Message: "do you ship to Alexandria?"
{}

Message: "looking for jackets under 3000"
{"budget_max": 3000, "mentioned_products": ["jacket"]}"""

_ALLOWED_KEYS = {
    "preferred_size",
    "preferred_color",
    "budget_max",
    "mentioned_products",
}

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503}


def _is_retryable(exc: Exception) -> bool:
    """True only for transient errors worth retrying on the SAME model."""
    if isinstance(exc, APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS_CODES
    return isinstance(exc, (APITimeoutError, APIConnectionError))


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_single_model(model: str, text: str) -> str:
    """
    One model, with per-model retry on transient errors. Raises if it
    still fails — the caller moves to the next model in the chain.
    """
    response = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _EXTRACTION_PROMPT},
            {"role": "user", "content": text},
        ],
        max_tokens=200,
        temperature=0.0,
    )
    
    if not response.choices:
        raise ValueError(f"Model {model} returned no choices in response.")

    return response.choices[0].message.content.strip()


def extract_facts(text: str) -> dict:
    """
    Extract explicitly-stated shopping preferences from a customer message.

    Tries each model in settings.response_model_chain in order, same
    failover pattern as response_generator.py. Returns {} if the text is
    empty, or if EVERY model in the chain fails.
    """
    if not text or not text.strip():
        return {}

    for model in settings.response_model_chain:
        try:
            raw = _call_single_model(model, text)
            return _parse_facts(raw)
        except Exception:
            logger.warning("Fact extraction model failed, trying next: %s", model, exc_info=True)
            continue

    logger.warning("Fact extraction: all models in the chain failed.")
    return {}


def _parse_facts(raw: str) -> dict:
    """
    Parse the model's response into a clean facts dict.
    """
    cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Fact extraction returned non-JSON: %r", raw[:200])
        return {}

    if not isinstance(parsed, dict):
        logger.warning("Fact extraction returned non-object JSON: %r", raw[:200])
        return {}

    facts = {}

    for key, value in parsed.items():
        if key not in _ALLOWED_KEYS:
            continue
        if value is None:
            continue

        if key == "budget_max":
            try:
                facts[key] = float(value)
            except (TypeError, ValueError):
                logger.warning("Bad budget_max value: %r", value)
        elif key == "mentioned_products":
            if isinstance(value, list):
                items = [str(v).strip() for v in value if str(v).strip()]
                if items:
                    facts[key] = items
        else:
            text_value = str(value).strip()
            if text_value:
                facts[key] = text_value

    return facts