"""
Fact extraction from a single customer message.

WHY THIS EXISTS:
Alongside the visible reply, we quietly build up a small profile of what
the customer said they want — their size, color preference, budget, and
which products they've mentioned. Doing this with a compact prompt to a
capable LLM handles both Arabic and English input without maintaining
regex/keyword rules per language.

WHY THIS RETURNS A DICT[str, ANY]:
The extractor returns a raw dict rather than a KnownFacts object because
the orchestrator layer decides how to MERGE the new facts with existing
memory. Keeping merge policy out of this module keeps extraction pure —
"look at the message, output what was stated." Nothing else.

WHY WE CHECK EACH FIELD BEFORE USING IT:
The LLM may return partial JSON, extra keys, or the wrong type. We only
accept known keys and validate types before returning them. Anything
malformed silently becomes {} — never a crash, never corrupted memory.

RESILIENCE:
Same two-layer model as response_generator.py:
  1. Per-model retries with backoff for transient errors
  2. Chain-level failover if a whole model is unusable

WHY A FAILURE HERE RETURNS {} INSTEAD OF RAISING:
Fact extraction is a background enrichment step — a failure should not
break the customer conversation. If we can't extract facts this turn,
we return {} and keep going. Memory just doesn't grow this turn.

STRUCTURED LOGGING NOTES:
Successful extraction: one log line with model + latency_ms + fact count.
Failed model call: one log line with failed_model + reason + latency,
before moving to the next model.
"""

import json
import logging
import re
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings
from app.industries.registry import get_fields_for_industry
from app.schemas.models import FieldDefinition

logger = logging.getLogger(__name__)


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


_client = OpenAI(
    base_url=_OPENROUTER_BASE_URL,
    api_key=settings.OPENROUTER_API_KEY,
    timeout=30.0,
)


def _build_extraction_prompt(industry_id: str) -> str:
    """Build extraction prompt dynamically from industry's FieldDefinitions."""
    fields = get_fields_for_industry(industry_id)

    field_lines = []
    for f in fields:
        line = f"- {f.canonical_name}: {f.field_type}"
        if f.unit:
            line += f" ({f.unit})"
        if f.allowed_values:
            line += f" — one of {f.allowed_values}"
        if f.extraction_hint:
            line += f". {f.extraction_hint}"
        field_lines.append(line)

    fields_block = "\n".join(field_lines)

    return f"""You are an extraction assistant. Read the customer's message
and extract facts they stated about their preferences.

For each fact, classify it as HARD or SOFT:
- HARD = mandatory constraint. Words like "must", "need", "only",
  "at least", "under $X", "no less than". Filter must enforce it.
- SOFT = preference that influences ranking but doesn't exclude.
  Words like "prefer", "would like", "if possible", "nice to have".

Return a valid JSON object with this exact structure:
{{
  "hard_constraints": {{ ... }},
  "soft_preferences": {{ ... }}
}}

Where each inner dict may include any of these keys:
{fields_block}

Rules:
- Only include a key if the message clearly states it.
- Do NOT guess. Do NOT invent.
- If unsure whether hard or soft, treat as soft (safer — no exclusion).
- If nothing is stated, return: {{"hard_constraints": {{}}, "soft_preferences": {{}}}}
- Output ONLY the JSON. No prose, no code fences.
"""






# ---------------------------------------------------------------------------
# LOW-LEVEL: single-model call with retry
# ---------------------------------------------------------------------------

def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return 500 <= exc.status_code < 600
    if isinstance(exc, APIError):
        return True
    return False


def _classify_failure(exc: BaseException) -> str:
    if isinstance(exc, APITimeoutError):
        return "timeout"
    if isinstance(exc, RateLimitError):
        return "rate_limit"
    if isinstance(exc, APIConnectionError):
        return "connection_error"
    if isinstance(exc, APIStatusError):
        return f"http_{exc.status_code}"
    if isinstance(exc, ValueError):
        return "empty_response"
    if isinstance(exc, APIError):
        return "api_error"
    return type(exc).__name__


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(_is_retryable),
)

def _call_single_model(model: str, text: str, industry_id: str) -> tuple[str, dict]:
    """..."""
    system_prompt = _build_extraction_prompt(industry_id)
    response = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    
    if not response.choices:
        raise ValueError(f"Model {model} returned no choices in response.")

    reply_text = response.choices[0].message.content.strip()

    usage: dict = {}
    if response.usage is not None:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return reply_text, usage


# ---------------------------------------------------------------------------
# PUBLIC
# ---------------------------------------------------------------------------

def extract_facts(text: str, industry_id: str | None) -> dict[str, Any]:
    """
    Extract stated facts from a single customer message. Walks the
    configured model chain and stops at the first success.
    """
    # Lenient V1: no industry means no industry-specific extraction.
    # Return {} without calling the LLM (saves cost + latency).
    if industry_id is None:
        return {"hard_constraints": {}, "soft_preferences": {}}
    
    
    if not text or not text.strip():
        return {"hard_constraints": {}, "soft_preferences": {}}

    for model in settings.response_model_chain:
        started = time.monotonic()
        try:
            raw, usage = _call_single_model(model, text, industry_id)
            latency_ms = int((time.monotonic() - started) * 1000)

            parsed = _parse_facts(raw, industry_id)
            logger.info(
                "Facts extracted",
                extra={
                    "model": model,
                    "latency_ms": latency_ms,
                    "fact_count": len(parsed["hard_constraints"]) + len(parsed["soft_preferences"]),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                },
            )
            return parsed

        except Exception as exc:
            latency_ms = int((time.monotonic() - started) * 1000)
            logger.warning(
                "Fact extraction model failed, trying next in chain",
                extra={
                    "failed_model": model,
                    "failure_reason": _classify_failure(exc),
                    "failure_detail": str(exc)[:200],
                    "latency_ms": latency_ms,
                },
            )
            continue

    logger.error(
        "All models in chain failed for fact extraction; returning {}",
        extra={"chain_length": len(settings.response_model_chain)},
    )
    return {"hard_constraints": {}, "soft_preferences": {}}




# ---------------------------------------------------------------------------
# VALUE NORMALIZATION & VALIDATION
# ---------------------------------------------------------------------------



def _normalize_value(value: Any, field) -> Any:
    """
    Coerce and normalize a raw value against its FieldDefinition.

    Returns the normalized value, or None if coercion is impossible
    (wrong type entirely, e.g. list where int expected).
    """
    rules = field.normalization_rules or {}

    # Type coercion
    if field.field_type == "int":
        try:
            if isinstance(value, str):
                # Strip common unit suffixes ("8 GB" -> "8")
                if rules.get("strip_units"):
                    value = re.sub(r"[a-zA-Z\s]+$", "", value).strip()
                return int(float(value))
            if isinstance(value, (int, float)):
                return int(value)
            return None
        except (ValueError, TypeError):
            return None

    if field.field_type == "float":
        try:
            if isinstance(value, str):
                if rules.get("strip_units"):
                    value = re.sub(r"[a-zA-Z\s]+$", "", value).strip()
                return float(value)
            if isinstance(value, (int, float)):
                return float(value)
            return None
        except (ValueError, TypeError):
            return None

    if field.field_type == "boolean":
        if isinstance(value, bool):
            return value
        return None

    if field.field_type in ("string", "enum"):
        if not isinstance(value, str):
            return None
        s = value

        if rules.get("trim", True):
            s = s.strip()

        case = rules.get("case")
        if case == "lower":
            s = s.lower()
        elif case == "upper":
            s = s.upper()
        elif case == "title":
            s = s.title()

        # Apply alias mapping (e.g. "large" -> "L")
        aliases = rules.get("aliases", {})
        if s.lower() in {k.lower() for k in aliases}:
            for k, v in aliases.items():
                if k.lower() == s.lower():
                    s = v
                    break

        if not s:
            return None
        return s

    # Unknown field_type — refuse rather than guess
    return None


def _validate_value(value: Any, field) -> bool:
    """
    Check a normalized value against the FieldDefinition's rules.

    Returns True if valid, False otherwise.
    """
    rules = field.validation_rules or {}

    # Enum: value must be in allowed_values
    if field.field_type == "enum":
        if field.allowed_values and value not in field.allowed_values:
            return False

    # Numeric bounds
    if field.field_type in ("int", "float"):
        if "min" in rules and value < rules["min"]:
            return False
        if "max" in rules and value > rules["max"]:
            return False

    # String length bounds
    if field.field_type == "string":
        if "min_length" in rules and len(value) < rules["min_length"]:
            return False
        if "max_length" in rules and len(value) > rules["max_length"]:
            return False

    return True







# ---------------------------------------------------------------------------
# PARSING HELPERS
# ---------------------------------------------------------------------------

def _parse_facts(raw: str, industry_id: str) -> dict[str, dict[str, Any]]:
    """
    Parse LLM output into {'hard_constraints': {...}, 'soft_preferences': {...}}.
    Both dicts validated against industry's FieldDefinitions.
    """
    empty = {"hard_constraints": {}, "soft_preferences": {}}

    if not raw:
        return empty

    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match is None:
        logger.debug("Fact extraction returned non-JSON; ignoring")
        return empty

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.debug("Fact extraction returned malformed JSON; ignoring")
        return empty

    if not isinstance(parsed, dict):
        return empty

    industry_fields = {
        f.canonical_name: f
        for f in get_fields_for_industry(industry_id)
    }

    def _validate_bucket(bucket: Any) -> dict[str, Any]:
        if not isinstance(bucket, dict):
            return {}
        result: dict[str, Any] = {}
        for key, value in bucket.items():
            field = industry_fields.get(key)
            if field is None:
                continue
            normalized = _normalize_value(value, field)
            if normalized is None:
                continue
            if not _validate_value(normalized, field):
                continue
            result[key] = normalized
        return result

    return {
        "hard_constraints": _validate_bucket(parsed.get("hard_constraints", {})),
        "soft_preferences": _validate_bucket(parsed.get("soft_preferences", {})),
    }