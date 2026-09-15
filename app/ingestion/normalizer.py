"""
Value normalization based on FieldDefinition rules.

WHY THIS EXISTS:
Merchant data is messy. "8 GB", "8gb", "8" all mean the same thing.
"Large", "L", "large" all mean size L. This module applies each
field's normalization_rules to produce canonical values.

DESIGN:
- Reuses same logic as fact_extractor._normalize_value (shared canonical form)
- Runs at ingestion time to store clean values
- Returns None if value cannot be normalized (caller decides what to do)
"""

import re
from typing import Any

from app.schemas.models import FieldDefinition


def normalize(value: Any, field: FieldDefinition) -> Any:
    """
    Coerce and clean a raw value against its FieldDefinition.
    Returns normalized value, or None if impossible.
    """
    if value is None:
        return None

    rules = field.normalization_rules or {}

    # ---- Numeric types ----
    if field.field_type == "int":
        try:
            if isinstance(value, str):
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

    # ---- Boolean ----
    if field.field_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("true", "yes", "1", "y", "in stock", "available"):
                return True
            if s in ("false", "no", "0", "n", "out of stock", "unavailable"):
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return None

    # ---- String / Enum ----
    if field.field_type in ("string", "enum"):
        if not isinstance(value, str):
            value = str(value)
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

        # Alias mapping (case-insensitive lookup)
        aliases = rules.get("aliases", {})
        if aliases:
            lower_map = {k.lower(): v for k, v in aliases.items()}
            if s.lower() in lower_map:
                s = lower_map[s.lower()]

        if not s:
            return None
        return s

    return None