"""
Field lookup = industry fields + built-in fields that apply to every industry.

Built-in `price` lets customers state a budget ("under 500 EGP") in any
industry. Extraction and recommendation both use this module, so a budget
is extracted AND enforced consistently. Industry configs stay untouched;
an industry that defines its own `price` field takes precedence.
"""

from app.industries.registry import get_field as _registry_get_field
from app.industries.registry import get_fields_for_industry as _registry_fields
from app.schemas.models import FieldDefinition

PRICE_FIELD = FieldDefinition.model_construct(
    canonical_name="price",
    display_name="Budget",
    field_type="float",
    comparison_operator="<=",
    unit=None,
    allowed_values=None,
    extraction_hint=(
        "Only when the customer states a maximum budget or price limit. "
        "Output just the number in the store's currency, e.g. 500."
    ),
    normalization_rules={"strip_units": True},
    validation_rules={"min": 0},
)

BUILTIN_FIELDS: list = [PRICE_FIELD]


def get_fields_for_industry(industry_id: str) -> list:
    fields = list(_registry_fields(industry_id))
    names = {f.canonical_name for f in fields}
    return fields + [f for f in BUILTIN_FIELDS if f.canonical_name not in names]


def get_field(industry_id: str, canonical_name: str):
    """Industry field first; built-in fallback. Raises KeyError if neither exists."""
    try:
        return _registry_get_field(industry_id, canonical_name)
    except KeyError:
        for f in BUILTIN_FIELDS:
            if f.canonical_name == canonical_name:
                return f
        raise