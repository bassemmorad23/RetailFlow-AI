"""
Registry for FieldDefinitions and IndustryConfigs.

WHY THIS EXISTS:
Every pipeline module that needs to know "what fields does this industry
use?" or "what's the definition for 'ram_gb' in smartphones?" comes here.
Centralizing the lookup means:
  - One place to catch typos in industry/field IDs (fail loudly at import)
  - One place to enforce consistency between IndustryConfig.field_ids
    and actual FieldDefinition instances
  - One place to swap the backing store later (Python -> MongoDB) if needed

ARCHITECTURE — FULL INDUSTRY ISOLATION:
Fields are scoped to a single industry. Two industries that both define
a field named "color" have TWO separate FieldDefinitions — one per
industry, with potentially different aliases, allowed values, or
validation rules. This matches the folder structure (fashion/fields.py,
smartphones/fields.py) where each industry owns its own definitions.

Consequence: field lookups REQUIRE an industry_id. There is no
"the color field" — only "the color field for fashion" or "the color
field for smartphones".

WHY NOT MONGODB YET:
Per Q4 decision, FieldDefinitions live in Python code for V1. This means
adding a new industry requires a code deploy — an acceptable trade-off
until merchants demand custom fields. Migration path is preserved:
this registry can be swapped for a MongoDB-backed one later without
changing any calling code, as long as the public API (get_field,
get_industry, etc.) stays the same.

HOW INDUSTRIES REGISTER:
Each industry folder (fashion/, smartphones/, laptops/) exposes:
  - FIELDS: list[FieldDefinition]  (in fields.py)
  - CONFIG: IndustryConfig         (in config.py)

The registry imports these from each industry module at startup and
validates them once. Failure at import time is intentional — we'd
rather crash on deploy than serve wrong data.
"""

from types import MappingProxyType

from app.schemas.models import FieldDefinition, IndustryConfig


# ---------------------------------------------------------------------------
# Internal storage
# ---------------------------------------------------------------------------

# Nested: industry_id -> canonical_name -> FieldDefinition
# Populated at module load time by _load_all_industries().
# Never accessed directly outside this module — use get_field() etc.
_FIELDS_BY_INDUSTRY: dict[str, dict[str, FieldDefinition]] = {}

# industry_id -> IndustryConfig
_INDUSTRIES_BY_ID: dict[str, IndustryConfig] = {}


# ---------------------------------------------------------------------------
# Public read-only views
# ---------------------------------------------------------------------------

# Read-only public view of the industries dict. Downstream code can
# iterate this directly (e.g. `for industry_id in INDUSTRIES:`), but
# any attempt to mutate raises TypeError.
INDUSTRIES: MappingProxyType[str, IndustryConfig] = MappingProxyType(_INDUSTRIES_BY_ID)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RegistryError(Exception):
    """
    Raised when the registry detects a configuration inconsistency
    at load time. Fatal by design — the app should not start with
    a broken industry config.
    """


# ---------------------------------------------------------------------------
# Internal registration helpers (called only during _load_all_industries)
# ---------------------------------------------------------------------------


def _register_field(industry_id: str, field: FieldDefinition) -> None:
    """
    Add a FieldDefinition to the registry, scoped to a specific industry.

    Under full-isolation architecture, the same canonical_name can
    legitimately appear in multiple industries (e.g. "color" exists in
    both fashion and smartphones). What we reject is duplicate
    canonical_name WITHIN the same industry — that would silently
    override the earlier definition and cause impossible-to-debug bugs.

    Also validates that the field's own 'industries' list actually
    includes the industry it's being registered under. This catches
    the case where someone copies a field between industry files but
    forgets to update its `industries` attribute.
    """
    # Ensure the field acknowledges this industry.
    if industry_id not in field.industries:
        raise RegistryError(
            f"Attempting to register field '{field.canonical_name}' "
            f"under industry '{industry_id}', but the field's own "
            f"'industries' list is {field.industries}. Add "
            f"'{industry_id}' to the field's industries, or register "
            f"it under the correct industry."
        )

    # Ensure no duplicate within this industry.
    industry_fields = _FIELDS_BY_INDUSTRY.setdefault(industry_id, {})
    if field.canonical_name in industry_fields:
        raise RegistryError(
            f"Duplicate FieldDefinition for '{field.canonical_name}' "
            f"within industry '{industry_id}'. Each canonical_name "
            f"may appear at most once per industry."
        )

    industry_fields[field.canonical_name] = field


def _register_industry(config: IndustryConfig) -> None:
    """
    Add an IndustryConfig to the registry.

    Rejects duplicates by industry_id. Validates that every field_id
    the config references exists as a real FieldDefinition already
    registered under this industry.

    Order of operations matters: _load_all_industries() registers
    fields BEFORE the industry config, so by the time this runs the
    cross-check can look up any referenced field.
    """
    if config.industry_id in _INDUSTRIES_BY_ID:
        raise RegistryError(
            f"Duplicate IndustryConfig for industry_id "
            f"'{config.industry_id}'"
        )

    industry_fields = _FIELDS_BY_INDUSTRY.get(config.industry_id, {})

    # Every field_id must resolve to a real FieldDefinition registered
    # under THIS industry. A typo or missing field means downstream
    # code would silently skip an intended field.
    missing = [
        field_id
        for field_id in config.field_ids
        if field_id not in industry_fields
    ]
    if missing:
        raise RegistryError(
            f"IndustryConfig '{config.industry_id}' references field_ids "
            f"that are not registered under this industry: {missing}. "
            f"Register the FieldDefinitions in "
            f"app/industries/{config.industry_id}/fields.py before "
            f"registering the config, or fix the typo. "
            f"Available fields for this industry: "
            f"{sorted(industry_fields.keys())}"
        )

    _INDUSTRIES_BY_ID[config.industry_id] = config


# ---------------------------------------------------------------------------
# Industry loading
# ---------------------------------------------------------------------------


def _load_all_industries() -> None:
    """
    Import every industry module and register its fields + config.

    Deliberately explicit rather than auto-discovering folders —
    reading this function tells you exactly which industries exist.
    Adding a new industry means adding one block of import + register
    lines here.

    Order matters inside each block: fields must be registered BEFORE
    the industry config that references them, because _register_industry
    cross-checks against already-registered fields.
    """
    # Fashion
    from app.industries.fashion import fields as _fashion_fields
    from app.industries.fashion import config as _fashion_config
    for field in _fashion_fields.FIELDS:
        _register_field("fashion", field)
    _register_industry(_fashion_config.CONFIG)

    # Smartphones
    from app.industries.smartphones import fields as _smartphones_fields
    from app.industries.smartphones import config as _smartphones_config
    for field in _smartphones_fields.FIELDS:
        _register_field("smartphones", field)
    _register_industry(_smartphones_config.CONFIG)

    # Laptops — module not built yet. When adding:
    #   from app.industries.laptops import fields as _laptops_fields
    #   from app.industries.laptops import config as _laptops_config
    #   for field in _laptops_fields.FIELDS:
    #       _register_field("laptops", field)
    #   _register_industry(_laptops_config.CONFIG)


# ---------------------------------------------------------------------------
# Public lookup API
# ---------------------------------------------------------------------------


def get_field(industry_id: str, canonical_name: str) -> FieldDefinition:
    """
    Look up a FieldDefinition scoped to a specific industry.

    Under full-isolation architecture, a canonical_name is unique
    only within its industry — two industries can each have their
    own "color" field with different rules. The caller must state
    which industry's field they want.

    Raises KeyError with a helpful message if the industry or field
    is not registered. Never returns None — that would let bugs
    propagate silently through downstream code.
    """
    if industry_id not in _FIELDS_BY_INDUSTRY:
        available = sorted(_FIELDS_BY_INDUSTRY.keys())
        raise KeyError(
            f"No fields registered for industry '{industry_id}'. "
            f"Registered industries: {available}"
        )

    industry_fields = _FIELDS_BY_INDUSTRY[industry_id]
    if canonical_name not in industry_fields:
        available = sorted(industry_fields.keys())
        raise KeyError(
            f"No FieldDefinition '{canonical_name}' registered for "
            f"industry '{industry_id}'. Available fields for this "
            f"industry: {available}"
        )
    return industry_fields[canonical_name]


def get_fields_for_industry(industry_id: str) -> list[FieldDefinition]:
    """
    Return every FieldDefinition scoped to the given industry.

    Uses the IndustryConfig.field_ids as the source of truth — a field
    physically present in fields.py but NOT referenced by the industry's
    config will be excluded. Config wins because it represents the
    intentional choice of which fields to expose for this industry.

    Returns them in the order defined by config.field_ids, so callers
    can rely on stable ordering (useful for prompt building).
    """
    if industry_id not in _INDUSTRIES_BY_ID:
        available = sorted(_INDUSTRIES_BY_ID.keys())
        raise KeyError(
            f"No IndustryConfig registered for industry_id "
            f"'{industry_id}'. Registered industries: {available}"
        )

    config = _INDUSTRIES_BY_ID[industry_id]
    industry_fields = _FIELDS_BY_INDUSTRY[industry_id]
    return [industry_fields[field_id] for field_id in config.field_ids]


def get_industry(industry_id: str) -> IndustryConfig:
    """
    Look up an IndustryConfig by its ID.

    Same failure semantics as get_field(): raise, never return None.
    """
    if industry_id not in _INDUSTRIES_BY_ID:
        available = sorted(_INDUSTRIES_BY_ID.keys())
        raise KeyError(
            f"No IndustryConfig registered for industry_id "
            f"'{industry_id}'. Registered industries: {available}"
        )
    return _INDUSTRIES_BY_ID[industry_id]


def list_industries() -> list[str]:
    """
    Return all registered industry IDs, sorted alphabetically.
    Useful for merchant onboarding dropdowns and for tests.
    """
    return sorted(_INDUSTRIES_BY_ID.keys())


def list_fields_for_industry(industry_id: str) -> list[str]:
    """
    Return all canonical field names registered under the given
    industry, sorted alphabetically.

    Useful for validation, debugging, and tests. Distinct from
    get_fields_for_industry() which respects the config's field
    ordering — this one is a flat sorted list of names.
    """
    if industry_id not in _FIELDS_BY_INDUSTRY:
        available = sorted(_FIELDS_BY_INDUSTRY.keys())
        raise KeyError(
            f"No fields registered for industry '{industry_id}'. "
            f"Registered industries: {available}"
        )
    return sorted(_FIELDS_BY_INDUSTRY[industry_id].keys())


def list_all_fields() -> dict[str, list[str]]:
    """
    Return every registered field name, grouped by industry.

    Debugging aid — shows the whole registry contents at a glance.
    Not for production hot paths; each call builds a fresh dict.
    """
    return {
        industry_id: sorted(fields.keys())
        for industry_id, fields in _FIELDS_BY_INDUSTRY.items()
    }


# ---------------------------------------------------------------------------
# Test support
# ---------------------------------------------------------------------------


def _reset_registry() -> None:
    """
    Wipe the registry and re-run initialization.

    FOR TESTS ONLY. Never call in production code.

    Test isolation problem: because the registry loads at module
    import time and stores state in module-level dicts, tests that
    register additional industries (to test error handling, edge
    cases, etc.) leak state into every subsequent test.

    Calling _reset_registry() at the start of a test guarantees a
    known-clean state. Test teardown can call it again to prevent
    leaking test data into other tests.

    Naming convention: leading underscore signals "not part of the
    public API". Production code should never need this.
    """
    _FIELDS_BY_INDUSTRY.clear()
    _INDUSTRIES_BY_ID.clear()
    _load_all_industries()


# ---------------------------------------------------------------------------
# Module initialization
# ---------------------------------------------------------------------------

# Load everything at import time. If any industry has a bad config,
# the app fails to start — better than silently serving wrong data.
_load_all_industries()