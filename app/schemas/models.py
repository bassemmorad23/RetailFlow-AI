"""Pydantic data models used across the StoreFlow AI pipeline."""

from typing import Any, Literal, Optional
from enum import Enum
from pydantic import ConfigDict, BaseModel, Field, field_validator
import re
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EmotionLabel(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SADNESS = "sadness"
    ANGRY = "angry"
    CONFUSED = "confused"
    EXCITED = "excited"


class IntentLabel(str, Enum):
    """Application-level intent labels used across the pipeline."""

    BROWSING = "browsing"
    ASKING_DETAILS = "asking_details"
    ASKING_PRICE = "asking_price"
    ASKING_AVAILABILITY = "asking_availability"
    ASKING_SIZE_ADVICE = "asking_size_advice"
    WANTS_RECOMMENDATION = "wants_recommendation"
    READY_TO_BUY = "ready_to_buy"
    COMPLAINT = "complaint"
    COMPARE_PRODUCTS = "compare_products"
    ORDER_STATUS = "order_status"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------


# Where the customer message came from.
Channel = Literal["web", "instagram", "messenger", "whatsapp"]


# Supported primitive types for FieldDefinition values.
# Used by the multi-industry configuration system.
FieldType = Literal["int", "float", "string", "enum", "boolean"]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


_ID_FIELDS = ("conversation_id", "customer_id", "store_id")


def _reject_blank_identifiers(value: str, field_name: str) -> str:
    """
    Ensure identifier-like string fields are non-empty and non-whitespace.

    Empty or whitespace-only identifiers silently corrupt downstream
    memory lookups and multi-tenant scoping; reject them at the
    validation boundary.
    """
    if value is None or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MAX_MESSAGE_LENGTH = 2000

def _clean_customer_text(value: str) -> str:
    """
    Sanitize the customer's message before it enters the pipeline.

    - Strip leading/trailing whitespace.
    - Remove ASCII control characters (keeping \\n and \\t).
    - Cap the length to protect downstream tokenization + LLM cost.
    """
    if value is None:
        raise ValueError("text must be a non-empty string")

    cleaned = value.strip()
    cleaned = _CONTROL_CHAR_RE.sub("", cleaned)

    if not cleaned:
        raise ValueError("text must contain non-whitespace characters")

    # Hard cap: 2000 chars is well beyond any legitimate chat message
    # and protects us from paste-bombs.
    if len(cleaned) > MAX_MESSAGE_LENGTH:
        cleaned = cleaned[:MAX_MESSAGE_LENGTH]

    return cleaned


# ---------------------------------------------------------------------------
# Incoming: message from the customer
# ---------------------------------------------------------------------------


class CustomerMessage(BaseModel):
    """
    A single message from a customer.

    This is the input to the orchestrator, whatever the surface
    (widget, Instagram, WhatsApp) it arrived through.
    """

    conversation_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    store_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    channel: Channel

    @field_validator("conversation_id", "customer_id", "store_id")
    @classmethod
    def _identifiers_must_be_non_blank(cls, v: str, info) -> str:
        return _reject_blank_identifiers(v, info.field_name)

    @field_validator("text")
    @classmethod
    def _text_must_be_clean(cls, v: str) -> str:
        return _clean_customer_text(v)

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Emotion + intent results
# ---------------------------------------------------------------------------


class EmotionResult(BaseModel):
    """Emotion detected from the customer's message."""

    label: EmotionLabel
    confidence: float = Field(ge=0.0, le=1.0)
    scores: dict[str, float] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class IntentResult(BaseModel):
    """Intent detected from the customer's message."""

    label: IntentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    scores: dict[str, float] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

# ---------------------------------------------------------------------------
# Product Variant Schema
# ---------------------------------------------------------------------------
class Variant(BaseModel):
    """
    A specific purchasable version of a Product.

    Variants exist for products where the customer picks size, color,
    storage, etc. Each variant has its own SKU, price, and stock status.
    Attributes and specifications may differ per variant (e.g. iPhone
    128GB vs 256GB have different storage_gb specs and different prices).

    For products WITHOUT variants (single SKU, single price), the parent
    Product's price/stock are used and variants is an empty list.
    """

    sku: str = Field(min_length=1, description="Unique identifier per variant.")
    price: float = Field(ge=0, description="Variant's price. May differ from parent.")
    stock_available: bool = Field(default=True, description="Whether variant is in stock.")
    stock_status: Literal["in_stock", "out_of_stock", "unknown"] = Field(
        default="unknown",
        description="Synced stock status. 'unknown' = no stock data; never shown as available.",
    )
    stock_quantity: int | None = Field(
        default=None, ge=0,
        description="Synced quantity when the platform tracks inventory. Never shown to customers.",
    )
    external_id: str | None = Field(default=None, description="Platform id of this variant (Shopify variant GID / WooCommerce variation id).")
    
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Soft attributes like color, size — canonical field names.",
    )
    specifications: dict[str, Any] = Field(
        default_factory=dict,
        description="Technical specs that differ per variant (e.g. storage_gb).",
    )
    image_url: str | None = Field(default=None, description="Variant-specific image.")

    model_config = ConfigDict(extra="forbid")




# ---------------------------------------------------------------------------
# Product + recommendation
# ---------------------------------------------------------------------------


class Product(BaseModel):
    """
    A single product carried by a store.

        MULTI-INDUSTRY DESIGN:
    - Industry is NOT stored on the product — it lives on
      store_settings (one store = one industry). Pipeline code looks
      it up via store_id when needed.
    - `category` locates this product within its industry's category
      list (e.g. "dresses" for fashion, "flagship" for smartphones).
      Must appear in the industry's IndustryConfig.categories.
      Optional for now (V1 refactor is additive).
    - `attributes` holds customer-facing traits (color, size, material).
      Keys should be canonical FieldDefinition names.
    - `specifications` holds technical specs (ram_gb, cpu, storage_gb).
      Keys should be canonical FieldDefinition names.
    - `original_data` preserves the merchant's raw input verbatim
      (original column names, original values) so nothing is lost
      when the ingestion pipeline normalizes into canonical form.

    LEGACY FIELDS:
    - `size` and `color` remain as top-level fields for backward
      compatibility with existing fashion data. New industries should
      write these into `attributes` instead. These will be removed in
      Week 4 (R8 deprecation cleanup) once all data is migrated.
    """

    # ---- Identity ----

    product_id: str = Field(min_length=1)
    store_id: str = Field(min_length=1)

    
    category: Optional[str] = Field(
        default=None,
        description=(
            "The product's category within its industry (e.g. 'dresses' "
            "for fashion, 'flagship' for smartphones). Must appear in "
            "the industry's IndustryConfig.categories list. Optional "
            "during V1 refactor."
        ),
    )

    # ---- Core product data ----

    name: str = Field(min_length=1)
    description: Optional[str] = None
    price: float = Field(ge=0)
    currency: str = "USD"

    # ---- Legacy top-level attributes (deprecated, migrating to `attributes`) ----

    size: Optional[str] = None
    color: Optional[str] = None

    # ---- Multi-industry dynamic fields ----

    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Customer-facing dynamic attributes (e.g. Fashion: "
            "{'color': 'red', 'material': 'cotton'}; Smartphones: "
            "{'color': 'black'}). Keys must be canonical FieldDefinition "
            "names where possible. Unknown keys are allowed under "
            "lenient V1 rules but ignored by extraction and filtering."
        ),
    )

    specifications: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Technical specifications (e.g. Smartphones: "
            "{'ram_gb': 8, 'storage_gb': 256}; Laptops: "
            "{'ram_gb': 16, 'cpu': 'Intel i7', 'gpu': 'RTX 3060'}). "
            "Keys must be canonical FieldDefinition names where "
            "is_specification=True."
        ),
    )

    original_data: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Verbatim merchant-provided data preserved before "
            "canonicalization (original column names, original values). "
            "Ensures no information is lost when the ingestion pipeline "
            "converts merchant fields to canonical schema. Useful for "
            "debugging mapping issues and for showing merchants what "
            "they originally sent."
        ),
    )
    
    
    variants: list[Variant] = Field(
        default_factory=list,
        description=(
            "Purchasable variations of this product. Empty for simple "
            "products (single SKU). Non-empty for products where "
            "customer picks size/color/storage — recommender filters "
            "and returns specific variants."
        ),
    )
    
    
    
    
    
    

    # ---- Stock + media ----

    stock_available: bool = True
    
    stock_status: Literal["in_stock", "out_of_stock", "unknown"] = Field(
        default="unknown",
        description="Synced stock status. 'unknown' = no stock data; never shown as available.",
    )
    stock_quantity: int | None = Field(
        default=None, ge=0,
        description="Synced quantity when the platform tracks inventory. Never shown to customers.",
    )
    external_id: str | None = Field(default=None, description="Platform id for live stock of a simple product.")
    external_parent_id: str | None = Field(default=None, description="Platform parent id (Shopify product GID / WooCommerce parent product id).")
    
    image_url: Optional[str] = None

    model_config = ConfigDict(extra="forbid")
        


class ProductRecommendation(BaseModel):
    """..."""
    product_id: str
    name: str
    price: float
    reason: str
    stock: Literal["in_stock", "low_stock", "out_of_stock", "unknown"] = "unknown"

    # NEW — populated when a specific variant matches
    variant_sku: str | None = Field(
        default=None,
        description=(
            "SKU of the matching variant, if this product has variants "
            "and one specifically fits the customer's constraints. "
            "None for simple products or when no specific variant fits."
        ),
    )
    variant_attrs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Attributes of the matching variant (e.g. {size: L, color: blue}). "
            "Empty for simple products."
        ),
    )

    model_config = ConfigDict(extra="forbid")
        
        
        
class ComparisonRow(BaseModel):
    """
    One row in a comparison — one field across all products being compared.

    Values dict maps product_id to that product's value for this field.
    A None value means the product doesn't have this field defined
    (never invented — always shown as unavailable in the response).
    """

    field_name: str = Field(description="Canonical field name (e.g. 'ram_gb', 'color').")
    display_name: str = Field(description="Human-readable name for the field.")
    values: dict[str, Any] = Field(
        default_factory=dict,
        description="{product_id: value or None}. None = field not present on that product.",
    )

    model_config = ConfigDict(extra="forbid")


class ComparisonResult(BaseModel):
    """
    Structured comparison of 2-3 products from the same store.

    Enforces:
    - 2-3 products maximum
    - All products from the same store (verified at build time)
    - not_found tracks mentions that didn't match catalog
    - alternatives suggests replacement products from same store

    Response generator receives this and produces natural-language output.
    Never sees fabricated data — missing values are None.
    """

    products: list[Product] = Field(
        description="Found products being compared (2-3, from same store).",
    )
    rows: list[ComparisonRow] = Field(
        default_factory=list,
        description="One row per comparison field, in industry-preferred order.",
    )
    not_found: list[str] = Field(
        default_factory=list,
        description="Product mentions from customer that didn't match anything in catalog.",
    )
    alternatives: list[Product] = Field(
        default_factory=list,
        description="Suggested alternatives from same store for not-found products.",
    )

    model_config = ConfigDict(extra="forbid")




        
        
class RetrievedChunk(BaseModel):
    """
    A single result from RAG retrieval — a semantically relevant chunk
    of content (product description, FAQ, policy) that may inform the
    response or seed a product recommendation.

    Kept lightweight — just enough for scoring and lookup. Full product
    data is fetched separately via product_store when needed.
    """

    source: str = Field(
        description=(
            "Identifier of the source document/product. For products, "
            "this is the product_id (used as the fetch key)."
        ),
    )
    content: str = Field(description="The text content of the chunk.")
    score: float = Field(
        ge=0.0, le=1.0,
        description="RAG similarity score (0.0 = irrelevant, 1.0 = perfect match).",
    )
    name: str | None = Field(default=None, description="Product name if this chunk is a product.")
    price: float | None = Field(default=None, description="Product price if applicable.")

    model_config = ConfigDict(extra="forbid")

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class ConversationTurn(BaseModel):
    """One turn (customer or agent) in a conversation."""

    role: Literal["customer", "agent"]
    text: str
    timestamp: str | None = None

    model_config = ConfigDict(extra="forbid")


class KnownFacts(BaseModel):
    """
    Structured facts we've inferred about the customer over time.
    
    Legacy top-level fields kept for backward compatibility with
    existing MongoDB data. New industry-specific facts (ram_gb, size,
    color, etc.) land in `preferences` dict, keyed by canonical
    FieldDefinition name.
    """
    # Legacy fields — kept until existing data migrated (Week 4 R8)
    preferred_size: str | None = None
    preferred_color: str | None = None
    budget_max: float | None = None
    mentioned_products: list[str] = Field(default_factory=list)

    # Dynamic per-industry preferences (canonical_name -> value)
    preferences: dict[str, Any] = Field(default_factory=dict)
    
    # Hard constraints — MUST be satisfied by recommended products
    hard_constraints: dict[str, Any] = Field(default_factory=dict)
    
    # Soft preferences — influence ranking, not filtering
    soft_preferences: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class MemoryState(BaseModel):
    """The full memory context for a conversation."""

    conversation_id: str
    history: list[ConversationTurn] = Field(default_factory=list)
    known_facts: KnownFacts = Field(default_factory=KnownFacts)

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Outgoing: reply to the customer
# ---------------------------------------------------------------------------


class AgentReply(BaseModel):
    """The agent's response to a customer message."""

    reply_text: str
    emotion: EmotionResult
    intent: IntentResult
    recommendations: list[ProductRecommendation] = Field(default_factory=list)
    retrieved_context: list[RetrievedChunk] = Field(default_factory=list)
    conversation_id: str

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Multi-industry configuration schemas
# ---------------------------------------------------------------------------


class FieldDefinition(BaseModel):
    """
    Definition of a single canonical field used by the multi-industry
    configuration system.

    A FieldDefinition captures everything StoreFlow needs to know about
    one product attribute or one customer preference:
      - What it's called canonically ("ram_gb")
      - What it's called by real people ("RAM", "Memory", "الرام")
      - Its type ("int", "string", "enum", etc.)
      - Its allowed values (for enums like clothing sizes)
      - Its unit ("GB", "cm", "kg")
      - How to clean up incoming values (normalization)
      - How to reject invalid values (validation)
      - Which industry uses it (scoping)
      - Whether it's a customer-facing attribute or a technical spec
      - A hint for LLM-based fact extraction

    Once defined, this object drives every downstream layer:
    ingestion mapping, normalization, validation, extraction prompts,
    and recommendation filtering.

    DESIGN PRINCIPLES:
    - One canonical name per concept (e.g. "ram_gb" — not "ram" or "RAM").
    - Aliases capture how merchants and customers actually refer to it
      ("RAM", "Memory", "الرام").
    - Normalization runs at read/use time — original merchant data is
      preserved in Product.original_data.
    - Industry scoping ensures fashion doesn't accidentally accept "cpu"
      and vice versa.

    WHY THIS LIVES HERE:
    Same file as other Pydantic schemas so all data contracts have one
    import path. Industry-specific FieldDefinition INSTANCES live in
    app/industries/<industry>/fields.py.
    """

    # ---- Identity ----

    canonical_name: str = Field(
        min_length=1,
        description=(
            "The single canonical name used internally by StoreFlow. "
            "Convention: lowercase snake_case, unit-suffixed for numeric "
            "quantities (e.g. 'ram_gb', 'screen_size_inches'). "
            "This is the name that appears inside Product.attributes and "
            "Product.specifications."
        ),
    )

    display_name: str = Field(
        min_length=1,
        description=(
            "Human-readable name shown in the merchant dashboard and, "
            "when appropriate, in AI-generated replies to customers "
            "(e.g. 'RAM', 'Storage', 'Color'). Not necessarily the same "
            "as canonical_name."
        ),
    )

    description: str = Field(
        min_length=1,
        description=(
            "Plain-language description of what this field represents. "
            "Used in LLM extraction prompts and in the merchant "
            "dashboard's field-help text."
        ),
    )

    # ---- Type + values ----

    field_type: FieldType = Field(
        description=(
            "Primitive type of the field's value. Drives type coercion "
            "during normalization and type checking during validation."
        ),
    )

    allowed_values: Optional[list[str]] = Field(
        default=None,
        description=(
            "For enum fields, the exhaustive list of allowed values "
            "(e.g. ['S', 'M', 'L', 'XL']). Must be None for non-enum "
            "field types. Normalization will attempt to coerce input "
            "into one of these values."
        ),
    )

    unit: Optional[str] = Field(
        default=None,
        description=(
            "Physical unit of the field's value, if applicable "
            "(e.g. 'GB', 'inches', 'cm', 'kg'). Used for display and "
            "for detecting alternate units during normalization "
            "(e.g. '8 GB' vs '8GB' vs '8gb')."
        ),
    )
    comparison_operator: Literal["==", ">=", "<=", "range"] = Field(
        default="==",
        description=(
            "How to compare product values against customer constraints. "
            "'>=' means bigger is better (RAM, storage, screen_size). "
            "'<=' means smaller is better (price, weight). "
            "'==' means exact match (color, brand, OS). "
            "'range' means match within a tolerance of the requested value."
        ),
    )
    

    # ---- Aliases (multilingual, multi-format) ----

    aliases: list[str] = Field(
        default_factory=list,
        description=(
            "Alternative names the field is known by in the wild — "
            "merchant column headers, customer message phrasings, "
            "and translations in supported languages. Case-insensitive "
            "matching is applied at lookup time. Examples for 'ram_gb': "
            "['RAM', 'Memory', 'ram', 'memory', 'الرام', 'الذاكرة']."
        ),
    )

    # ---- Normalization + validation ----

    normalization_rules: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Rules applied to raw input to produce a canonical value. "
            "Runs at read/use time, not write time — original merchant "
            "data is preserved in Product.original_data. Common keys: "
            "'trim' (bool), 'case' ('lower' | 'upper' | 'title'), "
            "'strip_units' (bool — drops trailing unit strings before "
            "type coercion). Downstream code is free to add its own "
            "keys; unknown keys are ignored."
        ),
    )

    validation_rules: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Rules that determine whether a normalized value is "
            "acceptable. Common keys: 'min' (numeric), 'max' (numeric), "
            "'min_length' (int, for strings), 'max_length' (int, for "
            "strings), 'pattern' (regex). A value failing any rule is "
            "rejected by the compatibility validator."
        ),
    )

    # ---- Industry scoping ----

    industries: list[str] = Field(
        min_length=1,
        description=(
            "The industries this field applies to (e.g. ['smartphones', "
            "'laptops']). Under lenient V1 rules, a product carrying a "
            "field outside its industry's scope is allowed but ignored "
            "by extraction, filtering, and recommendation. Must include "
            "at least one industry — a field with no industry is dead "
            "code."
        ),
    )

    # ---- Semantics ----

    is_specification: bool = Field(
        default=False,
        description=(
            "True if the field represents a technical specification "
            "shown to customers (cpu, ram_gb, storage_gb). False if it "
            "represents a customer-facing attribute (color, size). "
            "Determines whether values land in Product.specifications "
            "or Product.attributes."
        ),
    )

    extraction_hint: Optional[str] = Field(
        default=None,
        description=(
            "Optional hint injected into the fact-extraction LLM prompt "
            "to help the model identify this field in customer "
            "messages. Example for 'preferred_size': "
            "'A clothing size the customer explicitly stated they "
            "want, in S/M/L/XL or numeric form.'"
        ),
    )

    # ---- Validators ----

    @field_validator("canonical_name")
    @classmethod
    def _canonical_name_must_be_snake_case(cls, v: str) -> str:
        """
        Canonical names must be lowercase snake_case (letters, digits,
        underscores) so they're safe as dict keys, JSON field names,
        and Python identifiers.
        """
        if not v.replace("_", "").isalnum():
            raise ValueError(
                f"canonical_name must be lowercase alphanumeric with "
                f"underscores only, got '{v}'"
            )
        if v != v.lower():
            raise ValueError(
                f"canonical_name must be lowercase, got '{v}'"
            )
        return v

    @field_validator("allowed_values")
    @classmethod
    def _allowed_values_only_for_enum(
        cls, v: Optional[list[str]], info
    ) -> Optional[list[str]]:
        """
        allowed_values only makes sense for enum-typed fields. Setting
        it on other types is almost always a bug.
        """
        field_type = info.data.get("field_type")
        if v is not None and field_type != "enum":
            raise ValueError(
                f"allowed_values may only be set when field_type='enum', "
                f"got field_type='{field_type}'"
            )
        if v is not None and len(v) == 0:
            raise ValueError(
                "allowed_values must be non-empty when set"
            )
        return v

    @field_validator("industries")
    @classmethod
    def _industries_must_be_non_empty_and_snake_case(
        cls, v: list[str]
    ) -> list[str]:
        """
        Every industry ID must be a non-empty lowercase snake_case
        string. Enforced here rather than at the registry layer so
        malformed configs are caught at class instantiation.
        """
        if not v:
            raise ValueError("industries must contain at least one entry")
        for industry_id in v:
            if not industry_id:
                raise ValueError(
                    "industry entries must be non-empty strings"
                )
            if industry_id != industry_id.lower():
                raise ValueError(
                    f"industry IDs must be lowercase, got '{industry_id}'"
                )
        return v

    # Reject extra fields so typos in FieldDefinition construction fail loudly.
    model_config = ConfigDict(extra="forbid")
        
        
# ---------------------------------------------------------------------------
# Industry Configuration
# Defines the configuration and AI behavior for each retail industry.
# ---------------------------------------------------------------------------



class IndustryConfig(BaseModel):
    """
    Configuration for a single retail industry (e.g. "fashion",
    "smartphones", "laptops").

    An IndustryConfig ties together:
      - The industry's identity (id + display name)
      - Its product categories (used for filtering + organization)
      - AI context (system prompt injection for this industry's LLM calls)
      - Selling points (things the AI should highlight when recommending)
      - The canonical field names that apply to this industry

    Once loaded from a registry, this object is passed to every
    pipeline stage that needs industry-specific behavior:
      - Fact extractor uses it to build dynamic prompts
      - Response generator uses ai_context + selling_points
      - Compatibility validator uses field_ids to scope validation
      - Recommender uses field_ids to know which specs to filter on

    DESIGN PRINCIPLES:
    - Configs are pure data — no logic here. Behavior lives in
      registries and pipeline modules.
    - Every field on a product must correspond to a FieldDefinition
      whose canonical_name is listed in this config's field_ids.
    - Adding a new industry is: write a new IndustryConfig instance +
      its FieldDefinitions. No pipeline changes.

    WHY THIS LIVES HERE:
    Same file as FieldDefinition and other Pydantic schemas so all
    data contracts have one import path. Industry-specific INSTANCES
    of this class live in app/industries/<industry>/config.py.
    """

    # ---- Identity ----

    industry_id: str = Field(
        min_length=1,
        description=(
            "Unique lowercase snake_case identifier for the industry "
            "(e.g. 'fashion', 'smartphones', 'laptops'). "
            "This is what stores set on their store_settings document "
            "and what the registry uses as a lookup key."
        ),
    )

    display_name: str = Field(
        min_length=1,
        description=(
            "Human-readable industry name for the merchant dashboard "
            "(e.g. 'Fashion & Apparel', 'Smartphones', 'Laptops & "
            "Computers')."
        ),
    )

    # ---- Product categorization ----

    categories: list[str] = Field(
        min_length=1,
        description=(
            "The product categories that make sense for this industry "
            "(e.g. Fashion: ['dresses', 'tops', 'shoes']; Smartphones: "
            "['flagship', 'mid-range', 'budget', 'accessories']). "
            "Merchant dashboard uses this list to constrain the "
            "'category' field when uploading products. Must contain at "
            "least one category — an industry with no categories can't "
            "hold products."
        ),
    )

    # ---- Field scoping ----

    field_ids: list[str] = Field(
        min_length=1,
        description=(
            "Canonical names of all FieldDefinitions relevant to this "
            "industry (e.g. Smartphones might list ['brand', "
            "'ram_gb', 'storage_gb', 'color', 'screen_size_inches']). "
            "The registry validates at load time that every listed id "
            "resolves to a real FieldDefinition whose 'industries' list "
            "includes this industry's id. Must be non-empty — an "
            "industry with no fields can't describe products."
        ),
    )

    # ---- AI behavior ----

    ai_context: str = Field(
        min_length=1,
        description=(
            "Industry-specific context injected into the response "
            "generator's system prompt. Should be 1-3 sentences that "
            "tell the LLM how to think about this domain "
            "(e.g. Fashion: 'You help customers find clothing that "
            "fits their style, size, and budget. Prioritize the size "
            "guide and current promotions.'). This is where domain "
            "expertise goes."
        ),
    )

    selling_points: list[str] = Field(
        default_factory=list,
        description=(
            "Things worth highlighting when recommending products in "
            "this industry (e.g. Fashion: ['free returns', 'size "
            "guide', 'seasonal collections']; Laptops: ['warranty', "
            "'spec comparison', 'student discount']). The response "
            "generator may weave these into recommendations when "
            "relevant. Optional — an empty list means no automatic "
            "selling points."
        ),
    )

    # ---- Validators ----

    @field_validator("industry_id")
    @classmethod
    def _industry_id_must_be_snake_case(cls, v: str) -> str:
        """
        Industry IDs are used as dict keys, MongoDB values, folder
        names, and Python identifiers. Force lowercase snake_case
        (letters, digits, underscores) so all uses stay compatible.
        """
        if not v.replace("_", "").isalnum():
            raise ValueError(
                f"industry_id must be lowercase alphanumeric with "
                f"underscores only, got '{v}'"
            )
        if v != v.lower():
            raise ValueError(
                f"industry_id must be lowercase, got '{v}'"
            )
        return v

    @field_validator("field_ids")
    @classmethod
    def _field_ids_must_be_unique_and_snake_case(
        cls, v: list[str]
    ) -> list[str]:
        """
        Duplicate field IDs would silently cause a canonical name to
        be listed twice in extraction prompts and filter logic. Reject
        at load time so bugs stay near their source.
        """
        if len(v) != len(set(v)):
            duplicates = [name for name in v if v.count(name) > 1]
            raise ValueError(
                f"field_ids must be unique, found duplicates: "
                f"{sorted(set(duplicates))}"
            )
        for field_id in v:
            if not field_id.replace("_", "").isalnum():
                raise ValueError(
                    f"field_ids must be lowercase snake_case, got "
                    f"'{field_id}'"
                )
            if field_id != field_id.lower():
                raise ValueError(
                    f"field_ids must be lowercase, got '{field_id}'"
                )
        return v

    @field_validator("categories")
    @classmethod
    def _categories_must_be_unique(cls, v: list[str]) -> list[str]:
        """
        Duplicate category names would confuse both the merchant UI
        and downstream filtering. Deduplicate silently is worse than
        rejecting loudly — the merchant's intent isn't obvious.
        """
        if len(v) != len(set(v)):
            duplicates = [c for c in v if v.count(c) > 1]
            raise ValueError(
                f"categories must be unique, found duplicates: "
                f"{sorted(set(duplicates))}"
            )
        return v

    # Reject extra fields so typos in IndustryConfig construction fail loudly.
    model_config = ConfigDict(extra="forbid")
        
        


# ---------------------------------------------------------------------------
# Per-store settings
# ---------------------------------------------------------------------------

#helper function to get the default anchor day for billing
def _default_anchor_day() -> int:
    """Billing anchor = today's day of month, capped at 28 so every month has it."""
    return min(datetime.now(timezone.utc).day, 28)



class StoreSettings(BaseModel):
    """
    Per-store configuration stored in MongoDB.

    One document per store, keyed by store_id. Grows over time as
    new configurable behaviors are added — channels, sync credentials,
    branding, etc. Today it holds industry only.

    WHY THIS EXISTS AS A SCHEMA:
    - Multi-tenant scoping: every store's config is isolated
    - Validation: prevents typos in industry names, malformed channel toggles
    - Type safety: downstream code gets a real object, not a dict-of-anything
    - Migration path: adding fields is a schema change, not a data-shape guess

    WHY industry IS OPTIONAL:
    Lenient V1 rule — stores may exist before industry is assigned
    (e.g. store_002 was created before the multi-industry system).
    Pipeline modules that need industry treat None as "no industry-
    specific behavior, use generic defaults".
    """

    store_id: str = Field(
        min_length=1,
        description=(
            "Multi-tenant identifier — matches store_id used throughout "
            "the pipeline (CustomerMessage, Product, MemoryState, etc.)."
        ),
    )

    industry: Optional[str] = Field(
        default=None,
        description=(
            "The industry this store belongs to (must match an "
            "IndustryConfig.industry_id in the registry, e.g. 'fashion', "
            "'smartphones', 'laptops'). None means no industry assigned "
            "yet — pipeline uses generic behavior. Set once at merchant "
            "onboarding, rarely changed after."
        ),
    )
    
    
    display_name: str = Field(default="", max_length=100, description="Store name shown to the merchant.")
    country: str | None = Field(default=None, description="ISO 3166-1 alpha-2, e.g. EG.")
    currency: str | None = Field(default=None, description="ISO 4217, e.g. EGP.")
    
    

    plan: str = Field(
        default="starter",
        description="Subscription plan key: starter | pro | enterprise.",
    )

    enterprise_monthly_limit: int | None = Field(
        default=None,
        description="Custom monthly AI message limit. Only used when plan=enterprise.",
    )

    billing_anchor_day: int = Field(
        default_factory=_default_anchor_day,
        ge=1,
        le=28,
        description="Day of month the usage period resets.",
    )

    model_config = ConfigDict(extra="forbid")
        

# ---------------------------------------------------------------------------
# API request/response models
# ---------------------------------------------------------------------------


class WooCommerceCredentialsRequest(BaseModel):
    """Merchant-submitted WooCommerce credentials for their store."""

    site_url: str
    auth_method: Literal["application_password", "consumer_key"]
    username: str
    password: str

    model_config = ConfigDict(extra="forbid")