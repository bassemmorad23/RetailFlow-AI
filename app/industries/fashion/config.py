"""
IndustryConfig for the Fashion industry.

Configures how StoreFlow AI handles fashion stores — the categories
merchants can sell in, the fields relevant for filtering and
recommendation, and the AI context that shapes response generation.

DESIGN NOTES:
- field_ids MUST match canonical_names of FieldDefinitions registered
  in fashion/fields.py. Registry validates this at load time — if a
  field is listed here but not registered, the app fails to start.
- ai_context is injected into the response generator's system prompt
  to give the LLM domain knowledge without hardcoding it in prompts.
- selling_points guide what the AI highlights when recommending —
  fashion customers care about fit, style, and returns.
- categories is a starter set. Merchants can request more; add them
  here (pure data change).
"""

from app.schemas.models import IndustryConfig


CONFIG = IndustryConfig(
    industry_id="fashion",
    display_name="Fashion & Apparel",
    categories=[
        "dresses",
        "tops",
        "bottoms",
        "outerwear",
        "shoes",
        "accessories",
    ],
    field_ids=[
        "size",
        "color",
        "material",
        "brand",
    ],
    ai_context=(
        "You are helping a customer at a fashion store. Focus on "
        "fit, style, occasion, and how items work together. When a "
        "customer mentions a size or color, prioritize matching "
        "products. If a customer seems unsure about sizing, "
        "reference the store's size guide. Fashion is personal — "
        "avoid pushy sales language; be a helpful stylist, not a "
        "salesperson."
    ),
    selling_points=[
        "free returns within 30 days",
        "size guide available",
        "seasonal collections",
        "style advice available on request",
    ],
)