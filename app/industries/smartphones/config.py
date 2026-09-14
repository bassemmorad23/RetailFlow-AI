"""
IndustryConfig for the Smartphones industry.

Configures how StoreFlow AI handles smartphone stores — categories,
relevant fields, AI context, and selling points.

DESIGN NOTES:
- field_ids MUST match canonical_names of FieldDefinitions registered
  in smartphones/fields.py. Registry validates at load time.
- ai_context tells the LLM to think in specs and use cases — very
  different from fashion's style-focused tone.
- selling_points emphasize warranty, financing, and trade-in
  programs which are common decision factors for phone buyers.
"""

from app.schemas.models import IndustryConfig


CONFIG = IndustryConfig(
    industry_id="smartphones",
    display_name="Smartphones",
    categories=[
        "flagship",
        "mid-range",
        "budget",
        "gaming",
        "accessories",
    ],
    field_ids=[
        "brand",
        "ram_gb",
        "storage_gb",
        "screen_size_inches",
        "os",
        "color",
    ],
    ai_context=(
        "You are helping a customer at a smartphone store. Focus on "
        "specs (RAM, storage, screen size), use case (photography, "
        "gaming, work, everyday), and budget. Match customer needs "
        "to phones — a photographer needs a good camera, a gamer "
        "needs strong specs, a student may prioritize battery and "
        "price. If the customer mentions iOS or iPhone, only "
        "recommend iOS phones. Be honest about trade-offs: no "
        "phone is best at everything."
    ),
    selling_points=[
        "1-year warranty on all phones",
        "installment plans available",
        "trade-in your old phone for credit",
        "same-day activation with any carrier",
    ],
)