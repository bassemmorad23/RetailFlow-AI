"""
FieldDefinitions for the Fashion industry.

Covers clothing, footwear, and accessories.

DESIGN NOTES:
- Every field's `industries` list contains only "fashion" — full
  isolation per your architecture decision. Even fields conceptually
  similar to other industries (e.g. "brand") get their own fashion-
  specific FieldDefinition, defined in each industry's own file.
- Aliases include English, Arabic, and common merchant column names
  observed in Shopify and WooCommerce fashion exports.
- Sizes use enum with a reasonable clothing-size set. Numeric sizes
  (e.g. shoe sizes) go under a separate field if needed later.

WHY THESE 4 FIELDS:
Starter set covering the most common fashion attributes that appear
in real merchant catalogs. Additional fields (fit, season, gender,
occasion) can be added later without touching the pipeline — pure
data change.
"""

from app.schemas.models import FieldDefinition


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------


SIZE_FIELD = FieldDefinition(
    canonical_name="size",
    display_name="Size",
    description=(
        "Clothing size in standard letter notation "
        "(XS, S, M, L, XL, XXL). Numeric sizes (waist inches, "
        "shoe sizes) are not covered by this field."
    ),
    field_type="enum",
    allowed_values=["XS", "S", "M", "L", "XL", "XXL"],
    aliases=[
        "size", "Size", "SIZE",
        "clothing_size", "garment_size",
        "المقاس", "مقاس",
    ],
    normalization_rules={
        "trim": True,
        "case": "upper",
        # Map common long-form values to the enum. Applied by the
        # normalizer at read time — original merchant value stays in
        # Product.original_data.
        "aliases": {
            "extra small": "XS",
            "small": "S",
            "medium": "M",
            "large": "L",
            "extra large": "XL",
            "double xl": "XXL",
        },
    },
    validation_rules={},
    industries=["fashion"],
    is_specification=False,
    extraction_hint=(
        "A clothing size the customer explicitly stated they want. "
        "Look for XS/S/M/L/XL/XXL, or long-form 'small', 'medium', "
        "'large', 'extra large'. Ignore mentions in Arabic that "
        "reference sizes ('مقاس مديوم')."
    ),
)


COLOR_FIELD = FieldDefinition(
    canonical_name="color",
    display_name="Color",
    description=(
        "The primary color of the garment. Free-form string — "
        "not restricted to a fixed palette because fashion color "
        "names vary widely (ecru, mauve, burgundy, etc.)."
    ),
    field_type="string",
    aliases=[
        "color", "Color", "COLOR",
        "colour", "Colour",
        "shade", "hue",
        "اللون", "لون",
    ],
    normalization_rules={
        "trim": True,
        "case": "lower",
    },
    validation_rules={
        "min_length": 1,
        "max_length": 40,
    },
    industries=["fashion"],
    is_specification=False,
    extraction_hint=(
        "A color the customer explicitly stated they want. Common "
        "colors: red, blue, black, white, green, yellow, pink, "
        "purple, brown, gray, beige, navy. Also accept Arabic "
        "color names (أحمر، أزرق، أسود)."
    ),
)


MATERIAL_FIELD = FieldDefinition(
    canonical_name="material",
    display_name="Material",
    description=(
        "The primary fabric or material the garment is made of "
        "(cotton, linen, polyester, wool, silk, denim, leather)."
    ),
    field_type="string",
    aliases=[
        "material", "Material", "MATERIAL",
        "fabric", "Fabric", "FABRIC",
        "composition",
        "الخامة", "القماش",
    ],
    normalization_rules={
        "trim": True,
        "case": "lower",
    },
    validation_rules={
        "min_length": 1,
        "max_length": 60,
    },
    industries=["fashion"],
    is_specification=False,
    extraction_hint=(
        "A fabric or material the customer wants. Examples: cotton, "
        "linen, polyester, wool, silk, denim, leather. Also Arabic "
        "equivalents (قطن، كتان، جلد)."
    ),
)


BRAND_FIELD = FieldDefinition(
    canonical_name="brand",
    display_name="Brand",
    description=(
        "The brand or designer name (Zara, H&M, Nike, Adidas, etc.). "
        "Free-form string — brand list is unbounded."
    ),
    field_type="string",
    aliases=[
        "brand", "Brand", "BRAND",
        "manufacturer", "designer", "label",
        "الماركة", "الشركة",
    ],
    normalization_rules={
        "trim": True,
        # Brand names have their own capitalization (H&M, iPhone-era
        # brands), so we don't force-lowercase. Just trim whitespace.
    },
    validation_rules={
        "min_length": 1,
        "max_length": 80,
    },
    industries=["fashion"],
    is_specification=False,
    extraction_hint=(
        "A brand or designer the customer prefers (e.g. Zara, H&M, "
        "Nike, Adidas, Gucci). Preserve original capitalization."
    ),
)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

# The registry imports this list at startup. Add new fields here.
FIELDS = [
    SIZE_FIELD,
    COLOR_FIELD,
    MATERIAL_FIELD,
    BRAND_FIELD,
]