"""
FieldDefinitions for the Smartphones industry.

Covers iPhone, Samsung, Xiaomi, and other smartphone brands sold
in Egyptian and regional markets.

DESIGN NOTES:
- Fields split by intent: quantitative specs (ram_gb, storage_gb)
  use numeric types for range filtering; qualitative features
  (brand, color, os) use strings/enums.
- Unit-suffixed canonical names (ram_gb, storage_gb) make it
  unambiguous what unit the number represents — no guessing whether
  '8' means GB or MB.
- Feature-flag fields (face_id, esim) intentionally NOT included in
  this starter set. They fall under the "attributes" bag on Product
  and rely on semantic search until real merchant data shows they
  need dedicated FieldDefinitions.
- Aliases cover English, Arabic, and common merchant column names
  from Shopify/WooCommerce and typical Egyptian/regional catalogs.

WHY THESE 6 FIELDS:
Starter set covering the specs customers most often ask about and
filter by. Additional fields (camera_mp, battery_mah, refresh_rate)
can be added later — pure data change, no pipeline updates needed.
"""

from app.schemas.models import FieldDefinition


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------


BRAND_FIELD = FieldDefinition(
    canonical_name="brand",
    display_name="Brand",
    description=(
        "The phone manufacturer (Apple, Samsung, Xiaomi, Oppo, "
        "Realme, Huawei, Google, etc.). Free-form string — brand "
        "list is unbounded."
    ),
    field_type="string",
    aliases=[
        "brand", "Brand", "BRAND",
        "manufacturer", "make",
        "الماركة", "الشركة",
    ],
    normalization_rules={
        "trim": True,
        # Brand names have specific capitalization (Apple, OPPO,
        # OnePlus) — don't force-lowercase.
    },
    validation_rules={
        "min_length": 1,
        "max_length": 60,
    },
    industries=["smartphones"],
    is_specification=False,
    extraction_hint=(
        "A phone brand the customer prefers (e.g. Apple, Samsung, "
        "Xiaomi, Oppo, Realme, Huawei). Preserve original "
        "capitalization. Also accept Arabic brand mentions ('أبل', "
        "'سامسونج', 'شاومي')."
    ),
)


RAM_GB_FIELD = FieldDefinition(
    canonical_name="ram_gb",
    display_name="RAM",
    description=(
        "Random-access memory capacity in gigabytes. Typical "
        "smartphone values: 4, 6, 8, 12, 16."
    ),
    field_type="int",
    unit="GB",
    aliases=[
        "ram", "RAM", "Ram",
        "memory", "Memory", "MEMORY",
        "ram_gb", "RAM_GB", "ram_size",
        "الرام", "الذاكرة", "الذاكرة العشوائية",
    ],
    normalization_rules={
        "trim": True,
        # "8 GB", "8GB", "8gb" all should become 8. The normalizer
        # strips trailing unit strings and coerces to int at read time.
        "strip_units": True,
    },
    validation_rules={
        # No smartphone ships with under 1GB or over 64GB as of 2026.
        # Values outside this range are almost certainly typos or
        # miscategorized data.
        "min": 1,
        "max": 64,
    },
    industries=["smartphones"],
    is_specification=True,
    extraction_hint=(
        "The amount of RAM the customer wants, expressed in "
        "gigabytes. Look for numbers followed by 'GB', 'gigs', "
        "'giga' — or Arabic equivalents ('جيجا', 'جيجابايت'). "
        "Common values: 4, 6, 8, 12, 16. If the customer says "
        "'at least 8GB', extract 8 (comparison is handled downstream)."
    ),
)


STORAGE_GB_FIELD = FieldDefinition(
    canonical_name="storage_gb",
    display_name="Storage",
    description=(
        "Internal storage capacity in gigabytes. Typical smartphone "
        "values: 64, 128, 256, 512, 1024."
    ),
    field_type="int",
    unit="GB",
    aliases=[
        "storage", "Storage", "STORAGE",
        "rom", "ROM",
        "internal_storage", "internal_memory",
        "capacity",
        "storage_gb", "STORAGE_GB",
        "المساحة", "التخزين", "الذاكرة الداخلية",
    ],
    normalization_rules={
        "trim": True,
        "strip_units": True,
    },
    validation_rules={
        # No modern smartphone ships with under 16GB or over 2TB.
        "min": 16,
        "max": 2048,
    },
    industries=["smartphones"],
    is_specification=True,
    extraction_hint=(
        "The internal storage capacity the customer wants, in "
        "gigabytes. Look for numbers followed by 'GB', 'gigs' — or "
        "Arabic ('جيجا'). Common values: 64, 128, 256, 512, 1024 "
        "(1TB = 1024GB)."
    ),
)


SCREEN_SIZE_INCHES_FIELD = FieldDefinition(
    canonical_name="screen_size_inches",
    display_name="Screen Size",
    description=(
        "Diagonal display size in inches. Typical smartphone values: "
        "5.4 to 6.9 inches."
    ),
    field_type="float",
    unit="inches",
    aliases=[
        "screen_size", "screen size", "Screen Size",
        "display_size", "display size",
        "screen", "display",
        "size_inches", "screen_inches",
        "شاشة", "حجم الشاشة", "مقاس الشاشة",
    ],
    normalization_rules={
        "trim": True,
        "strip_units": True,
    },
    validation_rules={
        # Smartphones range roughly 4 to 8 inches. Outside means
        # tablet territory or measurement error.
        "min": 4.0,
        "max": 8.0,
    },
    industries=["smartphones"],
    is_specification=True,
    extraction_hint=(
        "The screen size the customer wants, in inches. Look for "
        "numbers followed by 'inch', 'inches', or the double-prime "
        "symbol. Common values: 6.1, 6.5, 6.7, 6.9."
    ),
)


OS_FIELD = FieldDefinition(
    canonical_name="os",
    display_name="Operating System",
    description=(
        "The phone's operating system. Almost always iOS or Android, "
        "with rare exceptions (HarmonyOS, KaiOS)."
    ),
    field_type="enum",
    allowed_values=["iOS", "Android", "HarmonyOS", "KaiOS", "Other"],
    aliases=[
        "os", "OS",
        "operating_system", "operating system",
        "platform",
        "نظام التشغيل", "النظام",
    ],
    normalization_rules={
        "trim": True,
        # Common OS name variants get normalized to the enum values.
        "aliases": {
            "ios": "iOS",
            "iphone": "iOS",
            "apple": "iOS",
            "android": "Android",
            "harmony": "HarmonyOS",
            "harmonyos": "HarmonyOS",
        },
    },
    validation_rules={},
    industries=["smartphones"],
    is_specification=False,
    extraction_hint=(
        "The operating system the customer wants. iOS means Apple "
        "iPhones only; Android covers Samsung, Xiaomi, Oppo, "
        "Realme, and most others. If the customer says 'iPhone', "
        "extract os='iOS'."
    ),
)


COLOR_FIELD = FieldDefinition(
    canonical_name="color",
    display_name="Color",
    description=(
        "The phone's color. Free-form string because manufacturers "
        "invent marketing color names (Deep Purple, Titanium Blue, "
        "Phantom Black)."
    ),
    field_type="string",
    aliases=[
        "color", "Color", "COLOR",
        "colour", "Colour",
        "finish",
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
    industries=["smartphones"],
    is_specification=False,
    extraction_hint=(
        "A color the customer wants. Common: black, white, blue, "
        "green, purple, gold, silver, titanium. Also accept Arabic "
        "color names (أسود، أبيض، أزرق)."
    ),
)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

# The registry imports this list at startup. Add new fields here.
FIELDS = [
    BRAND_FIELD,
    RAM_GB_FIELD,
    STORAGE_GB_FIELD,
    SCREEN_SIZE_INCHES_FIELD,
    OS_FIELD,
    COLOR_FIELD,
]