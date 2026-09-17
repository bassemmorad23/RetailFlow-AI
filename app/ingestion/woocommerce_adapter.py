"""
WooCommerce adapter.

Fetches products from a merchant's WooCommerce store via REST API.
Uses per-store credentials from store_credentials collection.

WHY REST API (not GraphQL):
WC has no GraphQL. REST API is standard, well-documented, works with
API keys (no OAuth complexity for MVP).

WHY sync (not async):
Matches CSVAdapter pattern. Async can be added later if a merchant
has 5000+ products and memory matters.
"""

import httpx

from app.ingestion.source_adapter import SourceAdapter
from app.settings.store_credentials import get_credentials


# WC returns many fields per product. We expose the ones that map to
# canonical Product fields plus attributes/specs commonly present.
_WC_FIXED_COLUMNS = [
    "id",
    "name",
    "description",
    "short_description",
    "price",
    "regular_price",
    "sale_price",
    "sku",
    "stock_status",
    "images",
    "categories",
    "attributes",
]


class WooCommerceAdapter(SourceAdapter):
    """
    WooCommerce REST API adapter.

    Requires credentials in store_credentials with source='woocommerce':
      { site_url, consumer_key, consumer_secret }
    """

    def get_source_name(self) -> str:
        return "woocommerce"

    def get_source_columns(self, store_id: str) -> list[str]:
        """
        WC has a fixed product schema — we know the columns without
        fetching. Attributes vary per store but appear as items inside
        the 'attributes' field and are flattened at fetch time.
        """
        return list(_WC_FIXED_COLUMNS)

    def fetch_raw_products(self, store_id: str) -> list[dict]:
        creds = get_credentials(store_id, "woocommerce")
        if creds is None:
            raise RuntimeError(
                f"No WooCommerce credentials for store '{store_id}'. "
                f"Save them with store_credentials.set_credentials()."
            )

        base_url = creds["site_url"].rstrip("/")
        auth = (creds["username"], creds["password"])

        all_products: list[dict] = []
        page = 1
        per_page = 100

        with httpx.Client(timeout=30.0) as client:
            while True:
                resp = client.get(
                    f"{base_url}/wp-json/wc/v3/products",
                    params={"page": page, "per_page": per_page},
                    auth=auth,
                )
                resp.raise_for_status()
                batch = resp.json()

                if not batch:
                    break

                for raw in batch:
                    all_products.append(_flatten_product(raw))

                if len(batch) < per_page:
                    break
                page += 1

        return all_products


def _flatten_product(raw: dict) -> dict:
    """
    Turn a WC product dict into a flat row the mapper can process.

    WC nests attributes like:
      "attributes": [{"name": "RAM", "options": ["8 GB"]}, ...]

    We flatten to top-level keys:
      "RAM": "8 GB"

    So the deterministic mapper matches "RAM" against ram_gb aliases.
    """
    flat = dict(raw)

    # Flatten attributes: [{name, options}] -> top-level {name: value}
    attrs = raw.get("attributes", [])
    if isinstance(attrs, list):
        for attr in attrs:
            if not isinstance(attr, dict):
                continue
            name = attr.get("name")
            options = attr.get("options") or []
            if name and options:
                # WC allows multiple options per attribute; join or take first
                flat[name] = options[0] if len(options) == 1 else ", ".join(options)

    # Normalize price to a single field
    if "regular_price" in raw and raw["regular_price"]:
        flat["price"] = raw["regular_price"]

    # Use SKU or id as product_id — SKU preferred (merchant-set)
    if raw.get("sku"):
        flat["product_id"] = raw["sku"]
    elif raw.get("id"):
        flat["product_id"] = str(raw["id"])

    return flat