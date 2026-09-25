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

VARIABLE PRODUCTS:
WC distinguishes simple products (single SKU) from variable products
(parent + variations). Variable products' variations are fetched
separately via /products/{parent_id}/variations. We emit each variation
as its own row sharing the parent's product_id, so the existing
to_canonical_grouped() naturally groups them into a parent Product with
Variants.
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
      { site_url, auth_method, username, password }
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
        """
        Fetch all products from WC.

        Variable products (parent + variations): we fetch the parent's
        variations via /products/{id}/variations and emit each variation
        as its own row sharing the parent's product_id. The existing
        to_canonical_grouped() then groups them naturally into a parent
        Product with Variants.

        Simple products emit one row each.
        """
        creds = get_credentials(store_id, "woocommerce")
        if creds is None:
            raise RuntimeError(
                f"No WooCommerce credentials for store '{store_id}'. "
                f"Save them via /stores/{{id}}/credentials/woocommerce."
            )

        base_url = creds["site_url"].rstrip("/")
        auth = (creds["username"], creds["password"])

        all_rows: list[dict] = []

        with httpx.Client(timeout=30.0) as client:
            # 1. Fetch all top-level products (parents + simple)
            page = 1
            per_page = 100
            parents: list[dict] = []
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
                parents.extend(batch)
                if len(batch) < per_page:
                    break
                page += 1

            # 2. Process each parent based on its type
            for parent in parents:
                if parent.get("type") == "variable":
                    # Fetch variations and emit one row per variation
                    variations = self._fetch_variations(
                        client, base_url, auth, parent["id"]
                    )
                    for var in variations:
                        row = _flatten_variation(parent, var)
                        all_rows.append(row)
                else:
                    # Simple product — one row
                    all_rows.append(_flatten_product(parent))

        return all_rows

    def _fetch_variations(self, client, base_url, auth, parent_id):
        """Fetch all variations for a variable product."""
        variations = []
        page = 1
        per_page = 100
        while True:
            resp = client.get(
                f"{base_url}/wp-json/wc/v3/products/{parent_id}/variations",
                params={"page": page, "per_page": per_page},
                auth=auth,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            variations.extend(batch)
            if len(batch) < per_page:
                break
            page += 1
        return variations


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

    
    flat["external_id"] = str(raw["id"]) if raw.get("id") is not None else None
    flat["external_parent_id"] = None
    flat["stock"] = _wc_stock(raw)
    return flat


def _flatten_variation(parent: dict, variation: dict) -> dict:
    """
    Turn a WC variation into a row that shares the parent's product_id.

    Uses parent's name, description, and product_id (via SKU or id fallback).
    Variation's own attributes (Size, Color) are flattened as top-level keys.
    Price/stock come from the variation.
    """
    flat = {
        "name": parent.get("name"),
        "description": parent.get("description"),
        "short_description": parent.get("short_description"),
    }

    # Use parent's SKU or id as product_id (SHARED across variations = group key)
    if parent.get("sku"):
        flat["product_id"] = parent["sku"]
    elif parent.get("id"):
        flat["product_id"] = str(parent["id"])

    # Variation's own price + stock
    if variation.get("regular_price"):
        flat["price"] = variation["regular_price"]
    elif variation.get("price"):
        flat["price"] = variation["price"]
    
    flat["stock"] = _wc_stock(variation)

    # Flatten variation's attributes (Size, Color, etc.)
    for attr in variation.get("attributes", []):
        if isinstance(attr, dict):
            name = attr.get("name")
            option = attr.get("option")
            if name and option:
                flat[name] = option

    
    
    flat["external_id"] = str(variation["id"]) if variation.get("id") is not None else None
    flat["external_parent_id"] = str(parent["id"]) if parent.get("id") is not None else None
    return flat


def _wc_stock(obj: dict):
    """Managed stock -> real quantity; otherwise the stock status; no data -> None."""
    if obj.get("manage_stock") and obj.get("stock_quantity") is not None:
        return max(int(obj["stock_quantity"]), 0)
    return {"instock": "in stock", "outofstock": "out of stock",
            "onbackorder": "in stock"}.get(obj.get("stock_status"))