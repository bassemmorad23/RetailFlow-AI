"""
Shopify adapter.

Fetches products via Shopify Admin GraphQL API using per-store OAuth token.

WHY GraphQL:
- REST deprecating variant endpoints Feb 2025
- One query fetches products + variants + attributes → no N+1 like WooCommerce
- Shopify's officially recommended API direction

VARIANT HANDLING:
Shopify products always have at least one variant (even simple products =
one default variant). We flatten each variant to its own row sharing the
parent product's `handle` as product_id. Downstream to_canonical_grouped
handles the rest.
"""

import httpx

from app.ingestion.source_adapter import SourceAdapter
from app.settings.store_credentials import get_credentials


_SHOPIFY_API_VERSION = "2026-07"

_PRODUCTS_QUERY = """
query getProducts($first: Int!, $cursor: String) {
  products(first: $first, after: $cursor) {
    edges {
      cursor
      node {
        id
        handle
        title
        description
        productType
        vendor
        status
        variants(first: 50) {
          edges {
            node {
              id
              sku
              price
              compareAtPrice
              inventoryQuantity
              inventoryPolicy
              inventoryItem {
                tracked
              }
              selectedOptions {
                name
                value
              }
              image {
                url
              }
            }
          }
        }
        images(first: 1) {
          edges {
            node {
              url
            }
          }
        }
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
"""


class ShopifyAdapter(SourceAdapter):
    """
    Shopify Admin GraphQL adapter.
    Requires credentials in store_credentials with source='shopify':
      { shop, access_token }
    """

    def get_source_name(self) -> str:
        return "shopify"

    def get_source_columns(self, store_id: str) -> list[str]:
        """
        Shopify has a fixed schema per selectedOption. Actual attribute
        names (Size, Color) vary per store but appear flattened as
        top-level keys after fetch, so mapper builds from actual row keys.
        """
        return [
            "product_id", "name", "description", "vendor", "product_type",
            "price", "sku", "stock", "image_url",
        ]

    def fetch_raw_products(self, store_id: str) -> list[dict]:
        """
        Fetch all products via GraphQL, paginating with cursor.
        Emits one row per variant, sharing parent's handle as product_id.
        """
        creds = get_credentials(store_id, "shopify")
        if creds is None:
            raise RuntimeError(
                f"No Shopify credentials for store '{store_id}'. "
                f"Run OAuth flow first."
            )

        shop = creds["shop"]
        token = creds["access_token"]
        url = f"https://{shop}/admin/api/{_SHOPIFY_API_VERSION}/graphql.json"
        headers = {
            "X-Shopify-Access-Token": token,
            "Content-Type": "application/json",
        }

        all_rows: list[dict] = []
        cursor = None
        page_size = 50

        with httpx.Client(timeout=30.0) as client:
            while True:
                resp = client.post(
                    url,
                    headers=headers,
                    json={
                        "query": _PRODUCTS_QUERY,
                        "variables": {"first": page_size, "cursor": cursor},
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                if "errors" in data:
                    raise RuntimeError(f"Shopify GraphQL error: {data['errors']}")

                products_page = data["data"]["products"]

                for edge in products_page["edges"]:
                    product = edge["node"]
                    rows = _flatten_product_with_variants(product)
                    all_rows.extend(rows)

                page_info = products_page["pageInfo"]
                if not page_info["hasNextPage"]:
                    break
                cursor = page_info["endCursor"]

        return all_rows


def _flatten_product_with_variants(product: dict) -> list[dict]:
    """
    Emit one row per variant, sharing parent's handle as product_id.
    Each variant carries its own price, sku, stock, and selectedOptions
    (Size, Color, etc.) flattened as top-level keys.
    """
    handle = product.get("handle") or product.get("id", "").split("/")[-1]
    parent_image = None
    if product.get("images", {}).get("edges"):
        parent_image = product["images"]["edges"][0]["node"]["url"]

    rows = []
    variant_edges = product.get("variants", {}).get("edges", [])

    for v_edge in variant_edges:
        v = v_edge["node"]
        row = {
            "product_id": handle,
            "name": product.get("title"),
            "description": product.get("description"),
            "vendor": product.get("vendor"),
            "product_type": product.get("productType"),
            "price": v.get("price"),
            "sku": v.get("sku") or v.get("id", "").split("/")[-1],
            "stock": _shopify_stock(v),
            "external_id": v.get("id"),
            "external_parent_id": product.get("id"),
        }

        # Variant image OR parent image fallback
        v_image = v.get("image") or {}
        row["image_url"] = v_image.get("url") or parent_image

        # Flatten selectedOptions (Size, Color, etc.) as top-level keys
        # Skip Shopify's default "Title=Default Title" for simple products.
        for opt in v.get("selectedOptions", []):
            name = opt.get("name")
            value = opt.get("value")
            if name and value and name != "Title":
                row[name] = value

        rows.append(row)

    return rows
  
  

def _shopify_stock(variant: dict):
    """
    Tracked -> real quantity. Not tracked, or 'continue selling when out
    of stock' -> in stock. No data -> None (unknown).
    """
    if variant.get("inventoryPolicy") == "CONTINUE":
        return "in stock"
    tracked = (variant.get("inventoryItem") or {}).get("tracked")
    if tracked is False:
        return "in stock"
    qty = variant.get("inventoryQuantity")
    return None if qty is None else max(int(qty), 0)