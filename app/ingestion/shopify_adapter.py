"""
Shopify adapter (Admin GraphQL, per-store OAuth token).

- fetch_raw_products: full catalog (used by the full sync / reconciliation)
- fetch_product_rows: ONE product, same fields + same flattening (used by webhooks)
- product_id_for_inventory_item: stock-change webhooks carry an inventory item id
Only ACTIVE products are imported; draft/archived products are treated as absent.

Shopify products always have >=1 variant; each variant becomes a row sharing the
product's handle as product_id, so to_canonical_grouped builds variants.
"""

import httpx

from app.ingestion.source_adapter import SourceAdapter
from app.settings.store_credentials import get_credentials

_SHOPIFY_API_VERSION = "2026-07"
_TIMEOUT = 30.0

# Shared by the catalog query and the single-product query (never duplicated).
_PRODUCT_FIELDS = """
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
              inventoryItem { tracked }
              selectedOptions { name value }
              image { url }
            }
          }
        }
        images(first: 1) { edges { node { url } } }
"""

_PRODUCTS_QUERY = """
query getProducts($first: Int!, $cursor: String) {
  products(first: $first, after: $cursor) {
    edges { cursor node { %s } }
    pageInfo { hasNextPage endCursor }
  }
}
""" % _PRODUCT_FIELDS

_PRODUCT_QUERY = "query getProduct($id: ID!) { product(id: $id) { %s } }" % _PRODUCT_FIELDS

_INVENTORY_ITEM_QUERY = "query inv($id: ID!) { inventoryItem(id: $id) { variant { product { id } } } }"


class ShopifyAdapter(SourceAdapter):
    """Requires store_credentials source='shopify': {shop, access_token}."""

    def get_source_name(self) -> str:
        return "shopify"

    def get_source_columns(self, store_id: str) -> list[str]:
        return ["product_id", "name", "description", "vendor", "product_type", "price", "sku", "stock", "image_url"]

    def fetch_raw_products(self, store_id: str) -> list[dict]:
        rows: list[dict] = []
        cursor = None
        with httpx.Client(timeout=_TIMEOUT) as client:
            while True:
                data = _graphql(store_id, client, _PRODUCTS_QUERY, {"first": 50, "cursor": cursor})
                page = data["products"]
                for edge in page["edges"]:
                    if _is_active(edge["node"]):
                        rows.extend(_flatten_product_with_variants(edge["node"]))
                if not page["pageInfo"]["hasNextPage"]:
                    break
                cursor = page["pageInfo"]["endCursor"]
        return rows

    def fetch_product_rows(self, store_id: str, product_gid: str, client: httpx.Client | None = None) -> list[dict] | None:
        """Rows for one product, or None if it no longer exists or isn't active."""
        http = client or httpx.Client(timeout=_TIMEOUT)
        try:
            node = _graphql(store_id, http, _PRODUCT_QUERY, {"id": product_gid}).get("product")
        finally:
            if client is None:
                http.close()
        if not node or not _is_active(node):
            return None
        return _flatten_product_with_variants(node)

    def product_id_for_inventory_item(self, store_id: str, inventory_item_gid: str,
                                      client: httpx.Client | None = None) -> str | None:
        http = client or httpx.Client(timeout=_TIMEOUT)
        try:
            item = _graphql(store_id, http, _INVENTORY_ITEM_QUERY, {"id": inventory_item_gid}).get("inventoryItem")
        finally:
            if client is None:
                http.close()
        return (((item or {}).get("variant") or {}).get("product") or {}).get("id")


def _graphql(store_id: str, client: httpx.Client, query: str, variables: dict) -> dict:
    creds = get_credentials(store_id, "shopify")
    if creds is None:
        raise RuntimeError(f"No Shopify credentials for store '{store_id}'. Run OAuth flow first.")
    resp = client.post(
        f"https://{creds['shop']}/admin/api/{_SHOPIFY_API_VERSION}/graphql.json",
        headers={"X-Shopify-Access-Token": creds["access_token"], "Content-Type": "application/json"},
        json={"query": query, "variables": variables},
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"Shopify GraphQL error: {data['errors']}")
    return data.get("data") or {}


def _is_active(product: dict) -> bool:
    return (product.get("status") or "ACTIVE") == "ACTIVE"


def _shopify_stock(variant: dict):
    """Tracked -> real quantity. Not tracked / 'continue selling' -> in stock. No data -> None."""
    if variant.get("inventoryPolicy") == "CONTINUE":
        return "in stock"
    tracked = (variant.get("inventoryItem") or {}).get("tracked")
    if tracked is False:
        return "in stock"
    qty = variant.get("inventoryQuantity")
    return None if qty is None else max(int(qty), 0)


def _flatten_product_with_variants(product: dict) -> list[dict]:
    """One row per variant sharing the product's handle as product_id."""
    handle = product.get("handle") or product.get("id", "").split("/")[-1]
    parent_image = None
    if (product.get("images") or {}).get("edges"):
        parent_image = product["images"]["edges"][0]["node"]["url"]

    rows = []
    for v_edge in (product.get("variants") or {}).get("edges", []):
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
        row["image_url"] = (v.get("image") or {}).get("url") or parent_image
        for opt in v.get("selectedOptions", []):
            name, value = opt.get("name"), opt.get("value")
            if name and value and name != "Title":  # skip Shopify's "Default Title"
                row[name] = value
        rows.append(row)
    return rows