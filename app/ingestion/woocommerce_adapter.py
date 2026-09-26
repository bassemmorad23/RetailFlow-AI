"""
WooCommerce adapter (REST API, per-store credentials).

- fetch_raw_products: full catalog (full sync / reconciliation)
- fetch_product_rows: ONE product (+ its variations), same flattening (webhooks)
Only published products are imported; drafts/private/pending are treated as absent.

Variable products: each variation becomes a row sharing the parent's product_id,
so to_canonical_grouped builds a parent Product with Variants.
"""

import httpx

from app.ingestion.source_adapter import SourceAdapter
from app.settings.store_credentials import get_credentials

_WC_FIXED_COLUMNS = ["id", "name", "description", "short_description", "price", "regular_price",
                     "sale_price", "sku", "stock_status", "images", "categories", "attributes"]
_TIMEOUT = 30.0
_PER_PAGE = 100


class WooCommerceAdapter(SourceAdapter):
    """Requires store_credentials source='woocommerce': {site_url, auth_method, username, password}."""

    def get_source_name(self) -> str:
        return "woocommerce"

    def get_source_columns(self, store_id: str) -> list[str]:
        return list(_WC_FIXED_COLUMNS)

    def fetch_raw_products(self, store_id: str) -> list[dict]:
        base, auth = _conn(store_id)
        rows: list[dict] = []
        with httpx.Client(timeout=_TIMEOUT) as client:
            for parent in _paged(client, f"{base}/products", auth, {"status": "publish"}):
                rows.extend(_rows_for(client, base, auth, parent))
        return rows

    def fetch_product_rows(self, store_id: str, product_id: str, client: httpx.Client | None = None) -> list[dict] | None:
        """Rows for one product, or None if it no longer exists or isn't published."""
        base, auth = _conn(store_id)
        http = client or httpx.Client(timeout=_TIMEOUT)
        try:
            resp = http.get(f"{base}/products/{product_id}", auth=auth)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            product = resp.json()
            if product.get("status") != "publish":
                return None
            return _rows_for(http, base, auth, product)
        finally:
            if client is None:
                http.close()


def _conn(store_id: str) -> tuple[str, tuple[str, str]]:
    creds = get_credentials(store_id, "woocommerce")
    if creds is None:
        raise RuntimeError(f"No WooCommerce credentials for store '{store_id}'. "
                           f"Save them via /stores/{{id}}/credentials/woocommerce.")
    return creds["site_url"].rstrip("/") + "/wp-json/wc/v3", (creds["username"], creds["password"])


def _paged(client: httpx.Client, url: str, auth, params: dict | None = None) -> list[dict]:
    items, page = [], 1
    while True:
        resp = client.get(url, params={**(params or {}), "page": page, "per_page": _PER_PAGE}, auth=auth)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        items.extend(batch)
        if len(batch) < _PER_PAGE:
            break
        page += 1
    return items


def _rows_for(client: httpx.Client, base: str, auth, product: dict) -> list[dict]:
    if product.get("type") == "variable":
        variations = _paged(client, f"{base}/products/{product['id']}/variations", auth)
        return [_flatten_variation(product, v) for v in variations]
    return [_flatten_product(product)]


def _wc_stock(obj: dict):
    """Managed stock -> real quantity; otherwise the stock status; no data -> None."""
    if obj.get("manage_stock") and obj.get("stock_quantity") is not None:
        return max(int(obj["stock_quantity"]), 0)
    return {"instock": "in stock", "outofstock": "out of stock", "onbackorder": "in stock"}.get(obj.get("stock_status"))


def _flatten_product(raw: dict) -> dict:
    """WC product -> flat row. Attributes [{name, options}] become top-level keys."""
    flat = dict(raw)
    for attr in raw.get("attributes") or []:
        if isinstance(attr, dict) and attr.get("name") and attr.get("options"):
            options = attr["options"]
            flat[attr["name"]] = options[0] if len(options) == 1 else ", ".join(options)
    if raw.get("regular_price"):
        flat["price"] = raw["regular_price"]
    if raw.get("sku"):
        flat["product_id"] = raw["sku"]
    elif raw.get("id"):
        flat["product_id"] = str(raw["id"])
    flat["stock"] = _wc_stock(raw)
    flat["external_id"] = str(raw["id"]) if raw.get("id") is not None else None
    flat["external_parent_id"] = None
    return flat


def _flatten_variation(parent: dict, variation: dict) -> dict:
    """A variation row sharing the parent's product_id; price/stock/attributes from the variation."""
    flat = {"name": parent.get("name"), "description": parent.get("description"),
            "short_description": parent.get("short_description")}
    if parent.get("sku"):
        flat["product_id"] = parent["sku"]
    elif parent.get("id"):
        flat["product_id"] = str(parent["id"])
    if variation.get("regular_price"):
        flat["price"] = variation["regular_price"]
    elif variation.get("price"):
        flat["price"] = variation["price"]
    flat["stock"] = _wc_stock(variation)
    for attr in variation.get("attributes") or []:
        if isinstance(attr, dict) and attr.get("name") and attr.get("option"):
            flat[attr["name"]] = attr["option"]
    flat["external_id"] = str(variation["id"]) if variation.get("id") is not None else None
    flat["external_parent_id"] = str(parent["id"]) if parent.get("id") is not None else None
    return flat