"""
Live stock lookups (no cache, by decision): ask the platform right now.

Batched:
- Shopify: one GraphQL `nodes` call for all variant ids
- WooCommerce: one call for simple products + one per variable parent

Returns {external_id: (status, quantity)} using the same parser as
ingestion. Ids the platform doesn't return are simply absent.
Any transport/API failure raises LiveStockError: the caller must treat
the item as "can't confirm right now", never as available.
"""

from collections import defaultdict

import httpx

from app.ingestion.shopify_adapter import _SHOPIFY_API_VERSION, _shopify_stock
from app.ingestion.stock_parser import parse_stock
from app.ingestion.woocommerce_adapter import _wc_stock
from app.settings.store_credentials import get_credentials

LIVE_TIMEOUT_SECONDS = 5.0
_SHOPIFY_BATCH = 250
_WC_BATCH = 100

_NODES_QUERY = """
query stock($ids: [ID!]!) {
  nodes(ids: $ids) {
    ... on ProductVariant {
      id
      inventoryQuantity
      inventoryPolicy
      inventoryItem { tracked }
    }
  }
}
"""


class LiveStockError(Exception):
    pass


def _client(client: httpx.Client | None) -> httpx.Client:
    return client or httpx.Client(timeout=LIVE_TIMEOUT_SECONDS)


def shopify_live_stock(store_id: str, variant_ids: list[str],
                       client: httpx.Client | None = None) -> dict[str, tuple[str, int | None]]:
    if not variant_ids:
        return {}
    creds = get_credentials(store_id, "shopify")
    if creds is None:
        raise LiveStockError("Shopify not connected")

    url = f"https://{creds['shop']}/admin/api/{_SHOPIFY_API_VERSION}/graphql.json"
    headers = {"X-Shopify-Access-Token": creds["access_token"], "Content-Type": "application/json"}
    http = _client(client)
    results: dict[str, tuple[str, int | None]] = {}
    try:
        for i in range(0, len(variant_ids), _SHOPIFY_BATCH):
            batch = variant_ids[i:i + _SHOPIFY_BATCH]
            resp = http.post(url, headers=headers, json={"query": _NODES_QUERY, "variables": {"ids": batch}})
            resp.raise_for_status()
            data = resp.json()
            if data.get("errors"):
                raise LiveStockError("Shopify GraphQL error")
            for node in (data.get("data") or {}).get("nodes") or []:
                if node and node.get("id"):
                    results[node["id"]] = parse_stock(_shopify_stock(node))
    except httpx.HTTPError as exc:
        raise LiveStockError(f"Shopify request failed: {type(exc).__name__}") from exc
    finally:
        if client is None:
            http.close()
    return results


def woocommerce_live_stock(store_id: str, units: list[tuple[str | None, str]],
                           client: httpx.Client | None = None) -> dict[str, tuple[str, int | None]]:
    """units: (parent_id or None, id). Variations need their parent id."""
    if not units:
        return {}
    creds = get_credentials(store_id, "woocommerce")
    if creds is None:
        raise LiveStockError("WooCommerce not connected")

    base = creds["site_url"].rstrip("/") + "/wp-json/wc/v3"
    auth = (creds["username"], creds["password"])

    simple_ids = [uid for parent, uid in units if not parent]
    by_parent: dict[str, list[str]] = defaultdict(list)
    for parent, uid in units:
        if parent:
            by_parent[parent].append(uid)

    http = _client(client)
    results: dict[str, tuple[str, int | None]] = {}

    def fetch(path: str, ids: list[str]) -> None:
        for i in range(0, len(ids), _WC_BATCH):
            batch = ids[i:i + _WC_BATCH]
            resp = http.get(f"{base}{path}", auth=auth,
                            params={"include": ",".join(batch), "per_page": _WC_BATCH})
            resp.raise_for_status()
            for item in resp.json():
                results[str(item.get("id"))] = parse_stock(_wc_stock(item))

    try:
        if simple_ids:
            fetch("/products", simple_ids)
        for parent, ids in by_parent.items():
            fetch(f"/products/{parent}/variations", ids)
    except (httpx.HTTPError, ValueError) as exc:
        raise LiveStockError(f"WooCommerce request failed: {type(exc).__name__}") from exc
    finally:
        if client is None:
            http.close()
    return results