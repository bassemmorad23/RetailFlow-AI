"""
Create an approved StoreFlow order in the store's platform.

Shopify (GraphQL orderCreate):
- payment pending, tags StoreFlow / COD / sf-<order id>
- price = what the customer confirmed; stock deducted obeying the product's policy
- before creating, looks up our tag -> an earlier successful attempt is reused
- shop currency must equal the order currency

WooCommerce (REST POST /orders):
- cod, status "processing" (WooCommerce deducts stock)
- price = what the customer confirmed
- a timeout is reported as "check_platform_before_retry": WooCommerce has no
  reliable lookup, so the merchant checks their admin before retrying

Every item must have its platform id (synced from the platform). Errors are
returned as PushResult(ok=False, error=<code>), never raised.
"""

from dataclasses import dataclass

import httpx

from app.ingestion.shopify_adapter import _SHOPIFY_API_VERSION
from app.products.product_store import fetch_products
from app.settings.store_credentials import get_credentials

TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class PushResult:
    ok: bool
    platform_order_id: str | None = None
    platform_order_number: str | None = None
    error: str | None = None


class _ShopifyError(Exception):
    pass


def push_order(store_id: str, platform: str, order: dict, client: httpx.Client | None = None) -> PushResult:
    http = client or httpx.Client(timeout=TIMEOUT_SECONDS)
    try:
        if platform == "shopify":
            return _shopify(store_id, order, http)
        if platform == "woocommerce":
            return _woocommerce(store_id, order, http)
        return PushResult(False, error="unsupported_platform")
    except httpx.TimeoutException:
        return PushResult(False, error="timeout_check_platform_before_retry")
    except httpx.HTTPError as exc:
        return PushResult(False, error=f"network_{type(exc).__name__}")
    finally:
        if client is None:
            http.close()


# ---------------------------------------------------------------- shared

def _platform_units(store_id: str, order: dict):
    """[(item, product, variant_or_None)] with platform ids, or None if any id is missing."""
    products = {p.product_id: p for p in fetch_products(store_id, list({i["product_id"] for i in order["items"]}))}
    units = []
    for item in order["items"]:
        product = products.get(item["product_id"])
        if product is None:
            return None
        variant = None
        if item.get("variant_sku"):
            variant = next((v for v in product.variants if v.sku == item["variant_sku"]), None)
            if variant is None or not variant.external_id:
                return None
        elif not product.external_id:
            return None
        units.append((item, product, variant))
    return units


def _split_name(full: str) -> tuple[str, str]:
    parts = full.strip().split(maxsplit=1)
    return (parts[0], parts[1]) if len(parts) == 2 else (full.strip(), "")


def _money(value: float) -> str:
    return f"{value:.2f}"


# ---------------------------------------------------------------- Shopify

_LOOKUP = """
query lookup($q: String!) {
  shop { currencyCode }
  orders(first: 1, query: $q) { edges { node { id name } } }
}
"""

_CREATE = """
mutation create($order: OrderCreateOrderInput!, $options: OrderCreateOptionsInput) {
  orderCreate(order: $order, options: $options) {
    order { id name }
    userErrors { field message }
  }
}
"""


def _shopify(store_id: str, order: dict, http: httpx.Client) -> PushResult:
    creds = get_credentials(store_id, "shopify")
    if creds is None:
        return PushResult(False, error="not_connected")
    url = f"https://{creds['shop']}/admin/api/{_SHOPIFY_API_VERSION}/graphql.json"
    headers = {"X-Shopify-Access-Token": creds["access_token"], "Content-Type": "application/json"}
    tag = f"sf-{order['id']}"

    def gql(query: str, variables: dict) -> dict:
        resp = http.post(url, headers=headers, json={"query": query, "variables": variables})
        if resp.status_code >= 400:
            raise _ShopifyError(f"http_{resp.status_code}")
        data = resp.json()
        if data.get("errors"):
            raise _ShopifyError("graphql_error: " + str(data["errors"])[:150])
        return data.get("data") or {}

    try:
        found = gql(_LOOKUP, {"q": f"tag:'{tag}'"})
        edges = (found.get("orders") or {}).get("edges") or []
        if edges:  # an earlier attempt succeeded: reuse it, never duplicate
            node = edges[0]["node"]
            return PushResult(True, node["id"], node.get("name"))

        shop_currency = (found.get("shop") or {}).get("currencyCode")
        if shop_currency and shop_currency != order["currency"]:
            return PushResult(False, error=f"currency_mismatch: shop uses {shop_currency}, order is {order['currency']}")

        units = _platform_units(store_id, order)
        if units is None:
            return PushResult(False, error="missing_platform_ids_resync_products")

        c = order["customer"]
        first, last = _split_name(c["name"])

        def money(v: float) -> dict:
            return {"shopMoney": {"amount": _money(v), "currencyCode": order["currency"]}}

        payload = {
            "lineItems": [
                {"variantId": (v or p).external_id, "quantity": i["quantity"], "priceSet": money(i["unit_price"])}
                for i, p, v in units
            ],
            "financialStatus": "PENDING",
            "phone": c["phone"],
            "shippingAddress": {
                "firstName": first, "lastName": last, "phone": c["phone"],
                "address1": c["address_line"], "city": c["city"],
                "countryCode": c["country"], "provinceCode": c["region_code"].split("-", 1)[-1],
            },
            "tags": ["StoreFlow", "COD", tag],
            "note": f"StoreFlow order {order['number']} — cash on delivery."
                    + (f" Customer notes: {c['notes']}" if c.get("notes") else ""),
        }
        if order.get("shipping_fee") is not None:
            payload["shippingLines"] = [{"title": "Shipping", "priceSet": money(order["shipping_fee"])}]

        created = gql(_CREATE, {"order": payload, "options": {
            "inventoryBehaviour": "DECREMENT_OBEYING_POLICY", "sendReceipt": False, "sendFulfillmentReceipt": False}})
        result = created.get("orderCreate") or {}
        if result.get("userErrors"):
            return PushResult(False, error="rejected: " + str(result["userErrors"][0].get("message", ""))[:150])
        node = result.get("order") or {}
        if not node.get("id"):
            return PushResult(False, error="no_order_returned")
        return PushResult(True, node["id"], node.get("name"))
    except _ShopifyError as exc:
        return PushResult(False, error=str(exc))


# ---------------------------------------------------------------- WooCommerce
def _wc_state(country: str, region_code: str) -> str:
    """WooCommerce state codes: Egypt uses 'EG' + ISO suffix (EGGZ); others the ISO suffix."""
    suffix = region_code.split("-", 1)[-1]
    return f"{country}{suffix}" if country == "EG" else suffix




def _woocommerce(store_id: str, order: dict, http: httpx.Client) -> PushResult:
    creds = get_credentials(store_id, "woocommerce")
    if creds is None:
        return PushResult(False, error="not_connected")
    units = _platform_units(store_id, order)
    if units is None:
        return PushResult(False, error="missing_platform_ids_resync_products")

    c = order["customer"]
    first, last = _split_name(c["name"])
    address = {"first_name": first, "last_name": last, "address_1": c["address_line"], "city": c["city"],
               "state": _wc_state(c["country"], c["region_code"]), "country": c["country"], "phone": c["phone"]}

    line_items = []
    try:
        for item, product, variant in units:
            amount = _money(item["unit_price"] * item["quantity"])
            line = {"quantity": item["quantity"], "subtotal": amount, "total": amount}
            if variant is not None:
                line["product_id"] = int(product.external_parent_id)
                line["variation_id"] = int(variant.external_id)
            else:
                line["product_id"] = int(product.external_id)
            line_items.append(line)
    except (TypeError, ValueError):
        return PushResult(False, error="invalid_platform_ids_resync_products")

    body = {
        "payment_method": "cod", "payment_method_title": "Cash on delivery", "set_paid": False,
        "status": "processing", "billing": address, "shipping": address, "line_items": line_items,
        "customer_note": c.get("notes") or "",
        "meta_data": [{"key": "_storeflow_order_id", "value": order["id"]},
                      {"key": "storeflow_order_number", "value": order["number"]}],
    }
    if order.get("shipping_fee") is not None:
        body["shipping_lines"] = [{"method_id": "flat_rate", "method_title": "Shipping",
                                   "total": _money(order["shipping_fee"])}]

    base = creds["site_url"].rstrip("/")
    resp = http.post(f"{base}/wp-json/wc/v3/orders", json=body, auth=(creds["username"], creds["password"]))
    if resp.status_code >= 400:
        try:
            message = resp.json().get("message", "")
        except ValueError:
            message = ""
        return PushResult(False, error=f"http_{resp.status_code}" + (f": {message[:150]}" if message else ""))
    data = resp.json()
    return PushResult(True, str(data.get("id")), str(data.get("number") or data.get("id")))