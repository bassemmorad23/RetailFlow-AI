"""
Shopify calculated shipping rates for the exact items + delivery address.

Uses draftOrderCalculate (nothing is created in the store). Returns the
available options, or None when rates can't be obtained (then the store's
configured fallback is used). Rates must be in the order's currency.
"""

import httpx

from app.commerce.models import ShippingOption
from app.ingestion.shopify_adapter import _SHOPIFY_API_VERSION
from app.products.product_store import fetch_products
from app.settings.store_credentials import get_credentials

TIMEOUT_SECONDS = 10.0

_CALCULATE = """
mutation calc($input: DraftOrderInput!) {
  draftOrderCalculate(input: $input) {
    calculatedDraftOrder {
      availableShippingRates { handle title price { amount currencyCode } }
    }
    userErrors { field message }
  }
}
"""


def shopify_shipping_rates(store_id: str, items: list, customer, currency: str,
                           client: httpx.Client | None = None) -> list[ShippingOption] | None:
    """items: OrderItem list; customer: DraftCustomer with country + region_code set."""
    creds = get_credentials(store_id, "shopify")
    if creds is None or not items:
        return None

    products = {p.product_id: p for p in fetch_products(store_id, list({i.product_id for i in items}))}
    line_items = []
    for i in items:
        p = products.get(i.product_id)
        v = next((x for x in p.variants if x.sku == i.variant_sku), None) if p and i.variant_sku else None
        variant_id = (v or p).external_id if p else None
        if not variant_id:
            return None
        line_items.append({"variantId": variant_id, "quantity": i.quantity})

    address = {"address1": customer.address_line or "", "city": customer.city or "",
               "countryCode": customer.country, "provinceCode": (customer.region_code or "").split("-", 1)[-1]}
    http = client or httpx.Client(timeout=TIMEOUT_SECONDS)
    try:
        resp = http.post(
            f"https://{creds['shop']}/admin/api/{_SHOPIFY_API_VERSION}/graphql.json",
            headers={"X-Shopify-Access-Token": creds["access_token"], "Content-Type": "application/json"},
            json={"query": _CALCULATE, "variables": {"input": {"lineItems": line_items, "shippingAddress": address}}},
        )
        if resp.status_code >= 400:
            return None
        data = resp.json()
        result = ((data.get("data") or {}).get("draftOrderCalculate") or {})
        if data.get("errors") or result.get("userErrors"):
            return None
        rates = ((result.get("calculatedDraftOrder") or {}).get("availableShippingRates")) or []
        options = []
        for r in rates:
            price = r.get("price") or {}
            if r.get("handle") and r.get("title") and price.get("currencyCode") == currency:
                options.append(ShippingOption(handle=r["handle"], title=r["title"], price=round(float(price["amount"]), 2)))
        return options or None
    except (httpx.HTTPError, ValueError, TypeError):
        return None
    finally:
        if client is None:
            http.close()