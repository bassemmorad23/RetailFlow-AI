"""
Stock service — the single source of stock answers for the whole system.

- Platform stores (Shopify/WooCommerce) + product has a platform id: LIVE
  lookup, batched in one call per message (no cache, by decision).
- CSV stores, or products without a platform id: synced value.
- Live check failed / not found / no data -> "unknown": never presented
  as available.
- Exact quantities are internal only; customers see levels.
"""

from dataclasses import dataclass
from typing import Literal

from app.commerce.stock_live import LiveStockError, shopify_live_stock, woocommerce_live_stock
from app.products.product_store import fetch_products
from app.schemas.models import Product, ProductRecommendation
from app.settings.store_credentials import get_credentials

StockLevel = Literal["in_stock", "low_stock", "out_of_stock", "unknown"]
LOW_STOCK_THRESHOLD = 3


@dataclass(frozen=True)
class StockCheck:
    level: StockLevel
    source: Literal["live", "synced", "none"]
    reason: str | None = None  # live_check_failed | not_found_on_platform | no_data


def store_platform(store_id: str) -> str | None:
    if get_credentials(store_id, "shopify"):
        return "shopify"
    if get_credentials(store_id, "woocommerce"):
        return "woocommerce"
    return None


def _level(status: str, quantity: int | None) -> StockLevel:
    if status == "in_stock":
        if quantity is not None and quantity <= LOW_STOCK_THRESHOLD:
            return "low_stock"
        return "in_stock"
    if status == "out_of_stock":
        return "out_of_stock"
    return "unknown"


def _aggregate(levels: list[StockLevel]) -> StockLevel:
    if "in_stock" in levels:
        return "in_stock"
    if "low_stock" in levels:
        return "low_stock"
    if levels and all(lv == "out_of_stock" for lv in levels):
        return "out_of_stock"
    return "unknown"


def check_stock(store_id: str, requests: list[tuple[str, str | None]]) -> list[StockCheck]:
    """
    requests: (product_id, variant_sku or None). A product with variants and
    no sku is checked across all its variants (in stock if any variant is).
    Returns one StockCheck per request, in order.
    """
    products = {p.product_id: p for p in fetch_products(store_id, list({pid for pid, _ in requests}))}
    platform = store_platform(store_id)

    # Expand each request into purchasable units: (product, variant or None)
    expanded: list[list[tuple[Product, object]]] = []
    for pid, sku in requests:
        product = products.get(pid)
        if product is None:
            expanded.append([])
        elif sku:
            variant = next((v for v in product.variants if v.sku == sku), None)
            expanded.append([(product, variant)] if variant else [])
        elif product.variants:
            expanded.append([(product, v) for v in product.variants])
        else:
            expanded.append([(product, None)])

    live_results: dict[str, tuple[str, int | None]] = {}
    live_failed = False
    if platform:
        units = [u for group in expanded for u in group]
        try:
            if platform == "shopify":
                ids = [(v or p).external_id for p, v in units if (v or p).external_id]
                live_results = shopify_live_stock(store_id, list(dict.fromkeys(ids)))
            else:
                wc_units = [((p.external_parent_id if v else None), (v or p).external_id)
                            for p, v in units if (v or p).external_id]
                live_results = woocommerce_live_stock(store_id, list(dict.fromkeys(wc_units)))
        except LiveStockError:
            live_failed = True

    def unit_check(product: Product, variant) -> StockCheck:
        unit = variant or product
        if platform and unit.external_id:
            if live_failed:
                return StockCheck("unknown", "none", "live_check_failed")
            found = live_results.get(unit.external_id)
            if found is None:
                return StockCheck("unknown", "none", "not_found_on_platform")
            return StockCheck(_level(*found), "live")
        level = _level(unit.stock_status, unit.stock_quantity)
        return StockCheck(level, "synced", "no_data" if level == "unknown" else None)

    results: list[StockCheck] = []
    for group in expanded:
        if not group:
            results.append(StockCheck("unknown", "none", "no_data"))
            continue
        checks = [unit_check(p, v) for p, v in group]
        if len(checks) == 1:
            results.append(checks[0])
        else:
            source = "live" if any(c.source == "live" for c in checks) else checks[0].source
            results.append(StockCheck(_aggregate([c.level for c in checks]), source))
    return results


_ORDER = {"in_stock": 0, "low_stock": 0, "unknown": 1, "out_of_stock": 2}


def attach_stock(store_id: str, recommendations: list[ProductRecommendation]) -> list[ProductRecommendation]:
    """Add stock level to each recommendation; available first, out of stock last."""
    if not recommendations:
        return recommendations
    checks = check_stock(store_id, [(r.product_id, r.variant_sku) for r in recommendations])
    enriched = [r.model_copy(update={"stock": c.level}) for r, c in zip(recommendations, checks)]
    return sorted(enriched, key=lambda r: _ORDER[r.stock])  # stable: keeps ranking within a level