"""
Product writes shared by full syncs and real-time webhooks.

MongoDB is the source of truth; Qdrant (AI search) always follows it.
- upsert_platform_products: convert raw platform rows (same mapping + conversion as
  the full sync) -> save -> upsert just those vectors. Idempotent.
- remove_platform_product: delete ONE platform-sourced product (has a platform id).
- remove_missing_platform_products: reconciliation after a full platform sync,
  with guards against wiping a catalog on a partial API response.
CSV products (no platform id) are never removed by any of these.
"""

import logging

from app.ingestion.canonical_converter import to_canonical_grouped
from app.ingestion.deterministic_mapper import map_columns
from app.products.product_store import _get_collection, fetch_products
from app.rag.indexer import delete_products_from_qdrant, upsert_products_to_qdrant
from app.schemas.models import Product
from app.settings.store_settings import get_industry

logger = logging.getLogger(__name__)

MAX_REMOVAL_SHARE = 0.5
_PLATFORM_SOURCED = {"$or": [{"external_id": {"$ne": None}}, {"external_parent_id": {"$ne": None}}]}


def save_products(store_id: str, products: list[Product]) -> tuple[int, int]:
    """Upsert into MongoDB by product_id. Returns (created, updated)."""
    created = updated = 0
    col = _get_collection()
    for product in products:
        write = col.update_one({"store_id": store_id, "product_id": product.product_id},
                               {"$set": product.model_dump()}, upsert=True)
        if write.upserted_id is not None:
            created += 1
        elif write.matched_count:
            updated += 1
    return created, updated


def upsert_platform_products(store_id: str, rows: list[dict]) -> list[str]:
    """Rows for one or a few products, exactly as the platform adapter produces them."""
    if not rows:
        return []
    industry_id = get_industry(store_id)
    keys = sorted({k for r in rows for k in r if not k.startswith("external_")})
    products = to_canonical_grouped(rows, map_columns(keys, industry_id), industry_id, store_id)
    save_products(store_id, products)
    upsert_products_to_qdrant(store_id, products)
    return [p.product_id for p in products]


def remove_platform_product(store_id: str, platform_id: str) -> list[str]:
    """Delete the product whose platform product/variant id matches. Never touches CSV products."""
    col = _get_collection()
    query = {"store_id": store_id, "$or": [{"external_parent_id": platform_id}, {"external_id": platform_id}]}
    ids = [d["product_id"] for d in col.find(query, {"_id": 0, "product_id": 1})]
    if ids:
        col.delete_many({"store_id": store_id, "product_id": {"$in": ids}})
        delete_products_from_qdrant(store_id, ids)
        logger.info("Platform product removed", extra={"removed_products": ids})
    return ids


def remove_missing_platform_products(store_id: str, fetched_ids: set[str]) -> list[str]:
    """
    After a FULL platform sync: remove platform-sourced products the platform no longer has.
    Skipped when the platform returned nothing, or when it would remove more than half of
    the store's platform products (protects against partial/broken API responses).
    """
    if not fetched_ids:
        logger.warning("Reconciliation skipped: platform returned no products")
        return []
    col = _get_collection()
    platform_ids = [d["product_id"] for d in col.find({"store_id": store_id, **_PLATFORM_SOURCED},
                                                      {"_id": 0, "product_id": 1})]
    missing = [pid for pid in platform_ids if pid not in fetched_ids]
    if not missing:
        return []
    if len(missing) > MAX_REMOVAL_SHARE * len(platform_ids):
        logger.warning("Reconciliation skipped: would remove too many products",
                       extra={"would_remove": len(missing), "platform_products": len(platform_ids)})
        return []
    col.delete_many({"store_id": store_id, "product_id": {"$in": missing}})
    delete_products_from_qdrant(store_id, missing)
    logger.info("Removed products deleted on the platform", extra={"removed_products": missing})
    return missing


def all_store_products(store_id: str) -> list[Product]:
    return fetch_products(store_id, _get_collection().distinct("product_id", {"store_id": store_id}))