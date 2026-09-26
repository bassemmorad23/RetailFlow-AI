"""
Products API for the merchant dashboard (owner only).

- List: name search, category and stock filters, page-based (catalogs are small).
- Detail: full product with variants and stock; platform ids stay internal.
- Delete: CSV stores only (on Shopify/WooCommerce the next sync would bring the
  product back — the merchant manages it there). Removes from MongoDB, then
  re-runs the search-index snapshot sync, which drops the product's vector.
"""

import logging
import re
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.auth.dependencies import require_store_member
from app.commerce.stock import store_platform
from app.products.product_store import _get_collection, fetch_products
from app.rag.indexer import sync_products_to_qdrant
from app.settings.store_settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/stores/{store_id}/products", tags=["products"])

_INTERNAL = {"external_id", "external_parent_id", "store_id", "stock_available"}
_SUMMARY_FIELDS = {"_id": 0, "product_id": 1, "name": 1, "price": 1, "category": 1,
                   "stock_status": 1, "image_url": 1, "variants.sku": 1}


class ProductSummary(BaseModel):
    product_id: str
    name: str
    price: float
    currency: str
    category: str | None = None
    stock_status: str = "unknown"
    variants_count: int = 0
    image_url: str | None = None


class ProductPage(BaseModel):
    products: list[ProductSummary]
    page: int
    limit: int
    total: int


@router.get("", response_model=ProductPage)
def list_products(
    store_id: str = Depends(require_store_member),
    q: str | None = Query(default=None, max_length=100),
    category: str | None = Query(default=None, max_length=100),
    stock: Literal["in_stock", "out_of_stock", "unknown"] | None = None,
    page: int = Query(default=1, ge=1, le=1000),
    limit: int = Query(default=25, ge=1, le=100),
) -> dict:
    query: dict = {"store_id": store_id}
    if q and q.strip():
        query["name"] = {"$regex": re.escape(q.strip()), "$options": "i"}
    if category:
        query["category"] = category
    if stock == "unknown":
        query["stock_status"] = {"$in": ["unknown", None]}
    elif stock:
        query["stock_status"] = stock

    col = _get_collection()
    currency = get_settings(store_id).currency or "EGP"
    docs = col.find(query, _SUMMARY_FIELDS).sort([("name", 1), ("product_id", 1)]).skip((page - 1) * limit).limit(limit)
    products = [{
        "product_id": d["product_id"], "name": d["name"], "price": d["price"], "currency": currency,
        "category": d.get("category"), "stock_status": d.get("stock_status") or "unknown",
        "variants_count": len(d.get("variants") or []), "image_url": d.get("image_url"),
    } for d in docs]
    return {"products": products, "page": page, "limit": limit, "total": col.count_documents(query)}


@router.get("/{product_id}")
def get_product(product_id: str, store_id: str = Depends(require_store_member)) -> dict:
    found = fetch_products(store_id, [product_id])
    if not found:
        raise HTTPException(status_code=404, detail="Product not found")
    data = found[0].model_dump(exclude=_INTERNAL)
    for v in data.get("variants") or []:
        v.pop("external_id", None)
        v.pop("stock_available", None)
    data["currency"] = get_settings(store_id).currency or "EGP"
    return data


@router.delete("/{product_id}", status_code=204)
def delete_product(product_id: str, store_id: str = Depends(require_store_member)) -> None:
    platform = store_platform(store_id)
    if platform:
        raise HTTPException(status_code=409, detail={
            "code": "managed_by_platform",
            "message": f"This product comes from {platform.capitalize()}. Delete it there; the next sync removes it here.",
        })
    result = _get_collection().delete_one({"store_id": store_id, "product_id": product_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Product not found")

    col = _get_collection()
    remaining = fetch_products(store_id, col.distinct("product_id", {"store_id": store_id}))
    try:
        sync_products_to_qdrant(store_id, remaining)
    except Exception:  # MongoDB is the source of truth; the next sync repairs the index
        logger.exception("Search index refresh after delete failed", extra={"deleted_product": product_id})