"""
Product lookup from MongoDB, scoped by store_id.

WHY THIS EXISTS:
The recommender needs full Product data (specs, attributes) to apply
hard filters and soft ranking. RAG only returns lightweight
RetrievedChunks (name + price). This module fills the gap: given
product_ids from RAG, fetch the full Products from MongoDB.
"""

from functools import lru_cache
from typing import Iterable
from pymongo import MongoClient
from pymongo.collection import Collection

from app.config import settings
from app.schemas.models import Product


@lru_cache(maxsize=1)
def _get_collection() -> Collection:
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB]
    return db["products"]


def fetch_products(store_id: str, product_ids: Iterable[str]) -> list[Product]:
    """
    Fetch full Product records by IDs, scoped to the store.
    Products not found are silently skipped.
    """
    ids = list(product_ids)
    if not ids:
        return []

    col = _get_collection()
    docs = col.find({"store_id": store_id, "product_id": {"$in": ids}})

    products: list[Product] = []
    for doc in docs:
        doc.pop("_id", None)
        # Filter to only Product schema fields — MongoDB may have extras
        # (like 'sizes', 'brand' from legacy schema) that Product doesn't
        # know about yet.
        allowed = set(Product.model_fields.keys())
        clean = {k: v for k, v in doc.items() if k in allowed}
        # Ensure defaults for required-but-optional-with-default fields
        try:
            products.append(Product(**clean))
        except Exception:
            # Malformed product — skip rather than crash the pipeline
            continue

    return products