"""
Sync products from MongoDB → Qdrant.

Called at end of every ingestion. Ensures RAG stays in sync with catalog.

STRATEGY:
- One Qdrant point per Product (variants folded into embedding text)
- Deterministic UUID from (store_id, product_id) — upserts, no duplicates
- Snapshot diff: delete Qdrant points that no longer exist in Mongo
- Failures logged, not raised — ingestion succeeds even if Qdrant fails
"""

import logging
import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings
from app.rag.retriever import _load_model
from app.schemas.models import Product


logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # any fixed UUID


@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)


def _point_id(store_id: str, product_id: str) -> str:
    """Deterministic UUID from store+product. Same inputs → same UUID."""
    return str(uuid.uuid5(_NAMESPACE, f"{store_id}:{product_id}"))


def _build_text(product: Product) -> str:
    """
    Full-text embedding input: name + description + attributes + specs +
    variant attributes (so search matches 'size L cotton' on parent product).
    """
    parts = [product.name]
    if product.description:
        parts.append(product.description)

    for k, v in {**product.attributes, **product.specifications}.items():
        parts.append(f"{k}: {v}")

    # Fold variant attrs (unique values only — avoid noise)
    if product.variants:
        variant_keys: dict[str, set] = {}
        for v in product.variants:
            for k, val in {**v.attributes, **v.specifications}.items():
                variant_keys.setdefault(k, set()).add(str(val))
        for k, vals in variant_keys.items():
            parts.append(f"Available {k}: {', '.join(sorted(vals))}")

    return ". ".join(str(p) for p in parts if p)


def _ensure_collection(store_id: str) -> None:
    """Create Qdrant collection for store if it doesn't exist."""
    c = _client()
    existing = {col.name for col in c.get_collections().collections}
    if store_id in existing:
        return
    c.create_collection(
        collection_name=store_id,
        vectors_config=qmodels.VectorParams(
            size=_EMBEDDING_DIM,
            distance=qmodels.Distance.COSINE,
        ),
    )
    logger.info("Created Qdrant collection: %s", store_id)


def sync_products_to_qdrant(store_id: str, products: list[Product]) -> dict:
    """
    Upsert all products to Qdrant + delete orphans (products in Qdrant
    that are no longer in the given list).

    Returns {'upserted': N, 'deleted': N} for logging.
    Never raises — errors logged, ingestion continues.
    """
    result = {"upserted": 0, "deleted": 0}

    try:
        _ensure_collection(store_id)
        c = _client()
        model = _load_model()

        # 1. Upsert current products
        if products:
            texts = [_build_text(p) for p in products]
            vectors = model.encode(texts, normalize_embeddings=True).tolist()

            points = [
                qmodels.PointStruct(
                    id=_point_id(store_id, p.product_id),
                    vector=vec,
                    payload={
                        "source": p.product_id,
                        "content": text,
                        "name": p.name,
                        "price": p.price,
                    },
                )
                for p, text, vec in zip(products, texts, vectors)
            ]
            c.upsert(collection_name=store_id, points=points)
            result["upserted"] = len(points)

        # 2. Delete orphans (in Qdrant but not in current products)
        current_ids = {_point_id(store_id, p.product_id) for p in products}
        # Scroll all existing point IDs in the collection
        all_qdrant_ids: set[str] = set()
        offset = None
        while True:
            batch, offset = c.scroll(
                collection_name=store_id,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            all_qdrant_ids.update(str(pt.id) for pt in batch)
            if offset is None:
                break

        orphans = all_qdrant_ids - current_ids
        if orphans:
            c.delete(
                collection_name=store_id,
                points_selector=qmodels.PointIdsList(points=list(orphans)),
            )
            result["deleted"] = len(orphans)

        logger.info(
            "Qdrant sync complete for %s: %d upserted, %d deleted",
            store_id, result["upserted"], result["deleted"],
        )

    except Exception:
        logger.exception("Qdrant sync failed for store %s", store_id)
        # Best-effort — don't raise

    return result