"""
Build (or rebuild) a store's embeddings in Qdrant Cloud.

WHY THIS EXISTS:
Reads all of a single store's data from MongoDB (products, policies, faq,
store_info), computes embeddings locally with sentence-transformers, and
pushes each point into a per-store Qdrant collection named after the
store_id. This is the multi-tenant successor to the old pickle-file
approach.

IDEMPOTENT:
Re-running for the same store fully recreates the Qdrant collection —
drops it if it exists, creates it fresh, re-inserts everything. No stale
data survives a rebuild.

CONTENT-BUILDING PATTERNS (kept from the old code — these tested well in
the RAG eval and shouldn't change):
  products    → "{name} by {brand} {description} Available sizes: ...
                 Category: ... Color: ... Price: ... Stock: ..."
  policies    → "{title}. {body}"
  faq         → "Q: {question} A: {answer}"
  store_info  → one flat sentence per meaningful field (name, address,
                 hours, shipping, payment methods, contacts)

HOW TO RUN:
    python -m app.rag.build_embeddings --store-id store_001
"""

import argparse
import logging
import sys
import uuid
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_VECTOR_SIZE = 384  # dimension of all-MiniLM-L6-v2 output


def _build_product_content(p: dict) -> str:
    sizes = ",".join(p.get("sizes") or [])
    return (
        f"{p.get('name', '')} by {p.get('brand', '')} "
        f"{p.get('description', '')} Available sizes: {sizes}. "
        f"Category: {p.get('category', '')}. "
        f"Color: {p.get('color', '')}. "
        f"Price: {p.get('price', '')}. "
        f"Stock: {p.get('stock', 0)} units."
    ).strip()


def _build_policy_content(pol: dict) -> str:
    return f"{pol.get('title', '')}. {pol.get('body', '')}".strip()


def _build_faq_content(f: dict) -> str:
    return f"Q: {f.get('question', '')} A: {f.get('answer', '')}".strip()


def _store_info_chunks(info: dict) -> list[dict]:
    """
    Turn the single store_info document into several separate points, one
    per meaningful field, so the retriever can match specific facts.
    Each chunk gets a distinct `source` id like store_address, store_hours.
    """
    chunks = []

    if info.get("name"):
        chunks.append({
            "source": "store_name",
            "content": f"store_name: {info['name']}",
        })

    if info.get("description"):
        chunks.append({
            "source": "store_description",
            "content": f"description: {info['description']}",
        })

    if info.get("address"):
        chunks.append({
            "source": "store_address",
            "content": info["address"],
        })

    hours = info.get("business_hours") or {}
    if hours:
        hours_str = ", ".join(f"{d}: {t}" for d, t in hours.items())
        chunks.append({
            "source": "store_business_hours",
            "content": f"business_hours: {hours_str}",
        })

    shipping = info.get("shipping") or {}
    if shipping:
        countries = shipping.get("countries", [])
        delivery = shipping.get("estimated_delivery", "")
        chunks.append({
            "source": "store_shipping",
            "content": f"shipping: countries: {countries}, estimated_delivery: {delivery}",
        })

    payment = info.get("payment_methods") or []
    if payment:
        chunks.append({
            "source": "store_payment_methods",
            "content": f"payment methods: {', '.join(payment)}",
        })

    contact_parts = []
    if info.get("phone"):     contact_parts.append(f"Phone: {info['phone']}")
    if info.get("email"):     contact_parts.append(f"Email: {info['email']}")
    if info.get("instagram"): contact_parts.append(f"Instagram: {info['instagram']}")
    if info.get("facebook"):  contact_parts.append(f"Facebook: {info['facebook']}")
    if contact_parts:
        chunks.append({
            "source": "store_contact",
            "content": "Store contact information: " + ". ".join(contact_parts) + ".",
        })

    return chunks


def _load_all_chunks_from_mongo(db, store_id: str) -> list[dict]:
    """
    Pull all of a store's data out of MongoDB and turn it into a flat list
    of documents ready to embed. Each item: content, source, type,
    plus name/price when applicable.
    """
    chunks = []

    for p in db["products"].find({"store_id": store_id}):
        chunks.append({
            "content": _build_product_content(p),
            "source": p["product_id"],
            "type": "product",
            "name": p.get("name"),
            "price": p.get("price"),
        })

    for pol in db["policies"].find({"store_id": store_id}):
        chunks.append({
            "content": _build_policy_content(pol),
            "source": pol["policy_id"],
            "type": "policy",
            "name": None,
            "price": None,
        })

    for f in db["faq"].find({"store_id": store_id}):
        chunks.append({
            "content": _build_faq_content(f),
            "source": f["faq_id"],
            "type": "faq",
            "name": None,
            "price": None,
        })

    info = db["store_info"].find_one({"store_id": store_id})
    if info:
        for c in _store_info_chunks(info):
            chunks.append({
                "content": c["content"],
                "source": c["source"],
                "type": "store_info",
                "name": None,
                "price": None,
            })

    return chunks


def _recreate_collection(qdrant: QdrantClient, collection_name: str) -> None:
    """Drop-and-create so rebuilds never leave stale points behind."""
    existing = [c.name for c in qdrant.get_collections().collections]
    if collection_name in existing:
        logger.info("  dropping existing collection: %s", collection_name)
        qdrant.delete_collection(collection_name)

    logger.info("  creating collection: %s (size=%d)", collection_name, _VECTOR_SIZE)
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
    )


def build(store_id: str) -> None:
    mongo = MongoClient(settings.MONGO_URI)
    db = mongo[settings.MONGO_DB]

    qdrant = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    collection_name = store_id

    logger.info("Loading chunks for store_id=%s from MongoDB...", store_id)
    chunks = _load_all_chunks_from_mongo(db, store_id)
    logger.info("  loaded %d chunks", len(chunks))

    if not chunks:
        logger.error("No chunks found for store_id=%s. Aborting.", store_id)
        sys.exit(1)

    logger.info("Loading embedding model: %s", _EMBEDDING_MODEL)
    model = SentenceTransformer(_EMBEDDING_MODEL)

    logger.info("Computing embeddings...")
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    logger.info("Preparing Qdrant collection: %s", collection_name)
    _recreate_collection(qdrant, collection_name)

    logger.info("Uploading %d points to Qdrant...", len(chunks))
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i].tolist(),
            payload={
                "content": chunk["content"],
                "source": chunk["source"],
                "type": chunk["type"],
                "name": chunk["name"],
                "price": chunk["price"],
                "store_id": store_id,
            },
        )
        for i, chunk in enumerate(chunks)
    ]
    qdrant.upsert(collection_name=collection_name, points=points)

    logger.info("Done. store_id=%s now has %d searchable points.", store_id, len(points))


def main():
    parser = argparse.ArgumentParser(description="Build a store's embeddings in Qdrant.")
    parser.add_argument("--store-id", required=True, help="Store identifier, e.g. store_001")
    args = parser.parse_args()

    build(store_id=args.store_id)


if __name__ == "__main__":
    main()