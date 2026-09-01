"""
Runtime semantic search against Qdrant Cloud (multi-tenant).

WHY THIS EXISTS:
Each store's data lives in its own Qdrant collection, named after the
store_id. This module takes a store_id + query, computes the query
embedding locally with the same model used at build time, searches ONLY
that store's collection, and returns matches above the similarity
threshold. Cross-store leaks are impossible by design — a Store A query
never sees Store B's collection.

KEPT FROM THE OLD RETRIEVER:
  - MIN_SIMILARITY_SCORE = 0.35 (validated in the RAG evaluation)
  - RetrievedChunk shape unchanged (content, source, score, name, price)
  - Fail-graceful: any error returns [] instead of raising
  - The embedding model is loaded once and cached (lru_cache)

CHANGED:
  - No more local pickle file; corpus lives in Qdrant Cloud
  - Signature now requires store_id as the first argument
"""

import logging
from functools import lru_cache

import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.schemas.models import RetrievedChunk


logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_TOP_K = 3
MIN_SIMILARITY_SCORE = 0.35
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load the embedding model once per process."""
    logger.info("Loading embedding model: %s", _EMBEDDING_MODEL)
    return SentenceTransformer(_EMBEDDING_MODEL, device=DEVICE)


@lru_cache(maxsize=1)
def _qdrant() -> QdrantClient:
    """One Qdrant client, reused for every query."""
    return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)


def retrieve_context(
    store_id: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievedChunk]:
    """
    Search the specified store's Qdrant collection for the top-k
    documents matching the query, filtered by similarity threshold.
    """
    if query is None:
        return []

    query = query.strip()
    if not query:
        return []

    if not store_id:
        logger.warning("retrieve_context called without a store_id")
        return []

    try:
        model = _load_model()
        client = _qdrant()

        query_embedding = model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        top_k = max(1, top_k)

        search_results = client.query_points(
            collection_name=store_id,
            query=query_embedding,
            limit=top_k,
        ).points

        retrieved_chunks: list[RetrievedChunk] = []
        for hit in search_results:
            score = float(hit.score)
            if score < MIN_SIMILARITY_SCORE:
                continue

            payload = hit.payload or {}
            retrieved_chunks.append(
                RetrievedChunk(
                    content=payload.get("content", ""),
                    source=payload.get("source", ""),
                    score=score,
                    name=payload.get("name"),
                    price=payload.get("price"),
                )
            )

        logger.info("Retrieved %d chunks for store_id=%s", len(retrieved_chunks), store_id)
        return retrieved_chunks

    except Exception:
        logger.exception("Retriever failed for store_id=%s", store_id)
        return []