import logging
import pickle
from functools import lru_cache
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util

from app.schemas.models import RetrievedChunk


logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_TOP_K = 3
MIN_SIMILARITY_SCORE = 0.35

EMBEDDINGS_PATH = (
    Path(__file__).parent.parent.parent
    / "data"
    / "embeddings"
    / "store_embeddings.pkl"
)


@lru_cache(maxsize=1)
def _load_corpus() -> dict:
    """
    Load the precomputed embedding corpus from disk.
    """

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {EMBEDDINGS_PATH}"
        )

    with open(EMBEDDINGS_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _load_model() :
    """
    Load the SentenceTransformer model once.
    """

    corpus = _load_corpus()
    
    

    logger.info("Loading embedding model: %s", corpus["model_name"])

    return SentenceTransformer(
        corpus["model_name"],
        device=DEVICE,
    )


def retrieve_context(
    query: str,
    top_k: int = DEFAULT_TOP_K,):

    """
    Retrieve the most relevant documents using cosine similarity.
    """

    if query is None:
        return []

    query = query.strip()

    if not query:
        return []

    try:
        corpus = _load_corpus()
        model = _load_model()

        top_k = max(1, top_k)
        top_k = min(top_k, len(corpus["docs"]))

        query_embedding = model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        
        
        print(query_embedding.device)
        

        similarity_scores = util.cos_sim(
            query_embedding,
            corpus["embeddings"].to(DEVICE),
        )[0]

        top_results = similarity_scores.topk(k=top_k)

        retrieved_chunks = []

        for score, idx in zip(
            top_results.values,
            top_results.indices,
        ):

            score = float(score)

            if score < MIN_SIMILARITY_SCORE:
                continue

            document = corpus["docs"][int(idx)]

            retrieved_chunks.append(
                RetrievedChunk(
                    content=document["content"],
                    source=document["source"],
                    score=score,
                    name=document.get("name"),
                    price=document.get("price"),
                )
            )

        logger.info(
            "Retrieved %d chunks for query.",
            len(retrieved_chunks),
        )

        return retrieved_chunks

    except Exception:
        logger.exception("Retriever failed.")
        return []