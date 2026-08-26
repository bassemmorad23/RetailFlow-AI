"""
One-off utility: dump the full contents of the built RAG index.

WHY THIS EXISTS:
Building a RAG evaluation set requires knowing exactly what documents are
actually indexed — guessing at the catalog contents would produce ground
truth that doesn't match reality, making the eval worse than useless. This
script prints every (source, content) pair currently in
data/embeddings/store_embeddings.pkl so a real evaluation set can be built
against the real corpus, not assumptions about it.

HOW TO RUN:
    python -m eval.dump_index

This is a read-only inspection tool, not part of the app or the eval
suite itself — safe to delete after use, or keep for future re-inspection
whenever the catalog changes.
"""

import pickle
from pathlib import Path

_EMBEDDINGS_PATH = Path(__file__).parent.parent / "data" / "embeddings" / "store_embeddings.pkl"


def dump_index() -> None:
    with open(_EMBEDDINGS_PATH, "rb") as f:
        corpus = pickle.load(f)

    docs = corpus["docs"]
    print(f"Model used to build this index: {corpus['model_name']}")
    print(f"Total documents indexed: {len(docs)}\n")

    for i, doc in enumerate(docs, start=1):
        print(f"{i}. source: {doc['source']}")
        print(f"   content: {doc['content']}")
        print()


if __name__ == "__main__":
    dump_index()