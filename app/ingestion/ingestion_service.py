"""
Orchestrates CSV/Excel ingestion end-to-end.
Adapter → column mapping → per-row canonical conversion → MongoDB upsert.
Returns a summary the merchant can act on.
"""

import logging
from typing import Any
from unittest import result

from pydantic import BaseModel, Field

from app.ingestion.canonical_converter import to_canonical_grouped
from app.ingestion.deterministic_mapper import map_columns
from app.ingestion.source_adapter import SourceAdapter
from app.products.product_store import _get_collection
from app.rag.indexer import sync_products_to_qdrant
from app.ingestion.ai_column_mapper import ai_map_columns
from datetime import datetime, timezone
from app.ingestion.history import record_run



logger = logging.getLogger(__name__)


class IngestionResult(BaseModel):
    """Merchant-facing summary of an ingestion run."""

    created: int = 0
    updated: int = 0
    failed: int = 0
    total_rows: int = 0
    unmapped_columns: list[str] = Field(default_factory=list)
    conflict_columns: dict[str, list[str]] = Field(default_factory=dict)
    source: str = ""


def _ingest_products_impl(
    adapter: SourceAdapter,
    store_id: str,
    industry_id: str,
) -> IngestionResult:
    """
    Run full ingestion pipeline. Upserts by product_id.
    """
    result = IngestionResult(source=adapter.get_source_name())

    # 1. Fetch raw rows first (needed to know actual columns)
    rows = adapter.fetch_raw_products(store_id)
    result.total_rows = len(rows)
    
    if not rows:
        return result

    # 2. Build column mapping from union of keys across all rows
    # (adapters like WC produce dynamic keys per product via flattened attributes)
    
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    
    
    all_keys = {k for k in all_keys if not k.startswith("external_")}
    mapping = map_columns(sorted(all_keys), industry_id)
    result.unmapped_columns = mapping.unmapped
    result.conflict_columns = mapping.conflicts
    
    # AI-assisted mapping: try to resolve unmapped columns via LLM
    if mapping.unmapped and industry_id is not None and adapter.get_source_name() == "csv":
        ai_mappings = ai_map_columns(mapping.unmapped, rows, industry_id)
        if ai_mappings:
            builtin_targets = {"product_id", "name", "description", "price", "category", "image_url"}
            # Add AI-resolved mappings into the existing mapping
            for src, tgt in ai_mappings.items():
                
                if tgt in mapping.builtin.values():
                    continue  # never override an already-mapped built-in
                
                if tgt in builtin_targets:
                    mapping.builtin[src] = tgt
                else:
                    mapping.mapped[src] = tgt
                    
            # Remove AI-resolved columns from the unmapped list
            mapping.unmapped = [c for c in mapping.unmapped if c not in ai_mappings]
            result.unmapped_columns = mapping.unmapped
            logger.info(
                "AI mapping resolved %d additional columns",
                len(ai_mappings),
                extra={"ai_mapped_columns": list(ai_mappings.keys())},
            )
    

    
    # 3. Group rows into Products (handling variants) then upsert
    products = to_canonical_grouped(rows, mapping, industry_id, store_id)
    
    # Failed = rows that didn't produce any Product
    # (e.g. missing product_id/name/price). Rough estimate.
    result.failed = max(0, len(rows) - sum(max(len(p.variants), 1) for p in products))

    col = _get_collection()
    for product in products:
        doc = product.model_dump()
        write = col.update_one(
             {"store_id": store_id, "product_id": product.product_id},
             {"$set": doc},
            upsert=True,
        )
        
        if write.upserted_id is not None:
            result.created += 1
        elif write.matched_count > 0:
            result.updated += 1
    
    
    # Sync to Qdrant — RAG must reflect the catalog
    qdrant_result = sync_products_to_qdrant(store_id, products)
    logger.info(
        "Qdrant sync during ingestion",
        extra={
            "qdrant_upserted": qdrant_result["upserted"],
            "qdrant_deleted": qdrant_result["deleted"],
        },
    )
    
    
      
    logger.info(
    "Ingestion complete",
    extra={
        "ingest_store_id": store_id,
        "ingest_source": result.source,
        "ingest_created": result.created,
        "ingest_updated": result.updated,
        "ingest_failed": result.failed,
        "ingest_unmapped": len(result.unmapped_columns),
    },
)
    return result



def ingest_products(adapter, store_id: str, *args, **kwargs):
    """Every import is recorded in the sync history (success or failure)."""
    started = datetime.now(timezone.utc)
    source = adapter.get_source_name()
    try:
        result = _ingest_products_impl(adapter, store_id, *args, **kwargs)
    except Exception as exc:
        record_run(store_id, source, started, error=f"{type(exc).__name__}: {exc}")
        raise
    record_run(store_id, source, started, result=result)
    return result