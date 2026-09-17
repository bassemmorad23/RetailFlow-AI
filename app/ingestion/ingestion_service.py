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


def ingest_products(
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
        
    mapping = map_columns(sorted(all_keys), industry_id)
    result.unmapped_columns = mapping.unmapped
    result.conflict_columns = mapping.conflicts
    

    
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