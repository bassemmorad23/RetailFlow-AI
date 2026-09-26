"""
Product sync history: every import (CSV upload, Shopify, WooCommerce) is recorded
— when, source, result counts, or the error. Kept 90 days. Owner-only API.
"""

from datetime import datetime, timedelta, timezone
from functools import lru_cache

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from pymongo import DESCENDING, MongoClient
from pymongo.collection import Collection

from app.auth.dependencies import require_store_member
from app.config import settings

RETENTION = timedelta(days=90)
router = APIRouter(prefix="/stores/{store_id}/syncs", tags=["products"])


@lru_cache(maxsize=1)
def _col() -> Collection:
    col = MongoClient(settings.MONGO_URI, tz_aware=True)[settings.MONGO_DB]["sync_runs"]
    col.create_index([("store_id", 1), ("started_at", DESCENDING)])
    col.create_index("expires_at", expireAfterSeconds=0)
    return col


def record_run(store_id: str, source: str, started_at: datetime, *, result=None, error: str | None = None) -> None:
    now = datetime.now(timezone.utc)
    doc = {
        "store_id": store_id, "source": source, "started_at": started_at, "finished_at": now,
        "duration_ms": int((now - started_at).total_seconds() * 1000),
        "status": "failed" if error else "succeeded", "error": error[:300] if error else None,
        "created": getattr(result, "created", 0) if result else 0,
        "updated": getattr(result, "updated", 0) if result else 0,
        "failed": getattr(result, "failed", 0) if result else 0,
        "total_rows": getattr(result, "total_rows", 0) if result else 0,
        "unmapped_columns": list(getattr(result, "unmapped_columns", []) or []) if result else [],
        "expires_at": now + RETENTION,
    }
    _col().insert_one(doc)


class SyncRun(BaseModel):
    source: str
    status: str
    started_at: datetime
    finished_at: datetime
    duration_ms: int
    created: int
    updated: int
    failed: int
    total_rows: int
    unmapped_columns: list[str] = []
    error: str | None = None


class SyncHistory(BaseModel):
    runs: list[SyncRun]


@router.get("", response_model=SyncHistory)
def list_syncs(store_id: str = Depends(require_store_member), limit: int = Query(default=20, ge=1, le=100)) -> dict:
    runs = _col().find({"store_id": store_id}, {"_id": 0, "store_id": 0, "expires_at": 0}) \
        .sort([("started_at", DESCENDING), ("_id", DESCENDING)]).limit(limit)
    return {"runs": list(runs)}