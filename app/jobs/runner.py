"""
Scheduled-job runner — infrastructure-independent.

Triggered by a CLI (`python -m app.jobs <job>`) so any scheduler can run it
(AWS EventBridge Scheduler -> ECS task in production). Guarantees:
- No overlapping runs: MongoDB lease lock per job; a crashed run's lease simply expires.
- Idempotent re-triggers: each store is skipped if it succeeded within the job's interval.
- One store failing never stops the others; errors are recorded per store.
- Time budget: stops starting new stores when the budget is used; stores are processed
  least-recently-attempted first, so the next run continues where this one stopped.
- Every run is recorded (30 days).
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Callable

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

from app.config import settings

logger = logging.getLogger(__name__)

LEASE = timedelta(minutes=30)
RUN_RETENTION = timedelta(days=30)


@dataclass(frozen=True)
class JobSpec:
    name: str
    stores: Callable[[], list[str]]       # which stores this job applies to
    run_for_store: Callable[[str], str]   # returns a short outcome; raises on failure
    min_interval: timedelta               # skip a store that succeeded more recently than this


@dataclass
class RunSummary:
    job: str
    status: str = "succeeded"             # succeeded | partial | failed | skipped_locked
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    stopped_early: bool = False
    stores: list[dict] = field(default_factory=list)


@lru_cache(maxsize=1)
def _db():
    db = MongoClient(settings.MONGO_URI, tz_aware=True)[settings.MONGO_DB]
    db["job_runs"].create_index("expires_at", expireAfterSeconds=0)
    db["job_runs"].create_index([("job", 1), ("started_at", -1)])
    return db


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------- lease lock

def acquire_lock(job: str, owner: str, lease: timedelta = LEASE) -> bool:
    now = _now()
    try:
        _db()["job_locks"].update_one(
            {"_id": job, "$or": [{"locked_until": {"$lt": now}}, {"locked_until": {"$exists": False}}]},
            {"$set": {"locked_until": now + lease, "owner": owner, "acquired_at": now}},
            upsert=True,
        )
        return True
    except DuplicateKeyError:  # the lock document exists and is still held
        return False


def release_lock(job: str, owner: str) -> None:
    _db()["job_locks"].update_one({"_id": job, "owner": owner}, {"$set": {"locked_until": _now()}})


# ---------------------------------------------------------------- per-store state

def _state_key(job: str, store_id: str) -> str:
    return f"{job}:{store_id}"


def _ordered(job: str, store_ids: list[str]) -> list[str]:
    states = {d["_id"]: d for d in _db()["job_store_state"].find(
        {"_id": {"$in": [_state_key(job, s) for s in store_ids]}})}
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return sorted(dict.fromkeys(store_ids),
                  key=lambda s: states.get(_state_key(job, s), {}).get("last_attempt_at") or epoch)


def _recently_succeeded(job: str, store_id: str, interval: timedelta) -> bool:
    doc = _db()["job_store_state"].find_one({"_id": _state_key(job, store_id)}, {"last_success_at": 1})
    last = (doc or {}).get("last_success_at")
    return bool(last and last > _now() - interval)


def _mark(job: str, store_id: str, ok: bool, outcome: str | None, error: str | None) -> None:
    now = _now()
    update = {"last_attempt_at": now, "last_outcome": outcome, "last_error": error}
    if ok:
        update["last_success_at"] = now
    _db()["job_store_state"].update_one({"_id": _state_key(job, store_id)},
                                        {"$set": {**update, "job": job, "store_id": store_id}}, upsert=True)


# ---------------------------------------------------------------- run

def run_job(spec: JobSpec, *, budget: timedelta = timedelta(minutes=10), only_store: str | None = None) -> RunSummary:
    summary = RunSummary(job=spec.name)
    owner = uuid.uuid4().hex
    started = _now()
    if not acquire_lock(spec.name, owner):
        summary.status = "skipped_locked"
        logger.info("Job already running elsewhere; skipped", extra={"job": spec.name})
        _record(summary, started)
        return summary

    deadline = time.monotonic() + budget.total_seconds()
    try:
        stores = [only_store] if only_store else _ordered(spec.name, spec.stores())
        for store_id in stores:
            if time.monotonic() >= deadline:
                summary.stopped_early = True
                break
            if not only_store and _recently_succeeded(spec.name, store_id, spec.min_interval):
                summary.skipped += 1
                continue
            try:
                outcome = spec.run_for_store(store_id)
                _mark(spec.name, store_id, True, outcome, None)
                summary.processed += 1
                summary.stores.append({"store_id": store_id, "ok": True, "outcome": outcome})
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
                _mark(spec.name, store_id, False, None, error)
                summary.failed += 1
                summary.stores.append({"store_id": store_id, "ok": False, "error": error})
                logger.exception("Job failed for store", extra={"job": spec.name, "job_store": store_id})
    finally:
        release_lock(spec.name, owner)

    if summary.failed:
        summary.status = "failed" if not summary.processed else "partial"
    _record(summary, started)
    logger.info("Job finished", extra={"job": spec.name, "job_status": summary.status,
                                       "job_processed": summary.processed, "job_failed": summary.failed,
                                       "job_skipped": summary.skipped, "job_stopped_early": summary.stopped_early})
    return summary


def _record(summary: RunSummary, started: datetime) -> None:
    now = _now()
    _db()["job_runs"].insert_one({
        "job": summary.job, "status": summary.status, "started_at": started, "finished_at": now,
        "processed": summary.processed, "skipped": summary.skipped, "failed": summary.failed,
        "stopped_early": summary.stopped_early, "stores": summary.stores[:200],
        "expires_at": now + RUN_RETENTION,
    })