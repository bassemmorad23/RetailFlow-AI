"""Jobs: lease lock, idempotent re-runs, per-store isolation, time budget, CLI exit codes, job logic."""

import types
import uuid
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from app.jobs import __main__ as cli
from app.jobs import runner, tasks
from app.jobs.runner import JobSpec, acquire_lock, release_lock, run_job
from app.settings.store_credentials import _get_collection as creds_col
from app.settings.store_credentials import get_credentials


def _spec(stores, fail=(), name=None):
    calls = []

    def run(store_id):
        calls.append(store_id)
        if store_id in fail:
            raise RuntimeError("boom")
        return "ok"
    return JobSpec(name or f"job_{uuid.uuid4().hex[:6]}", lambda: stores, run, timedelta(hours=1)), calls


# ---------------------------------------------------------------- runner

def test_lease_lock():
    job = f"lock_{uuid.uuid4().hex[:6]}"
    assert acquire_lock(job, "a") is True
    assert acquire_lock(job, "b") is False
    release_lock(job, "a")
    assert acquire_lock(job, "b") is True
    assert acquire_lock(f"{job}_x", "c", lease=timedelta(seconds=-1)) is True   # an expired lease...
    assert acquire_lock(f"{job}_x", "d") is True                                # ...is taken over (crash-safe)


def test_rerun_is_idempotent_and_failures_isolated():
    spec, calls = _spec(["s1", "s2", "s3"], fail={"s2"})
    first = run_job(spec)
    assert (first.status, first.processed, first.failed) == ("partial", 2, 1)
    second = run_job(spec)                       # immediate re-trigger (e.g. scheduler retry)
    assert (second.processed, second.skipped, second.failed) == (0, 2, 1)   # only the failed store is retried
    assert calls == ["s1", "s2", "s3", "s2"]
    assert runner._db()["job_runs"].count_documents({"job": spec.name}) == 2


def test_only_store_ignores_interval():
    spec, calls = _spec(["s1"])
    run_job(spec)
    assert run_job(spec, only_store="s1").processed == 1 and calls == ["s1", "s1"]


def test_time_budget_and_all_failed():
    spec, calls = _spec(["s1", "s2"])
    s = run_job(spec, budget=timedelta(seconds=0))
    assert s.stopped_early and calls == []
    spec, _ = _spec(["s1"], fail={"s1"})
    assert run_job(spec).status == "failed"


def test_locked_job_is_skipped():
    spec, calls = _spec(["s1"])
    acquire_lock(spec.name, "someone-else")
    assert run_job(spec).status == "skipped_locked" and calls == []


def test_cli_exit_codes(monkeypatch):
    ok, _ = _spec(["s1"], name="ok_job")
    partial, _ = _spec(["s1", "s2"], fail={"s2"}, name="partial_job")
    monkeypatch.setattr(cli, "JOBS", {"ok_job": ok, "partial_job": partial})
    assert cli.main(["ok_job"]) == 0
    assert cli.main(["partial_job"]) == 1
    assert cli.main(["nope"]) == 3


# ---------------------------------------------------------------- Instagram token refresh

def _ig_store(**extra):
    sid = f"store_job_{uuid.uuid4().hex[:8]}"
    creds_col().insert_one({"store_id": sid, "source": "instagram", "credentials": {
        "access_token": "OLD", "instagram_business_account_id": "1", "username": "shop", **extra}})
    return sid


def _client(response):
    return httpx.Client(transport=httpx.MockTransport(lambda r: response))


NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)


def test_token_not_due():
    sid = _ig_store(token_expires_at=(NOW + timedelta(days=40)).isoformat())
    assert tasks.refresh_instagram_token(sid, client=_client(httpx.Response(500)), now=NOW) == "not due"


def test_token_refreshed_and_expiry_saved():
    sid = _ig_store()                                                       # unknown expiry -> refresh
    ok = httpx.Response(200, json={"access_token": "NEW", "token_type": "bearer", "expires_in": 5184000})
    assert "refreshed" in tasks.refresh_instagram_token(sid, client=_client(ok), now=NOW)
    creds = get_credentials(sid, "instagram")
    assert creds["access_token"] == "NEW" and creds["token_status"] == "ok"
    assert creds["token_expires_at"].startswith("2026-11-25")               # +60 days


def test_refused_refresh_flags_reconnect_only_when_urgent():
    soon = _ig_store(token_expires_at=(NOW + timedelta(days=3)).isoformat())
    with pytest.raises(RuntimeError):
        tasks.refresh_instagram_token(soon, client=_client(httpx.Response(400, json={"error": {}})), now=NOW)
    assert get_credentials(soon, "instagram")["token_status"] == "needs_reconnect"

    later = _ig_store(token_expires_at=(NOW + timedelta(days=12)).isoformat())
    with pytest.raises(RuntimeError):
        tasks.refresh_instagram_token(later, client=_client(httpx.Response(400, json={"error": {}})), now=NOW)
    assert "token_status" not in get_credentials(later, "instagram")        # retried tomorrow, no alarm yet


# ---------------------------------------------------------------- sync_products (reconciliation)

def test_sync_store_products_runs_sync_and_webhooks(monkeypatch):
    sid = f"store_job_{uuid.uuid4().hex[:8]}"
    creds_col().insert_one({"store_id": sid, "source": "shopify", "credentials": {"shop": "t.myshopify.com", "access_token": "x"}})
    monkeypatch.setattr(tasks, "get_industry", lambda s: "fashion")
    monkeypatch.setattr(tasks, "ingest_products",
                        lambda adapter, s, industry: types.SimpleNamespace(created=1, updated=2))
    monkeypatch.setattr(tasks, "ensure_shopify_webhooks", lambda s: {"ok": True, "created": []})
    assert tasks.sync_store_products(sid) == "shopify: 1 new, 2 updated, webhooks ok"
    assert sid in tasks._stores_with("shopify", "woocommerce")

    monkeypatch.setattr(tasks, "ensure_shopify_webhooks", lambda s: {"ok": False, "error": "HTTPError"})
    with pytest.raises(RuntimeError):
        tasks.sync_store_products(sid)


def test_expired_instagram_shows_in_needs_attention(monkeypatch):
    from app.analytics import metrics
    monkeypatch.setattr(metrics, "check_usage", lambda sid: types.SimpleNamespace(
        monthly_used=0, monthly_limit=100, daily_used=0, daily_cap=10, period_start="2026-09-01"))
    sid = _ig_store(token_status="needs_reconnect")
    assert any(i["type"] == "instagram_reconnect" and i["severity"] == "high" for i in metrics.needs_attention(sid))